"""Parallel audiobook generation across multiple GPUs."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from pydub import AudioSegment

from .vastai import VastAIInstance, VastAIManager


@dataclass
class SegmentRange:
    """A range of segments assigned to a GPU."""

    start: int
    end: int  # inclusive

    @property
    def count(self) -> int:
        return self.end - self.start + 1

    @property
    def estimated_hours(self) -> float:
        return (self.count * 25) / 3600  # ~25 sec per segment


class ParallelOrchestrator:
    """Orchestrates parallel audiobook generation across multiple GPUs."""

    def __init__(
        self,
        book_id: str,
        output_dir: Path | None = None,
        verify: bool = True,
        whisper_model: str = "base",
    ):
        self.book_id = book_id
        self.output_dir = output_dir or Path("books") / book_id / "audio"
        self.verify = verify
        self.whisper_model = whisper_model
        self.manager = VastAIManager()

    def get_segment_count(self) -> int:
        """Get total segment count from book info."""
        result = subprocess.run(
            ["uv", "run", "audiobook", "info", "--book", self.book_id],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get book info: {result.stderr}")

        # Parse "Total: X chapters, Y segments, Z characters"
        import re

        match = re.search(r"Total:.*?(\d+)\s+segments", result.stdout.replace(",", ""))
        if match:
            return int(match.group(1))
        raise RuntimeError("Could not parse segment count from book info")

    def distribute_segments(self, total: int, gpu_count: int) -> list[SegmentRange]:
        """Distribute segments evenly across GPUs."""
        per_gpu = total // gpu_count
        remainder = total % gpu_count

        ranges = []
        start = 0
        for i in range(gpu_count):
            count = per_gpu + (1 if i < remainder else 0)
            if count > 0:
                ranges.append(SegmentRange(start=start, end=start + count - 1))
                start += count
        return ranges

    def run_on_instance(
        self, instance: VastAIInstance, seg_range: SegmentRange
    ) -> tuple[bool, str | None]:
        """Run generation for a segment range on an instance."""
        verify_flag = "--verify" if self.verify else "--no-verify"
        cmd = (
            f"cd /workspace/audiobook && source ~/.local/bin/env && "
            f"uv run audiobook generate --book {self.book_id} "
            f"--segment-range {seg_range.start}-{seg_range.end} {verify_flag}"
        )

        logger.info(
            f"[{instance.instance_id}] Generating segments {seg_range.start}-{seg_range.end}"
        )

        try:
            timeout = max(1800, seg_range.count * 35)  # ~35s per segment with buffer
            result = instance.run_ssh(f"bash -c '{cmd}'", timeout=timeout, check=False)
            if result.returncode == 0:
                return True, None
            return False, result.stderr[:300] if result.stderr else "Unknown error"
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)

    def download_and_combine(self, instances: list[VastAIInstance]) -> Path:
        """Download segments from all instances and combine into audiobook."""
        segments_dir = self.output_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        # Download from each instance
        for inst in instances:
            try:
                inst.scp_download_dir(
                    f"/workspace/audiobook/books/{self.book_id}/audio/segments/",
                    segments_dir.parent,
                )
            except Exception as e:
                logger.error(f"Download failed from {inst.instance_id}: {e}")

        # Load manifests and group by chapter
        manifests = []
        for f in segments_dir.glob("manifest_*.json"):
            manifests.extend(json.load(open(f)))
        manifests.sort(key=lambda m: m["global_index"])

        chapters: dict[int, list] = {}
        for seg in manifests:
            chapters.setdefault(seg["chapter"], []).append(seg)

        # Combine segments into chapters
        part_dir = self.output_dir / "partie_1"
        part_dir.mkdir(parents=True, exist_ok=True)
        silence = AudioSegment.silent(duration=500)

        for ch_num, segs in sorted(chapters.items()):
            combined = None
            for seg in sorted(segs, key=lambda s: s["global_index"]):
                audio = AudioSegment.from_file(segments_dir / seg["file"])
                combined = audio if combined is None else combined + silence + audio
            if combined:
                combined.export(str(part_dir / f"chapitre_{ch_num}_full.mp3"), format="mp3")

        # Combine chapters into final audiobook
        chapter_files = sorted(
            part_dir.glob("chapitre_*_full.mp3"), key=lambda f: int(f.stem.split("_")[1])
        )
        final = AudioSegment.from_file(chapter_files[0])
        for f in chapter_files[1:]:
            final += AudioSegment.silent(duration=1000) + AudioSegment.from_file(f)

        final_path = self.output_dir / "audiobook_complete.mp3"
        final.export(str(final_path), format="mp3", bitrate="192k")
        return final_path

    def run(
        self,
        gpu_count: int = 20,
        gpu_type: str = "RTX_4090",
        max_cost: float = 0.40,
        keep_instances: bool = False,
    ) -> Path:
        """Run parallel generation across multiple GPUs.

        Returns path to final audiobook.
        """
        total_segments = self.get_segment_count()
        ranges = self.distribute_segments(total_segments, gpu_count)

        logger.info(f"Distributing {total_segments} segments across {len(ranges)} GPUs")
        for i, r in enumerate(ranges):
            logger.info(f"  GPU {i+1}: segments {r.start}-{r.end} ({r.count} segs)")

        # Rent and setup instances
        instances = self.manager.rent_instances(len(ranges), gpu_type, max_cost)
        ready = self.manager.wait_for_ready(instances, timeout=600)

        if not ready:
            self.manager.destroy_all()
            raise RuntimeError("No instances became ready")

        # Setup in parallel
        with ThreadPoolExecutor(max_workers=len(ready)) as ex:
            setup_ok = [
                inst for inst, ok in zip(ready, ex.map(self.manager.setup_instance, ready)) if ok
            ]

        if not setup_ok:
            self.manager.destroy_all()
            raise RuntimeError("No instances set up successfully")

        # Redistribute if we lost instances
        if len(setup_ok) < len(ranges):
            ranges = self.distribute_segments(total_segments, len(setup_ok))

        # Run generation in parallel
        results = []
        with ThreadPoolExecutor(max_workers=len(setup_ok)) as ex:
            futures = {
                ex.submit(self.run_on_instance, inst, r): (inst, r)
                for inst, r in zip(setup_ok, ranges)
            }
            for future in as_completed(futures):
                inst, r = futures[future]
                success, error = future.result()
                results.append({"range": f"{r.start}-{r.end}", "success": success, "error": error})
                if not success:
                    logger.error(f"Failed {r.start}-{r.end}: {error}")

        # Download and combine
        final_path = self.download_and_combine(setup_ok)

        if not keep_instances:
            self.manager.destroy_all()

        succeeded = sum(1 for r in results if r["success"])
        logger.info(f"Complete: {succeeded}/{len(results)} succeeded. Output: {final_path}")
        return final_path


def estimate(book_id: str = "absalon", gpu_count: int = 20, cost_per_hour: float = 0.30) -> dict:
    """Estimate time and cost for parallel generation."""
    orch = ParallelOrchestrator(book_id)
    total = orch.get_segment_count()
    ranges = orch.distribute_segments(total, gpu_count)

    max_time = max(r.estimated_hours for r in ranges)
    total_gpu_hours = sum(r.estimated_hours for r in ranges)
    single_gpu_time = total * 25 / 3600

    return {
        "segments": total,
        "gpus": len(ranges),
        "ranges": [
            {"gpu": i + 1, "start": r.start, "end": r.end, "count": r.count}
            for i, r in enumerate(ranges)
        ],
        "wall_time_hours": max_time,
        "wall_time_minutes": max_time * 60,
        "total_cost": total_gpu_hours * cost_per_hour,
        "single_gpu_hours": single_gpu_time,
        "speedup": single_gpu_time / max_time if max_time > 0 else 0,
    }
