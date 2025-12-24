"""Parallel audiobook generation across multiple GPUs."""

import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

from loguru import logger
from pydub import AudioSegment

from .vastai import VastAIInstance, VastAIManager


@dataclass
class SegmentRange:
    """A range of segments assigned to a GPU."""

    start: int
    end: int  # inclusive
    assigned_to: int | None = None  # instance_id when assigned

    @property
    def count(self) -> int:
        return self.end - self.start + 1

    @property
    def estimated_hours(self) -> float:
        return (self.count * 30) / 3600  # ~30 sec per segment


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
        """Get total segment count after preprocessing.

        This must match what the CLI uses with --segment-range,
        which preprocesses segments (splitting long ones).
        """
        from src.audio.pipeline import preprocess_segments
        from src.book.books.absolon import AbsalonBook

        # Load book
        book = AbsalonBook(use_chapter_files=True)

        # Count preprocessed segments (same logic as CLI)
        total = 0
        for part in book.parts:
            for chapter in part.chapters:
                processed = preprocess_segments(chapter.segments)
                total += len(processed)

        return total

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
            timeout = max(1800, seg_range.count * 40)  # ~40s per segment with buffer
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

    def _process_instance(
        self,
        instance: VastAIInstance,
        work_queue: Queue,
        results: list,
        results_lock: threading.Lock,
        completed_instances: list,
        timeout: int = 300,
    ) -> None:
        """Process a single instance: wait for ready → setup → generate.

        Each instance runs through the full pipeline independently.
        """
        instance_id = instance.instance_id
        start_time = time.time()

        # Phase 1: Wait for this instance to be ready
        logger.info(f"[{instance_id}] Waiting for instance to be ready...")
        while (time.time() - start_time) < timeout:
            result = subprocess.run(
                ["vastai", "show", "instances", "--raw"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                try:
                    all_instances = json.loads(result.stdout)
                    for inst_data in all_instances:
                        if inst_data.get("id") == instance_id:
                            status = inst_data.get("actual_status", "")
                            ssh_addr = inst_data.get("ssh_host", "")
                            ssh_port = inst_data.get("ssh_port", 0)

                            if status == "running" and ssh_addr and ssh_port:
                                instance.ssh_host = ssh_addr
                                instance.ssh_port = ssh_port
                                instance.status = "ready"
                                logger.info(f"[{instance_id}] Ready at {ssh_addr}:{ssh_port}")
                                break
                except json.JSONDecodeError:
                    pass

            if instance.status == "ready":
                break
            time.sleep(10)

        if instance.status != "ready":
            logger.warning(f"[{instance_id}] Did not become ready in {timeout}s, destroying")
            self.manager.destroy_instance(instance)
            return

        # Phase 2: Setup the instance
        logger.info(f"[{instance_id}] Starting setup...")
        if not self.manager.setup_instance(instance):
            logger.error(f"[{instance_id}] Setup failed, destroying")
            self.manager.destroy_instance(instance)
            return

        logger.info(f"[{instance_id}] Setup complete, ready for work!")

        # Phase 3: Grab work from queue and execute
        while True:
            try:
                seg_range = work_queue.get_nowait()
            except Exception:
                # Queue empty, no more work
                break

            seg_range.assigned_to = instance_id
            logger.info(f"[{instance_id}] Generating segments {seg_range.start}-{seg_range.end}")

            success, error = self.run_on_instance(instance, seg_range)

            with results_lock:
                results.append({
                    "instance_id": instance_id,
                    "range": f"{seg_range.start}-{seg_range.end}",
                    "success": success,
                    "error": error,
                })

            if success:
                logger.info(f"[{instance_id}] Completed segments {seg_range.start}-{seg_range.end}")
            else:
                logger.error(f"[{instance_id}] Failed {seg_range.start}-{seg_range.end}: {error}")
                # Put back in queue for another instance to try
                work_queue.put(seg_range)

        # Mark as completed for download
        with results_lock:
            completed_instances.append(instance)

        logger.info(f"[{instance_id}] All work done")

    def run(
        self,
        gpu_count: int = 20,
        gpu_type: str = "RTX_4090",
        max_cost: float = 0.40,
        keep_instances: bool = False,
        instance_timeout: int = 300,
        segment_limit: int | None = None,
    ) -> Path:
        """Run parallel generation across multiple GPUs.

        Uses streaming pipeline: each GPU starts work immediately when ready,
        without waiting for other GPUs.

        Returns path to final audiobook.
        """
        total_segments = self.get_segment_count()
        if segment_limit is not None:
            total_segments = min(total_segments, segment_limit)
        ranges = self.distribute_segments(total_segments, gpu_count)

        logger.info(f"Distributing {total_segments} segments across {gpu_count} GPUs")
        for i, r in enumerate(ranges):
            logger.debug(f"  Range {i+1}: segments {r.start}-{r.end} ({r.count} segs)")

        # Create work queue with all segment ranges
        work_queue: Queue = Queue()
        for r in ranges:
            work_queue.put(r)

        # Rent instances
        logger.info(f"Renting {gpu_count} {gpu_type} instances...")
        instances = self.manager.rent_instances(gpu_count, gpu_type, max_cost)

        if not instances:
            raise RuntimeError("Failed to rent any instances")

        logger.info(f"Rented {len(instances)} instances, starting streaming pipeline...")

        # Shared state for results
        results: list = []
        results_lock = threading.Lock()
        completed_instances: list = []

        # Process each instance in parallel - each goes through full pipeline independently
        with ThreadPoolExecutor(max_workers=len(instances)) as executor:
            futures = [
                executor.submit(
                    self._process_instance,
                    inst,
                    work_queue,
                    results,
                    results_lock,
                    completed_instances,
                    instance_timeout,
                )
                for inst in instances
            ]

            # Wait for all to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Instance processing error: {e}")

        # Check if all work was completed
        if not work_queue.empty():
            remaining = work_queue.qsize()
            logger.error(f"{remaining} segment ranges were not completed!")

        if not completed_instances:
            self.manager.destroy_all()
            raise RuntimeError("No instances completed successfully")

        # Download and combine
        logger.info(f"Downloading from {len(completed_instances)} instances...")
        final_path = self.download_and_combine(completed_instances)

        if not keep_instances:
            self.manager.destroy_all()

        succeeded = sum(1 for r in results if r["success"])
        logger.info(f"Complete: {succeeded}/{len(results)} succeeded. Output: {final_path}")
        return final_path


# Seconds per segment by GPU type (based on benchmarks)
GPU_SPEED = {
    "RTX_4090": 30,
    "RTX_4080": 35,
    "RTX_3090": 40,
    "RTX_3080": 50,
    "RTX_A6000": 35,
    "A100": 25,
}


def estimate(
    book_id: str = "absalon",
    gpu_count: int = 10,
    gpu_type: str = "RTX_3090",
    max_cost: float = 0.20,
    segment_limit: int | None = None,
) -> dict:
    """Estimate time and cost for parallel generation.

    Fetches current market prices from VastAI for accurate estimates.
    """
    sec_per_segment = GPU_SPEED.get(gpu_type, 40)  # default to 40 if unknown

    orch = ParallelOrchestrator(book_id)
    total = orch.get_segment_count()
    if segment_limit is not None:
        total = min(total, segment_limit)
    ranges = orch.distribute_segments(total, gpu_count)

    # Fetch current market prices
    manager = VastAIManager()
    offers = manager.search_instances(gpu_name=gpu_type, max_cost=max_cost, limit=gpu_count)

    if offers:
        prices = [o.get("dph_total", o.get("dph", max_cost)) for o in offers]
        avg_price = sum(prices) / len(prices)
        min_price = min(prices)
        max_price = max(prices)
        available = len(offers)
    else:
        avg_price = max_cost
        min_price = max_cost
        max_price = max_cost
        available = 0

    # Setup overhead per GPU: ~4 min (rsync + deps) + ~2 min (model download on first gen)
    setup_hours_per_gpu = 6 / 60  # 0.1 hours

    # Calculate time based on GPU speed
    max_segments_per_gpu = max(r.count for r in ranges)
    max_time = (max_segments_per_gpu * sec_per_segment) / 3600
    generation_gpu_hours = (total * sec_per_segment) / 3600
    setup_gpu_hours = len(ranges) * setup_hours_per_gpu
    total_gpu_hours = generation_gpu_hours + setup_gpu_hours

    single_gpu_time = (total * sec_per_segment) / 3600

    return {
        "segments": total,
        "gpus": len(ranges),
        "gpu_type": gpu_type,
        "available": available,
        "price_per_hour": avg_price,
        "price_range": (min_price, max_price),
        "ranges": [
            {"gpu": i + 1, "start": r.start, "end": r.end, "count": r.count}
            for i, r in enumerate(ranges)
        ],
        "wall_time_hours": max_time + setup_hours_per_gpu,
        "wall_time_minutes": (max_time + setup_hours_per_gpu) * 60,
        "total_cost": total_gpu_hours * avg_price,
        "setup_cost": setup_gpu_hours * avg_price,
        "generation_cost": generation_gpu_hours * avg_price,
        "single_gpu_hours": single_gpu_time,
        "speedup": single_gpu_time / max_time if max_time > 0 else 0,
    }
