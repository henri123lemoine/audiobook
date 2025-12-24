"""Parallel audiobook generation orchestrator."""

import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from loguru import logger

from .vastai import VastAIManager, VastAIInstance


@dataclass
class ChapterInfo:
    """Information about a chapter for scheduling."""
    number: int
    segment_count: int
    char_count: int
    estimated_hours: float = 0.0

    def __post_init__(self):
        # Estimate ~1.1h per 100k chars on RTX 4090
        self.estimated_hours = (self.char_count / 100_000) * 1.1


@dataclass
class SegmentRange:
    """A range of segments assigned to an instance."""
    start: int
    end: int  # inclusive
    segment_count: int = 0
    estimated_hours: float = 0.0

    def __post_init__(self):
        self.segment_count = self.end - self.start + 1
        # Estimate ~25 seconds per segment on average
        self.estimated_hours = (self.segment_count * 25) / 3600


@dataclass
class ChapterAssignment:
    """Assignment of chapters to an instance."""
    instance: VastAIInstance
    chapters: list[ChapterInfo]
    total_chars: int = 0
    estimated_hours: float = 0.0

    def __post_init__(self):
        self.total_chars = sum(c.char_count for c in self.chapters)
        self.estimated_hours = sum(c.estimated_hours for c in self.chapters)


@dataclass
class GenerationJob:
    """A chapter generation job."""
    chapter: ChapterInfo
    instance: VastAIInstance
    status: str = "pending"  # pending, running, completed, failed
    start_time: float | None = None
    end_time: float | None = None
    error: str | None = None
    output_path: Path | None = None


class ParallelOrchestrator:
    """Orchestrates parallel audiobook generation across multiple GPUs."""

    def __init__(
        self,
        book_id: str = "absalon",
        output_dir: Path | None = None,
        verify: bool = True,
        whisper_model: str = "base",
        dry_run: bool = False,
    ):
        """Initialize orchestrator.

        Args:
            book_id: Book identifier
            output_dir: Local output directory for results
            verify: Enable STT verification
            whisper_model: Whisper model for verification
            dry_run: If True, don't rent instances or generate
        """
        self.book_id = book_id
        self.output_dir = output_dir or Path("books") / book_id / "audio"
        self.verify = verify
        self.whisper_model = whisper_model
        self.dry_run = dry_run
        self.manager = VastAIManager(dry_run=dry_run)
        self.jobs: list[GenerationJob] = []

    def get_chapter_info(self) -> list[ChapterInfo]:
        """Get information about all chapters in the book.

        Returns:
            List of ChapterInfo sorted by chapter number
        """
        # Get book info from local CLI
        result = subprocess.run(
            ["uv", "run", "audiobook", "info", "--book", self.book_id],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to get book info: {result.stderr}")

        chapters = []
        lines = result.stdout.split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("Chapter "):
                # Parse: "Chapter 1: 100 segments, 53,757 chars"
                try:
                    # Use regex for robust parsing
                    import re
                    match = re.match(r'Chapter\s+(\d+):\s*(\d+)\s+segments,\s*([\d,]+)\s+chars', line)
                    if match:
                        chapter_num = int(match.group(1))
                        segments = int(match.group(2))
                        chars = int(match.group(3).replace(",", ""))

                        chapters.append(ChapterInfo(
                            number=chapter_num,
                            segment_count=segments,
                            char_count=chars,
                        ))
                    else:
                        logger.warning(f"Failed to parse chapter line (no match): {line}")
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse chapter line: {line}: {e}")

        return sorted(chapters, key=lambda c: c.number)

    def assign_chapters(
        self,
        chapters: list[ChapterInfo],
        instance_count: int,
    ) -> list[list[ChapterInfo]]:
        """Assign chapters to instances for load balancing.

        Uses a greedy algorithm that assigns the largest unassigned chapter
        to the instance with the least total work.

        Args:
            chapters: List of chapters to assign
            instance_count: Number of instances

        Returns:
            List of chapter lists (one per instance)
        """
        # Sort chapters by size (descending) for better load balancing
        sorted_chapters = sorted(chapters, key=lambda c: c.char_count, reverse=True)

        # Initialize assignment buckets with their totals
        assignments: list[tuple[int, list[ChapterInfo]]] = [(0, []) for _ in range(instance_count)]

        for chapter in sorted_chapters:
            # Find bucket with minimum total work
            min_idx = min(range(len(assignments)), key=lambda i: assignments[i][0])
            total, chapter_list = assignments[min_idx]
            chapter_list.append(chapter)
            assignments[min_idx] = (total + chapter.char_count, chapter_list)

        # Sort each bucket by chapter number for ordered processing
        return [sorted(bucket[1], key=lambda c: c.number) for bucket in assignments]

    def generate_on_instance(
        self,
        instance: VastAIInstance,
        chapter: ChapterInfo,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> GenerationJob:
        """Run generation for a chapter on an instance.

        Args:
            instance: VastAI instance to run on
            chapter: Chapter to generate
            progress_callback: Callback(chapter_num, status) for progress

        Returns:
            GenerationJob with result status
        """
        job = GenerationJob(chapter=chapter, instance=instance)

        try:
            job.status = "running"
            job.start_time = time.time()

            if progress_callback:
                progress_callback(chapter.number, "running")

            # Build generation command
            verify_flag = "--verify" if self.verify else ""
            whisper_flag = f"--whisper-model {self.whisper_model}" if self.verify else ""

            cmd = (
                f"cd /workspace/audiobook && "
                f"source ~/.local/bin/env && "
                f"uv run audiobook generate --book {self.book_id} --chapter {chapter.number} "
                f"{verify_flag} {whisper_flag}"
            )

            logger.info(f"[Instance {instance.instance_id}] Generating chapter {chapter.number}...")

            if self.dry_run:
                logger.info(f"[DRY RUN] Would run: {cmd}")
                time.sleep(2)  # Simulate work
                job.status = "completed"
                job.end_time = time.time()
                return job

            # Run generation (long timeout for large chapters)
            timeout = max(7200, int(chapter.estimated_hours * 3600 * 1.5))  # 1.5x estimated time
            result = instance.run_ssh(f"bash -c '{cmd}'", timeout=timeout, check=False)

            if result.returncode == 0:
                job.status = "completed"
                logger.info(f"[Instance {instance.instance_id}] Chapter {chapter.number} completed")
            else:
                job.status = "failed"
                job.error = result.stderr[:500] if result.stderr else "Unknown error"
                logger.error(f"[Instance {instance.instance_id}] Chapter {chapter.number} failed: {job.error}")

            job.end_time = time.time()

        except subprocess.TimeoutExpired:
            job.status = "failed"
            job.error = "Generation timed out"
            job.end_time = time.time()
            logger.error(f"[Instance {instance.instance_id}] Chapter {chapter.number} timed out")

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.end_time = time.time()
            logger.error(f"[Instance {instance.instance_id}] Chapter {chapter.number} error: {e}")

        if progress_callback:
            progress_callback(chapter.number, job.status)

        return job

    def download_results(
        self,
        instances: list[VastAIInstance],
        chapters: list[ChapterInfo],
    ) -> dict[int, Path]:
        """Download generated chapter audio from all instances.

        Args:
            instances: Instances that ran generation
            chapters: Chapters that were generated

        Returns:
            Dict mapping chapter number to local audio path
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        downloaded = {}

        # Group chapters by which instance generated them
        instance_chapters: dict[int, list[int]] = {}
        for instance in instances:
            for chapter_num in instance.assigned_chapters:
                instance_chapters.setdefault(instance.instance_id, []).append(chapter_num)

        # Download from each instance
        for instance in instances:
            chapter_nums = instance_chapters.get(instance.instance_id, [])

            for chapter_num in chapter_nums:
                remote_path = f"/workspace/audiobook/books/{self.book_id}/audio/partie_1/chapitre_{chapter_num}_full.mp3"
                local_path = self.output_dir / "partie_1" / f"chapitre_{chapter_num}_full.mp3"
                local_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    logger.info(f"Downloading chapter {chapter_num} from instance {instance.instance_id}...")

                    if self.dry_run:
                        logger.info(f"[DRY RUN] Would download {remote_path} to {local_path}")
                        continue

                    instance.scp_download(remote_path, local_path)
                    downloaded[chapter_num] = local_path
                    logger.info(f"Downloaded chapter {chapter_num}: {local_path}")

                except Exception as e:
                    logger.error(f"Failed to download chapter {chapter_num}: {e}")

                    # Try downloading the segment directory as fallback
                    try:
                        remote_seg_dir = f"/workspace/audiobook/books/{self.book_id}/audio/partie_1/chapitre_{chapter_num}"
                        local_seg_dir = self.output_dir / "partie_1" / f"chapitre_{chapter_num}"
                        instance.scp_download_dir(remote_seg_dir, local_seg_dir.parent)
                        logger.info(f"Downloaded chapter {chapter_num} segments to {local_seg_dir}")
                    except Exception as e2:
                        logger.error(f"Failed to download chapter {chapter_num} segments: {e2}")

        return downloaded

    def run_parallel(
        self,
        instance_count: int = 9,
        gpu_name: str = "RTX_4090",
        max_cost: float = 0.40,
        keep_instances: bool = False,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict[int, GenerationJob]:
        """Run parallel generation across multiple GPU instances.

        Args:
            instance_count: Number of GPU instances to rent
            gpu_name: GPU model to rent
            max_cost: Maximum cost per hour per instance
            keep_instances: If True, don't destroy instances after completion
            progress_callback: Callback(chapter_num, total, status) for progress

        Returns:
            Dict mapping chapter number to GenerationJob
        """
        # Get chapter info
        chapters = self.get_chapter_info()
        logger.info(f"Found {len(chapters)} chapters, {sum(c.char_count for c in chapters):,} total chars")

        # Limit instance count to chapter count
        actual_instance_count = min(instance_count, len(chapters))
        if actual_instance_count < instance_count:
            logger.info(f"Reduced instance count from {instance_count} to {actual_instance_count} (one per chapter)")

        # Assign chapters to instances
        assignments = self.assign_chapters(chapters, actual_instance_count)

        # Log assignment plan
        total_chars = sum(c.char_count for c in chapters)
        for i, chapter_list in enumerate(assignments):
            chars = sum(c.char_count for c in chapter_list)
            chapter_nums = [c.number for c in chapter_list]
            estimated_hours = sum(c.estimated_hours for c in chapter_list)
            logger.info(
                f"Instance {i+1}: chapters {chapter_nums}, "
                f"{chars:,} chars ({chars/total_chars*100:.1f}%), "
                f"~{estimated_hours:.1f}h"
            )

        # Calculate cost and time estimates
        max_time = max(sum(c.estimated_hours for c in a) for a in assignments)
        # Cost based on total GPU-hours, not max_time × instances
        # (instances that finish early stop costing money)
        total_gpu_hours = sum(sum(c.estimated_hours for c in a) for a in assignments)
        total_cost = total_gpu_hours * max_cost

        logger.info(f"Estimated wall time: ~{max_time:.1f}h")
        logger.info(f"Estimated total cost: ~${total_cost:.2f}")

        if self.dry_run:
            logger.info("[DRY RUN] Would rent instances and generate")
            return {}

        # Rent instances
        logger.info(f"Renting {actual_instance_count} {gpu_name} instances...")
        instances = self.manager.rent_instances(
            count=actual_instance_count,
            gpu_name=gpu_name,
            max_cost=max_cost,
        )

        if len(instances) < actual_instance_count:
            logger.warning(f"Only got {len(instances)} instances, redistributing chapters...")
            assignments = self.assign_chapters(chapters, len(instances))

        # Wait for instances to be ready
        ready_instances = self.manager.wait_for_ready(instances, timeout=600)
        if not ready_instances:
            logger.error("No instances became ready, aborting")
            self.manager.destroy_all()
            raise RuntimeError("No instances became ready")

        logger.info(f"{len(ready_instances)} instances ready")

        # Setup instances in parallel
        with ThreadPoolExecutor(max_workers=len(ready_instances)) as executor:
            setup_futures = {
                executor.submit(self.manager.setup_instance, inst): inst
                for inst in ready_instances
            }

            ready_for_gen = []
            for future in as_completed(setup_futures):
                inst = setup_futures[future]
                try:
                    if future.result():
                        ready_for_gen.append(inst)
                except Exception as e:
                    logger.error(f"Setup failed for instance {inst.instance_id}: {e}")

        if not ready_for_gen:
            logger.error("No instances set up successfully, aborting")
            self.manager.destroy_all()
            raise RuntimeError("No instances set up successfully")

        logger.info(f"{len(ready_for_gen)} instances set up and ready for generation")

        # Reassign chapters if we lost instances
        if len(ready_for_gen) < len(assignments):
            assignments = self.assign_chapters(chapters, len(ready_for_gen))

        # Assign chapters to ready instances
        for instance, chapter_list in zip(ready_for_gen, assignments):
            instance.assigned_chapters = [c.number for c in chapter_list]

        # Run generation on all instances in parallel
        all_jobs: dict[int, GenerationJob] = {}
        lock = threading.Lock()

        def run_instance_chapters(instance: VastAIInstance, chapter_list: list[ChapterInfo]):
            for chapter in chapter_list:
                def callback(ch_num, status):
                    if progress_callback:
                        with lock:
                            completed = sum(1 for j in all_jobs.values() if j.status == "completed")
                            progress_callback(ch_num, len(chapters), status)

                job = self.generate_on_instance(instance, chapter, callback)
                with lock:
                    all_jobs[chapter.number] = job
                    self.jobs.append(job)

        with ThreadPoolExecutor(max_workers=len(ready_for_gen)) as executor:
            futures = [
                executor.submit(run_instance_chapters, inst, chapter_list)
                for inst, chapter_list in zip(ready_for_gen, assignments)
                if chapter_list
            ]

            # Wait for all generation to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Generation thread failed: {e}")

        # Download results
        logger.info("Downloading generated audio files...")
        self.download_results(ready_for_gen, chapters)

        # Cleanup instances unless keeping
        if not keep_instances:
            logger.info("Destroying instances...")
            self.manager.destroy_all()

        # Summary
        completed = sum(1 for j in all_jobs.values() if j.status == "completed")
        failed = sum(1 for j in all_jobs.values() if j.status == "failed")
        logger.info(f"Generation complete: {completed} succeeded, {failed} failed")

        return all_jobs

    def combine_chapters(self) -> Path:
        """Combine all downloaded chapters into final audiobook.

        Returns:
            Path to combined audiobook
        """
        from pydub import AudioSegment

        part_dir = self.output_dir / "partie_1"
        if not part_dir.exists():
            raise FileNotFoundError(f"Part directory not found: {part_dir}")

        # Find all chapter files
        chapter_files = sorted(
            part_dir.glob("chapitre_*_full.mp3"),
            key=lambda f: int(f.stem.split("_")[1])
        )

        if not chapter_files:
            raise FileNotFoundError(f"No chapter files found in {part_dir}")

        logger.info(f"Combining {len(chapter_files)} chapters...")

        # Combine with silence between chapters
        silence = AudioSegment.silent(duration=1000)
        combined = AudioSegment.from_file(chapter_files[0])

        for chapter_file in chapter_files[1:]:
            chapter_audio = AudioSegment.from_file(chapter_file)
            combined += silence + chapter_audio

        # Export final audiobook
        final_path = self.output_dir / "audiobook_complete.mp3"
        combined.export(str(final_path), format="mp3", bitrate="192k")
        logger.info(f"Created final audiobook: {final_path}")

        return final_path

    # ==================== SEGMENT-LEVEL PARALLELIZATION ====================

    def get_total_segments(self) -> int:
        """Get total segment count for the book (after preprocessing).

        Returns:
            Total number of segments
        """
        chapters = self.get_chapter_info()
        return sum(c.segment_count for c in chapters)

    def assign_segment_ranges(self, total_segments: int, instance_count: int) -> list[SegmentRange]:
        """Assign segment ranges evenly across instances.

        Args:
            total_segments: Total number of segments
            instance_count: Number of GPU instances

        Returns:
            List of SegmentRange (one per instance)
        """
        segments_per_instance = total_segments // instance_count
        remainder = total_segments % instance_count

        ranges = []
        start = 0
        for i in range(instance_count):
            # Distribute remainder across first instances
            count = segments_per_instance + (1 if i < remainder else 0)
            if count > 0:
                end = start + count - 1
                ranges.append(SegmentRange(start=start, end=end))
                start = end + 1

        return ranges

    def generate_segment_range(
        self,
        instance: VastAIInstance,
        segment_range: SegmentRange,
    ) -> tuple[bool, str | None]:
        """Run segment-range generation on an instance.

        Args:
            instance: VastAI instance to run on
            segment_range: Range of segments to generate

        Returns:
            Tuple of (success, error_message)
        """
        try:
            verify_flag = "--verify" if self.verify else "--no-verify"
            whisper_flag = f"--whisper-model {self.whisper_model}" if self.verify else ""

            cmd = (
                f"cd /workspace/audiobook && "
                f"source ~/.local/bin/env && "
                f"uv run audiobook generate --book {self.book_id} "
                f"--segment-range {segment_range.start}-{segment_range.end} "
                f"{verify_flag} {whisper_flag}"
            )

            logger.info(
                f"[Instance {instance.instance_id}] Generating segments "
                f"{segment_range.start}-{segment_range.end} ({segment_range.segment_count} segments)..."
            )

            if self.dry_run:
                logger.info(f"[DRY RUN] Would run: {cmd}")
                time.sleep(2)
                return True, None

            # Estimate timeout: ~30s per segment + 5 min buffer
            timeout = max(1800, segment_range.segment_count * 30 + 300)
            result = instance.run_ssh(f"bash -c '{cmd}'", timeout=timeout, check=False)

            if result.returncode == 0:
                logger.info(
                    f"[Instance {instance.instance_id}] Completed segments "
                    f"{segment_range.start}-{segment_range.end}"
                )
                return True, None
            else:
                error = result.stderr[:500] if result.stderr else "Unknown error"
                logger.error(f"[Instance {instance.instance_id}] Failed: {error}")
                return False, error

        except subprocess.TimeoutExpired:
            logger.error(f"[Instance {instance.instance_id}] Timed out")
            return False, "Generation timed out"
        except Exception as e:
            logger.error(f"[Instance {instance.instance_id}] Error: {e}")
            return False, str(e)

    def download_segments(self, instances: list[VastAIInstance]) -> Path:
        """Download all generated segments from instances.

        Args:
            instances: Instances that ran generation

        Returns:
            Path to local segments directory
        """
        segments_dir = self.output_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        for instance in instances:
            remote_dir = f"/workspace/audiobook/books/{self.book_id}/audio/segments/"

            try:
                logger.info(f"Downloading segments from instance {instance.instance_id}...")

                if self.dry_run:
                    logger.info(f"[DRY RUN] Would download from {remote_dir}")
                    continue

                # Download all files from segments directory
                instance.scp_download_dir(remote_dir, segments_dir.parent)
                logger.info(f"Downloaded segments from instance {instance.instance_id}")

            except Exception as e:
                logger.error(f"Failed to download from instance {instance.instance_id}: {e}")

        return segments_dir

    def combine_segments_to_chapters(self, segments_dir: Path) -> Path:
        """Combine downloaded segments into chapter audio files.

        Reads manifests to determine chapter assignments, combines segments
        in order, and creates chapter audio files.

        Args:
            segments_dir: Directory containing segment files and manifests

        Returns:
            Path to output directory with chapter files
        """
        import json
        from pydub import AudioSegment

        # Load all manifests
        manifests = []
        for manifest_file in segments_dir.glob("manifest_*.json"):
            with open(manifest_file) as f:
                manifests.extend(json.load(f))

        if not manifests:
            raise FileNotFoundError(f"No manifests found in {segments_dir}")

        # Sort by global index
        manifests.sort(key=lambda m: m["global_index"])

        # Group by chapter
        chapter_segments: dict[int, list[dict]] = {}
        for seg in manifests:
            chapter_num = seg["chapter"]
            chapter_segments.setdefault(chapter_num, []).append(seg)

        # Create output directory
        part_dir = self.output_dir / "partie_1"
        part_dir.mkdir(parents=True, exist_ok=True)

        silence = AudioSegment.silent(duration=500)

        # Combine segments for each chapter
        for chapter_num, segs in sorted(chapter_segments.items()):
            logger.info(f"Combining {len(segs)} segments for chapter {chapter_num}...")

            # Sort by global index to ensure correct order
            segs.sort(key=lambda s: s["global_index"])

            # Load and combine
            combined = None
            for seg in segs:
                seg_file = segments_dir / seg["file"]
                if not seg_file.exists():
                    logger.warning(f"Missing segment file: {seg_file}")
                    continue

                audio = AudioSegment.from_file(seg_file)
                if combined is None:
                    combined = audio
                else:
                    combined += silence + audio

            if combined:
                chapter_path = part_dir / f"chapitre_{chapter_num}_full.mp3"
                combined.export(str(chapter_path), format="mp3", bitrate="192k")
                logger.info(f"Created chapter {chapter_num}: {chapter_path}")

        return part_dir

    def run_parallel_segments(
        self,
        instance_count: int = 9,
        gpu_name: str = "RTX_4090",
        max_cost: float = 0.40,
        keep_instances: bool = False,
    ) -> dict:
        """Run segment-level parallel generation across multiple GPUs.

        This distributes segments evenly across all GPUs for optimal load balancing.
        Unlike chapter-level parallelization, this allows scaling beyond the number
        of chapters.

        Args:
            instance_count: Number of GPU instances to rent
            gpu_name: GPU model to rent
            max_cost: Maximum cost per hour per instance
            keep_instances: If True, don't destroy instances after completion

        Returns:
            Dict with results summary
        """
        # Get total segment count
        chapters = self.get_chapter_info()
        total_segments = sum(c.segment_count for c in chapters)

        logger.info(f"Total segments: {total_segments}")
        logger.info(f"Requested instances: {instance_count}")

        # Create segment ranges
        ranges = self.assign_segment_ranges(total_segments, instance_count)

        # Log assignment plan
        for i, r in enumerate(ranges):
            logger.info(
                f"Instance {i+1}: segments {r.start}-{r.end} "
                f"({r.segment_count} segments, ~{r.estimated_hours:.2f}h)"
            )

        # Calculate estimates
        max_time = max(r.estimated_hours for r in ranges)
        total_gpu_hours = sum(r.estimated_hours for r in ranges)
        total_cost = total_gpu_hours * max_cost

        logger.info(f"Estimated wall time: ~{max_time:.2f}h ({max_time * 60:.0f} min)")
        logger.info(f"Estimated total cost: ~${total_cost:.2f}")

        if self.dry_run:
            logger.info("[DRY RUN] Would rent instances and generate")
            return {"dry_run": True, "ranges": ranges}

        # Rent instances
        logger.info(f"Renting {len(ranges)} {gpu_name} instances...")
        instances = self.manager.rent_instances(
            count=len(ranges),
            gpu_name=gpu_name,
            max_cost=max_cost,
        )

        if len(instances) < len(ranges):
            logger.warning(f"Only got {len(instances)} instances, redistributing...")
            ranges = self.assign_segment_ranges(total_segments, len(instances))

        # Wait for instances to be ready
        ready_instances = self.manager.wait_for_ready(instances, timeout=600)
        if not ready_instances:
            logger.error("No instances became ready, aborting")
            self.manager.destroy_all()
            raise RuntimeError("No instances became ready")

        logger.info(f"{len(ready_instances)} instances ready")

        # Setup instances in parallel
        with ThreadPoolExecutor(max_workers=len(ready_instances)) as executor:
            setup_futures = {
                executor.submit(self.manager.setup_instance, inst): inst
                for inst in ready_instances
            }

            ready_for_gen = []
            for future in as_completed(setup_futures):
                inst = setup_futures[future]
                try:
                    if future.result():
                        ready_for_gen.append(inst)
                except Exception as e:
                    logger.error(f"Setup failed for instance {inst.instance_id}: {e}")

        if not ready_for_gen:
            logger.error("No instances set up successfully, aborting")
            self.manager.destroy_all()
            raise RuntimeError("No instances set up successfully")

        logger.info(f"{len(ready_for_gen)} instances set up and ready")

        # Reassign ranges if we lost instances
        if len(ready_for_gen) < len(ranges):
            ranges = self.assign_segment_ranges(total_segments, len(ready_for_gen))

        # Run generation on all instances in parallel
        results = []
        with ThreadPoolExecutor(max_workers=len(ready_for_gen)) as executor:
            futures = {
                executor.submit(self.generate_segment_range, inst, seg_range): (inst, seg_range)
                for inst, seg_range in zip(ready_for_gen, ranges)
            }

            for future in as_completed(futures):
                inst, seg_range = futures[future]
                try:
                    success, error = future.result()
                    results.append({
                        "instance": inst.instance_id,
                        "range": f"{seg_range.start}-{seg_range.end}",
                        "success": success,
                        "error": error,
                    })
                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                    results.append({
                        "instance": inst.instance_id,
                        "range": f"{seg_range.start}-{seg_range.end}",
                        "success": False,
                        "error": str(e),
                    })

        # Download all segments
        logger.info("Downloading generated segments...")
        segments_dir = self.download_segments(ready_for_gen)

        # Combine segments into chapters
        logger.info("Combining segments into chapters...")
        self.combine_segments_to_chapters(segments_dir)

        # Combine chapters into final audiobook
        logger.info("Creating final audiobook...")
        final_path = self.combine_chapters()

        # Cleanup instances unless keeping
        if not keep_instances:
            logger.info("Destroying instances...")
            self.manager.destroy_all()

        # Summary
        succeeded = sum(1 for r in results if r["success"])
        failed = sum(1 for r in results if not r["success"])
        logger.info(f"Generation complete: {succeeded} succeeded, {failed} failed")

        return {
            "final_path": final_path,
            "results": results,
            "succeeded": succeeded,
            "failed": failed,
        }


def estimate_segment_parallel_run(
    book_id: str = "absalon",
    instance_count: int = 9,
    cost_per_hour: float = 0.30,
) -> dict:
    """Estimate time and cost for segment-level parallel generation.

    Args:
        book_id: Book identifier
        instance_count: Number of GPU instances
        cost_per_hour: Cost per instance per hour

    Returns:
        Dict with estimation details
    """
    orchestrator = ParallelOrchestrator(book_id=book_id, dry_run=True)
    chapters = orchestrator.get_chapter_info()

    total_segments = sum(c.segment_count for c in chapters)
    total_chars = sum(c.char_count for c in chapters)

    # Create segment ranges
    ranges = orchestrator.assign_segment_ranges(total_segments, instance_count)

    # Calculate time estimates (~25 sec per segment)
    max_time = max(r.estimated_hours for r in ranges)
    total_gpu_hours = sum(r.estimated_hours for r in ranges)
    total_cost = total_gpu_hours * cost_per_hour

    # Single GPU baseline
    single_gpu_segments_per_hour = 3600 / 25  # ~144 segments/hour
    single_gpu_time = total_segments / single_gpu_segments_per_hour
    single_gpu_cost = single_gpu_time * cost_per_hour

    return {
        "book_id": book_id,
        "chapters": len(chapters),
        "total_segments": total_segments,
        "total_chars": total_chars,
        "instance_count": len(ranges),
        "assignments": [
            {
                "instance": i + 1,
                "start": r.start,
                "end": r.end,
                "segment_count": r.segment_count,
                "estimated_hours": r.estimated_hours,
            }
            for i, r in enumerate(ranges)
        ],
        "estimated_wall_time_hours": max_time,
        "estimated_wall_time_minutes": max_time * 60,
        "estimated_total_cost": total_cost,
        "single_gpu_time_hours": single_gpu_time,
        "single_gpu_cost": single_gpu_cost,
        "speedup": single_gpu_time / max_time if max_time > 0 else 0,
    }


def estimate_parallel_run(
    book_id: str = "absalon",
    instance_count: int = 9,
    cost_per_hour: float = 0.30,
) -> dict:
    """Estimate time and cost for parallel generation.

    Args:
        book_id: Book identifier
        instance_count: Number of GPU instances
        cost_per_hour: Cost per instance per hour

    Returns:
        Dict with estimation details
    """
    orchestrator = ParallelOrchestrator(book_id=book_id, dry_run=True)
    chapters = orchestrator.get_chapter_info()

    total_chars = sum(c.char_count for c in chapters)
    total_estimated = sum(c.estimated_hours for c in chapters)

    # Limit instances to chapter count
    actual_instances = min(instance_count, len(chapters))
    assignments = orchestrator.assign_chapters(chapters, actual_instances)

    # Find slowest instance (limiting factor)
    max_time = max(sum(c.estimated_hours for c in a) for a in assignments)
    # Cost based on actual GPU-hours used (instances finish at different times)
    total_gpu_hours = sum(sum(c.estimated_hours for c in a) for a in assignments)
    total_cost = total_gpu_hours * cost_per_hour

    # Single GPU baseline
    single_gpu_time = total_estimated
    single_gpu_cost = single_gpu_time * cost_per_hour

    return {
        "book_id": book_id,
        "chapters": len(chapters),
        "total_chars": total_chars,
        "instance_count": actual_instances,
        "assignments": [
            {
                "instance": i + 1,
                "chapters": [c.number for c in a],
                "chars": sum(c.char_count for c in a),
                "estimated_hours": sum(c.estimated_hours for c in a),
            }
            for i, a in enumerate(assignments)
        ],
        "estimated_wall_time_hours": max_time,
        "estimated_total_cost": total_cost,
        "single_gpu_time_hours": single_gpu_time,
        "single_gpu_cost": single_gpu_cost,
        "speedup": single_gpu_time / max_time if max_time > 0 else 0,
        "cost_multiplier": total_cost / single_gpu_cost if single_gpu_cost > 0 else 0,
    }
