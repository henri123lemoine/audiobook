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
