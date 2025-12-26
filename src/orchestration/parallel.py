"""Parallel audiobook generation across multiple GPUs."""

import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from src.book.catalog import load_book

from .combine import combine_from_segments
from .throughput import collect_processed_segments, summarize_throughput
from .vastai import VastAIInstance, VastAIManager


@dataclass
class SegmentRange:
    """A range of segments assigned to a GPU."""

    start: int
    end: int  # inclusive
    status: str = "pending"  # pending, in_progress, completed, failed
    instance_id: int | None = None
    error: str | None = None

    @property
    def count(self) -> int:
        return self.end - self.start + 1

    def __str__(self) -> str:
        return f"{self.start}-{self.end}"


@dataclass
class RunResult:
    """Result of a parallel run."""

    completed_ranges: list[SegmentRange] = field(default_factory=list)
    failed_ranges: list[SegmentRange] = field(default_factory=list)
    instances: list[VastAIInstance] = field(default_factory=list)
    output_path: Path | None = None


class ParallelOrchestrator:
    """Orchestrates parallel audiobook generation across multiple GPUs.

    Uses fixed assignment model: each instance gets exactly one range at startup.
    No dynamic work-stealing. Failed ranges are reported for manual retry.
    """

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
        """Get total segment count after preprocessing."""
        from src.audio.pipeline import preprocess_segments

        book = load_book(self.book_id)
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

        logger.info(f"[{instance.instance_id}] Generating segments {seg_range}")

        try:
            timeout = max(1800, seg_range.count * 40)  # Scale timeout with segment count
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
                logger.info(f"Downloading segments from {inst.instance_id}...")
                inst.scp_download_dir(
                    f"/workspace/audiobook/books/{self.book_id}/audio/segments/",
                    segments_dir.parent,
                )
            except Exception as e:
                logger.error(f"Download failed from {inst.instance_id}: {e}")
        return combine_from_segments(self.book_id, self.output_dir)

    def _process_instance(
        self,
        instance: VastAIInstance,
        assigned_range: SegmentRange,
        results_lock: threading.Lock,
        completed_ranges: list[SegmentRange],
        failed_ranges: list[SegmentRange],
        completed_instances: list[VastAIInstance],
        ready_timeout: int = 300,
    ) -> None:
        """Process a single instance with its fixed assigned range.

        Each instance gets exactly one range. No work-stealing.
        """
        instance_id = instance.instance_id
        assigned_range.instance_id = instance_id
        assigned_range.status = "in_progress"
        start_time = time.time()

        # Phase 1: Wait for instance to be ready
        logger.info(f"[{instance_id}] Waiting for instance to be ready (range {assigned_range})...")
        while (time.time() - start_time) < ready_timeout:
            result = subprocess.run(
                ["vastai", "show", "instances", "--raw"], capture_output=True, text=True
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
            logger.warning(f"[{instance_id}] Did not become ready in {ready_timeout}s")
            assigned_range.status = "failed"
            assigned_range.error = "Instance never became ready"
            with results_lock:
                failed_ranges.append(assigned_range)
            self.manager.destroy_instance(instance)
            return

        # Phase 2: Setup the instance
        logger.info(f"[{instance_id}] Starting setup...")
        if not self.manager.setup_instance(instance):
            logger.error(f"[{instance_id}] Setup failed")
            assigned_range.status = "failed"
            assigned_range.error = "Setup failed"
            with results_lock:
                failed_ranges.append(assigned_range)
            self.manager.destroy_instance(instance)
            return

        logger.info(f"[{instance_id}] Setup complete, generating {assigned_range}...")

        # Phase 3: Generate the assigned range
        success, error = self.run_on_instance(instance, assigned_range)

        if success:
            assigned_range.status = "completed"
            logger.info(f"[{instance_id}] Completed {assigned_range}")
            with results_lock:
                completed_ranges.append(assigned_range)
                completed_instances.append(instance)
        else:
            assigned_range.status = "failed"
            assigned_range.error = error
            logger.error(f"[{instance_id}] Failed {assigned_range}: {error}")
            with results_lock:
                failed_ranges.append(assigned_range)

    def run(
        self,
        gpu_count: int = 10,
        gpu_type: str = "RTX_3090",
        keep_instances: bool = False,
        ready_timeout: int = 300,
        segment_limit: int | None = None,
    ) -> RunResult:
        """Run parallel generation with fixed assignment.

        Each instance gets exactly one range. If an instance fails,
        that range is marked failed and reported for manual retry.

        Returns RunResult with completed/failed ranges and instances.
        """
        total_segments = self.get_segment_count()
        if segment_limit is not None:
            total_segments = min(total_segments, segment_limit)
        ranges = self.distribute_segments(total_segments, gpu_count)

        logger.info(
            f"Distributing {total_segments} segments across {gpu_count} GPUs (fixed assignment)"
        )
        for i, r in enumerate(ranges):
            logger.debug(f"  Range {i+1}: {r} ({r.count} segs)")

        # Rent instances
        requested_gpus = gpu_count
        gpu_count = min(gpu_count, len(ranges))
        if gpu_count < requested_gpus:
            logger.info(f"Reducing GPU count to {gpu_count} (only {len(ranges)} ranges)")
        logger.info(f"Renting {gpu_count} {gpu_type} instances...")
        instances = self.manager.rent_instances(gpu_count, gpu_type)

        if not instances:
            raise RuntimeError("Failed to rent any instances")

        logger.info(f"Rented {len(instances)} instances, starting fixed-assignment pipeline...")

        # Assign ranges to instances (1:1 mapping)
        assignments = list(zip(instances, ranges))

        # Track results
        completed_ranges: list[SegmentRange] = []
        failed_ranges: list[SegmentRange] = []
        completed_instances: list[VastAIInstance] = []
        results_lock = threading.Lock()

        # Process each instance with its assigned range
        with ThreadPoolExecutor(max_workers=len(instances)) as executor:
            futures = [
                executor.submit(
                    self._process_instance,
                    inst,
                    seg_range,
                    results_lock,
                    completed_ranges,
                    failed_ranges,
                    completed_instances,
                    ready_timeout,
                )
                for inst, seg_range in assignments
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Instance processing error: {e}")

        # Report results
        logger.info(f"Completed: {len(completed_ranges)}/{len(ranges)} ranges")
        if failed_ranges:
            logger.warning("Failed ranges (retry with --segment-range):")
            for r in failed_ranges:
                logger.warning(f"  {r}: {r.error}")

        result = RunResult(
            completed_ranges=completed_ranges,
            failed_ranges=failed_ranges,
            instances=completed_instances,
        )

        # Download and combine if any succeeded
        if completed_instances:
            logger.info(f"Downloading from {len(completed_instances)} instances...")
            result.output_path = self.download_and_combine(completed_instances)
            logger.info(f"Output: {result.output_path}")

        if not keep_instances:
            self.manager.destroy_all()

        return result

    def run_ranges(
        self,
        ranges: list[tuple[int, int]],
        gpu_type: str = "RTX_3090",
        keep_instances: bool = False,
        ready_timeout: int = 300,
    ) -> RunResult:
        """Run specific segment ranges on new instances.

        Use this to retry failed ranges or add capacity mid-run.
        Each (start, end) tuple gets one instance.

        Example:
            orch.run_ranges([(3730, 4475), (4476, 5221)])
        """
        segment_ranges = [SegmentRange(start=s, end=e) for s, e in ranges]
        gpu_count = len(ranges)

        logger.info(f"Running {gpu_count} specific ranges on new instances...")
        for r in segment_ranges:
            logger.info(f"  {r}")

        # Rent instances
        instances = self.manager.rent_instances(gpu_count, gpu_type)

        if not instances:
            raise RuntimeError("Failed to rent any instances")

        if len(instances) < len(segment_ranges):
            logger.warning(f"Only got {len(instances)} instances for {len(segment_ranges)} ranges")

        # Assign and run
        assignments = list(zip(instances, segment_ranges[: len(instances)]))

        completed_ranges: list[SegmentRange] = []
        failed_ranges: list[SegmentRange] = segment_ranges[len(instances) :]
        for r in failed_ranges:
            r.status = "failed"
            r.error = "No instance available"
        completed_instances: list[VastAIInstance] = []
        results_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=len(instances)) as executor:
            futures = [
                executor.submit(
                    self._process_instance,
                    inst,
                    seg_range,
                    results_lock,
                    completed_ranges,
                    failed_ranges,
                    completed_instances,
                    ready_timeout,
                )
                for inst, seg_range in assignments
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Instance processing error: {e}")

        result = RunResult(
            completed_ranges=completed_ranges,
            failed_ranges=failed_ranges,
            instances=completed_instances,
        )

        if completed_instances:
            logger.info(f"Downloading from {len(completed_instances)} instances...")
            result.output_path = self.download_and_combine(completed_instances)

        if not keep_instances:
            self.manager.destroy_all()

        return result


def estimate(
    book_id: str = "absalon",
    gpu_count: int = 10,
    gpu_type: str = "RTX_3090",
    segment_limit: int | None = None,
) -> dict:
    """Estimate wall time and cost using manifest throughput and live pricing."""
    segments = collect_processed_segments(book_id, limit=segment_limit)
    total = len(segments)
    if total == 0:
        raise RuntimeError("No segments available for estimation")

    orch = ParallelOrchestrator(book_id)
    ranges = orch.distribute_segments(total, gpu_count)
    if not ranges:
        raise RuntimeError("No segments available for estimation")

    target_gpus = min(gpu_count, len(ranges))
    if target_gpus < gpu_count:
        ranges = ranges[:target_gpus]

    throughput_entry = summarize_throughput(book_id, gpu_type)
    audio_seconds_per_char = float(throughput_entry["audio_seconds_per_char"])
    audio_seconds_per_wall_second = float(throughput_entry["audio_seconds_per_wall_second"])

    if audio_seconds_per_char <= 0 or audio_seconds_per_wall_second <= 0:
        raise RuntimeError(f"Throughput entry for {gpu_type} is missing required metrics")

    total_chars = sum(len(segment.text) for segment in segments)
    if total_chars <= 0:
        raise RuntimeError("Total character count is zero")

    total_audio_seconds = total_chars * audio_seconds_per_char

    # Fetch current market prices
    manager = VastAIManager()
    offers = manager.search_instances(gpu_name=gpu_type, limit=target_gpus * 2)

    prices_with_offers = []
    for offer in offers:
        if "dph_total" in offer:
            price = float(offer["dph_total"])
        elif "dph" in offer:
            price = float(offer["dph"])
        else:
            raise RuntimeError(f"Offer {offer.get('id')} missing price data")
        prices_with_offers.append((price, offer))

    if len(prices_with_offers) < target_gpus:
        raise RuntimeError(f"Only {len(prices_with_offers)} offers found for {gpu_type}")

    prices_with_offers.sort(key=lambda item: item[0])
    selected = prices_with_offers[:target_gpus]
    prices = [price for price, _ in selected]

    avg_price = sum(prices) / len(prices)
    min_price = min(prices)
    max_price = max(prices)

    total_gpu_hours = 0.0
    total_cost = 0.0
    range_wall_seconds: list[float] = []
    range_rows = []

    for idx, seg_range in enumerate(ranges):
        range_segments = segments[seg_range.start : seg_range.end + 1]
        range_chars = sum(len(segment.text) for segment in range_segments)
        range_audio_seconds = range_chars * audio_seconds_per_char
        range_seconds = range_audio_seconds / audio_seconds_per_wall_second
        range_hours = range_seconds / 3600
        range_cost = prices[idx] * range_hours

        range_wall_seconds.append(range_seconds)
        total_gpu_hours += range_hours
        total_cost += range_cost

        range_rows.append(
            {
                "gpu": idx + 1,
                "start": seg_range.start,
                "end": seg_range.end,
                "count": seg_range.count,
                "chars": range_chars,
                "audio_seconds": range_audio_seconds,
                "wall_seconds": range_seconds,
                "price_per_hour": prices[idx],
                "cost": range_cost,
            }
        )

    wall_seconds = max(range_wall_seconds) if range_wall_seconds else 0.0

    return {
        "segments": total,
        "total_chars": total_chars,
        "gpus": len(ranges),
        "gpu_type": gpu_type,
        "offers_considered": len(prices_with_offers),
        "price_per_hour": avg_price,
        "price_range": (min_price, max_price),
        "total_hourly_cost": sum(prices),
        "throughput_entry": throughput_entry,
        "estimate": {
            "total_audio_seconds": total_audio_seconds,
            "wall_seconds": wall_seconds,
            "total_gpu_hours": total_gpu_hours,
            "total_cost": total_cost,
        },
        "ranges": range_rows,
    }
