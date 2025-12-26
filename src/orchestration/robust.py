"""Robust parallel audiobook generation - designed to handle failures gracefully.

Architecture:
- Segment files ARE the state (no database, no queue)
- Workers are stateless and can die/restart anytime
- Local machine is source of truth
- Idempotent: existing segments are skipped
- Progress = count of files in segments/ directory
"""

import json
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from src.book.catalog import load_book

from .combine import combine_from_segments
from .vastai import VastAIInstance, VastAIManager


@dataclass
class WorkerStatus:
    """Status of a remote worker."""

    instance_id: int
    range_start: int
    range_end: int
    status: str = "pending"  # pending, setup, generating, completed, failed
    segments_done: int = 0
    last_update: float = field(default_factory=time.time)
    error: str | None = None


class RobustOrchestrator:
    """Robust orchestrator that survives failures.

    Key design:
    - Workers generate segments and rsync them back to local
    - Local segments/ directory is the source of truth
    - Any worker can be killed and restarted - it just picks up where it left off
    - Progress is tracked by counting local files, not by process state
    """

    def __init__(
        self,
        book_id: str,
        segments_dir: Path | None = None,
        verify: bool = True,
        segment_limit: int | None = None,
        reference_audio: str | None = None,
        no_reference_audio: bool = False,
        language: str = "fr",
        whisper_model: str = "base",
    ):
        if reference_audio and no_reference_audio:
            raise ValueError("Use either reference_audio or no_reference_audio, not both")
        self.book_id = book_id
        self.segments_dir = segments_dir or Path("books") / book_id / "audio" / "segments"
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self.verify = verify
        self.segment_limit = segment_limit
        self.reference_audio = reference_audio
        self.no_reference_audio = no_reference_audio
        self.language = language
        self.whisper_model = whisper_model
        self.manager = VastAIManager()
        self.workers: dict[int, WorkerStatus] = {}
        self._stop_sync = threading.Event()

    def _build_generate_cmd(self, start: int, end: int) -> str:
        args = [
            "audiobook",
            "generate",
            "--book",
            self.book_id,
            "--segment-range",
            f"{start}-{end}",
            "--verify" if self.verify else "--no-verify",
            "--language",
            self.language,
        ]

        if self.reference_audio:
            args.extend(["--reference-audio", self.reference_audio])
        elif self.no_reference_audio:
            args.append("--no-reference-audio")

        if self.verify and self.whisper_model:
            args.extend(["--whisper-model", self.whisper_model])

        return " ".join(shlex.quote(arg) for arg in args)

    def get_total_segments(self) -> int:
        """Get total segment count for the book (respects segment_limit)."""
        from src.audio.pipeline import preprocess_segments

        book = load_book(self.book_id)
        total = 0
        for part in book.parts:
            for chapter in part.chapters:
                processed = preprocess_segments(chapter.segments)
                total += len(processed)

        if self.segment_limit is not None:
            return min(total, self.segment_limit)
        return total

    def get_completed_segments(self) -> set[int]:
        """Get set of segment indices that exist locally."""
        completed = set()
        for f in self.segments_dir.glob("segment_*.mp3"):
            try:
                parts = f.stem.split("_")
                if len(parts) != 2 or parts[0] != "segment" or not parts[1].isdigit():
                    continue
                completed.add(int(parts[1]))
            except (IndexError, ValueError):
                continue
        return completed

    def get_missing_segments(self) -> list[int]:
        """Get list of segment indices that need to be generated."""
        total = self.get_total_segments()  # Already respects segment_limit
        completed = self.get_completed_segments()
        # Only consider segments up to total (which may be limited)
        relevant_completed = {s for s in completed if s < total}
        return sorted(set(range(total)) - relevant_completed)

    def get_missing_ranges(self, max_range_size: int = 500) -> list[tuple[int, int]]:
        """Get contiguous ranges of missing segments."""
        missing = self.get_missing_segments()
        if not missing:
            return []

        ranges = []
        start = missing[0]
        end = missing[0]

        for idx in missing[1:]:
            if idx == end + 1 and (end - start + 1) < max_range_size:
                end = idx
            else:
                ranges.append((start, end))
                start = idx
                end = idx
        ranges.append((start, end))

        return ranges

    def _setup_worker(self, instance: VastAIInstance, max_retries: int = 3) -> bool:
        """Setup worker instance with code and dependencies.

        Uses pip instead of uv sync to skip torch/cuda packages that are
        already in the base pytorch image.
        """
        local_repo = Path(__file__).parent.parent.parent

        # Wait a bit for SSH to be fully ready
        time.sleep(10)

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"[{instance.instance_id}] Retry {attempt + 1}/{max_retries}")
                    time.sleep(15)

                if not self.manager.install_system_deps(instance):
                    raise RuntimeError("Failed to install system dependencies")

                # Quick setup commands
                for cmd in [
                    "rm -rf /workspace/audiobook && mkdir -p /workspace/audiobook",
                ]:
                    result = instance.run_ssh(f"bash -c '{cmd}'", timeout=120, check=False)
                    if result.returncode != 0:
                        logger.error(
                            f"[{instance.instance_id}] Setup command failed: {result.stderr[:200]}"
                        )
                        raise RuntimeError("SSH command failed")

                # Upload code
                instance.rsync_upload(local_repo, "/workspace/audiobook")

                # Install deps using pip (skips torch/cuda that are in base image)
                # Force reinstall torchvision to fix version mismatch with transformers
                pip_install_cmd = (
                    "cd /workspace/audiobook && "
                    "pip uninstall -y torchvision 2>/dev/null; "
                    "pip install --quiet --no-cache-dir -r requirements-remote.txt && "
                    "pip install --quiet --no-cache-dir -e ."
                )
                result = instance.run_ssh(
                    f"bash -c '{pip_install_cmd}'",
                    timeout=600,  # 10 min should be enough without torch
                    check=False,
                )
                if result.returncode != 0:
                    logger.error(
                        f"[{instance.instance_id}] pip install failed: {result.stderr[:500]}"
                    )
                    raise RuntimeError("pip install failed")

                logger.info(f"[{instance.instance_id}] Setup complete")
                return True

            except Exception as e:
                logger.warning(f"[{instance.instance_id}] Setup attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(
                        f"[{instance.instance_id}] Setup failed after {max_retries} attempts"
                    )
                    return False
        return False

    def _generate_on_worker(
        self,
        instance: VastAIInstance,
        start: int,
        end: int,
        sync_interval: int = 60,
    ) -> bool:
        """Run generation on a worker with periodic sync back to local.

        The worker generates segments and we periodically rsync them back.
        If the worker dies, we have partial progress locally.
        """
        # Start generation in background on the worker
        # Uses system python since we installed with pip install -e .
        generate_cmd = self._build_generate_cmd(start, end)
        gen_cmd = (
            f"cd /workspace/audiobook && "
            f"nohup {generate_cmd} "
            f"> /tmp/gen.log 2>&1 &"
        )

        logger.info(f"[{instance.instance_id}] Starting generation {start}-{end} in background")
        instance.run_ssh(f"bash -c '{gen_cmd}'", timeout=300, check=False)

        # Give it a moment to start and verify
        time.sleep(10)

        # Check if process started and log any startup errors
        check_result = instance.run_ssh(
            "bash -c 'pgrep -f \"audiobook generate\" && head -30 /tmp/gen.log 2>/dev/null || cat /tmp/gen.log 2>/dev/null'",
            timeout=30,
            check=False,
        )
        if check_result.stdout.strip():
            logger.info(f"[{instance.instance_id}] Startup check: {check_result.stdout[:500]}")
        if check_result.stderr.strip():
            logger.warning(f"[{instance.instance_id}] Startup stderr: {check_result.stderr[:300]}")

        # Periodically sync segments back to local
        last_count = 0
        stall_count = 0
        max_stalls = 10  # Give up after 10 minutes of no progress

        while True:
            time.sleep(sync_interval)

            # Check if generation process is still running
            ps_result = instance.run_ssh(
                "bash -c 'pgrep -f \"audiobook generate\" || echo DONE'", timeout=30, check=False
            )

            # Sync segments back to local
            try:
                rsync_cmd = [
                    "rsync",
                    "-az",
                    "-e",
                    f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {instance.ssh_port}",
                    f"root@{instance.ssh_host}:/workspace/audiobook/books/{self.book_id}/audio/segments/",
                    str(self.segments_dir) + "/",
                ]
                subprocess.run(rsync_cmd, capture_output=True, timeout=120)
            except Exception as e:
                logger.warning(f"[{instance.instance_id}] Sync failed: {e}")

            # Check progress
            current_count = sum(
                1
                for idx in range(start, end + 1)
                if (self.segments_dir / f"segment_{idx:05d}.mp3").exists()
            )

            if current_count > last_count:
                logger.info(f"[{instance.instance_id}] Progress: {current_count} segments locally")
                last_count = current_count
                stall_count = 0
            else:
                stall_count += 1

            # Check if done
            if "DONE" in ps_result.stdout:
                logger.info(f"[{instance.instance_id}] Generation process completed")
                # Final sync
                try:
                    subprocess.run(rsync_cmd, capture_output=True, timeout=120)
                except:
                    pass
                missing = [
                    idx
                    for idx in range(start, end + 1)
                    if not (self.segments_dir / f"segment_{idx:05d}.mp3").exists()
                ]
                if missing:
                    logger.warning(
                        f"[{instance.instance_id}] Missing {len(missing)} segments after completion"
                    )
                    return False
                return True

            # Check for stall
            if stall_count >= max_stalls:
                logger.warning(
                    f"[{instance.instance_id}] No progress for {max_stalls} minutes, assuming failed"
                )
                return False

    def _process_worker(
        self,
        instance: VastAIInstance,
        start: int,
        end: int,
        status: WorkerStatus,
        ready_timeout: int = 300,
        keep_instances: bool = False,
    ) -> None:
        """Full worker lifecycle: wait ready → setup → generate → sync."""
        instance_id = instance.instance_id
        start_time = time.time()

        # Wait for ready
        status.status = "pending"
        logger.info(f"[{instance_id}] Waiting for instance (range {start}-{end})...")

        while (time.time() - start_time) < ready_timeout:
            result = subprocess.run(
                ["vastai", "show", "instances", "--raw"], capture_output=True, text=True
            )
            if result.returncode == 0:
                try:
                    for inst_data in json.loads(result.stdout):
                        if inst_data.get("id") == instance_id:
                            if (
                                inst_data.get("actual_status") == "running"
                                and inst_data.get("ssh_host")
                                and inst_data.get("ssh_port")
                            ):
                                instance.ssh_host = inst_data["ssh_host"]
                                instance.ssh_port = inst_data["ssh_port"]
                                instance.status = "ready"
                                break
                except json.JSONDecodeError:
                    pass

            if instance.status == "ready":
                break
            time.sleep(10)

        if instance.status != "ready":
            status.status = "failed"
            status.error = "Instance never became ready"
            logger.warning(f"[{instance_id}] Timeout waiting for ready")
            self.manager.destroy_instance(instance)
            return

        logger.info(f"[{instance_id}] Ready at {instance.ssh_host}:{instance.ssh_port}")

        # Setup
        status.status = "setup"
        if not self._setup_worker(instance):
            status.status = "failed"
            status.error = "Setup failed"
            self.manager.destroy_instance(instance)
            return

        logger.info(f"[{instance_id}] Setup complete")

        # Generate with continuous sync
        status.status = "generating"
        success = self._generate_on_worker(instance, start, end)

        if success:
            status.status = "completed"
            logger.info(f"[{instance_id}] Completed range {start}-{end}")
        else:
            status.status = "failed"
            status.error = "Generation failed or stalled"
            logger.error(f"[{instance_id}] Failed range {start}-{end}")

        # Always destroy when done
        if not keep_instances:
            self.manager.destroy_instance(instance)

    def run(
        self,
        gpu_count: int = 10,
        gpu_type: str = "RTX_3090",
        max_range_size: int = 500,
        keep_instances: bool = False,
    ) -> dict:
        """Run generation for all missing segments.

        Returns dict with completed/failed/remaining counts.
        """
        # Get what needs to be done
        missing_ranges = self.get_missing_ranges(max_range_size)

        if not missing_ranges:
            logger.info("All segments already exist!")
            return {"completed": self.get_total_segments(), "remaining": 0}

        total_missing = sum(e - s + 1 for s, e in missing_ranges)
        logger.info(
            f"Need to generate {total_missing} segments across {len(missing_ranges)} ranges"
        )

        # Limit to gpu_count ranges at a time
        ranges_to_run = missing_ranges[:gpu_count]

        # Rent instances
        logger.info(f"Renting {len(ranges_to_run)} {gpu_type} instances...")
        instances = self.manager.rent_instances(len(ranges_to_run), gpu_type)

        if not instances:
            raise RuntimeError("Failed to rent any instances")

        # Assign ranges to instances
        assignments = list(zip(instances, ranges_to_run))

        # Track workers
        for inst, (start, end) in assignments:
            self.workers[inst.instance_id] = WorkerStatus(
                instance_id=inst.instance_id,
                range_start=start,
                range_end=end,
            )

        # Start workers in parallel
        threads = []
        for inst, (start, end) in assignments:
            status = self.workers[inst.instance_id]
            t = threading.Thread(
                target=self._process_worker,
                args=(inst, start, end, status),
                kwargs={"keep_instances": keep_instances},
            )
            t.start()
            threads.append(t)

        # Wait for all workers
        for t in threads:
            t.join()

        # Report results
        completed = len(self.get_completed_segments())
        total = self.get_total_segments()
        remaining = total - completed
        output_path = None

        if remaining == 0:
            output_path = combine_from_segments(self.book_id, self.segments_dir.parent)
            logger.info(f"Combined audiobook: {output_path}")

        logger.info(f"Completed: {completed}/{total} segments ({remaining} remaining)")

        if remaining > 0:
            logger.info("Run again to generate remaining segments")

        return {
            "completed": completed,
            "total": total,
            "remaining": remaining,
            "output_path": str(output_path) if output_path else None,
            "workers": {
                wid: {"status": w.status, "error": w.error} for wid, w in self.workers.items()
            },
        }

    def status(self) -> dict:
        """Get current progress status."""
        completed = self.get_completed_segments()
        total = self.get_total_segments()
        missing = self.get_missing_ranges()

        return {
            "completed": len(completed),
            "total": total,
            "remaining": total - len(completed),
            "percent": len(completed) / total * 100 if total > 0 else 0,
            "missing_ranges": missing,
        }


def robust_generate(
    book_id: str = "absalon",
    gpu_count: int = 10,
    gpu_type: str = "RTX_3090",
    verify: bool = True,
) -> dict:
    """Run robust generation - can be called multiple times to complete."""
    orch = RobustOrchestrator(book_id, verify=verify)
    return orch.run(gpu_count, gpu_type)


def status(book_id: str = "absalon") -> dict:
    """Get generation status for a book."""
    orch = RobustOrchestrator(book_id)
    return orch.status()
