"""VastAI instance management for parallel generation."""

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from loguru import logger


@dataclass
class VastAIInstance:
    """Represents a rented VastAI instance."""

    instance_id: int
    ssh_host: str
    ssh_port: int
    gpu_name: str
    cost_per_hour: float
    status: str = "pending"
    assigned_chapters: list[int] = field(default_factory=list)

    @property
    def ssh_command(self) -> str:
        """SSH command to connect to this instance."""
        return f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {self.ssh_port} root@{self.ssh_host}"

    def run_ssh(
        self, command: str, timeout: int = 300, check: bool = True
    ) -> subprocess.CompletedProcess:
        """Run command on instance via SSH."""
        ssh_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=30",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=4",
            "-o",
            "BatchMode=yes",
            "-p",
            str(self.ssh_port),
            f"root@{self.ssh_host}",
            command,
        ]
        logger.debug(f"[{self.instance_id}] Running: {command[:100]}...")
        return subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout, check=check)

    def scp_download(self, remote_path: str, local_path: Path) -> None:
        """Download file from instance via SCP."""
        scp_cmd = [
            "scp",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-P",
            str(self.ssh_port),
            f"root@{self.ssh_host}:{remote_path}",
            str(local_path),
        ]
        subprocess.run(scp_cmd, check=True, capture_output=True)

    def scp_download_dir(self, remote_path: str, local_path: Path) -> None:
        """Download directory from instance via SCP."""
        scp_cmd = [
            "scp",
            "-r",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-P",
            str(self.ssh_port),
            f"root@{self.ssh_host}:{remote_path}",
            str(local_path),
        ]
        subprocess.run(scp_cmd, check=True, capture_output=True)

    def rsync_upload(self, local_path: Path, remote_path: str, timeout: int = 300) -> None:
        """Upload directory to instance via rsync (excludes .venv, .git, etc)."""
        ssh_opts = (
            f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            f"-o ConnectTimeout=30 -o ServerAliveInterval=15 "
            f"-o BatchMode=yes -p {self.ssh_port}"
        )
        rsync_cmd = [
            "rsync",
            "-az",
            "--delete",
            "--timeout=60",
            "-e",
            ssh_opts,
            "--exclude=.venv",
            "--exclude=.git",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=books/*/audio",
            f"{local_path}/",
            f"root@{self.ssh_host}:{remote_path}",
        ]
        subprocess.run(rsync_cmd, check=True, capture_output=True, timeout=timeout)


class VastAIManager:
    """Manages VastAI instances for parallel generation."""

    # Default search criteria
    DEFAULT_SEARCH = {
        "gpu_name": "RTX_3090",
        "disk_space": 30,
        "reliability": 0.90,
    }

    SYSTEM_PACKAGES = ["curl", "git", "rsync", "ffmpeg"]

    # Docker image for instances (Python 3.11 required by our deps)
    DOCKER_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"

    def __init__(self, dry_run: bool = False):
        """Initialize VastAI manager.

        Args:
            dry_run: If True, don't actually rent instances
        """
        self.dry_run = dry_run
        self.instances: list[VastAIInstance] = []
        self._verify_cli()

    def _verify_cli(self) -> None:
        """Verify vastai CLI is available and authenticated."""
        try:
            result = subprocess.run(["vastai", "show", "user"], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("VastAI CLI not authenticated. Run 'vastai set api-key <key>'")
            logger.debug("VastAI CLI verified")
        except FileNotFoundError:
            raise RuntimeError("VastAI CLI not found. Install with: pip install vastai")

    def search_instances(
        self,
        gpu_name: str | None = None,
        min_disk: float = 30,
        min_reliability: float = 0.90,
        limit: int = 20,
    ) -> list[dict]:
        """Search for available VastAI instances.

        Args:
            gpu_name: GPU model to search for (e.g., "RTX_4090", "RTX_3090")
            min_disk: Minimum disk space in GB
            min_reliability: Minimum reliability score (0-1)
            limit: Maximum number of results

        Returns:
            List of available offers sorted by price
        """
        # Build search query
        query_parts = [
            f"disk_space>={min_disk}",
            f"reliability>={min_reliability}",
            "rented=false",
            "cuda_vers>=12.0",
        ]

        if gpu_name:
            query_parts.append(f"gpu_name={gpu_name}")

        query = " ".join(query_parts)

        cmd = ["vastai", "search", "offers", query, "-o", "dph", "--raw"]
        logger.debug(f"Search command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"VastAI search failed: {result.stderr.strip()}")

        try:
            offers = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse offers: {result.stdout[:200]}") from exc

        return offers[:limit]

    def rent_instances(
        self,
        count: int,
        gpu_name: str | None = None,
        on_start_cmd: str | None = None,
    ) -> list[VastAIInstance]:
        """Rent multiple VastAI instances.

        Args:
            count: Number of instances to rent
            gpu_name: GPU model to search for
            on_start_cmd: Command to run on instance startup

        Returns:
            List of rented instances
        """
        logger.info(f"Searching for {count} instances (GPU: {gpu_name or 'any'})...")

        offers = self.search_instances(gpu_name=gpu_name, limit=count * 2)

        if len(offers) < count:
            raise RuntimeError(f"Only {len(offers)} offers found, requested {count}")

        instances = []
        for i, offer in enumerate(offers[:count]):
            offer_id = offer["id"]
            if "dph_total" in offer:
                cost = offer["dph_total"]
            elif "dph" in offer:
                cost = offer["dph"]
            else:
                raise RuntimeError(f"Offer {offer_id} missing price data")

            if "gpu_name" not in offer:
                raise RuntimeError(f"Offer {offer_id} missing gpu_name")
            gpu = offer["gpu_name"]

            logger.info(f"Renting instance {i+1}/{count}: offer {offer_id} ({gpu}, ${cost:.3f}/hr)")

            if self.dry_run:
                logger.info("[DRY RUN] Would rent instance")
                continue

            # Rent the instance
            cmd = [
                "vastai",
                "create",
                "instance",
                str(offer_id),
                "--image",
                self.DOCKER_IMAGE,
                "--disk",
                "30",
                "--raw",
            ]

            if on_start_cmd:
                cmd.extend(["--onstart-cmd", on_start_cmd])

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Failed to rent offer {offer_id}: {result.stderr}")
                continue

            try:
                response = json.loads(result.stdout)
                instance_id = response.get("new_contract")
                if instance_id:
                    instance = VastAIInstance(
                        instance_id=instance_id,
                        ssh_host="",  # Will be filled when ready
                        ssh_port=0,
                        gpu_name=gpu,
                        cost_per_hour=cost,
                        status="pending",
                    )
                    instances.append(instance)
                    logger.info(f"Rented instance {instance_id}")
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Failed to parse rental response: {result.stdout[:200]}"
                ) from exc

        if self.dry_run:
            return []

        if len(instances) < count:
            raise RuntimeError(f"Only rented {len(instances)}/{count} instances")

        self.instances.extend(instances)
        return instances

    def wait_for_ready(
        self,
        instances: list[VastAIInstance] | None = None,
        timeout: int = 600,
        poll_interval: int = 15,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[VastAIInstance]:
        """Wait for instances to be ready (SSH accessible).

        Args:
            instances: Instances to wait for (default: all managed instances)
            timeout: Maximum wait time in seconds
            poll_interval: Seconds between status checks
            progress_callback: Callback(ready_count, total_count) for progress

        Returns:
            List of ready instances
        """
        if instances is None:
            instances = self.instances

        if not instances:
            return []

        start_time = time.time()
        pending = {i.instance_id: i for i in instances}
        ready = []

        logger.info(f"Waiting for {len(pending)} instances to be ready (timeout: {timeout}s)...")

        while pending and (time.time() - start_time) < timeout:
            # Get current status
            result = subprocess.run(
                ["vastai", "show", "instances", "--raw"], capture_output=True, text=True
            )

            if result.returncode == 0:
                try:
                    all_instances = json.loads(result.stdout)
                    for inst_data in all_instances:
                        inst_id = inst_data.get("id")
                        if inst_id in pending:
                            status = inst_data.get("actual_status", "")
                            ssh_addr = inst_data.get("ssh_host", "")
                            ssh_port = inst_data.get("ssh_port", 0)

                            if status == "running" and ssh_addr and ssh_port:
                                instance = pending.pop(inst_id)
                                instance.ssh_host = ssh_addr
                                instance.ssh_port = ssh_port
                                instance.status = "ready"
                                ready.append(instance)
                                logger.info(f"Instance {inst_id} ready: {ssh_addr}:{ssh_port}")

                                if progress_callback:
                                    progress_callback(len(ready), len(instances))
                except json.JSONDecodeError:
                    pass

            if pending:
                logger.debug(f"Waiting for {len(pending)} instances...")
                time.sleep(poll_interval)

        if pending:
            logger.warning(f"{len(pending)} instances did not become ready in time")
            for inst in pending.values():
                inst.status = "timeout"

        return ready

    def install_system_deps(self, instance: VastAIInstance, timeout: int = 300) -> bool:
        """Install required system dependencies on a fresh instance."""
        packages = " ".join(self.SYSTEM_PACKAGES)
        cmd = "apt-get update -qq && " f"apt-get install -y -qq --no-install-recommends {packages}"
        result = instance.run_ssh(f"bash -c '{cmd}'", timeout=timeout, check=False)
        if result.returncode != 0:
            logger.error(
                f"System deps install failed on {instance.instance_id}: {result.stderr[:200]}"
            )
            return False
        return True

    def setup_instance(self, instance: VastAIInstance) -> bool:
        """Setup instance with required software and local code.

        Uploads local code via SCP (no git clone), so local changes work immediately.

        Args:
            instance: Instance to setup

        Returns:
            True if setup successful
        """
        logger.info(f"Setting up instance {instance.instance_id}...")

        # Get local repo path
        local_repo = Path(__file__).parent.parent.parent

        try:
            if not self.install_system_deps(instance):
                return False

            # Install uv first
            for cmd in [
                "curl -LsSf https://astral.sh/uv/install.sh | sh",
                "source ~/.local/bin/env && echo 'source ~/.local/bin/env' >> ~/.bashrc",
                "rm -rf /workspace/audiobook && mkdir -p /workspace/audiobook",
            ]:
                result = instance.run_ssh(f"bash -c '{cmd}'", timeout=300)
                if result.returncode != 0:
                    logger.error(f"Setup failed on {instance.instance_id}: {result.stderr[:200]}")
                    return False

            # Upload local code via rsync (uses current local code, not GitHub)
            logger.info(f"[{instance.instance_id}] Uploading local code via rsync...")
            instance.rsync_upload(local_repo, "/workspace/audiobook")

            # Install dependencies and verify
            for cmd in [
                "cd /workspace/audiobook && source ~/.local/bin/env && uv sync",
                "cd /workspace/audiobook && source ~/.local/bin/env && uv run audiobook list-books",
            ]:
                result = instance.run_ssh(f"bash -c '{cmd}'", timeout=600)
                if result.returncode != 0:
                    logger.error(f"Setup failed on {instance.instance_id}: {result.stderr[:200]}")
                    return False

        except subprocess.TimeoutExpired:
            logger.error(f"Setup timed out on {instance.instance_id}")
            return False
        except Exception as e:
            logger.error(f"Setup error on {instance.instance_id}: {e}")
            return False

        instance.status = "setup_complete"
        logger.info(f"Instance {instance.instance_id} setup complete")
        return True

    def destroy_instance(self, instance: VastAIInstance) -> bool:
        """Destroy a rented instance.

        Args:
            instance: Instance to destroy

        Returns:
            True if destroyed successfully
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would destroy instance {instance.instance_id}")
            return True

        result = subprocess.run(
            ["vastai", "destroy", "instance", str(instance.instance_id)],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            instance.status = "destroyed"
            logger.info(f"Destroyed instance {instance.instance_id}")
            return True
        else:
            logger.error(f"Failed to destroy instance {instance.instance_id}: {result.stderr}")
            return False

    def destroy_all(self) -> None:
        """Destroy all managed instances."""
        logger.info(f"Destroying {len(self.instances)} instances...")
        for instance in self.instances:
            if instance.status != "destroyed":
                self.destroy_instance(instance)

    def get_running_instances(self) -> list[dict]:
        """Get list of currently running instances."""
        result = subprocess.run(
            ["vastai", "show", "instances", "--raw"], capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to list instances: {result.stderr.strip()}")

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse instances output: {result.stdout[:200]}") from exc
