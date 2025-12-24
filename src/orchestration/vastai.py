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

    def run_ssh(self, command: str, timeout: int = 300, check: bool = True) -> subprocess.CompletedProcess:
        """Run command on instance via SSH."""
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=30",
            "-p", str(self.ssh_port),
            f"root@{self.ssh_host}",
            command
        ]
        logger.debug(f"[{self.instance_id}] Running: {command[:100]}...")
        return subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout, check=check)

    def scp_download(self, remote_path: str, local_path: Path) -> None:
        """Download file from instance via SCP."""
        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(self.ssh_port),
            f"root@{self.ssh_host}:{remote_path}",
            str(local_path)
        ]
        subprocess.run(scp_cmd, check=True, capture_output=True)

    def scp_download_dir(self, remote_path: str, local_path: Path) -> None:
        """Download directory from instance via SCP."""
        scp_cmd = [
            "scp",
            "-r",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(self.ssh_port),
            f"root@{self.ssh_host}:{remote_path}",
            str(local_path)
        ]
        subprocess.run(scp_cmd, check=True, capture_output=True)


class VastAIManager:
    """Manages VastAI instances for parallel generation."""

    # Default search criteria
    DEFAULT_SEARCH = {
        "gpu_name": "RTX_4090",
        "disk_space": 30,
        "reliability": 0.95,
        "max_cost": 0.50,
    }

    # Docker image for instances
    DOCKER_IMAGE = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"

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
        min_reliability: float = 0.95,
        max_cost: float = 0.50,
        limit: int = 20,
    ) -> list[dict]:
        """Search for available VastAI instances.

        Args:
            gpu_name: GPU model to search for (e.g., "RTX_4090", "RTX_3090")
            min_disk: Minimum disk space in GB
            min_reliability: Minimum reliability score (0-1)
            max_cost: Maximum cost per hour in USD
            limit: Maximum number of results

        Returns:
            List of available offers sorted by price
        """
        # Build search query
        query_parts = [
            f"disk_space>={min_disk}",
            f"reliability>={min_reliability}",
            f"dph<={max_cost}",
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
            logger.error(f"Search failed: {result.stderr}")
            return []

        try:
            offers = json.loads(result.stdout)
            return offers[:limit]
        except json.JSONDecodeError:
            logger.error(f"Failed to parse offers: {result.stdout[:200]}")
            return []

    def rent_instances(
        self,
        count: int,
        gpu_name: str | None = None,
        max_cost: float = 0.50,
        on_start_cmd: str | None = None,
    ) -> list[VastAIInstance]:
        """Rent multiple VastAI instances.

        Args:
            count: Number of instances to rent
            gpu_name: GPU model to search for
            max_cost: Maximum cost per hour per instance
            on_start_cmd: Command to run on instance startup

        Returns:
            List of rented instances
        """
        logger.info(f"Searching for {count} instances (GPU: {gpu_name or 'any'}, max ${max_cost}/hr)...")

        offers = self.search_instances(gpu_name=gpu_name, max_cost=max_cost, limit=count * 2)

        if len(offers) < count:
            logger.warning(f"Only {len(offers)} offers found, requested {count}")
            if not offers:
                raise RuntimeError("No suitable instances found")

        instances = []
        for i, offer in enumerate(offers[:count]):
            offer_id = offer["id"]
            cost = offer.get("dph_total", offer.get("dph", 0))
            gpu = offer.get("gpu_name", "Unknown")

            logger.info(f"Renting instance {i+1}/{count}: offer {offer_id} ({gpu}, ${cost:.3f}/hr)")

            if self.dry_run:
                logger.info("[DRY RUN] Would rent instance")
                continue

            # Rent the instance
            cmd = [
                "vastai", "create", "instance", str(offer_id),
                "--image", self.DOCKER_IMAGE,
                "--disk", "30",
                "--raw"
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
                        status="pending"
                    )
                    instances.append(instance)
                    logger.info(f"Rented instance {instance_id}")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse rental response: {result.stdout[:200]}")

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
                ["vastai", "show", "instances", "--raw"],
                capture_output=True, text=True
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

    def setup_instance(self, instance: VastAIInstance, repo_url: str = "https://github.com/henri123lemoine/audiobook.git") -> bool:
        """Setup instance with required software and repository.

        Args:
            instance: Instance to setup
            repo_url: Git repository URL to clone

        Returns:
            True if setup successful
        """
        logger.info(f"Setting up instance {instance.instance_id}...")

        setup_commands = [
            # Install uv
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "source ~/.local/bin/env && echo 'source ~/.local/bin/env' >> ~/.bashrc",
            # Clone repo
            f"rm -rf /workspace/audiobook && git clone {repo_url} /workspace/audiobook",
            # Install dependencies
            "cd /workspace/audiobook && source ~/.local/bin/env && uv sync",
            # Verify setup
            "cd /workspace/audiobook && source ~/.local/bin/env && uv run audiobook list-books",
        ]

        for cmd in setup_commands:
            try:
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
            capture_output=True, text=True
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
            ["vastai", "show", "instances", "--raw"],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return []
        return []
