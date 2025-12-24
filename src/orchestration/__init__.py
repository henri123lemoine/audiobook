"""Multi-GPU orchestration for parallel audiobook generation."""

from .parallel import ParallelOrchestrator, estimate
from .vastai import VastAIInstance, VastAIManager

__all__ = ["VastAIManager", "VastAIInstance", "ParallelOrchestrator", "estimate"]
