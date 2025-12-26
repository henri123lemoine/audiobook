"""Multi-GPU orchestration for parallel audiobook generation."""

from .combine import combine_from_segments
from .parallel import ParallelOrchestrator, estimate
from .vastai import VastAIInstance, VastAIManager

__all__ = [
    "VastAIManager",
    "VastAIInstance",
    "ParallelOrchestrator",
    "combine_from_segments",
    "estimate",
]
