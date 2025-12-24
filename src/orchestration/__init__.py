"""Multi-GPU orchestration for parallel audiobook generation."""

from .vastai import VastAIManager, VastAIInstance
from .parallel import ParallelOrchestrator, ChapterAssignment

__all__ = [
    "VastAIManager",
    "VastAIInstance",
    "ParallelOrchestrator",
    "ChapterAssignment",
]
