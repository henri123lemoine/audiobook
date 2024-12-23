from abc import ABC
from pathlib import Path


class ContentProcessor(ABC):
    """Base class for all content processors."""

    def __init__(self, path: Path):
        """
        Initialize content processor.

        Args:
            path: Path to the content file
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
