from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
        self._cached_content: Optional[str] = None
        self._cached_metadata: Optional[Dict[str, Any]] = None

    @abstractmethod
    def extract_metadata(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract title and author from content.

        Returns:
            Tuple of (title, author)
        """
        pass

    @abstractmethod
    def extract_content(self, **kwargs) -> str:
        """
        Extract and preprocess text content.

        Args:
            **kwargs: Format-specific extraction options

        Returns:
            Preprocessed text content
        """
        pass

    @abstractmethod
    def extract_chapters(self, content: str) -> List[Tuple[str, str]]:
        """
        Attempt to identify chapter breaks in the content.

        Args:
            content: Raw text content

        Returns:
            List of (chapter_title, chapter_content) tuples
        """
        pass

    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and normalizing line endings.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Remove excess whitespace
        text = " ".join(text.split())

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Normalize multiple newlines
        text = "\n\n".join(line.strip() for line in text.split("\n") if line.strip())

        return text

    def _identify_language(self, text: str, sample_size: int = 1000) -> str:
        """
        Attempt to identify the language of the text.

        Args:
            text: Text to analyze
            sample_size: Number of characters to sample

        Returns:
            ISO language code (e.g., "fr", "en")
        """
        # This is a simple implementation that could be enhanced with langdetect or similar
        sample = text[:sample_size].lower()

        # Count frequency of common French words
        french_words = {"le", "la", "les", "un", "une", "des", "et", "en", "dans"}
        french_count = sum(1 for word in sample.split() if word in french_words)

        # Count frequency of common English words
        english_words = {"the", "a", "an", "and", "in", "of", "to", "for", "is"}
        english_count = sum(1 for word in sample.split() if word in english_words)

        return "fr" if french_count > english_count else "en"

    def invalidate_cache(self) -> None:
        """Clear any cached content."""
        self._cached_content = None
        self._cached_metadata = None

    def get_statistics(self, content: Optional[str] = None) -> Dict[str, int]:
        """
        Get basic statistics about the content.

        Args:
            content: Content to analyze (uses cached content if None)

        Returns:
            Dictionary of statistics
        """
        if content is None:
            if self._cached_content is None:
                content = self.extract_content()
            else:
                content = self._cached_content

        words = content.split()

        return {
            "total_chars": len(content),
            "total_words": len(words),
            "avg_word_length": sum(len(word) for word in words) // len(words),
            "total_paragraphs": content.count("\n\n") + 1,
        }
