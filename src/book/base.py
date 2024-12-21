from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


@dataclass
class Character:
    """Represents a character in the book."""

    name: str
    voice_id: str
    language: Optional[str] = None
    description: Optional[str] = None


@dataclass
class Segment:
    """Represents a segment of text to be narrated."""

    text: str
    voice_id: str
    character: Optional[Character] = None
    emphasis: Optional[str] = None  # e.g., "bold", "italic"
    language: Optional[str] = None


@dataclass
class Chapter:
    """Represents a chapter in the book."""

    number: int
    title: Optional[str]
    segments: List[Segment]


@dataclass
class Part:
    """Represents a part/section of the book."""

    number: int
    title: Optional[str]
    chapters: List[Chapter]


class Book(ABC):
    """
    Abstract base class for book processing.

    This class provides the foundation for processing books into a format suitable
    for audiobook generation. Each specific book implementation should subclass this
    and implement the required abstract methods.
    """

    def __init__(
        self,
        input_path: Path,
        title: Optional[str] = None,
        author: Optional[str] = None,
        language: str = "fr",
        narrator_voice_id: Optional[str] = None,
    ):
        """
        Initialize a new book processor.

        Args:
            input_path: Path to the book's source file (PDF, TXT, etc.)
            title: Book title (optional, can be extracted from content)
            author: Book author (optional, can be extracted from content)
            language: Primary language of the book (ISO code)
            narrator_voice_id: Voice ID for the default narrator
        """
        self.input_path = Path(input_path)
        self._raw_content = None
        self.title = title
        self.author = author
        self.language = language
        self.narrator_voice_id = narrator_voice_id

        # These will be populated by process()
        self.characters: Dict[str, Character] = {}
        self.parts: List[Part] = []
        self._processed = False

    @abstractmethod
    def _load_characters(self) -> Dict[str, Character]:
        """
        Load character definitions for the book.

        Returns:
            Dictionary mapping character names to Character objects
        """
        pass

    @abstractmethod
    def _process_content(self) -> List[Part]:
        """
        Process the book content into parts and chapters.

        Returns:
            List of Part objects containing the processed content
        """
        pass

    def process(self) -> None:
        """Process the book into a structured format ready for audio generation."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Book file not found: {self.input_path}")

        logger.info(f"Processing book: {self.title}")

        # Load character definitions
        logger.debug("Loading character definitions")
        self.characters = self._load_characters()

        # Process content into parts and chapters
        logger.debug("Processing book content")
        self.parts = self._process_content()

        self._processed = True
        logger.info(f"Finished processing book: {self.title}")

    def get_segments(self) -> List[Segment]:
        """Get all segments in reading order."""
        if not self._processed:
            raise RuntimeError("Book must be processed before getting segments")

        segments = []
        for part in self.parts:
            if part.title:
                segments.append(
                    Segment(text=part.title, voice_id=self.narrator_voice_id, emphasis="bold")
                )

            for chapter in part.chapters:
                if chapter.title:
                    segments.append(
                        Segment(
                            text=chapter.title, voice_id=self.narrator_voice_id, emphasis="bold"
                        )
                    )
                segments.extend(chapter.segments)

        return segments

    def validate(self) -> bool:
        """
        Validate the processed book structure.

        Returns:
            True if validation passes, raises exception otherwise
        """
        if not self._processed:
            raise RuntimeError("Book must be processed before validation")

        # Check for empty book
        if not self.parts:
            raise ValueError("Book has no content")

        # Validate voice IDs
        voice_ids = {self.narrator_voice_id} if self.narrator_voice_id else set()
        voice_ids.update(char.voice_id for char in self.characters.values())

        for segment in self.get_segments():
            if segment.voice_id not in voice_ids:
                raise ValueError(f"Unknown voice ID: {segment.voice_id}")

        return True

    def get_statistics(self) -> Dict:
        """
        Get statistics about the processed book.

        Returns:
            Dictionary containing various statistics
        """
        if not self._processed:
            raise RuntimeError("Book must be processed before getting statistics")

        segments = self.get_segments()

        return {
            "total_parts": len(self.parts),
            "total_chapters": sum(len(part.chapters) for part in self.parts),
            "total_segments": len(segments),
            "total_characters": len(self.characters),
            "total_words": sum(len(segment.text.split()) for segment in segments),
            "voices_used": len({segment.voice_id for segment in segments}),
        }
