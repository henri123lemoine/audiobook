from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger

from .processors.base import ContentProcessor


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

    This class provides the foundation for extracting and structuring book content.
    Each specific book implementation should subclass this and implement the
    required abstract methods.
    """

    def __init__(
        self,
        content_processor: ContentProcessor,
        title: Optional[str] = None,
        author: Optional[str] = None,
        language: str = "fr",
    ):
        """
        Initialize book processor.

        Args:
            content_processor: Processor for extracting book content
            title: Book title (optional, extracted from content if None)
            author: Book author (optional, extracted from content if None)
            language: Primary language of the book (ISO code)
        """
        self.processor = content_processor
        self.language = language

        # These will be populated by process()
        self.characters: Dict[str, Character] = {}
        self.parts: List[Part] = []
        self._processed = False

        # Try to get metadata from processor if not provided
        if title is None or author is None:
            extracted_title, extracted_author = self.processor.extract_metadata()
            self.title = title or extracted_title
            self.author = author or extracted_author
        else:
            self.title = title
            self.author = author

        logger.debug(f"Initialized book: {self.title} by {self.author}")

    @abstractmethod
    def _load_characters(self) -> Dict[str, Character]:
        """
        Load character definitions for the book.

        Returns:
            Dictionary mapping character names to Character objects
        """
        pass

    @abstractmethod
    def _structure_content(self, raw_content: str) -> List[Part]:
        """Convert raw content into structured parts and chapters."""
        pass

    def process(self) -> None:
        """Process the book into a structured format."""
        logger.info(f"Processing book: {self.title}")

        # Load character definitions
        logger.debug("Loading character definitions")
        self.characters = self._load_characters()

        # Get raw content
        logger.debug("Extracting content")
        raw_content = self.processor.extract_content()

        # Structure content
        logger.debug("Structuring content")
        self.parts = self._structure_content(raw_content)

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
                    Segment(text=part.title, voice_id=self.characters["narrator"].voice_id)
                )

            for chapter in part.chapters:
                if chapter.title:
                    segments.append(
                        Segment(text=chapter.title, voice_id=self.characters["narrator"].voice_id)
                    )
                segments.extend(chapter.segments)

        return segments

    def validate(self) -> bool:
        """Validate the processed book structure."""
        if not self._processed:
            raise RuntimeError("Book must be processed before validation")

        # Check for empty book
        if not self.parts:
            raise ValueError("Book has no content")

        # Check for required narrator
        if "narrator" not in self.characters:
            raise ValueError("Book must have a narrator character defined")

        # Validate character references in segments
        valid_voices = {char.voice_id for char in self.characters.values()}

        for segment in self.get_segments():
            if segment.voice_id not in valid_voices:
                raise ValueError(f"Unknown voice ID in segment: {segment.voice_id}")
            if segment.character and segment.character.name not in self.characters:
                raise ValueError(f"Unknown character in segment: {segment.character.name}")

        return True

    def get_statistics(self) -> Dict:
        """Get statistics about the processed book."""
        if not self._processed:
            raise RuntimeError("Book must be processed before getting statistics")

        segments = self.get_segments()

        return {
            "title": self.title,
            "author": self.author,
            "language": self.language,
            "total_parts": len(self.parts),
            "total_chapters": sum(len(part.chapters) for part in self.parts),
            "total_segments": len(segments),
            "total_characters": len(self.characters),
            "total_words": sum(len(segment.text.split()) for segment in segments),
            "voices_used": len({segment.voice_id for segment in segments}),
        }
