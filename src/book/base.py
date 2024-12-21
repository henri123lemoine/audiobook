from pathlib import Path
from typing import Any

from loguru import logger

from .processors.content import ContentParser
from .types import Character, Part


class Book:
    """Base class for processing and structuring book content."""

    def __init__(
        self,
        input_path: Path,
        title: str | None = None,
        author: str | None = None,
        language: str = "fr",
    ):
        """Initialize book processor.

        Args:
            input_path: Path to book content file
            title: Book title (optional)
            author: Book author (optional)
            language: Primary language (ISO code)
        """
        self.path = Path(input_path)
        self.language = language
        self.title = title
        self.author = author

        # Will be populated during processing
        self.characters: dict[str, Character] = {}
        self.parts: list[Part] = []
        self._processed = False

        # Initialize processors
        self._content_parser = ContentParser()

        logger.debug(f"Initialized book processor for: {self.path.name}")

    def _load_characters(self) -> dict[str, Character]:
        """Load character definitions. Override in subclasses."""
        return {"narrator": Character(name="narrator")}

    def _extract_content(self) -> str:
        """Extract raw content from file. Override in subclasses."""
        raise NotImplementedError

    def process(self) -> None:
        """Process the book into structured content."""
        logger.info(f"Processing book: {self.path.name}")

        # Load character definitions first
        self.characters = self._load_characters()

        # Extract and parse content
        raw_content = self._extract_content()
        self.parts = self._content_parser.parse_content(raw_content)

        self._processed = True
        logger.info(f"Finished processing book: {self.path.name}")

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the processed book."""
        if not self._processed:
            raise RuntimeError("Book must be processed before getting statistics")

        total_chapters = sum(len(part.chapters) for part in self.parts)
        total_paragraphs = sum(
            len(chapter.paragraphs) for part in self.parts for chapter in part.chapters
        )
        total_segments = sum(
            len(paragraph.segments)
            for part in self.parts
            for chapter in part.chapters
            for paragraph in chapter.paragraphs
        )

        return {
            "title": self.title,
            "author": self.author,
            "language": self.language,
            "total_parts": len(self.parts),
            "total_chapters": total_chapters,
            "total_paragraphs": total_paragraphs,
            "total_segments": total_segments,
            "total_characters": len(self.characters),
        }

    def validate(self) -> bool:
        """Validate the processed book structure."""
        if not self._processed:
            raise RuntimeError("Book must be processed before validation")

        if not self.parts:
            raise ValueError("Book has no content")

        if "narrator" not in self.characters:
            raise ValueError("Book must have a narrator character defined")

        # Validate structure
        for part_num, part in enumerate(self.parts, 1):
            if part.number != part_num:
                raise ValueError(f"Invalid part number: {part.number} != {part_num}")

            for chapter_num, chapter in enumerate(part.chapters, 1):
                if chapter.number != chapter_num:
                    raise ValueError(
                        f"Invalid chapter number in part {part_num}: "
                        f"{chapter.number} != {chapter_num}"
                    )

        return True

    def __str__(self):
        string = ""
        string += "\nParsed Structure:"
        string += "\n" + "-" * 40
        for part in self.parts:
            string += f"\nPart {part.number}: {part.title or 'Untitled'}"
            for chapter in part.chapters:
                string += f"\n\tChapter {chapter.number}:"
                for para_num, para in enumerate(chapter.paragraphs, 1):
                    string += f"\n\t\tParagraph {para_num}:"
                    for seg_num, seg in enumerate(para.segments, 1):
                        string += (
                            f"\n\t\t\tSegment {seg_num}: "
                            f"{'[DIALOGUE]' if seg.is_dialogue else ''} "
                            f"{seg.text[:50]}..."
                        )
        return string


if __name__ == "__main__":
    # Test with a simple text file
    test_file = Path("test_book.txt")

    # Create test content
    test_content = """
PREMIÈRE PARTIE TEST
    
1
This is chapter one.
It has multiple segments.

2
Chapter two begins.
<quote name="character1">Some dialogue here.</quote>
More content follows.
"""

    # Write test content
    test_file.write_text(test_content)

    # Create minimal book subclass for testing
    class TestBook(Book):
        def _extract_content(self) -> str:
            return self.path.read_text()

    # Process and validate
    book = TestBook(test_file)
    book.process()
    book.validate()

    print(f"Statistics:")
    print("-" * 40)
    stats = book.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print(book)

    test_file.unlink()
