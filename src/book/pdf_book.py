from pathlib import Path

from .base import Book
from .processors.pdf import PDFProcessor


class PDFBook(Book):
    """Base class for books sourced from PDF files."""

    def __init__(
        self,
        input_path: Path,
        title: str | None = None,
        author: str | None = None,
        language: str = "fr",
        start_page: int = 0,
        end_page: int | None = None,
        cache_dir: Path | None = None,
    ):
        """Initialize PDF-based book processor.

        Args:
            input_path: Path to PDF file
            title: Book title (optional)
            author: Book author (optional)
            language: Primary language
            start_page: First page to process (0-based)
            end_page: Last page to process (inclusive)
            cache_dir: Directory for PDF cache
        """
        super().__init__(input_path, title, author, language)

        self.start_page = start_page
        self.end_page = end_page
        self._pdf_processor = PDFProcessor(input_path, cache_dir=cache_dir)

    def _extract_content(self) -> str:
        """Extract content from PDF using cached processor."""
        return self._pdf_processor.extract_content(
            start_page=self.start_page,
            end_page=self.end_page,
        )


if __name__ == "__main__":
    from src.setting import DATA_PATH, L_INSOUTENABLE_PDF_PATH

    # Test with L'Insoutenable
    class TestPDFBook(PDFBook):
        def _load_characters(self):
            return {
                "narrator": {"name": "narrator"},
                "tomas": {"name": "tomas", "description": "Main character"},
            }

    # Test full extraction and parsing
    cache_dir = DATA_PATH / "tests" / "test_cache"
    book = TestPDFBook(
        input_path=L_INSOUTENABLE_PDF_PATH,
        cache_dir=cache_dir,
        start_page=7,  # Skip front matter
        end_page=10,  # Just test a few pages
    )

    # Process the book
    book.process()

    # Print structure
    print("\nProcessed Book Structure:")
    print("-" * 40)

    stats = book.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Print first part/chapter structure
    if book.parts:
        part = book.parts[0]
        print(f"\nFirst part: {part.title or 'Untitled'}")

        for chapter in part.chapters:
            print(f"\n  Chapter {chapter.number}:")
            for para_num, para in enumerate(chapter.paragraphs, 1):
                print(f"\n    Paragraph {para_num}:")
                for seg_num, seg in enumerate(para.segments, 1):
                    print(f"      Segment {seg_num}: " f"{seg.text[:50]}...")
