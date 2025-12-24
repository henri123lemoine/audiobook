"""Processor for 'Absalon, Absalon!' by William Faulkner."""

from pathlib import Path

from src.setting import DATA_PATH

from ..base import Book, split_if_too_long
from ..types import Chapter, Character, Part, Segment

# Directory containing the book files
BOOK_DIR = Path(__file__).parent.parent.parent.parent / "books" / "absalon"


class AbsalonBook(Book):
    """
    Processor for 'Absalon, Absalon!' by William Faulkner.
    French Version.

    Supports two modes:
    1. use_chapter_files=True: Load from individual chapitre_*.txt files
    2. use_chapter_files=False: Load from the main doc.txt file (original behavior)
    """

    TXT_PATH = DATA_PATH / "books" / "absalon" / "doc.txt"
    PDF_PATH = DATA_PATH / "books" / "absalon" / "doc.pdf"

    CHARACTERS: list[Character] = [
        Character(name="narrator"),
        Character(
            name="quentin_compson", description="Un étudiant de Harvard qui raconte l'histoire"
        ),
        Character(name="rosa_coldfield", description="La belle-sœur de Thomas Sutpen"),
        Character(
            name="thomas_sutpen",
            description="Le protagoniste principal, fondateur de la dynastie Sutpen",
        ),
        Character(name="henry_sutpen", description="Le fils de Thomas Sutpen"),
        Character(name="charles_bon", description="Ami d'Henry à l'université"),
        Character(name="judith_sutpen", description="La fille de Thomas Sutpen"),
        Character(name="ellen_coldfield", description="L'épouse de Thomas Sutpen"),
        Character(name="shreve_mccannon", description="Le colocataire de Quentin à Harvard"),
        Character(name="wash_jones", description="Un habitant pauvre de la propriété de Sutpen"),
        Character(name="mr_compson", description="Le père de Quentin"),
        Character(name="unknown"),
    ]

    def __init__(self, use_chapter_files: bool = True):
        """Initialize Absalon processor.

        Args:
            use_chapter_files: If True, load from individual chapitre_*.txt files
                             in the books/absalon/ directory. If False, use the
                             main doc.txt file (original behavior).
        """
        if use_chapter_files:
            self._init_from_chapter_files()
        else:
            self._init_from_main_file()

    def _init_from_main_file(self):
        """Initialize from the main doc.txt file (original behavior)."""
        input_path = self.TXT_PATH
        if not input_path.exists():
            raise FileNotFoundError(f"Text file not found: {input_path}")
        super().__init__(
            title="Absalon, Absalon!",
            author="William Faulkner",
            language="fr",
            path=input_path,
        )

    def _init_from_chapter_files(self):
        """Initialize from individual chapter files in books/absalon/."""
        # Set basic attributes without calling parent __init__ (which would call _process)
        self.title = "Absalon, Absalon!"
        self.author = "William Faulkner"
        self.language = "fr"
        self.path = BOOK_DIR
        self.parts = []
        self._processed = False

        # Find all chapter files (chapitre_*.txt or chapitre_*_tts.txt)
        chapter_files = self._find_chapter_files()

        if not chapter_files:
            raise FileNotFoundError(
                f"No chapter files found in {BOOK_DIR}. "
                "Expected files like chapitre_1.txt or chapitre_1_tts.txt"
            )

        # Load chapters
        chapters = []
        for chapter_num, chapter_path in sorted(chapter_files.items()):
            chapter = self._load_chapter(chapter_num, chapter_path)
            chapters.append(chapter)

        # Create a single part containing all chapters
        self.parts = [Part(number=1, title="", chapters=chapters)]
        self._processed = True

    def _find_chapter_files(self) -> dict[int, Path]:
        """Find chapter files in the book directory.

        Returns dict mapping chapter number to file path.
        Prefers *_tts.txt files over plain .txt files.
        """
        chapter_files = {}

        if not BOOK_DIR.exists():
            return chapter_files

        # Look for chapitre_N.txt or chapitre_N_tts.txt files
        for txt_file in BOOK_DIR.glob("chapitre_*.txt"):
            # Extract chapter number from filename
            name = txt_file.stem
            parts = name.split("_")

            if len(parts) >= 2 and parts[0] == "chapitre":
                try:
                    chapter_num = int(parts[1])

                    # Check if this is a _tts file
                    is_tts = len(parts) >= 3 and parts[2] == "tts"

                    # Prefer _tts files over plain files
                    if chapter_num not in chapter_files or is_tts:
                        chapter_files[chapter_num] = txt_file

                except ValueError:
                    continue

        return chapter_files

    def _load_chapter(self, chapter_num: int, chapter_path: Path) -> Chapter:
        """Load a single chapter from a text file.

        The file is expected to contain paragraphs separated by blank lines.
        Each paragraph becomes a segment.
        """
        text = chapter_path.read_text(encoding="utf-8")

        # Split into paragraphs (separated by one or more blank lines)
        paragraphs = []
        current = []

        for line in text.split("\n"):
            line = line.strip()
            if line:
                current.append(line)
            elif current:
                paragraphs.append(" ".join(current))
                current = []

        if current:
            paragraphs.append(" ".join(current))

        # Convert paragraphs to segments
        segments = [Segment.from_text(para, self.language) for para in paragraphs if para.strip()]

        # Split long segments
        segments = split_if_too_long(segments)

        return Chapter(number=chapter_num, segments=segments)


if __name__ == "__main__":
    # Test loading from chapter files
    print("Testing chapter file loading...")
    book = AbsalonBook(use_chapter_files=True)

    print(f"\nBook: {book.title} by {book.author}")
    print(f"Language: {book.language}")
    print(f"Parts: {len(book.parts)}")

    total_segments = 0
    total_chars = 0

    for part in book.parts:
        print(f"\nPart {part.number}:")
        for chapter in part.chapters:
            seg_count = len(chapter.segments)
            char_count = sum(len(s.text) for s in chapter.segments)
            print(f"  Chapter {chapter.number}: {seg_count} segments, {char_count:,} chars")
            total_segments += seg_count
            total_chars += char_count

    print(f"\nTotal: {total_segments} segments, {total_chars:,} characters")

    # Show first segment of first chapter
    if book.parts and book.parts[0].chapters:
        first_seg = book.parts[0].chapters[0].segments[0]
        print(f"\nFirst segment preview: {first_seg.text[:200]}...")
