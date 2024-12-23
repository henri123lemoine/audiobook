from pathlib import Path

from .types import Chapter, Part, Segment


class Book:
    def __init__(self, title: str, author: str, language: str, path: Path):
        self.title = title
        self.author = author
        self.language = language
        self.path = Path(path)
        self.parts: list[Part] = []
        self._processed = False
        self._process()

    def _process(self):
        with open(self.path, "r", encoding="utf-8") as f:
            text = f.read()

        raw_parts = text.split("\n\n\n")

        for part_num, raw_part in enumerate(raw_parts, 1):
            raw_chapters = raw_part.strip().split("\n\n")
            title = raw_chapters[0]
            if "PARTIE:" in title:
                title = title.split("PARTIE:")[1].strip()
            raw_chapters.pop(0)

            chapters = []
            for i, text in enumerate(raw_chapters, 1):
                text = text.strip()
                lines = text.split("\n")[1:]  # Skip chapter number

                segments = [
                    Segment.from_text(line, self.language) for line in lines if line.strip()
                ]
                chapters.append(Chapter(number=i, segments=segments))

            self.parts.append(Part(number=part_num, title=title, chapters=chapters))
        self._processed = True

    def validate(self) -> bool:
        """Validate the processed book structure."""
        if not self._processed:
            raise RuntimeError("Book must be processed before validation")

        if not self.parts:
            raise ValueError("Book has no content")

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
                for seg_num, seg in enumerate(chapter.segments, 1):
                    character = f"[{seg.character.name}]" if seg.character else ""
                    text = seg.text[:50] + "..." if len(seg.text) > 50 else seg.text
                    string += f"\n\t\tSegment {seg_num}: " f"{character} {text}"
        return string


if __name__ == "__main__":
    test_file = Path("test_book.txt")

    # Create test content
    test_content = """PREMIÈRE PARTIE: asjdfjaerfja;ejrif

Chapitre 1
This is chapter one.
It has multiple segments.

Chapitre 2
Chapter two begins.
<quote name="character1">Some dialogue here.</quote>
More content follows.


DEUXIÈME PARTIE: le nom de la partie

Chapitre 1
Hii

Chapitre 2
Hello"""

    test_file.write_text(test_content)

    class TestBook(Book):
        def _extract_content(self) -> str:
            return self.path.read_text()

    book = TestBook("title", "author", "French", test_file)
    book.validate()

    print(book)

    test_file.unlink()
