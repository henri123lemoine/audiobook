import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from loguru import logger

from ..base import Book
from ..processors.pdf import PDFProcessor
from ..types import Chapter, Character, Part, Segment


@dataclass
class ParsedContent:
    """Track parsing state and results."""

    current_part: int = 1
    current_chapter: int = 1
    parts: list[Part] = field(default_factory=list)
    current_segments: list[Segment] = field(default_factory=list)

    def start_new_part(self, title: str | None = None) -> None:
        """Start a new part, completing the previous one if it exists."""
        if self.current_segments and self.parts:
            # Complete current chapter
            self.parts[-1].chapters.append(
                Chapter(number=self.current_chapter, title=None, segments=self.current_segments)
            )

        # Reset for new part
        self.current_segments = []
        self.current_chapter = 1
        self.parts.append(Part(number=self.current_part, title=title, chapters=[]))
        self.current_part += 1


class InsoutenableBook(Book):
    """Processor for 'L'Insoutenable Légèreté de l'Être'."""

    def __init__(
        self,
        input_path: Path,
        start_page: int = 7,  # Skip front matter
        end_page: int = 394,  # Skip back matter
    ) -> None:
        processor = PDFProcessor(input_path)
        super().__init__(
            content_processor=processor,
            title="L'Insoutenable Légèreté de l'Être",
            author="Milan Kundera",
            language="fr",
        )
        self.start_page = start_page
        self.end_page = end_page

    def _load_characters(self) -> dict[str, Character]:
        """Define the book's characters."""
        return {
            "narrator": Character(name="narrator"),
            "tomas": Character(
                name="tomas",
                description="Principal male character",
            ),
            "tereza": Character(
                name="tereza",
                description="Principal female character",
            ),
            "sabina": Character(
                name="sabina",
                description="Tomas's mistress",
            ),
        }

    def _structure_content(self, raw_content: str) -> Sequence[Part]:
        """Convert raw content into properly structured parts and chapters."""
        parsed = ParsedContent()

        # Split into potential part boundaries first
        part_pattern = (
            r"(?P<part_header>(?:PREMI[ÈE]RE|DEUXI[ÈE]ME|TROISI[ÈE]ME|"
            r"QUATRI[ÈE]ME|CINQUI[ÈE]ME|SIXI[ÈE]ME|SEPTI[ÈE]ME)\s+PARTIE)"
            r"(?:\s+(?P<part_title>[^\n]+))?"
            r"(?P<content>.*?)"
            r"(?=\s*(?:PREMI[ÈE]RE|DEUXI[ÈE]ME|TROISI[ÈE]ME|"
            r"QUATRI[ÈE]ME|CINQUI[ÈE]ME|SIXI[ÈE]ME|SEPTI[ÈE]ME)\s+PARTIE|$)"
        )

        parts = list(re.finditer(part_pattern, raw_content, re.DOTALL))

        if not parts:
            logger.warning("No parts found, treating as single part")
            parsed.start_new_part()
            self._process_content_block(raw_content, parsed)
            return parsed.parts

        for part_match in parts:
            part_title = part_match.group("part_title")
            content = part_match.group("content").strip()

            parsed.start_new_part(title=part_title)
            self._process_content_block(content, parsed)

        return parsed.parts

    def _process_content_block(self, content: str, parsed: ParsedContent) -> None:
        """Process a block of content, handling chapters and segments."""
        # Split into chapters first
        chapter_splits = re.split(r"\n\s*(\d+)\s*\n", content)

        if len(chapter_splits) == 1:
            # No chapter markers found, treat as single chapter
            self._process_chapter_content(content, parsed)
            return

        # Process each chapter
        for i in range(1, len(chapter_splits), 2):
            chapter_num = int(chapter_splits[i])
            chapter_content = chapter_splits[i + 1].strip()

            if chapter_num != parsed.current_chapter:
                # Complete previous chapter if we have segments
                if parsed.current_segments:
                    parsed.parts[-1].chapters.append(
                        Chapter(
                            number=parsed.current_chapter,
                            title=None,
                            segments=parsed.current_segments,
                        )
                    )
                    parsed.current_segments = []

                parsed.current_chapter = chapter_num

            self._process_chapter_content(chapter_content, parsed)

    def _process_chapter_content(self, content: str, parsed: ParsedContent) -> None:
        """Process chapter content into segments."""
        # Split into paragraphs first
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        for paragraph in paragraphs:
            if paragraph.startswith("<quote"):
                # Handle quote blocks
                quote_segments = self._process_quote(paragraph)
                parsed.current_segments.extend(quote_segments)
            else:
                # Handle narrative paragraphs - split if too long
                words = paragraph.split()
                max_words = 200  # Reasonable segment length

                for i in range(0, len(words), max_words):
                    segment_words = words[i : i + max_words]
                    parsed.current_segments.append(
                        Segment(text=" ".join(segment_words), character=self.characters["narrator"])
                    )

    def _process_quote(self, quote_text: str) -> list[Segment]:
        """Process a quote block into segments."""
        match = re.search(r'<quote name="([^"]+)"(?:\s+language="([^"]+)")?>([^<]+)', quote_text)

        if not match:
            logger.warning(f"Invalid quote format: {quote_text[:100]}...")
            return [Segment(text=quote_text, character=self.characters["narrator"])]

        name, language, text = match.groups()
        character = self.characters.get(name, self.characters["narrator"])

        return [Segment(text=text.strip(), character=character, language=language)]


if __name__ == "__main__":
    from ...setting import L_INSOUTENABLE_PDF_PATH

    book = InsoutenableBook(input_path=L_INSOUTENABLE_PDF_PATH)
    book.process()
    book.validate()

    book.parts[0].chapters[0].segments[0].text

    stats = book.get_statistics()
    print("\nBook Statistics:")
    print("-" * 40)
    for key, value in stats.items():
        print(f"{key}: {value}")
