import re
from typing import Iterator

from loguru import logger

from ..types import Chapter, Paragraph, Part, Segment


class ContentParser:
    """Parse raw text content into structured parts/chapters/paragraphs/segments."""

    # Part markers in French books
    PART_PATTERN = re.compile(
        r"(?:PREMI[ÈE]RE|DEUXI[ÈE]ME|TROISI[ÈE]ME|"
        r"QUATRI[ÈE]ME|CINQUI[ÈE]ME|SIXI[ÈE]ME|SEPTI[ÈE]ME)\s+PARTIE"
        r"(?:\s+(?P<title>[^\n]+))?",
        re.IGNORECASE,
    )

    # Chapter markers (number followed by optional title)
    CHAPTER_PATTERN = re.compile(r"(?:^|\n)\s*(\d+)\s*(?:\n|$)")

    # Quote/dialogue markers
    QUOTE_PATTERN = re.compile(r'<quote name="([^"]+)"(?:\s+language="([^"]+)")?>([^<]+)</quote>')

    def parse_content(self, content: str) -> list[Part]:
        """Parse raw content into structured parts."""
        # Split into parts first
        part_splits = self._split_into_parts(content)

        parts: list[Part] = []
        for part_num, (part_title, part_content) in enumerate(part_splits, 1):
            chapters = self._parse_chapters(part_content)
            parts.append(Part(number=part_num, title=part_title, chapters=chapters))

        if not parts:
            # If no parts found, treat entire content as single part
            logger.warning("No parts found, treating content as single part")
            chapters = self._parse_chapters(content)
            parts.append(Part(number=1, title=None, chapters=chapters))

        return parts

    def _split_into_parts(self, content: str) -> list[tuple[str | None, str]]:
        """Split content into part title/content pairs."""
        parts: list[tuple[str | None, str]] = []

        # Find all part markers
        matches = list(self.PART_PATTERN.finditer(content))
        if not matches:
            return []

        # Extract each part's content
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i < len(matches) - 1 else len(content)

            part_title = match.group("title")
            part_content = content[start:end].strip()
            parts.append((part_title, part_content))

        return parts

    def _parse_chapters(self, content: str) -> list[Chapter]:
        """Parse content into chapters."""
        chapters: list[Chapter] = []

        # Split by chapter markers
        chapter_splits = self.CHAPTER_PATTERN.split(content)
        if len(chapter_splits) <= 1:
            # No chapter markers found
            paragraphs = self._parse_paragraphs(content)
            chapters.append(Chapter(number=1, title=None, paragraphs=paragraphs))
            return chapters

        # Process each chapter
        current_chapter = 0
        for i in range(1, len(chapter_splits), 2):
            chapter_num = int(chapter_splits[i])
            chapter_content = chapter_splits[i + 1].strip()

            # Validate chapter numbers are sequential
            if chapter_num != current_chapter + 1:
                logger.warning(
                    f"Non-sequential chapter numbers: {current_chapter} -> {chapter_num}"
                )
            current_chapter = chapter_num

            paragraphs = self._parse_paragraphs(chapter_content)
            chapters.append(Chapter(number=chapter_num, title=None, paragraphs=paragraphs))

        return chapters

    def _parse_paragraphs(self, content: str) -> list[Paragraph]:
        """Split content into paragraphs and parse segments."""
        paragraphs: list[Paragraph] = []

        # Split on double newlines for paragraphs
        for para_text in content.split("\n\n"):
            if not (para_text := para_text.strip()):
                continue

            segments = list(self._parse_segments(para_text))
            if segments:
                paragraphs.append(Paragraph(segments=segments))

        return paragraphs

    def _parse_segments(self, paragraph: str) -> Iterator[Segment]:
        """Parse paragraph into segments (sentences/dialogue)."""
        # First check for quote blocks
        quote_match = self.QUOTE_PATTERN.search(paragraph)
        if quote_match:
            # Handle structured quote
            _, language, text = quote_match.groups()
            yield Segment(text=text.strip(), is_dialogue=True, language=language)
            return

        # Otherwise split into natural segments
        current_text: list[str] = []

        for line in paragraph.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Split on major punctuation but keep together logical phrases
            chunks = re.split(r"([.!?]+(?=\s|$))", line)

            for i in range(0, len(chunks), 2):
                text = chunks[i]
                punctuation = chunks[i + 1] if i + 1 < len(chunks) else ""

                current_text.append(text + punctuation)

                # Yield complete segments
                if punctuation and len(current_text) > 0:
                    segment_text = " ".join(current_text).strip()
                    if segment_text:
                        yield Segment(text=segment_text)
                    current_text = []

        # Yield any remaining text
        if current_text:
            segment_text = " ".join(current_text).strip()
            if segment_text:
                yield Segment(text=segment_text)


if __name__ == "__main__":
    # Test the parser with sample content
    sample_content = """
PREMIÈRE PARTIE LES PARADOXES

1
L'éternel retour est une idée mystérieuse et, avec elle, Nietzsche a mis bien des philosophes dans l'embarras.

2
Imaginez maintenant que les mains de votre montre avancent en sens inverse. Non, elles ne reviennent pas lentement en arrière, comme elles le feraient dans le miroir.

<quote name="tomas" language="german">Ich liebe dich</quote>

3
C'est arrivé il y a trois semaines. Il l'avait rencontrée par hasard dans une autre ville. Ils avaient passé la nuit ensemble. Elle était partie le matin même pour Prague et lui, il prenait son service à l'hôpital à midi.
    """

    parser = ContentParser()
    parts = parser.parse_content(sample_content)

    # Print structure
    print("\nParsed Structure:")
    print("-" * 40)

    for part in parts:
        print(f"\nPart {part.number}: {part.title or 'Untitled'}")
        for chapter in part.chapters:
            print(f"\n  Chapter {chapter.number}:")
            for para_num, para in enumerate(chapter.paragraphs, 1):
                print(f"\n    Paragraph {para_num}:")
                for seg_num, seg in enumerate(para.segments, 1):
                    print(
                        f"      Segment {seg_num}: "
                        f"{'[DIALOGUE]' if seg.is_dialogue else ''} "
                        f"{seg.text[:50]}..."
                    )
