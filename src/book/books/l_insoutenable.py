import re
from pathlib import Path
from typing import Dict, List

from loguru import logger

from ..base import Book, Chapter, Character, Part, Segment
from ..processors.pdf import PDFProcessor


class InsoutenableBook(Book):
    """Processor for 'L'Insoutenable Légèreté de l'Être'."""

    def __init__(
        self,
        input_path: Path,
        start_page: int = 7,  # Skip front matter
        end_page: int = 394,  # Skip back matter
    ):
        processor = PDFProcessor(input_path)
        super().__init__(
            content_processor=processor,
            title="L'Insoutenable Légèreté de l'Être",
            author="Milan Kundera",
            language="fr",
        )
        self.start_page = start_page
        self.end_page = end_page

    def _load_characters(self) -> Dict[str, Character]:
        """Load character definitions."""
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

    def _structure_content(self, raw_content: str) -> List[Part]:
        """Convert raw content into structured parts and chapters."""
        # Look for part markers using French ordinals
        part_pattern = r"(?m)^\s*(PREMI[ÈE]RE|DEUXI[ÈE]ME|TROISI[ÈE]ME|QUATRI[ÈE]ME|CINQUI[ÈE]ME|SIXI[ÈE]ME|SEPTI[ÈE]ME)\s+PARTIE\s*$(.*?)(?=\s*(?:PREMI[ÈE]RE|DEUXI[ÈE]ME|TROISI[ÈE]ME|QUATRI[ÈE]ME|CINQUI[ÈE]ME|SIXI[ÈE]ME|SEPTI[ÈE]ME)\s+PARTIE|$)"

        parts = []
        ordinal_to_number = {
            "PREMIERE": 1,
            "PREMIÈRE": 1,
            "DEUXIEME": 2,
            "DEUXIÈME": 2,
            "TROISIEME": 3,
            "TROISIÈME": 3,
            "QUATRIEME": 4,
            "QUATRIÈME": 4,
            "CINQUIEME": 5,
            "CINQUIÈME": 5,
            "SIXIEME": 6,
            "SIXIÈME": 6,
            "SEPTIEME": 7,
            "SEPTIÈME": 7,
        }

        # Find all part matches
        part_matches = list(re.finditer(part_pattern, raw_content, re.DOTALL))

        if not part_matches:
            logger.warning("No parts found using French ordinals, treating as single part")
            parts = [self._process_part(1, None, raw_content)]
        else:
            logger.info(f"Found {len(part_matches)} parts")
            for match in part_matches:
                ordinal = match.group(1)
                content = match.group(2).strip()
                part_num = ordinal_to_number.get(ordinal.replace("È", "E"), 0)

                # Extract part title (text after "PARTIE" until next newline)
                title_match = re.search(r"(?s)PARTIE\s*\n(.*?)(?:\n|$)", match.group(0))
                title = title_match.group(1).strip() if title_match else None

                logger.debug(f"Processing part {part_num}: {title}")
                parts.append(self._process_part(part_num, title, content))

        return parts

    def _process_part(self, number: int, title: str, content: str) -> Part:
        """Process a part into chapters."""
        # Look for single numbers at start of lines, ignoring page numbers
        chapter_pattern = r"(?m)^\s*(\d+)\s*$(?!\d)"  # Matches isolated numbers

        chapters = []
        chapter_matches = list(re.finditer(chapter_pattern, content))

        if not chapter_matches:
            logger.warning(f"No chapters found in part {number}, treating as single chapter")
            chapters = [self._process_chapter(1, content)]
        else:
            logger.info(f"Found {len(chapter_matches)} chapters in part {number}")
            for i in range(len(chapter_matches)):
                start = chapter_matches[i].start()
                end = (
                    chapter_matches[i + 1].start() if i < len(chapter_matches) - 1 else len(content)
                )

                chapter_num = int(chapter_matches[i].group(1))
                chapter_content = content[start:end].strip()

                # Skip if chapter seems too short (might be a page number)
                if len(chapter_content.split()) < 10:
                    logger.debug(f"Skipping suspiciously short chapter {chapter_num}")
                    continue

                logger.debug(
                    f"Processing chapter {chapter_num} ({len(chapter_content.split())} words)"
                )
                chapters.append(self._process_chapter(chapter_num, chapter_content))

        return Part(number=number, title=title, chapters=chapters)

    def _process_chapter(self, number: int, content: str) -> Chapter:
        """Process a chapter into segments."""
        segments = []
        max_segment_words = 1000  # Maximum words per segment

        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        current_segment = []
        current_word_count = 0

        for paragraph in paragraphs:
            # Check for dialogue markers
            if paragraph.startswith("<quote"):
                # First add any accumulated narrative text
                if current_segment:
                    segment_text = " ".join(current_segment)
                    segments.append(
                        Segment(text=segment_text, character=self.characters["narrator"])
                    )
                    current_segment = []
                    current_word_count = 0

                # Add the dialogue segments
                segments.extend(self._process_quote(paragraph))
            else:
                # Count words in paragraph
                words = len(paragraph.split())

                # If adding this paragraph would exceed limit, create new segment
                if current_word_count + words > max_segment_words and current_segment:
                    segment_text = " ".join(current_segment)
                    segments.append(
                        Segment(text=segment_text, character=self.characters["narrator"])
                    )
                    current_segment = []
                    current_word_count = 0

                current_segment.append(paragraph)
                current_word_count += words

        # Add any remaining text
        if current_segment:
            segment_text = " ".join(current_segment)
            segments.append(Segment(text=segment_text, character=self.characters["narrator"]))

        logger.debug(f"Chapter {number}: {len(segments)} segments")
        return Chapter(number=number, title=None, segments=segments)

    def _process_quote(self, quote_text: str) -> List[Segment]:
        """Process a quote block into segments."""
        match = re.search(r'<quote name="([^"]+)"(?:\s+language="([^"]+)")?>([^<]+)', quote_text)

        if not match:
            logger.warning(f"Invalid quote format: {quote_text[:100]}...")
            return [Segment(text=quote_text, character=self.characters["narrator"])]

        name, language, text = match.groups()
        character = self.characters.get(name, self.characters["narrator"])

        return [
            Segment(
                text=text.strip(),
                character=character,
                language=language,
            )
        ]


if __name__ == "__main__":
    from ...setting import L_INSOUTENABLE_PDF_PATH

    book = InsoutenableBook(input_path=L_INSOUTENABLE_PDF_PATH)
    book.process()
    book.validate()

    stats = book.get_statistics()
    print("\nBook Statistics:")
    print("-" * 40)
    for key, value in stats.items():
        print(f"{key}: {value}")
