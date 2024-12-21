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
        narrator_voice_id: str = "fr-FR-AlainNeural",
        start_page: int = 7,  # Skip front matter
        end_page: int = 394,  # Skip back matter
    ):
        """
        Initialize L'Insoutenable processor.

        Args:
            input_path: Path to PDF file
            narrator_voice_id: Voice ID for narrator
            start_page: First page to process (0-based)
            end_page: Last page to process
        """
        # Initialize processor with specific page range
        processor = PDFProcessor(input_path)

        super().__init__(
            content_processor=processor,
            title="L'Insoutenable Légèreté de l'Être",
            author="Milan Kundera",
            language="fr",
        )

        self.start_page = start_page
        self.end_page = end_page
        self.narrator_voice_id = narrator_voice_id

    def _load_characters(self) -> Dict[str, Character]:
        """Load character definitions."""
        return {
            "narrator": Character(name="narrator", voice_id=self.narrator_voice_id, language="fr"),
            "tomas": Character(
                name="tomas",
                voice_id="fr-FR-HenriNeural",
                language="fr",
                description="Principal male character",
            ),
            "tereza": Character(
                name="tereza",
                voice_id="fr-FR-DeniseNeural",
                language="fr",
                description="Principal female character",
            ),
            "sabina": Character(
                name="sabina",
                voice_id="fr-FR-ClaudeNeural",
                language="fr",
                description="Tomas's mistress",
            ),
        }

    def _structure_content(self, raw_content: str) -> List[Part]:
        """Convert raw content into structured parts and chapters."""
        # First try splitting into parts
        parts = self._extract_parts(raw_content)
        if not parts:
            # No parts found, treat as single part
            logger.debug("No parts found, treating as single part")
            parts = [self._process_part(1, None, raw_content)]

        return parts

    def _extract_parts(self, content: str) -> List[Part]:
        """Extract parts from content."""
        parts = []

        # Try to find part markers
        part_matches = list(re.finditer(r"(?i)^\s*partie\s+(\d+)\s*$", content, re.MULTILINE))

        if not part_matches:
            return []

        # Process each part
        for i in range(len(part_matches)):
            start = part_matches[i].start()
            end = part_matches[i + 1].start() if i < len(part_matches) - 1 else len(content)

            part_num = int(part_matches[i].group(1))
            part_content = content[start:end].strip()

            parts.append(self._process_part(part_num, None, part_content))

        return parts

    def _process_part(self, number: int, title: str, content: str) -> Part:
        """Process a part into chapters."""
        chapters = []

        # Split into chapters
        chapter_matches = list(
            re.finditer(r"(?i)^\s*(?:chapitre\s+)?(\d+)\s*$", content, re.MULTILINE)
        )

        if not chapter_matches:
            # No chapters found, treat as single chapter
            logger.debug(f"No chapters found in part {number}, treating as single chapter")
            chapters = [self._process_chapter(1, content)]
        else:
            # Process each chapter
            for i in range(len(chapter_matches)):
                start = chapter_matches[i].start()
                end = (
                    chapter_matches[i + 1].start() if i < len(chapter_matches) - 1 else len(content)
                )

                chapter_num = int(chapter_matches[i].group(1))
                chapter_content = content[start:end].strip()

                chapters.append(self._process_chapter(chapter_num, chapter_content))

        return Part(number=number, title=title, chapters=chapters)

    def _process_chapter(self, number: int, content: str) -> Chapter:
        """Process a chapter into segments."""
        segments = []

        # Split into paragraphs
        paragraphs = content.split("\n\n")

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Check for dialogue markers
            if paragraph.startswith("<quote"):
                segments.extend(self._process_quote(paragraph))
            else:
                # Regular narrative
                segments.append(
                    Segment(text=paragraph, voice_id=self.characters["narrator"].voice_id)
                )

        return Chapter(number=number, title=None, segments=segments)

    def _process_quote(self, quote_text: str) -> List[Segment]:
        """Process a quote block into segments."""
        # Parse quote structure
        match = re.search(r'<quote name="([^"]+)"(?:\s+language="([^"]+)")?>([^<]+)', quote_text)

        if not match:
            logger.warning(f"Invalid quote format: {quote_text[:100]}...")
            return [Segment(text=quote_text, voice_id=self.characters["narrator"].voice_id)]

        name, language, text = match.groups()
        character = self.characters.get(name, self.characters["narrator"])

        return [
            Segment(
                text=text.strip(),
                voice_id=character.voice_id,
                character=character,
                language=language or character.language,
            )
        ]


if __name__ == "__main__":
    from ...setting import L_INSOUTENABLE_PDF_PATH

    book = InsoutenableBook(input_path=L_INSOUTENABLE_PDF_PATH)
    book.process()
    book.validate()

    stats = book.get_statistics()
    print(stats)
