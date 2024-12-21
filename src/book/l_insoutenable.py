import re
from pathlib import Path
from typing import Dict, List

from loguru import logger

from .base import Book, Chapter, Character, Part, Segment


class InsoutenableBook(Book):
    """Processor for 'L'Insoutenable Légèreté de l'Être'."""

    def __init__(self, input_path: Path, narrator_voice_id: str):
        """Initialize the book processor."""
        super().__init__(
            input_path=input_path,
            title="L'Insoutenable Légèreté de l'Être",
            author="Milan Kundera",
            language="fr",
            narrator_voice_id=narrator_voice_id,
        )

    def _load_characters(self) -> Dict[str, Character]:
        """Load character definitions."""
        # These could be loaded from a config file in practice
        return {
            "narrator": Character(name="narrator", voice_id=self.narrator_voice_id, language="fr"),
            "tomas": Character(name="tomas", voice_id="fr-FR-HenriNeural", language="fr"),
            "tereza": Character(name="tereza", voice_id="fr-FR-DeniseNeural", language="fr"),
            "sabina": Character(name="sabina", voice_id="fr-FR-ClaudeNeural", language="fr"),
            # Add other characters as needed
        }

    def _process_content(self) -> List[Part]:
        """Process the book content into parts and chapters."""
        with open(self.input_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split into parts (if any)
        part_splits = re.split(r"(?i)^\s*partie\s+(\d+)\s*$", content, flags=re.MULTILINE)
        if len(part_splits) == 1:  # No parts found
            # Treat the whole book as a single part
            parts = [self._process_part(1, None, content)]
        else:
            # Process each part
            parts = []
            for i in range(1, len(part_splits), 2):
                part_num = int(part_splits[i])
                part_content = part_splits[i + 1].strip()
                parts.append(self._process_part(part_num, None, part_content))

        return parts

    def _process_part(self, number: int, title: str, content: str) -> Part:
        """Process a part of the book into chapters."""
        # Split into chapters
        chapter_splits = re.split(r"(?i)^\s*(?:chapitre\s+)?(\d+)\s*$", content, flags=re.MULTILINE)

        chapters = []
        for i in range(1, len(chapter_splits), 2):
            chapter_num = int(chapter_splits[i])
            chapter_content = chapter_splits[i + 1].strip()
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

            # Check for character dialogue
            if paragraph.startswith("<quote"):
                segments.extend(self._process_quote(paragraph))
            else:
                # Regular narrative
                segments.append(Segment(text=paragraph, voice_id=self.narrator_voice_id))

        return Chapter(number=number, title=None, segments=segments)

    def _process_quote(self, quote_text: str) -> List[Segment]:
        """Process a quote block into one or more segments."""
        segments = []

        # Match basic quote with optional language
        match = re.search(r'<quote name="([^"]+)"(?:\s+language="([^"]+)")?>(.+)', quote_text)
        if not match:
            logger.warning(f"Invalid quote format: {quote_text[:100]}...")
            return [Segment(text=quote_text, voice_id=self.narrator_voice_id)]

        name, language, text = match.groups()
        character = self.characters.get(name, self.characters["narrator"])

        # Split into multiple segments if needed (e.g., for emphasis)
        # For now, just create a single segment
        segments.append(
            Segment(
                text=text.strip(),
                voice_id=character.voice_id,
                character=character,
                language=language or character.language,
            )
        )

        return segments


if __name__ == "__main__":
    book = InsoutenableBook(
        input_path=Path("data/books/l_insoutenable.txt"), narrator_voice_id="fr-FR-AlainNeural"
    )
    book.process()
    book.validate()

    stats = book.get_statistics()
    print(stats)

    segments = book.get_segments()
