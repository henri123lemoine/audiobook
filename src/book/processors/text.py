import re

from loguru import logger

from .base import ContentProcessor


class TextProcessor(ContentProcessor):
    """Process content from text files."""

    def extract_metadata(self) -> tuple[str | None, str | None]:
        """Extract title and author from text content."""
        if self._cached_metadata is None:
            content = self.extract_content()

            # Look for title/author in first few lines
            lines = content.split("\n")[:10]
            title = author = None

            # Try to find title/author patterns
            for line in lines:
                title_match = re.match(r"(?i)^title:\s*(.+)$", line)
                if title_match:
                    title = title_match.group(1).strip()
                    continue

                author_match = re.match(r"(?i)^(?:author|by):\s*(.+)$", line)
                if author_match:
                    author = author_match.group(1).strip()
                    continue

            # If no explicit metadata, try to guess from filename/content
            if not title:
                title = self.path.stem.replace("_", " ").title()

            self._cached_metadata = {"title": title, "author": author}

        return self._cached_metadata["title"], self._cached_metadata["author"]

    def extract_content(
        self,
        encoding: str = "utf-8",
        skip_metadata: bool = True,
        clean_formatting: bool = True,
        remove_quote_marks: bool = False,
    ) -> str:
        """
        Extract and preprocess text content.

        Args:
            encoding: File encoding
            skip_metadata: Whether to skip metadata lines at start
            clean_formatting: Whether to clean up formatting
            remove_quote_marks: Whether to remove quotation marks

        Returns:
            Preprocessed text content
        """
        if self._cached_content is not None:
            return self._cached_content

        logger.info(f"Reading text file: {self.path.name}")

        with open(self.path, "r", encoding=encoding) as f:
            content = f.read()

        if skip_metadata:
            # Skip metadata lines at start
            lines = content.split("\n")
            content_start = 0

            for i, line in enumerate(lines[:10]):
                if re.match(r"(?i)^(title|author|by|date|language):\s*.+$", line):
                    content_start = i + 1
                else:
                    break

            content = "\n".join(lines[content_start:])

        if clean_formatting:
            # Remove multiple spaces
            content = re.sub(r" +", " ", content)

            # Normalize quotes
            content = content.replace(""", '"').replace(""", '"')
            content = content.replace("'", "'").replace("'", "'")

            # Normalize dashes
            content = content.replace("—", "-").replace("–", "-")

            # Fix spacing around punctuation
            content = re.sub(r"\s+([,.!?;:])", r"\1", content)

            # Normalize paragraph breaks
            content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)
            content = content.strip()

        if remove_quote_marks:
            content = content.replace('"', "")
            content = content.replace("'", "")

        self._cached_content = content
        return content

    def extract_chapters(self, content: str) -> list[tuple[str, str]]:
        """Extract chapters from content."""
        # First look for quoted text markers
        chapters = self._extract_quoted_chapters(content)
        if chapters:
            return chapters

        # Then try numbered chapters
        chapters = self._extract_numbered_chapters(content)
        if chapters:
            return chapters

        # Finally try section markers
        chapters = self._extract_sections(content)
        if chapters:
            return chapters

        # If no chapters found, return entire content as one chapter
        logger.warning("No chapter markers found, treating content as single chapter")
        return [("1", content)]

    def _extract_quoted_chapters(self, content: str) -> list[tuple[str, str]]:
        """Extract chapters marked with XML-style quotes."""
        if not (content.count("<quote") > 0):
            return []

        pattern = r'<quote\s+name="([^"]+)"(?:\s+language="([^"]+)")?>([^<]+)</quote>'
        matches = list(re.finditer(pattern, content))

        if not matches:
            return []

        chapters = []
        for match in matches:
            name = match.group(1)
            language = match.group(2)  # might be None
            text = match.group(3).strip()

            title = f"Quote: {name}"
            if language:
                title += f" ({language})"

            chapters.append((title, text))

        return chapters

    def _extract_numbered_chapters(self, content: str) -> list[tuple[str, str]]:
        """Extract numbered chapters."""
        pattern = r"(?:^|\n)\s*(?:chapitre\s+)?([0-9]+)\s*(?:\n|$)"
        matches = list(re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE))

        if not matches:
            return []

        chapters = []
        for i in range(len(matches)):
            start = matches[i].start()
            end = matches[i + 1].start() if i < len(matches) - 1 else len(content)

            chapter_num = matches[i].group(1)
            chapter_content = content[start:end].strip()
            chapter_title = f"Chapitre {chapter_num}"

            chapters.append((chapter_title, chapter_content))

        return chapters

    def _extract_sections(self, content: str) -> list[tuple[str, str]]:
        """Extract sections using various markers."""
        patterns = [
            (r"(?:^|\n)\s*(?:partie\s+)([0-9]+|[IVX]+)\s*(?:\n|$)", "Partie"),
            (r"(?:^|\n)\s*([IVX]+)\s*(?:\n|$)", "Section"),
        ]

        for pattern, prefix in patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE))
            if matches:
                sections = []
                for i in range(len(matches)):
                    start = matches[i].start()
                    end = matches[i + 1].start() if i < len(matches) - 1 else len(content)

                    section_id = matches[i].group(1)
                    section_content = content[start:end].strip()
                    section_title = f"{prefix} {section_id}"

                    sections.append((section_title, section_content))
                return sections

        return []
