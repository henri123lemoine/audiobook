import re
from typing import List, Optional, Tuple

import pdfplumber
from loguru import logger

from .base import ContentProcessor


class PDFProcessor(ContentProcessor):
    """Process content from PDF files."""

    def extract_metadata(self) -> Tuple[Optional[str], Optional[str]]:
        """Extract title and author from PDF metadata."""
        if self._cached_metadata is None:
            with pdfplumber.open(self.path) as pdf:
                self._cached_metadata = pdf.metadata

        metadata = self._cached_metadata
        return metadata.get("title"), metadata.get("author")

    def extract_content(
        self,
        start_page: int = 0,
        end_page: Optional[int] = None,
        skip_headers: bool = True,
        skip_footers: bool = True,
        remove_hyphenation: bool = True,
    ) -> str:
        """
        Extract and preprocess text content from PDF.

        Args:
            start_page: First page to extract (0-based)
            end_page: Last page to extract (None for all pages)
            skip_headers: Whether to attempt to remove headers
            skip_footers: Whether to attempt to remove footers
            remove_hyphenation: Whether to join hyphenated words

        Returns:
            Preprocessed text content
        """
        # Return cached content if available
        if self._cached_content is not None:
            return self._cached_content

        logger.info(f"Extracting content from PDF: {self.path.name}")
        text_pages = []

        with pdfplumber.open(self.path) as pdf:
            pages = pdf.pages[start_page:end_page]

            for i, page in enumerate(pages):
                if (i + 1) % 25 == 0:
                    logger.debug(f"Processing page {start_page + i + 1}")

                text = page.extract_text()
                if not text.strip():
                    continue

                # Process the page text
                processed_text = self._process_page_text(
                    text, skip_headers, skip_footers, remove_hyphenation
                )
                if processed_text:
                    text_pages.append(processed_text)

        # Join pages and clean the text
        content = self._clean_text("\n\n".join(text_pages))
        self._cached_content = content
        return content

    def _process_page_text(
        self, text: str, skip_headers: bool, skip_footers: bool, remove_hyphenation: bool
    ) -> Optional[str]:
        """
        Process text from a single PDF page.

        Args:
            text: Raw page text
            skip_headers: Whether to remove headers
            skip_footers: Whether to remove footers
            remove_hyphenation: Whether to fix hyphenation

        Returns:
            Processed text or None if page should be skipped
        """
        lines = text.split("\n")

        # Skip empty pages
        if not lines:
            return None

        # Remove headers
        if skip_headers:
            while lines and self._is_header(lines[0]):
                lines.pop(0)
                if not lines:
                    return None

        # Remove footers
        if skip_footers:
            while lines and self._is_footer(lines[-1]):
                lines.pop()
                if not lines:
                    return None

        # Join lines
        text = "\n".join(lines)

        # Fix hyphenation
        if remove_hyphenation:
            text = self._fix_hyphenation(text)

        return text

    def _is_header(self, line: str) -> bool:
        """Check if a line is likely a header."""
        return bool(
            len(line) < 50  # Short lines
            or re.match(r"^\s*\d+\s*$", line)  # Just page numbers
            or line.isupper()  # ALL CAPS lines
            or re.match(r"^\s*[A-Z][a-z]+\s+\d+\s*$", line)  # Chapter headers
        )

    def _is_footer(self, line: str) -> bool:
        """Check if a line is likely a footer."""
        return bool(
            len(line) < 50  # Short lines
            or re.match(r"^\s*\d+\s*$", line)  # Just page numbers
            or re.match(r"^\s*[•\-\u2022]\s*\d+\s*[•\-\u2022]\s*$", line)  # Fancy page numbers
        )

    def _fix_hyphenation(self, text: str) -> str:
        """Fix hyphenation in text."""

        def should_join(m) -> str:
            word1, word2 = m.groups()

            # Don't join if either part is too short
            if len(word1) < 2 or len(word2) < 2:
                return f"{word1}-\n{word2}"

            # Don't join if it looks like a compound word
            if word1[-1].isupper() or word2[0].isupper():
                return f"{word1}-\n{word2}"

            # Join the parts
            return f"{word1}{word2}"

        return re.sub(r"(\w+)-\n(\w+)", should_join, text)

    def extract_chapters(self, content: str) -> List[Tuple[str, str]]:
        """Extract chapters from content."""
        # First look for numbered chapters
        chapters = self._extract_numbered_chapters(content)
        if chapters:
            return chapters

        # Then try other section markers
        chapters = self._extract_sections(content)
        if chapters:
            return chapters

        # If no chapters found, return entire content as one chapter
        logger.warning("No chapter markers found, treating content as single chapter")
        return [("1", content)]

    def _extract_numbered_chapters(self, content: str) -> List[Tuple[str, str]]:
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

    def _extract_sections(self, content: str) -> List[Tuple[str, str]]:
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
