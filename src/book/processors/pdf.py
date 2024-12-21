import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pdfplumber
from loguru import logger

from .base import ContentProcessor


class PDFProcessor(ContentProcessor):
    """Process content from PDF files with caching."""

    def __init__(self, path: Path, cache_dir: Path | None = None):
        """
        Initialize PDF processor with caching.

        Args:
            path: Path to PDF file
            cache_dir: Directory for cache files (default: .cache next to PDF)
        """
        super().__init__(path)

        # Set up cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else self.path.parent / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate cache keys based on file hash
        self._file_hash = self._compute_file_hash()
        self._content_cache_path = (
            self.cache_dir / f"{self.path.stem}_{self._file_hash}_content.txt"
        )
        self._metadata_cache_path = (
            self.cache_dir / f"{self.path.stem}_{self._file_hash}_metadata.json"
        )

    def _compute_file_hash(self) -> str:
        """Compute a hash of the PDF file for cache validation."""
        hasher = hashlib.sha256()
        with open(self.path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(64 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]  # First 12 chars are sufficient

    def extract_metadata(self) -> tuple[str | None, str | None]:
        """Extract title and author from PDF metadata with caching."""
        # Check cache first
        if self._metadata_cache_path.exists():
            try:
                with open(self._metadata_cache_path) as f:
                    cached = json.load(f)
                logger.debug(f"Using cached metadata from {self._metadata_cache_path}")
                return cached.get("title"), cached.get("author")
            except Exception as e:
                logger.warning(f"Failed to read metadata cache: {e}")

        # Extract fresh metadata
        with pdfplumber.open(self.path) as pdf:
            metadata = pdf.metadata or {}

        # Cache the results
        cache_data = {
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "cached_at": datetime.now().isoformat(),
        }
        try:
            with open(self._metadata_cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write metadata cache: {e}")

        return metadata.get("title"), metadata.get("author")

    def _load_from_cache(self) -> str | None:
        """Load content from cache file if it exists."""
        if self._cached_content is not None:
            return self._cached_content

        if self._content_cache_path.exists():
            try:
                with open(self._content_cache_path, encoding="utf-8") as f:
                    self._cached_content = f.read()
                    logger.info(f"Using cached content from {self._content_cache_path}")
                    return self._cached_content
            except Exception as e:
                logger.warning(f"Failed to read content cache: {e}")
        return None

    def _save_to_cache(self, content: str) -> None:
        """Save content to cache file."""
        try:
            with open(self._content_cache_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.debug(f"Cached content to {self._content_cache_path}")
        except Exception as e:
            logger.warning(f"Failed to write content cache: {e}")

    def extract_content(
        self,
        start_page: int = 0,
        end_page: int | None = None,
        skip_headers: bool = True,
        skip_footers: bool = True,
        remove_hyphenation: bool = True,
        force_refresh: bool = False,
    ) -> str:
        """
        Extract and preprocess text content from PDF with caching.

        Args:
            start_page: First page to extract (0-based)
            end_page: Last page to extract (None for all pages)
            skip_headers: Whether to attempt to remove headers
            skip_footers: Whether to attempt to remove footers
            remove_hyphenation: Whether to join hyphenated words
            force_refresh: Force extraction even if cache exists

        Returns:
            Preprocessed text content
        """
        # Try to use cache first
        if not force_refresh:
            cached_content = self._load_from_cache()
            if cached_content is not None:
                return cached_content

        # If we get here, either force_refresh is True or cache miss
        logger.info(f"Extracting content from PDF: {self.path.name}")
        text_pages = []

        with pdfplumber.open(self.path) as pdf:
            # Validate page range
            max_pages = len(pdf.pages)
            end_page = min(end_page or max_pages, max_pages)
            start_page = min(start_page, end_page)

            pages = pdf.pages[start_page:end_page]
            total_pages = len(pages)

            for i, page in enumerate(pages):
                if (i + 1) % 25 == 0:
                    logger.debug(
                        f"Processing page {start_page + i + 1}/{end_page} ({(i+1)/total_pages:.1%})"
                    )

                text = page.extract_text()
                if not text.strip():
                    continue

                processed_text = self._process_page_text(
                    text, skip_headers, skip_footers, remove_hyphenation
                )
                if processed_text:
                    text_pages.append(processed_text)

        # Join pages and clean the text
        content = self._clean_text("\n\n".join(text_pages))

        # Cache the content both in memory and on disk
        self._cached_content = content
        self._save_to_cache(content)
        return content

    def invalidate_cache(self) -> None:
        """Clear cached content and metadata."""
        super().invalidate_cache()
        try:
            if self._content_cache_path.exists():
                self._content_cache_path.unlink()
            if self._metadata_cache_path.exists():
                self._metadata_cache_path.unlink()
            logger.debug("Cache invalidated")
        except Exception as e:
            logger.warning(f"Failed to invalidate cache: {e}")

    def _process_page_text(
        self, text: str, skip_headers: bool, skip_footers: bool, remove_hyphenation: bool
    ) -> str | None:
        """Process text from a single PDF page."""
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

    def extract_chapters(self, content: str) -> list[tuple[str, str]]:
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

            if len(chapter_content.split()) < 10:  # Skip suspiciously short chapters
                continue

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


if __name__ == "__main__":
    from src.setting import L_INSOUTENABLE_PDF_PATH

    pdf_path = L_INSOUTENABLE_PDF_PATH
    processor = PDFProcessor(pdf_path)

    # First run - should extract and cache
    content1 = processor.extract_content()

    # Second run - should use cache  (TODO: but doesn't currently, idk why!)
    content2 = processor.extract_content()

    # Force refresh - should extract again
    content3 = processor.extract_content(force_refresh=True)

    # Test cache invalidation
    processor.invalidate_cache()
