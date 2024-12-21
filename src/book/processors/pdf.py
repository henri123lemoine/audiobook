import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import pdfplumber
from loguru import logger


class CacheMetadata(TypedDict):
    """Cache metadata structure."""

    file_hash: str
    cached_at: str
    page_range: tuple[int, int] | None


@dataclass
class CacheInfo:
    """Tracks cache state and metadata."""

    path: Path
    metadata: CacheMetadata
    is_valid: bool = True


class PDFProcessor:
    """Extract and cache PDF content with minimal processing."""

    def __init__(self, path: Path, cache_dir: Path | None = None, cache_ttl_days: int = 30):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        self.cache_dir = Path(cache_dir) if cache_dir else self.path.parent / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_days = cache_ttl_days

        # Cache setup
        self._file_hash = self._compute_file_hash()
        self._setup_cache()

    def _compute_file_hash(self) -> str:
        """Generate stable hash of PDF content."""
        hasher = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(64 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]

    def _setup_cache(self) -> None:
        """Initialize cache files and metadata."""
        stem = self.path.stem
        self._content_cache = CacheInfo(
            path=self.cache_dir / f"{stem}_{self._file_hash}_content.txt",
            metadata={"file_hash": "", "cached_at": "", "page_range": None},
        )
        self._validate_cache(self._content_cache)

    def _validate_cache(self, cache: CacheInfo) -> None:
        """Check if cache is valid and current."""
        if not (cache.path.exists() and cache.path.with_suffix(".meta").exists()):
            cache.is_valid = False
            return

        try:
            with open(cache.path.with_suffix(".meta")) as f:
                metadata = json.load(f)

            # Basic validation
            if metadata.get("file_hash") != self._file_hash:
                cache.is_valid = False
                return

            cached_at = datetime.fromisoformat(metadata["cached_at"])
            if (datetime.now() - cached_at).days > self.cache_ttl_days:
                cache.is_valid = False
                return

            cache.metadata = metadata
            cache.is_valid = True

        except (json.JSONDecodeError, ValueError, KeyError):
            cache.is_valid = False

    def _save_to_cache(self, content: str, page_range: tuple[int, int] | None = None) -> None:
        """Save extracted content and metadata to cache."""
        try:
            # Save content
            with open(self._content_cache.path, "w", encoding="utf-8") as f:
                f.write(content)

            # Save metadata
            metadata: CacheMetadata = {
                "file_hash": self._file_hash,
                "cached_at": datetime.now().isoformat(),
                "page_range": page_range,
            }

            with open(self._content_cache.path.with_suffix(".meta"), "w") as f:
                json.dump(metadata, f, indent=2)

            self._content_cache.metadata = metadata
            self._content_cache.is_valid = True

        except Exception as e:
            logger.error(f"Failed to write cache: {e}")
            self._content_cache.is_valid = False

    def extract_content(
        self,
        start_page: int = 0,
        end_page: int | None = None,
        force_refresh: bool = False,
    ) -> str:
        """Extract text content from PDF, using cache when valid."""
        page_range = (start_page, end_page if end_page else -1)

        # Check cache first
        if not force_refresh and self._content_cache.is_valid:
            if self._content_cache.metadata["page_range"] == page_range:
                with open(self._content_cache.path, encoding="utf-8") as f:
                    return f.read()

        logger.info(f"Extracting content from PDF: {self.path.name}")

        extracted_text: list[str] = []
        with pdfplumber.open(self.path) as pdf:
            # Validate page range
            max_pages = len(pdf.pages)
            end_page = min(end_page or max_pages, max_pages)
            start_page = min(start_page, end_page)

            # Extract text page by page
            for i, page in enumerate(pdf.pages[start_page:end_page]):
                if (i + 1) % 25 == 0:
                    logger.debug(
                        f"Processing page {start_page + i + 1}/{end_page} "
                        f"({(i+1)/(end_page-start_page):.1%})"
                    )

                text = page.extract_text()
                if text := text.strip():
                    extracted_text.append(text)

        # Basic cleaning only - structural parsing happens elsewhere
        content = "\n\n".join(extracted_text)
        self._save_to_cache(content, page_range)
        return content

    def invalidate_cache(self) -> None:
        """Clear the cache files."""
        try:
            if self._content_cache.path.exists():
                self._content_cache.path.unlink()
            meta_path = self._content_cache.path.with_suffix(".meta")
            if meta_path.exists():
                meta_path.unlink()
            self._content_cache.is_valid = False
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")


if __name__ == "__main__":
    from src.setting import DATA_PATH, L_INSOUTENABLE_PDF_PATH

    test_cache_dir = DATA_PATH / "tests" / "test_cache"
    processor = PDFProcessor(L_INSOUTENABLE_PDF_PATH, cache_dir=test_cache_dir)

    # First extraction (should create cache)
    content1 = processor.extract_content(start_page=7, end_page=10)
    print(f"Initial extraction length: {len(content1)} chars")

    # Second extraction (should use cache)
    content2 = processor.extract_content(start_page=7, end_page=10)
    print(f"Cached content length: {len(content2)} chars")
    print(f"Contents identical: {content1 == content2}")

    # Force refresh
    content3 = processor.extract_content(start_page=7, end_page=10, force_refresh=True)
    print(f"Force refresh length: {len(content3)} chars")
    print(f"Contents still identical: {content1 == content3}")
