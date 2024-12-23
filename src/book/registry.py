from pathlib import Path
from typing import Type

from loguru import logger

from .base import Book


class BookRegistry:
    """Registry for managing available book processors."""

    def __init__(self):
        self._registry: dict[str, Type[Book]] = {}

    def register(self, book_id: str, book_class: Type[Book]) -> None:
        """Register a book processor."""
        if book_id in self._registry:
            logger.warning(f"Overwriting existing book processor for {book_id}")
        self._registry[book_id] = book_class
        logger.debug(f"Registered book processor: {book_id}")

    def get_book(self, book_id: str, input_path: Path, **kwargs) -> Book | None:
        """Get an instance of a book processor."""
        book_class = self._registry.get(book_id)
        if not book_class:
            logger.error(f"No book processor registered for ID: {book_id}")
            return None

        return book_class(input_path=input_path, **kwargs)

    def list_available(self) -> list[str]:
        """Get list of registered book IDs."""
        return list(self._registry.keys())
