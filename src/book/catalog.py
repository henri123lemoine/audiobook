"""Shared book registry and loader helpers."""

from typing import Type

from .base import Book
from .books.absolon import AbsalonBook
from .books.l_insoutenable import InsoutenableBook

BOOK_REGISTRY: dict[str, Type[Book]] = {
    "absalon": AbsalonBook,
    "l_insoutenable": InsoutenableBook,
}


def get_book_class(book_id: str) -> Type[Book]:
    """Return the book class for a known identifier."""
    try:
        return BOOK_REGISTRY[book_id]
    except KeyError as exc:
        raise ValueError(f"Unknown book id: {book_id}") from exc


def load_book(book_id: str) -> Book:
    """Instantiate a book by id."""
    book_class = get_book_class(book_id)
    return book_class()
