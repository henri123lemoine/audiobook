from .books import InsoutenableBook
from .registry import BookRegistry

book_registry = BookRegistry()
book_registry.register("l_insoutenable", InsoutenableBook)

__all__ = ["registry", "Book"]
