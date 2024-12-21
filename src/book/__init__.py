from .l_insoutenable import InsoutenableBook
from .registry import BookRegistry

registry = BookRegistry()
registry.register("l_insoutenable", InsoutenableBook)

__all__ = ["registry", "Book"]
