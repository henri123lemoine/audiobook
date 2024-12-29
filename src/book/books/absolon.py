from src.setting import DATA_PATH

from ..base import Book
from ..types import Character


class AbsalonBook(Book):
    """
    Processor for 'Absalon, Absalon!' by William Faulkner.
    French Version.
    """
    
    TXT_PATH = DATA_PATH / "books" / "absalon" / "doc.txt"
    PDF_PATH = DATA_PATH / "books" / "absalon" / "doc.pdf"

    CHARACTERS: list[Character] = [
        Character(name="narrator"),
        Character(name="quentin_compson", description="Un étudiant de Harvard qui raconte l'histoire"),
        Character(name="rosa_coldfield", description="La belle-sœur de Thomas Sutpen"),
        Character(name="thomas_sutpen", description="Le protagoniste principal, fondateur de la dynastie Sutpen"),
        Character(name="henry_sutpen", description="Le fils de Thomas Sutpen"),
        Character(name="charles_bon", description="Ami d'Henry à l'université"),
        Character(name="judith_sutpen", description="La fille de Thomas Sutpen"),
        Character(name="ellen_coldfield", description="L'épouse de Thomas Sutpen"),
        Character(name="shreve_mccannon", description="Le colocataire de Quentin à Harvard"),
        Character(name="wash_jones", description="Un habitant pauvre de la propriété de Sutpen"),
        Character(name="mr_compson", description="Le père de Quentin"),
        Character(name="unknown")
    ]

    def __init__(self):
        """Initialize Absalon processor."""
        input_path = self.TXT_PATH
        if not input_path.exists():
            raise FileNotFoundError(f"Text file not found: {input_path}")
        super().__init__(
            title="Absalon, Absalon!",
            author="William Faulkner",
            language="fr",
            path=input_path,
        )


if __name__ == "__main__":
    book = AbsalonBook()
    book.parts = book.parts[:1]  # Only load the first part for testing
    book.parts[0].chapters = book.parts[0].chapters[:1]  # Only load the first chapter for testing
    book.validate()

    print(book)

    all_characters = set()
    for part in book.parts:
        for chapter in part.chapters:
            for segment in chapter.segments:
                if segment.character:
                    all_characters.add(segment.character.name)
    print(all_characters)
