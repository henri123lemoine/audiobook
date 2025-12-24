from src.setting import DATA_PATH

from ..base import Book
from ..types import Character


class InsoutenableBook(Book):
    """
    Processor for 'L'Insoutenable Légèreté de l'Être' by Milan Kundera.
    French Version.
    """

    TXT_PATH = DATA_PATH / "books" / "l_insoutenable" / "doc.txt"
    PDF_PATH = DATA_PATH / "books" / "l_insoutenable" / "doc.pdf"

    CHARACTERS: list[Character] = [
        Character(name="narrator"),
        Character(name="tomas", description="Le personnage principal, un médecin"),
        Character(name="tereza", description="L'amante/épouse de Tomas"),
        Character(name="sabina", description="La maîtresse et amie artiste de Tomas"),
        Character(
            name="cadavres", description="Les cadavres qui apparaissent dans le rêve de Tereza"
        ),
        Character(name="franz", description="Professeur universitaire genevois, amant de Sabina"),
        Character(name="marie-claude", description="L'épouse de Franz"),
        Character(name="mere_de_tereza", description="La mère de Tereza"),
        Character(name="photographe"),
        Character(name="redacteur"),
        Character(name="unknown"),
    ]

    def __init__(self):
        """Initialize L'Insoutenable processor."""
        input_path = self.TXT_PATH
        if not input_path.exists():
            raise FileNotFoundError(f"Text file not found: {input_path}")
        super().__init__(
            title="L'Insoutenable Légèreté de l'Être",
            author="Milan Kundera",
            language="fr",
            path=input_path,
        )


if __name__ == "__main__":
    book = InsoutenableBook()
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
