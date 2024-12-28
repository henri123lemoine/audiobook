from pathlib import Path

from ..base import Book
from ..types import Character


class InsoutenableBook(Book):
    """
    Processor for 'L'Insoutenable Légèreté de l'Être' by Milan Kundera.
    French Version.
    """

    # {'unknown', 'redacteur', 'sabina', 'cadavres', 'mere_de_tereza', 'franz', 'tereza', 'tomas', 'photographe'}
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

    def __init__(
        self,
        input_path: Path,
    ):
        """Initialize L'Insoutenable processor."""
        super().__init__(
            title="L'Insoutenable Légèreté de l'Être",
            author="Milan Kundera",
            language="fr",
            path=input_path,
        )


if __name__ == "__main__":
    from src.setting import L_INSOUTENABLE_TXT_PATH

    book = InsoutenableBook(input_path=L_INSOUTENABLE_TXT_PATH)
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
