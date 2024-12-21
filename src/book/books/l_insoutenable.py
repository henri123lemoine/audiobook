from pathlib import Path

from ..pdf_book import PDFBook
from ..types import Character


class InsoutenableBook(PDFBook):
    """Processor for 'L'Insoutenable Légèreté de l'Être'."""

    def __init__(
        self,
        input_path: Path,
        cache_dir: Path | None = None,
    ):
        """Initialize L'Insoutenable processor."""
        super().__init__(
            input_path=input_path,
            title="L'Insoutenable Légèreté de l'Être",
            author="Milan Kundera",
            language="fr",
            start_page=7,  # Skip front matter
            end_page=394,  # Skip back matter
            cache_dir=cache_dir,
        )

    def _load_characters(self) -> dict[str, Character]:
        """Define the book's characters."""
        return {
            "narrator": Character(name="narrator"),
            "tomas": Character(
                name="tomas",
                description="Principal male character",
            ),
            "tereza": Character(
                name="tereza",
                description="Principal female character",
            ),
            "sabina": Character(
                name="sabina",
                description="Tomas's mistress",
            ),
            # Additional characters from README
            "cadavre": Character(name="cadavre", description="Multiple cadavers that appear"),
            "beethoven": Character(name="beethoven", description="Beethoven references"),
            "tereza_mother": Character(name="tereza_mother", description="Tereza's mother"),
            "tereza_father": Character(name="tereza_father", description="Tereza's father"),
            "inconnu": Character(name="inconnu", description="Unknown/anonymous character"),
            "photographe": Character(name="photographe", description="Photographer character"),
            "redacteur": Character(name="redacteur", description="Editor character"),
        }


if __name__ == "__main__":
    from src.setting import L_INSOUTENABLE_PDF_PATH, PROJECT_PATH

    # Test the full pipeline
    cache_dir = PROJECT_PATH / "tests" / "test_cache"
    book = InsoutenableBook(
        input_path=L_INSOUTENABLE_PDF_PATH,
        cache_dir=cache_dir,
    )

    # Process the book
    book.process()

    # Print statistics
    print("\nBook Statistics:")
    print("-" * 40)
    stats = book.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Print first part structure as sample
    if book.parts:
        part = book.parts[0]
        print(f"\nFirst part: {part.title or 'Untitled'}")

        for chapter in part.chapters[:2]:  # Just first 2 chapters
            print(f"\n  Chapter {chapter.number}:")
            for para in chapter.paragraphs[:2]:  # Just first 2 paragraphs
                for seg in para.segments[:2]:  # Just first 2 segments
                    print(f"    {seg.text[:100]}...")
