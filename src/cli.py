"""CLI for audiobook generation with Chatterbox TTS."""

import click
from pathlib import Path
from loguru import logger

from src.audio.generators.chatterbox import ChatterboxGenerator
from src.audio.pipeline import AudiobookPipeline, VoiceCasting, AudioCombiner
from src.audio.types import Voice
from src.book.books.absolon import AbsalonBook


# Book registry for CLI access
BOOK_REGISTRY = {
    "absalon": AbsalonBook,
}


@click.group()
def cli():
    """Audiobook generation CLI using Chatterbox TTS."""
    pass


@cli.command()
@click.option(
    "--book", "-b",
    required=True,
    type=click.Choice(list(BOOK_REGISTRY.keys())),
    help="Book identifier"
)
@click.option(
    "--chapter", "-c",
    type=int,
    default=None,
    help="Specific chapter number (1-indexed, optional)"
)
@click.option(
    "--part", "-p",
    type=int,
    default=None,
    help="Specific part number (1-indexed, optional)"
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: books/<book>/audio)"
)
@click.option(
    "--device", "-d",
    type=click.Choice(["cuda", "cpu", "auto"]),
    default="auto",
    help="Device for inference (default: auto)"
)
@click.option(
    "--reference-audio", "-r",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Reference audio for voice cloning (~10-30 sec)"
)
@click.option(
    "--language", "-l",
    default="fr",
    help="Language code (default: fr)"
)
@click.option(
    "--silence-ms",
    type=int,
    default=500,
    help="Silence between segments in milliseconds (default: 500)"
)
def generate(
    book: str,
    chapter: int | None,
    part: int | None,
    output_dir: Path | None,
    device: str,
    reference_audio: Path | None,
    language: str,
    silence_ms: int,
):
    """Generate audiobook from text using Chatterbox TTS.

    Examples:

        # Generate full book
        uv run audiobook generate --book absalon

        # Generate specific chapter
        uv run audiobook generate --book absalon --chapter 1

        # With custom voice reference
        uv run audiobook generate --book absalon -r voice.wav
    """
    # Load book
    logger.info(f"Loading book: {book}")
    book_class = BOOK_REGISTRY[book]

    try:
        book_instance = book_class(use_chapter_files=True)
    except TypeError:
        # Fallback for books that don't support use_chapter_files
        book_instance = book_class()

    # Count total segments for progress reporting
    total_chapters = sum(len(p.chapters) for p in book_instance.parts)
    total_segments = sum(
        len(c.segments)
        for p in book_instance.parts
        for c in p.chapters
    )
    logger.info(f"Book loaded: {total_chapters} chapters, {total_segments} segments")

    # Filter to specific part/chapter if requested
    if part is not None:
        original_parts = len(book_instance.parts)
        book_instance.parts = [p for p in book_instance.parts if p.number == part]
        if not book_instance.parts:
            raise click.BadParameter(f"Part {part} not found (book has {original_parts} parts)")
        logger.info(f"Filtered to part {part}")

    if chapter is not None:
        for p in book_instance.parts:
            original_chapters = len(p.chapters)
            p.chapters = [c for c in p.chapters if c.number == chapter]
        book_instance.parts = [p for p in book_instance.parts if p.chapters]
        if not book_instance.parts:
            raise click.BadParameter(f"Chapter {chapter} not found")
        logger.info(f"Filtered to chapter {chapter}")

    # Setup output directory
    if output_dir is None:
        output_dir = Path("books") / book / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Find default reference audio if not specified
    if reference_audio is None:
        default_refs = [
            Path("assets/voices/default_french.wav"),
            Path("assets/voices/narrator.wav"),
        ]
        for ref in default_refs:
            if ref.exists():
                reference_audio = ref
                logger.info(f"Using default reference audio: {reference_audio}")
                break

    if reference_audio is None:
        logger.warning("No reference audio found - using Chatterbox default voice")

    # Initialize generator
    actual_device = None if device == "auto" else device
    logger.info(f"Initializing Chatterbox generator (device: {device})")

    generator = ChatterboxGenerator(
        device=actual_device,
        language_id=language,
        audio_prompt_path=reference_audio,
    )

    # Setup voice casting (single narrator for now)
    default_voice = Voice(
        voice_id="chatterbox_narrator",
        name="Narrateur",
        language=language,
        preview_url=str(reference_audio) if reference_audio else None,
    )
    casting = VoiceCasting(
        voice_map={},  # Empty for now - all characters use default
        default_voice=default_voice,
    )

    # Create and run pipeline
    pipeline = AudiobookPipeline(
        generator=generator,
        casting=casting,
        output_dir=output_dir,
        combiner=AudioCombiner(silence_duration_ms=silence_ms),
    )

    # Calculate expected segments for this run
    run_segments = sum(
        len(c.segments)
        for p in book_instance.parts
        for c in p.chapters
    )
    logger.info(f"Starting generation for {book} ({run_segments} segments)...")

    result_path = pipeline.process_book(book_instance)

    logger.info(f"Audiobook generated: {result_path}")
    click.echo(f"\nSuccess! Audiobook saved to: {result_path}")


@cli.command("list-books")
def list_books():
    """List available books."""
    click.echo("Available books:")
    for book_id, book_class in BOOK_REGISTRY.items():
        try:
            # Try to get book info without fully loading
            click.echo(f"  - {book_id}")
        except Exception:
            click.echo(f"  - {book_id} (error loading)")


@cli.command("list-voices")
@click.option(
    "--device", "-d",
    type=click.Choice(["cuda", "cpu", "auto"]),
    default="auto",
    help="Device for model loading"
)
def list_voices(device: str):
    """List available voices (reference audio files)."""
    actual_device = None if device == "auto" else device
    generator = ChatterboxGenerator(device=actual_device)

    voices = generator.get_available_voices()
    click.echo(f"Available voices ({len(voices)}):")
    for v in voices:
        ref = f" [{v.preview_url}]" if v.preview_url else ""
        click.echo(f"  - {v.name} ({v.voice_id}){ref}")


@cli.command("info")
@click.option(
    "--book", "-b",
    required=True,
    type=click.Choice(list(BOOK_REGISTRY.keys())),
    help="Book identifier"
)
def book_info(book: str):
    """Show information about a book."""
    book_class = BOOK_REGISTRY[book]

    try:
        book_instance = book_class(use_chapter_files=True)
    except TypeError:
        book_instance = book_class()

    click.echo(f"\nBook: {book_instance.title}")
    click.echo(f"Author: {book_instance.author}")
    click.echo(f"Language: {book_instance.language}")
    click.echo(f"Parts: {len(book_instance.parts)}")

    total_chapters = 0
    total_segments = 0
    total_chars = 0

    for p in book_instance.parts:
        click.echo(f"\n  Part {p.number}: {p.title or '(untitled)'}")
        click.echo(f"    Chapters: {len(p.chapters)}")
        for c in p.chapters:
            seg_count = len(c.segments)
            char_count = sum(len(s.text) for s in c.segments)
            click.echo(f"      Chapter {c.number}: {seg_count} segments, {char_count:,} chars")
            total_chapters += 1
            total_segments += seg_count
            total_chars += char_count

    click.echo(f"\nTotal: {total_chapters} chapters, {total_segments} segments, {total_chars:,} characters")

    # Estimate audio duration and cost
    words = total_chars / 5  # rough estimate
    minutes = words / 150  # 150 wpm
    hours = minutes / 60

    click.echo(f"\nEstimated audio: ~{hours:.1f} hours")
    click.echo(f"Estimated generation time (RTX 4090): ~{hours * 0.5:.1f} hours")
    click.echo(f"Estimated Vast.ai cost: ~${hours * 0.5 * 0.35:.2f}")


if __name__ == "__main__":
    cli()
