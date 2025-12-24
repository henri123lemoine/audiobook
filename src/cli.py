"""CLI for audiobook generation with Chatterbox TTS."""

from pathlib import Path

import click
from loguru import logger

from src.audio.generators.chatterbox import ChatterboxGenerator
from src.audio.pipeline import AudiobookPipeline, AudioCombiner, VoiceCasting
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
    "--book",
    "-b",
    required=True,
    type=click.Choice(list(BOOK_REGISTRY.keys())),
    help="Book identifier",
)
@click.option(
    "--chapter", "-c", type=int, default=None, help="Specific chapter number (1-indexed, optional)"
)
@click.option(
    "--part", "-p", type=int, default=None, help="Specific part number (1-indexed, optional)"
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: books/<book>/audio)",
)
@click.option(
    "--device",
    "-d",
    type=click.Choice(["cuda", "cpu", "auto"]),
    default="auto",
    help="Device for inference (default: auto)",
)
@click.option(
    "--reference-audio",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Reference audio for voice cloning (~10-30 sec)",
)
@click.option("--language", "-l", default="fr", help="Language code (default: fr)")
@click.option(
    "--silence-ms",
    type=int,
    default=500,
    help="Silence between segments in milliseconds (default: 500)",
)
@click.option(
    "--limit", "-n", type=int, default=None, help="Limit to first N segments (for testing)"
)
@click.option(
    "--test",
    "-t",
    is_flag=True,
    help="Quick test mode: only generate first 5 segments (~2-3 min on GPU)",
)
@click.option(
    "--segment-range",
    type=str,
    default=None,
    help="Generate specific segment range (e.g., '0-99' for segments 0-99). Used by parallel orchestrator.",
)
@click.option(
    "--verify",
    "-v",
    is_flag=True,
    help="Enable STT verification to detect and retry bad generations",
)
@click.option(
    "--whisper-model",
    type=click.Choice(["tiny", "base", "small", "medium", "large-v3"]),
    default="base",
    help="Whisper model size for verification (default: base)",
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
    limit: int | None,
    test: bool,
    segment_range: str | None,
    verify: bool,
    whisper_model: str,
):
    """Generate audiobook from text using Chatterbox TTS.

    Examples:

        # Quick test (~2-3 min on GPU)
        uv run audiobook generate --book absalon --test

        # Generate specific chapter
        uv run audiobook generate --book absalon --chapter 1

        # Generate first 10 segments only
        uv run audiobook generate --book absalon --limit 10

        # Generate full book
        uv run audiobook generate --book absalon
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
    total_segments = sum(len(c.segments) for p in book_instance.parts for c in p.chapters)
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

    # Apply segment limit (--test or --limit)
    segment_limit = limit if limit is not None else (5 if test else None)
    if segment_limit is not None:
        remaining = segment_limit
        for p in book_instance.parts:
            for c in p.chapters:
                if remaining <= 0:
                    c.segments = []
                elif len(c.segments) > remaining:
                    c.segments = c.segments[:remaining]
                    remaining = 0
                else:
                    remaining -= len(c.segments)
        # Remove empty chapters/parts
        for p in book_instance.parts:
            p.chapters = [c for c in p.chapters if c.segments]
        book_instance.parts = [p for p in book_instance.parts if p.chapters]
        logger.info(f"Limited to first {segment_limit} segments (test mode)")

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

    # Setup STT verification if enabled
    verifier = None
    if verify:
        from src.audio.verification import STTVerifier

        # Use CPU for Whisper to avoid cuDNN issues (still fast for short segments)
        logger.info(f"Initializing STT verification with Whisper {whisper_model} on CPU")
        verifier = STTVerifier(
            model_size=whisper_model,
            language=language,
            device="cpu",
        )

    # Handle segment-range mode (used by parallel orchestrator)
    if segment_range:
        import json

        from src.audio.pipeline import preprocess_segments

        # Parse range
        try:
            start, end = map(int, segment_range.split("-"))
        except ValueError:
            raise click.BadParameter(
                f"Invalid segment range: {segment_range}. Use format 'START-END'"
            )

        # Flatten all segments with chapter info
        all_segments = []
        for part in book_instance.parts:
            for chapter in part.chapters:
                processed = preprocess_segments(chapter.segments)
                for seg in processed:
                    all_segments.append(
                        {
                            "part": part.number,
                            "chapter": chapter.number,
                            "segment": seg,
                        }
                    )

        # Validate range
        if start < 0 or end >= len(all_segments) or start > end:
            raise click.BadParameter(
                f"Segment range {start}-{end} invalid. Book has {len(all_segments)} segments (0-{len(all_segments)-1})"
            )

        # Create segments output directory
        segments_dir = output_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        # Generate only the requested range
        target_segments = all_segments[start : end + 1]
        logger.info(f"Generating segments {start}-{end} ({len(target_segments)} segments)")

        from tqdm import tqdm

        from src.audio.types import GenerationStatus
        from src.audio.verification import generate_with_verification

        manifest = []
        pbar = tqdm(
            enumerate(target_segments), total=len(target_segments), desc="Segments", unit="seg"
        )

        for i, seg_info in pbar:
            global_idx = start + i
            segment = seg_info["segment"]
            voice = casting.get_voice_for_character(segment.character)
            output_path = segments_dir / f"segment_{global_idx:05d}.mp3"

            # Skip if already exists
            if output_path.exists():
                manifest.append(
                    {
                        "global_index": global_idx,
                        "part": seg_info["part"],
                        "chapter": seg_info["chapter"],
                        "file": output_path.name,
                    }
                )
                continue

            text_preview = segment.text[:40].replace("\n", " ")
            pbar.set_postfix_str(f"{text_preview}...")

            if verifier:
                try:
                    _, result = generate_with_verification(
                        generator, segment.text, voice, output_path, verifier
                    )
                except RuntimeError as e:
                    logger.error(f"Segment {global_idx} failed: {e}")
            else:
                audio = generator.generate(segment.text, voice, output_path)
                if audio.status != GenerationStatus.COMPLETED:
                    logger.warning(f"Failed to generate segment {global_idx}")

            manifest.append(
                {
                    "global_index": global_idx,
                    "part": seg_info["part"],
                    "chapter": seg_info["chapter"],
                    "file": output_path.name,
                }
            )

        pbar.close()

        # Save manifest
        manifest_path = segments_dir / f"manifest_{start}_{end}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Generated {len(target_segments)} segments to {segments_dir}")
        logger.info(f"Manifest saved to {manifest_path}")
        click.echo(f"\nSuccess! Generated segments {start}-{end} to: {segments_dir}")
        return

    # Standard mode: Create and run pipeline
    pipeline = AudiobookPipeline(
        generator=generator,
        casting=casting,
        output_dir=output_dir,
        combiner=AudioCombiner(silence_duration_ms=silence_ms),
        verifier=verifier,
    )

    # Calculate expected segments for this run
    run_segments = sum(len(c.segments) for p in book_instance.parts for c in p.chapters)
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
    "--device",
    "-d",
    type=click.Choice(["cuda", "cpu", "auto"]),
    default="auto",
    help="Device for model loading",
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


@cli.command("validate")
@click.option(
    "--book",
    "-b",
    required=True,
    type=click.Choice(list(BOOK_REGISTRY.keys())),
    help="Book identifier",
)
@click.option(
    "--min-chars",
    type=int,
    default=30,
    help="Minimum characters per segment (default: 30, based on Chatterbox limits)",
)
@click.option("--fix", "-f", is_flag=True, help="Show suggested fixes for each issue")
def validate(book: str, min_chars: int, fix: bool):
    """Validate book segments for TTS compatibility.

    Checks for segments that are too short for Chatterbox TTS.
    The model needs ~25-30+ characters (5+ tokens) to work reliably.

    Examples:
        uv run audiobook validate --book absalon
        uv run audiobook validate --book absalon --min-chars 40 --fix
    """
    import re

    book_class = BOOK_REGISTRY[book]
    try:
        book_instance = book_class(use_chapter_files=True)
    except TypeError:
        book_instance = book_class()

    issues = []
    total_segments = 0

    for part in book_instance.parts:
        for chapter in part.chapters:
            prev_segment = None
            for i, segment in enumerate(chapter.segments):
                total_segments += 1
                text = segment.text.strip()

                if not text:
                    issues.append(
                        {
                            "type": "empty",
                            "part": part.number,
                            "chapter": chapter.number,
                            "segment": i,
                            "text": "",
                            "prev_text": prev_segment.text[:50] if prev_segment else None,
                        }
                    )
                elif len(text) < min_chars:
                    # Detect if it's just a chapter/section number
                    is_number = re.match(r"^[\d\s\.\-]+$", text)
                    issues.append(
                        {
                            "type": "chapter_number" if is_number else "too_short",
                            "part": part.number,
                            "chapter": chapter.number,
                            "segment": i,
                            "text": text,
                            "length": len(text),
                            "prev_text": prev_segment.text[:50] if prev_segment else None,
                        }
                    )

                prev_segment = segment

    # Report results
    if not issues:
        click.echo(
            click.style(
                f"✓ All {total_segments} segments are valid (>= {min_chars} chars)", fg="green"
            )
        )
        return

    click.echo(
        click.style(f"Found {len(issues)} issues in {total_segments} segments:\n", fg="yellow")
    )

    for issue in issues:
        loc = f"Part {issue['part']}, Chapter {issue['chapter']}, Segment {issue['segment']}"

        if issue["type"] == "empty":
            click.echo(f"  [{loc}] EMPTY segment")
            if fix:
                click.echo(click.style("    → Remove from source file", fg="cyan"))

        elif issue["type"] == "chapter_number":
            click.echo(f"  [{loc}] Chapter number: {issue['text']!r} ({issue['length']} chars)")
            if fix:
                # Suggest transformation
                num = issue["text"].strip()
                click.echo(click.style(f"    → Transform to: 'Chapitre {num}'", fg="cyan"))

        else:  # too_short
            click.echo(f"  [{loc}] Too short: {issue['text']!r} ({issue['length']} chars)")
            if fix:
                if issue["prev_text"]:
                    click.echo(
                        click.style(
                            f"    → Merge with previous: '...{issue['prev_text'][-30:]}'", fg="cyan"
                        )
                    )
                else:
                    click.echo(click.style("    → Merge with next segment", fg="cyan"))

    click.echo(f"\nTo fix: edit the source text files in books/{book}/")
    click.echo("Short segments should be merged with adjacent text or transformed.")


@cli.command("generate-parallel")
@click.option(
    "--book",
    "-b",
    required=True,
    type=click.Choice(list(BOOK_REGISTRY.keys())),
    help="Book identifier",
)
@click.option("--gpus", "-g", type=int, default=10, help="Number of GPU instances (default: 10)")
@click.option("--gpu-type", type=str, default="RTX_3090", help="GPU model (default: RTX_3090)")
@click.option("--max-cost", type=float, default=0.15, help="Max cost per GPU hour (default: $0.15)")
@click.option("--limit", "-n", type=int, default=None, help="Limit total segments to generate (for testing)")
@click.option("--verify/--no-verify", default=True, help="Enable STT verification")
@click.option("--keep-instances", is_flag=True, help="Keep instances after completion")
@click.option("--dry-run", is_flag=True, help="Show plan without executing")
def generate_parallel(
    book: str,
    gpus: int,
    gpu_type: str,
    max_cost: float,
    limit: int | None,
    verify: bool,
    keep_instances: bool,
    dry_run: bool,
):
    """Generate audiobook using multiple GPUs in parallel.

    Distributes segments evenly across GPUs for optimal load balancing.
    Scales linearly - 20 GPUs = 20x speedup.

    Examples:
        uv run audiobook generate-parallel --book absalon --gpus 20 --dry-run
        uv run audiobook generate-parallel --book absalon --gpus 2 --limit 10
    """
    from src.orchestration.parallel import ParallelOrchestrator, estimate

    est = estimate(book, gpus, gpu_type, max_cost, limit)

    click.echo("\n=== Parallel Generation Plan ===\n")
    click.echo(f"Segments: {est['segments']:,}")
    click.echo(f"GPUs: {est['gpus']} × {gpu_type} ({est['available']} available)")
    click.echo(f"Price: ${est['price_per_hour']:.3f}/hr (range: ${est['price_range'][0]:.3f}-${est['price_range'][1]:.3f})")
    click.echo(f"Distribution: ~{est['ranges'][0]['count']} segments per GPU")
    click.echo()
    time_str = (
        f"{est['wall_time_minutes']:.0f} min"
        if est["wall_time_hours"] < 1
        else f"{est['wall_time_hours']:.1f}h"
    )
    click.echo(f"Estimated time: ~{time_str}")
    click.echo(f"Estimated cost: ~${est['total_cost']:.2f} (setup: ${est['setup_cost']:.2f}, generation: ${est['generation_cost']:.2f})")
    click.echo(f"Speedup: {est['speedup']:.0f}x")
    click.echo()

    if dry_run:
        click.echo(click.style("[DRY RUN] Would rent instances and generate", fg="yellow"))
        return

    if not click.confirm("Proceed?"):
        return

    try:
        orch = ParallelOrchestrator(book, verify=verify)
        final_path = orch.run(gpus, gpu_type, max_cost, keep_instances, segment_limit=limit)
        click.echo(click.style(f"\nSuccess! Audiobook: {final_path}", fg="green"))
    except Exception as e:
        click.echo(click.style(f"\nError: {e}", fg="red"))
        raise click.Abort()


@cli.command("instances")
@click.option("--destroy-all", is_flag=True, help="Destroy all running instances")
def manage_instances(destroy_all: bool):
    """List and manage Vast.ai GPU instances."""
    from src.orchestration.vastai import VastAIManager

    manager = VastAIManager()
    instances = manager.get_running_instances()

    if not instances:
        click.echo("No running instances.")
        return

    click.echo(f"\nRunning instances ({len(instances)}):\n")
    total_cost = 0
    for inst in instances:
        cost = inst.get("dph_total", inst.get("dph", 0))
        total_cost += cost
        click.echo(f"  [{inst.get('id')}] {inst.get('gpu_name')} - ${cost:.3f}/hr")

    click.echo(f"\nTotal: ${total_cost:.3f}/hr")

    if destroy_all and click.confirm(f"Destroy all {len(instances)} instances?"):
        import subprocess

        for inst in instances:
            subprocess.run(["vastai", "destroy", "instance", str(inst.get("id"))])
        click.echo("All instances destroyed.")


@cli.command("info")
@click.option(
    "--book",
    "-b",
    required=True,
    type=click.Choice(list(BOOK_REGISTRY.keys())),
    help="Book identifier",
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

    click.echo(
        f"\nTotal: {total_chapters} chapters, {total_segments} segments, {total_chars:,} characters"
    )

    # Estimate audio duration and cost
    words = total_chars / 5  # rough estimate
    minutes = words / 150  # 150 wpm
    hours = minutes / 60

    click.echo(f"\nEstimated audio: ~{hours:.1f} hours")
    click.echo(f"Estimated generation time (RTX 4090): ~{hours * 0.5:.1f} hours")
    click.echo(f"Estimated Vast.ai cost: ~${hours * 0.5 * 0.35:.2f}")


if __name__ == "__main__":
    cli()
