"""CLI for audiobook generation with Chatterbox TTS."""

import sys
from pathlib import Path

import click
from loguru import logger

# Configure logging BEFORE other imports: DEBUG to file, INFO+ to stderr
logger.remove()  # Remove default handler
Path("logs").mkdir(exist_ok=True)
logger.add(
    "logs/audiobook.log",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
)
logger.add(
    sys.stderr,
    level="INFO",
    format="<level>{level:<8}</level> | {message}",
)

from src.audio.generators.chatterbox import ChatterboxGenerator
from src.audio.pipeline import AudiobookPipeline, AudioCombiner, VoiceCasting
from src.audio.types import Voice
from src.book.catalog import BOOK_REGISTRY, load_book


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
    "--gpus",
    "-g",
    type=int,
    default=None,
    help="Number of Vast.ai GPU instances (enables parallel generation)",
)
@click.option(
    "--gpu-type",
    type=str,
    default=None,
    help="Vast.ai GPU model (required with --gpus)",
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
@click.option(
    "--no-reference-audio",
    is_flag=True,
    help="Disable voice cloning and use the model default voice",
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
    help="Quick test mode: only generate first 5 segments",
)
@click.option(
    "--segment-range",
    type=str,
    default=None,
    help="Generate specific segment range (e.g., '0-99' for segments 0-99). Used by parallel orchestrator.",
    hidden=True,
)
@click.option(
    "--verify/--no-verify",
    "-v",
    default=None,
    show_default=False,
    help="Enable or disable STT verification",
)
@click.option(
    "--whisper-model",
    type=click.Choice(["tiny", "base", "small", "medium", "large-v3"]),
    default="base",
    help="Whisper model size for verification (default: base)",
)
def generate(
    book: str,
    gpus: int | None,
    gpu_type: str | None,
    chapter: int | None,
    part: int | None,
    output_dir: Path | None,
    device: str,
    reference_audio: Path | None,
    no_reference_audio: bool,
    language: str,
    silence_ms: int,
    limit: int | None,
    test: bool,
    segment_range: str | None,
    verify: bool | None,
    whisper_model: str,
):
    """Generate audiobook from text using Chatterbox TTS.

    Examples:

        # Quick test (first 5 segments)
        uv run audiobook generate --book absalon --test

        # Generate specific chapter
        uv run audiobook generate --book absalon --chapter 1

        # Generate first 10 segments only
        uv run audiobook generate --book absalon --limit 10

        # Generate full book
        uv run audiobook generate --book absalon

        # Generate on Vast.ai (parallel)
        uv run audiobook generate --book absalon --gpus 10 --gpu-type RTX_3090
    """
    if gpus is None and gpu_type is not None:
        raise click.BadParameter("--gpu-type requires --gpus")

    if gpus is not None:
        if gpus <= 0:
            raise click.BadParameter("--gpus must be positive")
        if gpu_type is None or not gpu_type.strip():
            raise click.BadParameter("--gpu-type is required when --gpus is set")
        if segment_range:
            raise click.BadParameter("--segment-range cannot be used with --gpus")
        if chapter is not None or part is not None:
            raise click.BadParameter("Chapter/part filters are not supported with --gpus")
        if output_dir is not None:
            raise click.BadParameter("--output-dir is not supported with --gpus")
        if device != "auto":
            raise click.BadParameter("--device is not supported with --gpus")
        if silence_ms != 500:
            raise click.BadParameter("--silence-ms is not supported with --gpus")
        if reference_audio and no_reference_audio:
            raise click.BadParameter(
                "Use either --reference-audio or --no-reference-audio, not both"
            )

        verify_flag = True if verify is None else verify
        segment_limit = limit if limit is not None else (5 if test else None)

        remote_reference_audio: str | None = None
        if reference_audio is not None:
            repo_root = Path(__file__).resolve().parent.parent
            ref_path = reference_audio.resolve()
            try:
                remote_reference_audio = str(ref_path.relative_to(repo_root))
            except ValueError:
                raise click.BadParameter(
                    "--reference-audio must be inside the repo for --gpus runs"
                )

        from src.orchestration.robust import RobustOrchestrator

        orch = RobustOrchestrator(
            book,
            verify=verify_flag,
            segment_limit=segment_limit,
            reference_audio=remote_reference_audio,
            no_reference_audio=no_reference_audio,
            language=language,
            whisper_model=whisper_model,
        )

        status = orch.status()
        click.echo("\n=== Current Status ===")
        click.echo(
            f"Completed: {status['completed']}/{status['total']} segments ({status['percent']:.1f}%)"
        )
        click.echo(f"Remaining: {status['remaining']} segments")

        if status["remaining"] == 0:
            click.echo(click.style("\nAll segments complete!", fg="green"))
            return

        click.echo(f"\nMissing ranges: {len(status['missing_ranges'])}")
        for start, end in status["missing_ranges"][:5]:
            click.echo(f"  {start}-{end} ({end - start + 1} segments)")
        if len(status["missing_ranges"]) > 5:
            click.echo(f"  ... and {len(status['missing_ranges']) - 5} more")

        click.echo(f"\nWill rent {gpus} {gpu_type} instances")

        if not click.confirm("\nProceed?"):
            return

        try:
            result = orch.run(gpus, gpu_type)
            click.echo("\n=== Run Complete ===")
            click.echo(f"Completed: {result['completed']}/{result['total']} segments")
            click.echo(f"Remaining: {result['remaining']} segments")

            if result["remaining"] > 0:
                click.echo(
                    click.style("\nRun again to generate remaining segments", fg="yellow")
                )
            else:
                click.echo(click.style("\nAll segments complete!", fg="green"))
                if result.get("output_path"):
                    click.echo(f"Output: {result['output_path']}")
        except Exception as e:
            click.echo(click.style(f"\nError: {e}", fg="red"))
            raise click.Abort()

        return

    # Load book
    logger.info(f"Loading book: {book}")
    book_instance = load_book(book)

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

    if reference_audio and no_reference_audio:
        raise click.BadParameter("Use either --reference-audio or --no-reference-audio, not both")

    # Default reference audio for Chatterbox
    if no_reference_audio:
        reference_audio = None
    elif reference_audio is None:
        reference_audio = Path("assets/voices/default_french.wav")
        if not reference_audio.exists():
            raise click.BadParameter(
                "Default reference audio not found at assets/voices/default_french.wav. "
                "Provide --reference-audio or pass --no-reference-audio."
            )
        logger.info(f"Using default reference audio: {reference_audio}")

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
    verify_flag = False if verify is None else verify
    verifier = None
    if verify_flag:
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
        import time

        from pydub import AudioSegment as PydubSegment

        from src.audio.pipeline import preprocess_segments
        from src.orchestration.throughput import detect_gpu_type

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
                f"Segment range {start}-{end} invalid. Book has {len(all_segments)} segments (0-{len(all_segments) - 1})"
            )

        # Create segments output directory
        segments_dir = output_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        gpu_type_value, gpu_name_raw = detect_gpu_type(generator.device)

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

            start_time = time.perf_counter()
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
            wall_seconds = time.perf_counter() - start_time

            audio_seconds = None
            if output_path.exists():
                audio_segment = PydubSegment.from_file(output_path)
                audio_seconds = audio_segment.duration_seconds

            manifest.append(
                {
                    "global_index": global_idx,
                    "part": seg_info["part"],
                    "chapter": seg_info["chapter"],
                    "file": output_path.name,
                    "chars": len(segment.text),
                    "audio_seconds": audio_seconds,
                    "wall_seconds": wall_seconds,
                    "gpu_type": gpu_type_value,
                    "gpu_name_raw": gpu_name_raw,
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

    book_instance = load_book(book)

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


@cli.command("estimate")
@click.option(
    "--book",
    "-b",
    required=True,
    type=click.Choice(list(BOOK_REGISTRY.keys())),
    help="Book identifier",
)
@click.option("--gpus", "-g", type=int, required=True, help="Number of GPU instances")
@click.option("--gpu-type", type=str, required=True, help="GPU model to estimate (Vast.ai)")
def estimate(book: str, gpus: int, gpu_type: str):
    """Estimate time and cost using manifest throughput and live pricing.

    Requires throughput samples from prior generation on the target GPU.
    """
    if gpus <= 0:
        raise click.BadParameter("--gpus must be positive")
    from src.orchestration import estimate as estimate_parallel

    try:
        result = estimate_parallel(
            book_id=book,
            gpu_count=gpus,
            gpu_type=gpu_type,
        )
    except Exception as e:
        click.echo(click.style(f"\nError: {e}", fg="red"))
        raise click.Abort()

    def format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = seconds / 60
        if minutes < 60:
            return f"{minutes:.1f} min"
        hours = minutes / 60
        return f"{hours:.2f} hr"

    throughput = result["throughput_entry"]
    estimate_data = result["estimate"]

    click.echo(f"\nSegments: {result['segments']}")
    click.echo(f"Total chars: {result['total_chars']}")
    click.echo(f"GPUs: {result['gpus']} ({result['gpu_type']})")
    if throughput["gpu_name_raw"]:
        click.echo(f"GPU name: {', '.join(throughput['gpu_name_raw'])}")
    click.echo(
        "Throughput sample: "
        f"{throughput['segments']} segments from {throughput['segments_dir']}"
    )
    click.echo(
        f"Throughput: {throughput['audio_seconds_per_wall_second']:.2f} audio-sec/s, "
        f"{throughput['audio_seconds_per_char']:.4f} audio-sec/char"
    )
    click.echo(f"Estimated total audio: {format_duration(estimate_data['total_audio_seconds'])}")
    click.echo(f"Estimated wall time: {format_duration(estimate_data['wall_seconds'])}")
    click.echo(f"Estimated total GPU time: {estimate_data['total_gpu_hours']:.2f} GPU-hr")
    click.echo(
        f"Price/hr per GPU (snapshot): ${result['price_per_hour']:.3f} "
        f"(min ${result['price_range'][0]:.3f}, max ${result['price_range'][1]:.3f})"
    )
    click.echo(f"Estimated total cost: ${estimate_data['total_cost']:.2f}")
    click.echo(f"Offers considered: {result['offers_considered']}")
    click.echo("Note: excludes instance setup, sync, and retries.")


@cli.command("status")
@click.option(
    "--book",
    "-b",
    required=True,
    type=click.Choice(list(BOOK_REGISTRY.keys())),
    help="Book identifier",
)
def generation_status(book: str):
    """Check generation progress and running Vast.ai instances."""
    from src.orchestration.robust import RobustOrchestrator
    from src.orchestration.vastai import VastAIManager

    orch = RobustOrchestrator(book)
    status = orch.status()

    click.echo(f"\n=== {book} Generation Status ===")
    click.echo(f"Completed: {status['completed']}/{status['total']} segments")
    click.echo(f"Progress: {status['percent']:.1f}%")
    click.echo(f"Remaining: {status['remaining']} segments")

    if status["missing_ranges"]:
        click.echo(f"\nMissing ranges ({len(status['missing_ranges'])}):")
        total_missing = 0
        for start, end in status["missing_ranges"]:
            count = end - start + 1
            total_missing += count
            click.echo(f"  {start}-{end} ({count} segments)")
        click.echo(f"\nTotal missing: {total_missing} segments")
    else:
        click.echo(click.style("\nAll segments complete!", fg="green"))
    try:
        manager = VastAIManager()
        instances = manager.get_running_instances()
    except Exception as e:
        raise click.ClickException(f"Failed to fetch Vast.ai instances: {e}")
    click.echo(f"\n=== Vast.ai Instances ({len(instances)}) ===")
    if not instances:
        click.echo("No running instances.")
        return

    total_cost = 0.0
    for inst in instances:
        if "dph_total" in inst:
            cost = inst["dph_total"]
        elif "dph" in inst:
            cost = inst["dph"]
        else:
            raise click.ClickException(f"Instance {inst.get('id')} missing price data")
        if "gpu_name" not in inst:
            raise click.ClickException(f"Instance {inst.get('id')} missing gpu_name")
        total_cost += cost
        click.echo(f"  [{inst.get('id')}] {inst['gpu_name']} - ${cost:.3f}/hr")

    click.echo(f"\nTotal: ${total_cost:.3f}/hr")


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
    from src.audio.pipeline import preprocess_segments

    book_instance = load_book(book)

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
            processed = preprocess_segments(c.segments)
            seg_count = len(processed)
            char_count = sum(len(s.text) for s in processed)
            click.echo(f"      Chapter {c.number}: {seg_count} segments, {char_count:,} chars")
            total_chapters += 1
            total_segments += seg_count
            total_chars += char_count

    click.echo(
        f"\nTotal: {total_chapters} chapters, {total_segments} segments, {total_chars:,} characters"
    )


if __name__ == "__main__":
    cli()
