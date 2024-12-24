import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from loguru import logger

from ..book.base import Book
from ..book.types import Segment
from .generators.base import AudioGenerator
from .types import AudioSegment, GenerationStatus
from .voices.registry import VoiceRegistry


class InteractiveReviewer(Protocol):
    """Protocol for interactive review implementations."""

    async def review_segment(self, segment: AudioSegment) -> bool:
        """Review a generated segment and return whether it was approved."""
        ...

    async def handle_error(self, segment: AudioSegment) -> bool:
        """Handle a failed segment and return whether to retry."""
        ...


@dataclass
class GenerationProgress:
    """Tracks generation progress for resume capability."""

    total_segments: int = 0
    completed_segments: int = 0
    current_part: int = 0
    current_chapter: int = 0
    current_segment: int = 0

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_segments == 0:
            return 0.0
        return (self.completed_segments / self.total_segments) * 100

    def to_dict(self) -> dict:
        """Convert progress to dictionary for serialization."""
        return {
            "total_segments": self.total_segments,
            "completed_segments": self.completed_segments,
            "current_part": self.current_part,
            "current_chapter": self.current_chapter,
            "current_segment": self.current_segment,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationProgress":
        """Create progress from dictionary."""
        return cls(**data)


class AudiobookPipeline:
    """Pipeline for generating audiobook from text."""

    def __init__(
        self,
        book: Book,
        generator: AudioGenerator,
        voice_registry: VoiceRegistry,
        output_dir: Path,
        reviewer: InteractiveReviewer | None = None,
        max_retries: int = 3,
        progress_file: Path | None = None,
    ):
        """Initialize pipeline."""
        self.book = book
        self.generator = generator
        self.voice_registry = voice_registry
        self.output_dir = Path(output_dir)
        self.reviewer = reviewer
        self.max_retries = max_retries
        self.progress_file = progress_file or output_dir / "progress.json"
        self.progress = GenerationProgress()
        self._segments: list[AudioSegment] = []
        self._load_or_init_progress()

    def _calculate_total_segments(self) -> int:
        """Calculate total number of segments in book."""
        total = 0
        for part in self.book.parts:
            for chapter in part.chapters:
                total += len(chapter.segments)
        return total

    def _load_or_init_progress(self) -> None:
        """Load progress from file or initialize new progress."""
        try:
            if self.progress_file.exists():
                import json

                data = json.loads(self.progress_file.read_text())
                self.progress = GenerationProgress.from_dict(data)
            else:
                self.progress.total_segments = self._calculate_total_segments()
        except Exception as e:
            logger.error(f"Failed to load progress, starting fresh: {e}")
            self.progress = GenerationProgress(total_segments=self._calculate_total_segments())

    def _save_progress(self) -> None:
        """Save current progress to file."""
        try:
            import json

            self.progress_file.write_text(json.dumps(self.progress.to_dict(), indent=2))
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    def get_output_path(self, part_num: int, chapter_num: int, segment_num: int) -> Path:
        """Get output path for a segment."""
        segment_dir = self.output_dir / f"part_{part_num:02d}" / f"chapter_{chapter_num:02d}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        return segment_dir / f"segment_{segment_num:04d}.wav"

    async def _generate_segment(self, segment: Segment, output_path: Path) -> AudioSegment | None:
        """Generate audio for a single segment."""
        if not segment.character:
            logger.warning(f"Segment has no character: {segment.text[:50]}...")
            return None

        assignment = self.voice_registry.get_assignment(segment.character.name)
        if not assignment:
            logger.error(f"No voice assigned for character: {segment.character.name}")
            return None

        audio_segment = await self.generator.generate(
            text=segment.text,
            voice=assignment.voice,
            output_path=output_path,
        )

        return audio_segment

    async def _process_segment(self, segment: Segment, output_path: Path) -> AudioSegment | None:
        """Process a segment with retries and review."""
        retries = 0
        while retries <= self.max_retries:
            try:
                audio_segment = await self._generate_segment(segment, output_path)
                if not audio_segment:
                    return None

                if self.reviewer:
                    approved = await self.reviewer.review_segment(audio_segment)
                    if approved:
                        audio_segment.status = GenerationStatus.APPROVED
                        return audio_segment

                    audio_segment.status = GenerationStatus.REJECTED
                    retry = await self.reviewer.handle_error(audio_segment)
                    if not retry:
                        return None
                else:
                    # Auto-approve if no reviewer
                    audio_segment.status = GenerationStatus.APPROVED
                    return audio_segment

            except Exception as e:
                logger.error(f"Error generating segment: {e}")
                if self.reviewer:
                    retry = await self.reviewer.handle_error(
                        AudioSegment(
                            text=segment.text,
                            character_name=segment.character.name if segment.character else None,
                            status=GenerationStatus.FAILED,
                            error_message=str(e),
                        )
                    )
                    if not retry:
                        return None
                else:
                    logger.error("No reviewer to handle error, skipping segment")
                    return None

            retries += 1

        logger.error(f"Max retries exceeded for segment: {segment.text[:50]}...")
        return None


async def generate(self) -> None:
    """Generate full audiobook."""
    # Validate voice assignments
    is_valid, missing = self.voice_registry.validate_assignments()
    if not is_valid:
        raise ValueError(f"Missing voice assignments for: {missing}")

    try:
        # Process each segment
        for part_idx, part in enumerate(self.book.parts):
            if part_idx < self.progress.current_part:
                continue

            logger.info(f"Processing Part {part.number}: {part.title or 'Untitled'}")

            for chapter_idx, chapter in enumerate(part.chapters):
                if (
                    part_idx == self.progress.current_part
                    and chapter_idx < self.progress.current_chapter
                ):
                    continue

                logger.info(f"Processing Chapter {chapter.number}")

                for segment_idx, segment in enumerate(chapter.segments):
                    if (
                        part_idx == self.progress.current_part
                        and chapter_idx == self.progress.current_chapter
                        and segment_idx < self.progress.current_segment
                    ):
                        continue

                    # Update progress tracking
                    self.progress.current_part = part_idx
                    self.progress.current_chapter = chapter_idx
                    self.progress.current_segment = segment_idx
                    self._save_progress()

                    # Prepare output path
                    output_path = self.get_output_path(part.number, chapter.number, segment_idx + 1)

                    # Skip if already processed
                    if output_path.exists():
                        logger.info(f"Segment already exists: {output_path}")
                        self.progress.completed_segments += 1
                        continue

                    logger.info(
                        f"Generating segment {segment_idx + 1}/{len(chapter.segments)} "
                        f"of Chapter {chapter.number} in Part {part.number}"
                    )
                    logger.debug(f"Text: {segment.text[:100]}...")

                    # Process segment
                    audio_segment = await self._process_segment(segment, output_path)

                    if audio_segment and audio_segment.status == GenerationStatus.APPROVED:
                        self.progress.completed_segments += 1
                        logger.info(
                            f"Progress: {self.progress.progress_percentage:.1f}% "
                            f"({self.progress.completed_segments}/{self.progress.total_segments})"
                        )
                    else:
                        logger.warning(f"Failed to generate segment: {segment.text[:50]}...")

                # Reset segment counter when moving to next chapter
                self.progress.current_segment = 0

            # Reset chapter counter when moving to next part
            self.progress.current_chapter = 0

        logger.info("Audiobook generation complete!")

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        self._save_progress()  # Save progress before exiting
        raise

    finally:
        # Save final progress
        self._save_progress()


if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    from src.book.books.l_insoutenable import InsoutenableBook
    from src.setting import DATA_PATH, ELEVENLABS_CLIENT, L_INSOUTENABLE_TXT_PATH

    from .generators.eleven_labs import ElevenLabsGenerator
    from .voices.registry import VoiceRegistry

    class SimpleReviewer:
        """Simple interactive reviewer for testing."""

        async def review_segment(self, segment: AudioSegment) -> bool:
            """Play segment and get user approval."""
            if not segment.audio_path or not segment.audio_path.exists():
                return False

            print(f"\nReviewing segment: {segment.text[:100]}...")
            print(f"Character: {segment.character_name}")

            # Play audio
            audio_data = segment.audio_path.read_bytes()
            ElevenLabsGenerator.play_audio(None, audio_data)

            # Get user input
            response = input("\nApprove? (y/n): ").lower()
            return response.startswith("y")

        async def handle_error(self, segment: AudioSegment) -> bool:
            """Handle segment error."""
            print(f"\nError in segment: {segment.error_message}")
            response = input("Retry? (y/n): ").lower()
            return response.startswith("y")

    async def main():
        # Setup test environment
        test_dir = DATA_PATH / "test_audiobook"
        test_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        generator = ElevenLabsGenerator(client=ELEVENLABS_CLIENT, cache_dir=test_dir / "cache")

        registry = VoiceRegistry(
            generator,
            config_path=test_dir / "voices.json",
            required_characters=InsoutenableBook.CHARACTERS,
        )

        # Load book
        book = InsoutenableBook(input_path=L_INSOUTENABLE_TXT_PATH)
        # Limit to first part for testing
        book.parts = book.parts[:1]
        book.parts[0].chapters = book.parts[0].chapters[:1]

        try:
            # Setup voice assignments if needed
            if not registry.config_path.exists():
                print("\nSetting up voice assignments...")
                await registry.refresh_voices()
                voices = registry.get_available_voices()

                print("\nAvailable voices:")
                for i, voice in enumerate(voices, 1):
                    print(f"{i}. {voice.name}")

                # Assign voices to main characters
                main_chars = ["narrator", "tomas", "tereza"]
                for char in main_chars:
                    print(f"\nAssigning voice for {char}")
                    while True:
                        try:
                            idx = int(input("Enter voice number: ")) - 1
                            if 0 <= idx < len(voices):
                                registry.assign_voice(char, voices[idx])
                                break
                            print("Invalid selection")
                        except ValueError:
                            print("Please enter a number")

            # Initialize pipeline
            pipeline = AudiobookPipeline(
                book=book,
                generator=generator,
                voice_registry=registry,
                output_dir=test_dir / "output",
                reviewer=SimpleReviewer(),
            )

            # Generate audiobook
            await pipeline.generate()

        except KeyboardInterrupt:
            print("\nGeneration interrupted!")
        except Exception as e:
            print(f"Error during test: {e}")
        finally:
            # Keep files for inspection
            print(f"\nTest files available at: {test_dir}")

    asyncio.run(main())
