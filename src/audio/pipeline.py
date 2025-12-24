import re
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from pydub import AudioSegment
from pydub.silence import detect_silence
from src.book.types import Character, Segment

from src.audio.generators.base import AudioGenerator
from src.audio.types import GenerationStatus, Voice
from src.book.base import Book

# Chatterbox needs ~25-30+ chars (5+ tokens) for reliable alignment
MIN_SEGMENT_LENGTH = 30
# Chatterbox has ~40 sec output limit; ~300 chars is safe (~25 sec audio)
MAX_SEGMENT_LENGTH = 300


def split_long_text(text: str, max_length: int = MAX_SEGMENT_LENGTH) -> list[str]:
    """Recursively split long text at natural break points.

    Priority: sentence end (. ! ?) > clause (;) > phrase (: —) > comma > space
    """
    text = text.strip()
    if len(text) <= max_length:
        return [text] if text else []

    # Try splitting at different break points in priority order
    break_patterns = [
        r'(?<=[.!?])\s+',           # Sentence boundaries
        r'(?<=[;])\s*',              # Semicolon
        r'(?<=[:—–])\s*',            # Colon, em-dash
        r'(?<=[,])\s+',              # Comma
    ]

    for pattern in break_patterns:
        parts = re.split(pattern, text, maxsplit=1)
        if len(parts) == 2:
            left, right = parts[0].strip(), parts[1].strip()
            # Only accept if both parts are reasonable
            if len(left) >= MIN_SEGMENT_LENGTH and len(right) >= MIN_SEGMENT_LENGTH:
                # Recursively split each part if still too long
                return split_long_text(left, max_length) + split_long_text(right, max_length)

    # Last resort: split at middle space
    mid = len(text) // 2
    # Find nearest space to middle
    left_space = text.rfind(' ', 0, mid + 50)
    right_space = text.find(' ', mid - 50)

    if left_space > MIN_SEGMENT_LENGTH:
        split_point = left_space
    elif right_space > 0 and right_space < len(text) - MIN_SEGMENT_LENGTH:
        split_point = right_space
    else:
        # No good split point, return as-is (will be long but better than broken)
        logger.warning(f"Could not split long segment ({len(text)} chars): {text[:50]}...")
        return [text]

    left = text[:split_point].strip()
    right = text[split_point:].strip()

    return split_long_text(left, max_length) + split_long_text(right, max_length)


def cleanup_audio(audio: AudioSegment, silence_thresh_db: int = -40, min_silence_ms: int = 200) -> AudioSegment:
    """Remove trailing silence and artifacts from audio.

    Args:
        audio: Audio segment to clean
        silence_thresh_db: Silence threshold in dB (default -40)
        min_silence_ms: Minimum silence duration to detect (default 200ms)

    Returns:
        Cleaned audio with trailing silence removed
    """
    if len(audio) < min_silence_ms:
        return audio

    # Detect silent sections
    silent_ranges = detect_silence(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh_db
    )

    if not silent_ranges:
        return audio

    # Check if last silent range extends to the end
    last_silence_start, last_silence_end = silent_ranges[-1]

    # If silence at end, trim it (keep 100ms padding for natural fade)
    if last_silence_end >= len(audio) - 50:  # Within 50ms of end
        trim_point = last_silence_start + 100  # Keep 100ms of silence
        if trim_point < len(audio):
            trimmed = audio[:trim_point]
            trimmed_amount = len(audio) - len(trimmed)
            if trimmed_amount > 100:
                logger.debug(f"Trimmed {trimmed_amount}ms trailing silence")
            return trimmed

    return audio


@dataclass
class VoiceCasting:
    voice_map: dict[str, Voice]
    default_voice: Voice

    def get_voice_for_character(self, character: Character | None) -> Voice:
        if not character:
            return self.default_voice
        return self.voice_map.get(character.name, self.default_voice)


class AudioCombiner:
    def __init__(self, silence_duration_ms: int = 1000):
        self.silence = AudioSegment.silent(duration=silence_duration_ms)

    def combine_segments(self, segment_dir: Path, output_path: Path) -> None:
        # Natural sort by segment number to preserve narrative order
        segment_files = sorted(
            segment_dir.glob("segment_*.mp3"), key=lambda f: int(f.stem.split("_")[1])
        )
        if not segment_files:
            raise FileNotFoundError(f"No segments found in {segment_dir}")

        # Load and cleanup each segment
        segments = []
        for seg_file in segment_files:
            audio = AudioSegment.from_file(seg_file)
            cleaned = cleanup_audio(audio)
            segments.append(cleaned)

        # Combine with silence between
        combined = segments[0]
        for segment in segments[1:]:
            combined += self.silence + segment

        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.export(output_path, format="mp3")

    def combine_chapters(self, part_dir: Path, output_path: Path) -> None:
        # Natural sort by chapter number
        chapters = sorted(
            part_dir.glob("chapitre_*_full.mp3"), key=lambda f: int(f.stem.split("_")[1])
        )
        if not chapters:
            raise FileNotFoundError(f"No chapters found in {part_dir}")

        combined = AudioSegment.from_file(chapters[0])
        for chapter in chapters[1:]:
            combined += self.silence + AudioSegment.from_file(chapter)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.export(output_path, format="mp3")

    def combine_parts(self, book_dir: Path, output_path: Path) -> None:
        # Natural sort by part number
        parts = sorted(
            book_dir.glob("partie_*_complete.mp3"), key=lambda f: int(f.stem.split("_")[1])
        )
        if not parts:
            raise FileNotFoundError(f"No parts found in {book_dir}")

        combined = AudioSegment.from_file(parts[0])
        for part in parts[1:]:
            # Longer silence between parts
            combined += (self.silence * 3) + AudioSegment.from_file(part)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.export(output_path, format="mp3")


def preprocess_segments(segments: list[Segment]) -> list[Segment]:
    """Preprocess segments for Chatterbox TTS.

    1. Split long segments (>300 chars) at natural break points
    2. Merge short segments (<30 chars) with adjacent ones
    """
    if not segments:
        return []

    # First pass: split long segments
    split_segments = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        if len(text) > MAX_SEGMENT_LENGTH:
            chunks = split_long_text(text)
            logger.debug(f"Split long segment ({len(text)} chars) into {len(chunks)} chunks")
            for chunk in chunks:
                split_segments.append(Segment(text=chunk, character=segment.character))
        else:
            split_segments.append(segment)

    # Second pass: merge short segments
    result = []
    for segment in split_segments:
        text = segment.text.strip()

        if len(text) < MIN_SEGMENT_LENGTH:
            if result:
                prev = result[-1]
                merged_text = f"{prev.text} {text}"
                result[-1] = Segment(text=merged_text, character=prev.character)
                logger.debug(f"Merged short segment ({len(text)} chars): {text!r}")
            else:
                result.append(segment)
        else:
            if result and len(result[-1].text) < MIN_SEGMENT_LENGTH:
                prev = result[-1]
                merged_text = f"{prev.text} {text}"
                result[-1] = Segment(text=merged_text, character=segment.character)
                logger.debug(f"Merged previous short segment into current")
            else:
                result.append(segment)

    return result


class AudiobookPipeline:
    def __init__(
        self,
        generator: AudioGenerator,
        casting: VoiceCasting,
        output_dir: Path,
        combiner: AudioCombiner | None = None,
        verifier: "STTVerifier | None" = None,
    ):
        self.generator = generator
        self.casting = casting
        self.output_dir = output_dir
        self.combiner = combiner or AudioCombiner()
        self.verifier = verifier  # Optional STT verification

    def process_book(self, book: Book) -> Path:
        book_dir = self.output_dir
        book_dir.mkdir(parents=True, exist_ok=True)

        # Process each part
        for part in book.parts:
            part_path = self._process_part(book_dir, part)
            logger.info(f"Completed part {part.number}")

        # Combine all parts into final audiobook
        final_path = book_dir / "audiobook_complete.mp3"
        if not final_path.exists():
            self.combiner.combine_parts(book_dir, final_path)
            logger.info(f"Generated complete audiobook: {final_path}")

        return final_path

    def _process_part(self, book_dir: Path, part) -> Path:
        part_dir = book_dir / f"partie_{part.number}"
        part_path = part_dir.with_name(f"{part_dir.name}_complete.mp3")

        if part_path.exists():
            logger.info(f"Part {part.number} already exists: {part_path}")
            return part_path

        # Process each chapter
        for chapter in part.chapters:
            chapter_path = self._process_chapter(part_dir, chapter)
            logger.info(f"Completed chapter {chapter.number}")

        # Combine chapters into complete part
        self.combiner.combine_chapters(part_dir, part_path)
        return part_path

    def _process_chapter(self, part_dir: Path, chapter) -> Path:
        chapter_dir = part_dir / f"chapitre_{chapter.number}"
        chapter_path = chapter_dir.with_name(f"{chapter_dir.name}_full.mp3")

        if chapter_path.exists():
            logger.info(f"Chapter {chapter.number} already exists: {chapter_path}")
            return chapter_path

        chapter_dir.mkdir(parents=True, exist_ok=True)

        # Preprocess segments to handle short texts
        processed_segments = preprocess_segments(chapter.segments)
        logger.info(f"Processing {len(processed_segments)} segments (from {len(chapter.segments)} original)")

        # Generate audio for each segment
        retry_count = 0
        total_segments = len(processed_segments)

        for i, segment in enumerate(processed_segments):
            voice = self.casting.get_voice_for_character(segment.character)
            output_path = chapter_dir / f"segment_{i}.mp3"

            if not output_path.exists():
                if self.verifier:
                    # Use verification with retries
                    from src.audio.verification import generate_with_verification
                    try:
                        _, result = generate_with_verification(
                            self.generator, segment.text, voice, output_path, self.verifier
                        )
                        if result.attempt > 1:
                            retry_count += 1
                    except RuntimeError as e:
                        logger.error(f"Segment {i} failed after all retries: {e}")
                else:
                    # Direct generation without verification
                    audio = self.generator.generate(segment.text, voice, output_path)
                    if audio.status != GenerationStatus.COMPLETED:
                        logger.warning(
                            f"Failed to generate segment {i} in chapter {chapter.number}"
                        )

        if self.verifier and retry_count > 0:
            logger.info(f"Chapter {chapter.number}: {retry_count}/{total_segments} segments needed retries")

        # Combine segments into complete chapter
        self.combiner.combine_segments(chapter_dir, chapter_path)
        return chapter_path


if __name__ == "__main__":
    from src.book.books.l_insoutenable import InsoutenableBook
    from src.setting import DATA_PATH, ELEVENLABS_CLIENT
    

    generator = ElevenLabsGenerator(ELEVENLABS_CLIENT)
    voices = generator.get_available_voices()

    casting = VoiceCasting(
        voice_map={
            "narrator": next(v for v in voices if v.name == "Julien"),
            "tomas": next(v for v in voices if v.name == "Martin Dupont Intime"),
            "tereza": next(v for v in voices if v.name == "Adina - French young female"),
            "sabina": next(v for v in voices if v.name == "Emilie Lacroix"),
            "franz": next(v for v in voices if v.name == "Theo - Smart, warm, open"),
        },
        default_voice=next(v for v in voices if v.name == "Julien"),
    )

    pipeline = AudiobookPipeline(
        generator=generator, casting=casting, output_dir=DATA_PATH / "books"
    )

    BOOK_PATH = DATA_PATH / "books" / "l_insoutenable.txt"  # Example
    book = InsoutenableBook(BOOK_PATH)
    audiobook_path = pipeline.process_book(book)

    print(f"Generated audiobook at: {audiobook_path}")
