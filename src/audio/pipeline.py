from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from pydub import AudioSegment
from src.book.types import Character

from src.audio.generators.eleven_labs import ElevenLabsGenerator
from src.audio.types import Voice
from src.book.base import Book


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
        segments = sorted(
            segment_dir.glob("segment_*.mp3"), key=lambda f: int(f.stem.split("_")[1])
        )
        if not segments:
            raise FileNotFoundError(f"No segments found in {segment_dir}")

        combined = AudioSegment.from_file(segments[0])
        for segment in segments[1:]:
            combined += self.silence + AudioSegment.from_file(segment)

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


class AudiobookPipeline:
    def __init__(
        self,
        generator: ElevenLabsGenerator,
        casting: VoiceCasting,
        output_dir: Path,
        combiner: AudioCombiner | None = None,
    ):
        self.generator = generator
        self.casting = casting
        self.output_dir = output_dir
        self.combiner = combiner or AudioCombiner()

    def process_book(self, book: Book) -> Path:
        book_dir = self.output_dir / book.path.stem / "audio"
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

        # Generate audio for each segment
        for i, segment in enumerate(chapter.segments):
            if segment.text.strip():  # Skip empty segments
                voice = self.casting.get_voice_for_character(segment.character)
                output_path = chapter_dir / f"segment_{i}.mp3"

                if not output_path.exists():
                    audio = self.generator.generate(segment.text, voice, output_path)
                    if not audio.is_complete():
                        logger.warning(
                            f"Failed to generate segment {i} in chapter {chapter.number}"
                        )

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
