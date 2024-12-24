from pathlib import Path

from loguru import logger
from pydub import AudioSegment

from src.audio.generators.eleven_labs import ElevenLabsGenerator
from src.book.books.l_insoutenable import InsoutenableBook
from src.setting import DATA_PATH, ELEVENLABS_CLIENT, L_INSOUTENABLE_TXT_PATH

generator = ElevenLabsGenerator(ELEVENLABS_CLIENT)
voices = generator.get_available_voices()
print(voices)


book = InsoutenableBook(L_INSOUTENABLE_TXT_PATH)
book.parts = book.parts[0:1]  # Only load the first part for testing

VOICE_MAP = {
    "narrator": next((v for v in voices if v.name == "Julien"), None),
    "tomas": next((v for v in voices if v.name == "Martin Dupont Intime"), None),
    "tereza": next((v for v in voices if v.name == "Adina - French young female"), None),
    "sabina": next((v for v in voices if v.name == "Emilie Lacroix"), None),
    "franz": next((v for v in voices if v.name == "Theo - Smart, warm, open"), None),
}
VOICE_MAP


def join_chapter_segments(chapter_path: Path, output_file: Path) -> None:
    """
    Join all MP3 segments in a chapter into a single MP3 file.

    Args:
        chapter_path (Path): Path to the folder containing chapter segments.
        output_file (Path): Path where the merged chapter file will be saved.
    """
    try:
        # Find all MP3 files in the chapter directory
        segment_files = sorted(
            chapter_path.glob("segment_*.mp3"),
            key=lambda f: int(f.stem.split("_")[1]),  # Sort numerically by segment number
        )
        if not segment_files:
            raise FileNotFoundError(f"No MP3 segments found in {chapter_path}")

        logger.info(f"Joining {len(segment_files)} segments from {chapter_path}...")

        combined_audio = AudioSegment.from_file(segment_files[0])

        for segment_file in segment_files[1:]:
            segment_audio = AudioSegment.from_file(segment_file)
            combined_audio += segment_audio

        combined_audio.export(output_file, format="mp3")
        logger.info(f"Chapter audio saved to: {output_file}")

    except Exception as e:
        logger.error(f"Failed to join chapter segments: {e}")
        raise


generator = ElevenLabsGenerator(ELEVENLABS_CLIENT)

BOOK_PATH = DATA_PATH / "books" / "l_insoutenable" / "audio"
for part in book.parts:
    PART_PATH = BOOK_PATH / f"partie_{part.number}"
    for chapter in part.chapters:
        CHAPTER_PATH = PART_PATH / f"chapitre_{chapter.number}"
        CHAPTER_PATH.mkdir(parents=True, exist_ok=True)
        CHAPTER_FULL_PATH = CHAPTER_PATH.with_name(CHAPTER_PATH.stem + "_full.mp3")
        if CHAPTER_FULL_PATH.exists():
            logger.info(f"Chapter audio already generated: {CHAPTER_FULL_PATH}")
            continue
        for segment_number, segment in enumerate(chapter.segments):
            voice_name = segment.character.name if segment.character else "narrator"
            voice = VOICE_MAP[voice_name]
            output_path = CHAPTER_PATH / f"segment_{segment_number}.mp3"

            # print(segment.text)
            audio_segment = generator.generate(
                text=segment.text, voice=voice, output_path=output_path
            )
            if (
                audio_segment.is_complete()
                and audio_segment.audio_path
                and audio_segment.audio_path.exists()
            ):
                logger.info(f"\nGenerated audio saved to: {audio_segment.audio_path}")
        join_chapter_segments(
            chapter_path=CHAPTER_PATH,
            output_file=CHAPTER_FULL_PATH,
        )
