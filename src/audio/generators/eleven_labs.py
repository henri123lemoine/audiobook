from pathlib import Path
from typing import AsyncIterator

from elevenlabs import play
from elevenlabs.client import AsyncElevenLabs, ElevenLabs
from loguru import logger

from ..types import AudioSegment, GenerationStatus, Voice
from .base import AudioGenerator


class ElevenLabsGenerator(AudioGenerator):
    """ElevenLabs-based audio generator."""

    def __init__(
        self,
        client: ElevenLabs | AsyncElevenLabs,
        cache_dir: Path | None = None,
        model: str = "eleven_multilingual_v2",
    ):
        """Initialize with ElevenLabs client."""
        super().__init__(cache_dir)
        self.client = client
        self.model = model
        self._voice_cache: dict[str, Voice] | None = None

    async def generate(
        self, text: str, voice: Voice, output_path: Path | None = None
    ) -> AudioSegment:
        """Generate audio using ElevenLabs TTS."""
        segment = AudioSegment(
            text=text,
            character_name=None,  # We don't track character here
            voice=voice,
        )

        try:
            # Determine output path
            output_path = output_path or self.get_cache_path(segment)

            # Update status
            segment.status = GenerationStatus.IN_PROGRESS
            logger.debug(f"Generating audio for text: {text[:50]}...")

            # Get the generator from convert
            audio_gen = self.client.text_to_speech.convert(
                text=text, voice_id=voice.voice_id, model_id=self.model
            )

            # Collect all chunks
            chunks = []
            for chunk in audio_gen:
                if isinstance(chunk, bytes):
                    chunks.append(chunk)
                    logger.debug(f"Got chunk of {len(chunk)} bytes")

            if not chunks:
                raise ValueError("No audio chunks received")

            # Combine chunks and write
            audio_data = b"".join(chunks)
            logger.debug(f"Total audio size: {len(audio_data)} bytes")

            output_path.write_bytes(audio_data)
            segment.mark_complete(output_path)
            logger.info(f"Generated audio to {output_path}")

            return segment

        except Exception as e:
            logger.error(f"Failed to generate audio: {e}")
            segment.mark_failed(str(e))
            raise

    async def generate_stream(self, text: str, voice: Voice) -> AsyncIterator[bytes]:
        """Stream audio generation for preview."""
        try:
            audio_gen = self.client.text_to_speech.convert_as_stream(
                text=text, voice_id=voice.voice_id, model_id=self.model
            )

            for chunk in audio_gen:
                if isinstance(chunk, bytes):
                    yield chunk

        except Exception as e:
            logger.error(f"Failed to stream audio: {e}")
            raise

    async def get_available_voices(self) -> list[Voice]:
        """Get available ElevenLabs voices."""
        if self._voice_cache is not None:
            return list(self._voice_cache.values())

        try:
            response = self.client.voices.get_all()

            # Convert to our Voice type and cache
            self._voice_cache = {
                voice.voice_id: Voice(voice_id=voice.voice_id, name=voice.name)
                for voice in response.voices
            }

            return list(self._voice_cache.values())

        except Exception as e:
            logger.error(f"Failed to fetch voices: {e}")
            raise

    def play_audio(self, audio_data: bytes) -> None:
        """Play audio using ElevenLabs' player."""
        play(audio_data)


if __name__ == "__main__":
    import asyncio

    from src.setting import DATA_PATH, ELEVENLABS_CLIENT

    async def main():
        # Setup test environment
        output_dir = DATA_PATH / "test_audio"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "test.mp3"

        generator = ElevenLabsGenerator(client=ELEVENLABS_CLIENT, cache_dir=Path("test_cache"))

        try:
            # Get available voices
            voices = await generator.get_available_voices()
            print("\nAvailable voices:")
            for voice in voices:
                print(f"- {voice}")

            # Find a French voice
            french_voice = next(
                (v for v in voices if v.name.lower() in ["henri", "nicolas", "theo"]), voices[0]
            )
            print(f"\nTesting with voice: {french_voice}")

            # Test generation
            test_text = "Bonjour! Je suis une voix de test."
            print(f"Generating: '{test_text}'")

            segment = await generator.generate(
                text=test_text, voice=french_voice, output_path=output_file
            )

            if segment.is_complete() and segment.audio_path and segment.audio_path.exists():
                print(f"\nGenerated: {segment.audio_path}")
                audio = segment.audio_path.read_bytes()
                print("Playing audio...")
                generator.play_audio(audio)

            # Test streaming
            print("\nTesting streaming...")
            chunks = []
            async for chunk in generator.generate_stream(
                "Un autre test pour le streaming.", french_voice
            ):
                chunks.append(chunk)
                print(f"Received chunk: {len(chunk)} bytes")

            print(f"Total streamed size: {sum(len(c) for c in chunks)} bytes")

        except Exception as e:
            print(f"Error during test: {e}")

        finally:
            # Cleanup
            if generator.cache_dir.exists():
                for file in generator.cache_dir.glob("*"):
                    file.unlink()
                generator.cache_dir.rmdir()

    asyncio.run(main())
