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

            # Generate audio
            if isinstance(self.client, AsyncElevenLabs):
                response = await self.client.generate(text=text, voice=voice.voice_id)
            else:
                response = self.client.generate(text=text, voice=voice.voice_id)

            # Save to file
            output_path.write_bytes(response)
            segment.mark_complete(output_path)

            return segment

        except Exception as e:
            logger.error(f"Failed to generate audio: {e}")
            segment.mark_failed(str(e))
            raise

    async def generate_stream(self, text: str, voice: Voice) -> AsyncIterator[bytes]:
        """Stream audio generation for preview."""
        try:
            if isinstance(self.client, AsyncElevenLabs):
                stream = await self.client.generate_stream(text=text, voice=voice.voice_id)
                async for chunk in stream:
                    if isinstance(chunk, bytes):
                        yield chunk
            else:
                # For sync client, we need to convert to async
                stream = self.client.generate_stream(text=text, voice=voice.voice_id)
                for chunk in stream:
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
            if isinstance(self.client, AsyncElevenLabs):
                voices = await self.client.voices.get()
            else:
                voices = self.client.voices.get()

            # Convert to our Voice type and cache
            self._voice_cache = {}
            for v in voices:
                voice_id, name = v[:2]  # ElevenLabs returns tuples of (id, name, ...)
                self._voice_cache[voice_id] = Voice(
                    voice_id=voice_id,
                    name=name,
                )

            return list(self._voice_cache.values())

        except Exception as e:
            logger.error(f"Failed to fetch voices: {e}")
            raise

    def play_audio(self, audio_data: bytes) -> None:
        """Play audio using ElevenLabs' player."""
        play(audio_data)


if __name__ == "__main__":
    import asyncio

    from src.setting import ELEVENLABS_CLIENT

    async def main():
        # Initialize generator
        generator = ElevenLabsGenerator(client=ELEVENLABS_CLIENT, cache_dir=Path("test_cache"))

        try:
            # Get available voices
            voices = await generator.get_available_voices()
            print("\nAvailable voices:")
            for voice in voices:
                print(f"- {voice}")

            if not voices:
                print("No voices available!")
                return

            # Test generation with first voice
            test_voice = voices[0]
            print(f"\nTesting generation with voice: {test_voice}")

            segment = await generator.generate("Bonjour! Ceci est un test.", test_voice)
            print(f"Generated audio: {segment.audio_path}")

            # Test streaming
            print("\nTesting streaming...")
            chunks = []
            async for chunk in generator.generate_stream(
                "Un autre test pour le streaming.", test_voice
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
