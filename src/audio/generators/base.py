from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator, Protocol

from ..types import AudioSegment, Voice


class AudioPlayer(Protocol):
    """Protocol for audio playback implementations."""

    def play(self, audio_data: bytes) -> None:
        """Play audio data."""
        ...

    def stop(self) -> None:
        """Stop current playback."""
        ...


class AudioGenerator(ABC):
    """Base class for TTS implementations."""

    def __init__(self, cache_dir: Path | None = None):
        """Initialize generator with optional cache directory."""
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/audio")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def generate(
        self, text: str, voice: Voice, output_path: Path | None = None
    ) -> AudioSegment:
        """Generate audio for text using specified voice."""
        pass

    @abstractmethod
    async def generate_stream(self, text: str, voice: Voice) -> AsyncIterator[bytes]:
        """Stream audio generation for preview purposes."""
        pass

    @abstractmethod
    async def get_available_voices(self) -> list[Voice]:
        """Get list of available voices."""
        pass

    def get_cache_path(self, segment: AudioSegment) -> Path:
        """Get cache path for an audio segment."""
        return self.cache_dir / f"{segment.id}.wav"


if __name__ == "__main__":
    # Example implementation for testing
    class TestGenerator(AudioGenerator):
        async def generate(
            self, text: str, voice: Voice, output_path: Path | None = None
        ) -> AudioSegment:
            segment = AudioSegment(text=text, character_name=None, voice=voice)
            cache_path = output_path or self.get_cache_path(segment)
            # Simulate audio generation
            cache_path.write_text("test audio content")
            segment.mark_complete(cache_path)
            return segment

        async def generate_stream(self, text: str, voice: Voice) -> AsyncIterator[bytes]:
            # Simulate streaming
            chunk = b"test audio chunk"
            yield chunk

        async def get_available_voices(self) -> list[Voice]:
            return [Voice(voice_id="test", name="Test Voice", language="fr")]

    import asyncio

    async def main():
        generator = TestGenerator(cache_dir=Path("test_cache"))
        voice = Voice(voice_id="test", name="Test Voice")

        # Test generation
        segment = await generator.generate("Test text", voice)
        print(f"Generated audio at: {segment.audio_path}")

        # Test streaming
        async for chunk in generator.generate_stream("Test text", voice):
            print(f"Received chunk: {len(chunk)} bytes")

        # Test voice listing
        voices = await generator.get_available_voices()
        print(f"Available voices: {voices}")

        # Cleanup
        if segment.audio_path:
            segment.audio_path.unlink()
        generator.cache_dir.rmdir()

    asyncio.run(main())
