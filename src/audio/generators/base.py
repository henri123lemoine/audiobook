from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Protocol

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

    @abstractmethod
    def generate(self, text: str, voice: Voice, output_path: Path | None = None) -> AudioSegment:
        """Generate audio for text using specified voice."""
        pass

    def generate_stream(self, text: str, voice: Voice) -> Iterator[bytes]:
        """Stream audio generation for preview purposes. Optional."""
        raise NotImplementedError("Streaming not supported by this generator")

    @abstractmethod
    def get_available_voices(self) -> list[Voice]:
        """Get list of available voices."""
        pass


if __name__ == "__main__":

    class TestGenerator(AudioGenerator):
        def generate(
            self, text: str, voice: Voice, output_path: Path | None = None
        ) -> AudioSegment:
            segment = AudioSegment(text=text, character_name=None, voice=voice)
            output_path.write_text("test audio content")
            segment.mark_complete(output_path)
            return segment

        def generate_stream(self, text: str, voice: Voice) -> Iterator[bytes]:
            # Simulate streaming
            chunk = b"test audio chunk"
            yield chunk

        def get_available_voices(self) -> list[Voice]:
            return [Voice(voice_id="test", name="Test Voice", language="fr")]

    import asyncio

    async def main():
        generator = TestGenerator()
        voice = Voice(voice_id="test", name="Test Voice")

        # Test generation
        segment = generator.generate("Test text", voice)
        print(f"Generated audio at: {segment.audio_path}")

        # Test streaming
        for chunk in generator.generate_stream("Test text", voice):
            print(f"Received chunk: {len(chunk)} bytes")

        # Test voice listing
        voices = generator.get_available_voices()
        print(f"Available voices: {voices}")

        if segment.audio_path:
            segment.audio_path.unlink()

    asyncio.run(main())
