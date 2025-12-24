"""Chatterbox Multilingual TTS audio generator."""

from pathlib import Path
from typing import Iterator

import torch
import torchaudio
from loguru import logger

from ..types import AudioSegment, GenerationStatus, Voice
from .base import AudioGenerator


class ChatterboxGenerator(AudioGenerator):
    """Chatterbox Multilingual TTS audio generator.

    Uses the Chatterbox model for text-to-speech with optional voice cloning
    via reference audio files.
    """

    def __init__(
        self,
        device: str | None = None,
        language_id: str = "fr",
        audio_prompt_path: Path | str | None = None,
    ):
        """Initialize Chatterbox TTS generator.

        Args:
            device: Device to run on ("cuda", "cpu", or None for auto-detect)
            language_id: Default language code (e.g., "fr" for French)
            audio_prompt_path: Path to reference audio for voice cloning (~10-30 sec)
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language_id = language_id
        self.audio_prompt_path = Path(audio_prompt_path) if audio_prompt_path else None
        self._model = None  # Lazy loading to avoid VRAM usage until needed

    @property
    def model(self):
        """Lazy load the multilingual model on first use."""
        if self._model is None:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            logger.info(f"Loading Chatterbox Multilingual model on {self.device}...")
            self._model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            logger.info("Chatterbox Multilingual model loaded successfully")
        return self._model

    def generate(
        self,
        text: str,
        voice: Voice,
        output_path: Path | None = None
    ) -> AudioSegment:
        """Generate audio for text using Chatterbox TTS.

        Args:
            text: Text to synthesize
            voice: Voice configuration (uses preview_url for reference audio if available)
            output_path: Path to save the generated audio (MP3 or WAV)

        Returns:
            AudioSegment with generation status and output path
        """
        segment = AudioSegment(
            text=text,
            character_name=None,
            voice=voice,
        )

        try:
            if not output_path:
                raise ValueError("Output path must be specified")

            if output_path.exists() and output_path.is_dir():
                raise IsADirectoryError(f"Output path is a directory: {output_path}")

            segment.status = GenerationStatus.IN_PROGRESS
            logger.info(f"Generating audio for: {text[:50]}...")

            # Determine audio prompt path (voice-specific or default)
            prompt_path = self._get_audio_prompt(voice)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate audio with Chatterbox Multilingual
            wav = self.model.generate(
                text=text,
                audio_prompt_path=str(prompt_path) if prompt_path else None,
                language_id=self.language_id,
            )

            # Save as WAV first (Chatterbox native format)
            wav_path = output_path.with_suffix(".wav")
            torchaudio.save(str(wav_path), wav, self.model.sr)

            # Convert to MP3 if requested (for AudioCombiner compatibility)
            if output_path.suffix.lower() == ".mp3":
                self._convert_wav_to_mp3(wav_path, output_path)
                wav_path.unlink()  # Remove intermediate WAV
                final_path = output_path
            else:
                final_path = wav_path

            segment.mark_complete(final_path)
            logger.info(f"Generated audio: {final_path}")
            return segment

        except Exception as e:
            logger.error(f"Failed to generate audio: {e}")
            segment.mark_failed(str(e))
            raise

    def _get_audio_prompt(self, voice: Voice) -> Path | None:
        """Get audio prompt path for voice cloning.

        Priority:
        1. Voice-specific reference audio (voice.preview_url)
        2. Default audio prompt (self.audio_prompt_path)
        3. None (use model's default voice)
        """
        # Check if voice has a specific reference audio
        if voice.preview_url:
            voice_ref = Path(voice.preview_url)
            if voice_ref.exists():
                return voice_ref
            logger.warning(f"Voice reference not found: {voice_ref}")

        # Fall back to default audio prompt
        return self.audio_prompt_path

    def _convert_wav_to_mp3(self, wav_path: Path, mp3_path: Path) -> None:
        """Convert WAV to MP3 using pydub."""
        from pydub import AudioSegment as PydubSegment
        audio = PydubSegment.from_wav(str(wav_path))
        audio.export(str(mp3_path), format="mp3", bitrate="192k")

    def generate_stream(self, text: str, voice: Voice) -> Iterator[bytes]:
        """Streaming not supported by Chatterbox."""
        raise NotImplementedError("Streaming not supported by Chatterbox generator")

    def get_available_voices(self) -> list[Voice]:
        """Return available voices (reference audio-based voices).

        Chatterbox uses reference audio for voice cloning, so "voices" are
        defined by the available reference audio files.
        """
        voices = []

        # Default narrator voice (no reference = model default)
        voices.append(Voice(
            voice_id="chatterbox_default",
            name="Default (Chatterbox)",
            language=self.language_id,
            description="Default Chatterbox voice (no reference audio)",
        ))

        # Add voice from configured default audio prompt
        if self.audio_prompt_path and self.audio_prompt_path.exists():
            voices.append(Voice(
                voice_id="chatterbox_narrator",
                name="Narrateur",
                language=self.language_id,
                preview_url=str(self.audio_prompt_path),
                description=f"Voice from {self.audio_prompt_path.name}",
            ))

        # Add voices from assets/voices directory
        assets_dir = Path(__file__).parent.parent.parent.parent / "assets" / "voices"
        if assets_dir.exists():
            for wav_file in assets_dir.glob("*.wav"):
                voice_name = wav_file.stem.replace("_", " ").title()
                voices.append(Voice(
                    voice_id=f"chatterbox_{wav_file.stem}",
                    name=voice_name,
                    language=self.language_id,
                    preview_url=str(wav_file),
                    description=f"Custom voice from {wav_file.name}",
                ))

        return voices


if __name__ == "__main__":
    # Quick test of the generator
    import tempfile

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    generator = ChatterboxGenerator()

    # List available voices
    voices = generator.get_available_voices()
    print(f"\nAvailable voices: {len(voices)}")
    for v in voices:
        print(f"  - {v.name} ({v.voice_id})")

    # Test generation (only if we have a GPU or are willing to wait)
    if torch.cuda.is_available():
        with tempfile.TemporaryDirectory() as tmpdir:
            test_text = "Bonjour, ceci est un test de génération audio."
            output = Path(tmpdir) / "test.mp3"

            voice = voices[0]
            print(f"\nGenerating test audio with voice: {voice.name}")

            segment = generator.generate(test_text, voice, output)
            print(f"Generated: {segment.audio_path}")
            print(f"Status: {segment.status}")
