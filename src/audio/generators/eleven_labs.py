from pathlib import Path

from elevenlabs import play
from elevenlabs.client import ElevenLabs
from loguru import logger

from ..types import Voice


def generate_audio(client: ElevenLabs, text: str, voice_id: str, output_path: Path) -> bool:
    """Generate audio using ElevenLabs. Returns True if successful."""
    try:
        # Get the generator from convert
        print("\nDEBUG: Calling text_to_speech.convert")
        audio_gen = client.text_to_speech.convert(
            text=text, voice_id=voice_id, model_id="eleven_multilingual_v2"
        )

        # Collect all chunks into bytes
        print("DEBUG: Collecting chunks")
        chunks = []
        for chunk in audio_gen:
            if isinstance(chunk, bytes):
                chunks.append(chunk)
                print(f"Got chunk of {len(chunk)} bytes")

        if not chunks:
            print("DEBUG: No chunks received!")
            return False

        # Combine chunks and write
        audio_data = b"".join(chunks)
        print(f"DEBUG: Total audio size: {len(audio_data)} bytes")

        output_path.write_bytes(audio_data)
        logger.info(f"Generated audio to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to generate audio: {e}")
        print(f"\nDEBUG: Exception details: {repr(e)}")
        return False


def get_voices(client: ElevenLabs) -> list[Voice]:
    """Get available voices."""
    try:
        response = client.voices.get_all()
        return [Voice(voice_id=voice.voice_id, name=voice.name) for voice in response.voices]
    except Exception as e:
        logger.error(f"Failed to fetch voices: {e}")
        return []


if __name__ == "__main__":
    from src.setting import DATA_PATH, ELEVENLABS_CLIENT

    # Setup test
    output_dir = DATA_PATH / "test_audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "test.mp3"

    # Get voices
    voices = get_voices(ELEVENLABS_CLIENT)
    print("\nAvailable voices:")
    for voice in voices:
        print(f"- {voice}")

    # Find a French voice
    french_voice = next(
        (v for v in voices if v.name.lower() in ["henri", "nicolas", "theo"]), voices[0]
    )
    print(f"\nTesting with voice: {french_voice}")

    # Generate audio
    test_text = "Bonjour! Je suis une voix de test."
    print(f"Generating: '{test_text}'")

    success = generate_audio(
        client=ELEVENLABS_CLIENT,
        text=test_text,
        voice_id=french_voice.voice_id,
        output_path=output_file,
    )

    if success and output_file.exists():
        print(f"\nGenerated: {output_file}")
        audio = output_file.read_bytes()
        print("Playing audio...")
        play(audio)
