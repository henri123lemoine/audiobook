import json
from datetime import datetime
from pathlib import Path
from typing import Protocol

from loguru import logger

from ..types import Voice, VoiceAssignment


class VoiceProvider(Protocol):
    """Protocol for voice provider implementations."""

    async def get_available_voices(self) -> list[Voice]:
        """Get list of available voices."""
        ...


class VoiceRegistry:
    """Manages voice assignments for characters."""

    def __init__(self, voice_provider: VoiceProvider, config_path: Path | None = None):
        """Initialize registry with voice provider."""
        self.provider = voice_provider
        self.config_path = config_path or Path("config/voices.json")
        self._assignments: dict[str, VoiceAssignment] = {}
        self._available_voices: dict[str, Voice] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load voice assignments from config file."""
        if not self.config_path.exists():
            logger.info(f"No config file found at {self.config_path}")
            return

        try:
            data = json.loads(self.config_path.read_text())
            for char_name, voice_data in data.items():
                voice = Voice(
                    voice_id=voice_data["voice_id"],
                    name=voice_data["voice_name"],
                    language=voice_data.get("language"),
                )
                self._assignments[char_name] = VoiceAssignment(
                    character_name=char_name,
                    voice=voice,
                    language=voice_data.get("language", "fr"),
                    modified_at=datetime.fromisoformat(voice_data["modified_at"]),
                )
        except Exception as e:
            logger.error(f"Failed to load voice config: {e}")

    def _save_config(self) -> None:
        """Save voice assignments to config file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                char_name: {
                    "voice_id": assignment.voice.voice_id,
                    "voice_name": assignment.voice.name,
                    "language": assignment.language,
                    "modified_at": assignment.modified_at.isoformat(),
                }
                for char_name, assignment in self._assignments.items()
            }

            self.config_path.write_text(json.dumps(data, indent=2))

        except Exception as e:
            logger.error(f"Failed to save voice config: {e}")

    async def refresh_voices(self) -> None:
        """Refresh available voices from provider."""
        voices = await self.provider.get_available_voices()
        self._available_voices = {v.voice_id: v for v in voices}

    def get_assignment(self, character_name: str) -> VoiceAssignment | None:
        """Get voice assignment for a character."""
        return self._assignments.get(character_name)

    def assign_voice(self, character_name: str, voice: Voice, language: str = "fr") -> None:
        """Assign a voice to a character."""
        self._assignments[character_name] = VoiceAssignment(
            character_name=character_name,
            voice=voice,
            language=language,
        )
        self._save_config()

    def remove_assignment(self, character_name: str) -> None:
        """Remove voice assignment for a character."""
        if character_name in self._assignments:
            del self._assignments[character_name]
            self._save_config()

    def get_available_voices(self) -> list[Voice]:
        """Get list of available voices."""
        return list(self._available_voices.values())

    def get_all_assignments(self) -> dict[str, VoiceAssignment]:
        """Get all current voice assignments."""
        return self._assignments.copy()


if __name__ == "__main__":
    import asyncio

    from src.audio.generators.eleven_labs import ElevenLabsGenerator
    from src.setting import ELEVENLABS_CLIENT

    async def main():
        # Setup test environment
        test_config = Path("test_config/voices.json")
        if test_config.exists():
            test_config.unlink()

        # Initialize components
        generator = ElevenLabsGenerator(ELEVENLABS_CLIENT)
        registry = VoiceRegistry(generator, config_path=test_config)

        try:
            # Refresh available voices
            await registry.refresh_voices()
            voices = registry.get_available_voices()
            print("\nAvailable voices:")
            for voice in voices:
                print(f"- {voice}")

            if not voices:
                print("No voices available!")
                return

            # Test voice assignment
            test_voice = voices[0]
            print(f"\nAssigning voice {test_voice} to character 'tomas'")
            registry.assign_voice("tomas", test_voice)

            # Verify assignment
            assignment = registry.get_assignment("tomas")
            print(f"\nVoice assignment for 'tomas': {assignment}")

            # Test config persistence
            print(f"\nConfig saved to: {test_config}")
            print("Config contents:")
            if test_config.exists():
                print(test_config.read_text())

            # Remove assignment
            registry.remove_assignment("tomas")
            print("\nAssignment removed")

            assignments = registry.get_all_assignments()
            print(f"Remaining assignments: {assignments}")

        finally:
            # Cleanup
            if test_config.exists():
                test_config.unlink()
            if test_config.parent.exists():
                test_config.parent.rmdir()

    asyncio.run(main())
