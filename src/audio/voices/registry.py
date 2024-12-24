import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Protocol

from loguru import logger

from ...book.types import Character
from ..types import Voice, VoiceAssignment


class VoiceProvider(Protocol):
    """Protocol for voice provider implementations."""

    async def get_available_voices(self) -> list[Voice]:
        """Get list of available voices."""
        ...


class VoiceRegistry:
    """Manages voice assignments for characters."""

    def __init__(
        self,
        voice_provider: VoiceProvider,
        config_path: Path | None = None,
        required_characters: list[Character] | None = None,
    ):
        """Initialize registry with voice provider."""
        self.provider = voice_provider
        self.config_path = config_path or Path("config/voices.json")
        self._assignments: dict[str, VoiceAssignment] = {}
        self._available_voices: dict[str, Voice] = {}
        self.required_characters = required_characters or []
        self._load_config()

    def _load_config(self) -> None:
        """Load voice assignments from config file."""
        if not self.config_path.exists():
            logger.info(f"No config file found at {self.config_path}")
            return

        try:
            data = json.loads(self.config_path.read_text())
            for char_name, voice_data in data.items():
                if "voice_id" not in voice_data:
                    continue

                voice = Voice(
                    voice_id=voice_data["voice_id"],
                    name=voice_data["voice_name"],
                    language=voice_data.get("language"),
                    description=voice_data.get("description"),
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
                    "description": assignment.voice.description,
                    "modified_at": assignment.modified_at.isoformat(),
                }
                for char_name, assignment in self._assignments.items()
            }

            self.config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

        except Exception as e:
            logger.error(f"Failed to save voice config: {e}")

    async def refresh_voices(self) -> None:
        """Refresh available voices from provider."""
        voices = await self.provider.get_available_voices()
        self._available_voices = {v.voice_id: v for v in voices}

    def get_assignment(self, character_name: str) -> VoiceAssignment | None:
        """Get voice assignment for a character."""
        return self._assignments.get(character_name)

    def assign_voice(
        self,
        character_name: str,
        voice: Voice,
        language: str = "fr",
        save: bool = True,
    ) -> None:
        """Assign a voice to a character."""
        self._assignments[character_name] = VoiceAssignment(
            character_name=character_name,
            voice=voice,
            language=language,
        )
        if save:
            self._save_config()

    def batch_assign(self, assignments: dict[str, Voice]) -> None:
        """Assign multiple voices at once."""
        for char_name, voice in assignments.items():
            self.assign_voice(char_name, voice, save=False)
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

    def get_unassigned_characters(self) -> list[Character]:
        """Get list of required characters without voice assignments."""
        return [char for char in self.required_characters if char.name not in self._assignments]

    def validate_assignments(self) -> tuple[bool, list[str]]:
        """
        Validate that all required characters have voices assigned.

        Returns:
            Tuple of (is_valid, list of missing character names)
        """
        if not self.required_characters:
            return True, []

        missing = []
        for char in self.required_characters:
            if char.name not in self._assignments:
                missing.append(char.name)

        return len(missing) == 0, missing


if __name__ == "__main__":
    import asyncio

    from src.audio.generators.eleven_labs import ElevenLabsGenerator
    from src.book.books.l_insoutenable import InsoutenableBook
    from src.setting import ELEVENLABS_CLIENT

    async def main():
        # Setup test environment
        test_config = Path("test_config/voices.json")
        if test_config.exists():
            test_config.unlink()

        # Initialize components
        generator = ElevenLabsGenerator(ELEVENLABS_CLIENT)
        registry = VoiceRegistry(
            generator,
            config_path=test_config,
            required_characters=InsoutenableBook.CHARACTERS,
        )

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

            # Check unassigned characters
            unassigned = registry.get_unassigned_characters()
            print("\nUnassigned characters:")
            for char in unassigned:
                print(f"- {char.name}: {char.description or 'No description'}")

            # Validate assignments
            is_valid, missing = registry.validate_assignments()
            print(f"\nAssignments valid: {is_valid}")
            if missing:
                print("Missing assignments for:", missing)

        finally:
            # Cleanup
            if test_config.exists():
                test_config.unlink()
            if test_config.parent.exists():
                test_config.parent.rmdir()

    asyncio.run(main())
