from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4


class GenerationStatus(Enum):
    """Status of an audio generation attempt."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class Voice:
    """Represents a TTS voice with its metadata."""

    voice_id: str
    name: str
    language: str | None = None
    description: str | None = None
    preview_url: str | None = None

    def __str__(self) -> str:
        return f"{self.name} ({self.voice_id})"


@dataclass
class VoiceAssignment:
    """Maps a character to a specific voice."""

    character_name: str
    voice: Voice
    language: str = "fr"  # Default to French
    modified_at: datetime = field(default_factory=datetime.now)


@dataclass
class AudioSegment:
    """Represents a generated audio piece with its metadata."""

    text: str
    character_name: str | None
    id: UUID = field(default_factory=uuid4)
    voice: Voice | None = None
    audio_path: Path | None = None
    created_at: datetime = field(default_factory=datetime.now)
    status: GenerationStatus = GenerationStatus.PENDING
    generation_attempts: int = 0
    error_message: str | None = None

    def __post_init__(self):
        """Ensure audio_path is a Path object."""
        if isinstance(self.audio_path, str):
            self.audio_path = Path(self.audio_path)

    @property
    def duration(self) -> float | None:
        """Get audio duration in seconds if available."""
        # TODO: Implement using wave/ffmpeg
        return None

    def is_complete(self) -> bool:
        """Check if segment is successfully generated and approved."""
        return self.status == GenerationStatus.APPROVED

    def mark_failed(self, error_msg: str):
        """Mark segment as failed with error message."""
        self.status = GenerationStatus.FAILED
        self.error_message = error_msg
        self.generation_attempts += 1

    def mark_complete(self, audio_path: Path | str):
        """Mark segment as complete with generated audio path."""
        self.audio_path = Path(audio_path)
        self.status = GenerationStatus.COMPLETED
        self.error_message = None


if __name__ == "__main__":
    # Test Voice
    voice = Voice(
        voice_id="test_id",
        name="Test Voice",
        language="fr",
        description="A test voice",
    )
    print(f"Created voice: {voice}")

    # Test VoiceAssignment
    assignment = VoiceAssignment(
        character_name="tomas",
        voice=voice,
    )
    print(f"Created assignment: {assignment}")

    # Test AudioSegment lifecycle
    segment = AudioSegment(
        text="Bonjour, comment allez-vous?",
        character_name="tomas",
        voice=voice,
    )
    print(f"Initial segment status: {segment.status}")

    # Test status transitions
    segment.mark_failed("API Error")
    print(f"After failure: {segment.status}, Attempts: {segment.generation_attempts}")

    segment.mark_complete("test.wav")
    print(f"After completion: {segment.status}, Path: {segment.audio_path}")

    # Verify Path conversion
    assert isinstance(segment.audio_path, Path)
