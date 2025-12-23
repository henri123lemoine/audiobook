"""Audio generators for TTS."""

from .base import AudioGenerator, AudioPlayer
from .chatterbox import ChatterboxGenerator
from .eleven_labs import ElevenLabsGenerator

__all__ = [
    "AudioGenerator",
    "AudioPlayer",
    "ChatterboxGenerator",
    "ElevenLabsGenerator",
]
