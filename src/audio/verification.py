"""STT-based verification for TTS quality assurance."""

from pathlib import Path
from dataclasses import dataclass

import Levenshtein
from loguru import logger


@dataclass
class VerificationResult:
    """Result of STT verification."""
    original_text: str
    transcription: str
    distance: float  # Normalized edit distance (0-1)
    passed: bool
    attempt: int


class STTVerifier:
    """Verify TTS output using speech-to-text comparison.

    Uses Whisper to transcribe generated audio and compares with original text.
    Retries generation if transcription differs too much from original.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str = "fr",
        error_threshold: float = 0.15,
        max_retries: int = 5,
        device: str = "cuda",
    ):
        """Initialize STT verifier.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            language: Language code for transcription
            error_threshold: Max allowed normalized edit distance (0-1)
            max_retries: Maximum retry attempts before using best result
            device: Device for Whisper (cuda or cpu)
        """
        self.model_size = model_size
        self.language = language
        self.error_threshold = error_threshold
        self.max_retries = max_retries
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy load Whisper model."""
        if self._model is None:
            from faster_whisper import WhisperModel
            logger.info(f"Loading Whisper {self.model_size} model on {self.device}...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            logger.info("Whisper model loaded")
        return self._model

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe audio file to text."""
        segments, _ = self.model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments)

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison (lowercase, strip punctuation)."""
        import re
        # Lowercase and remove punctuation except apostrophes
        text = text.lower()
        text = re.sub(r"[^\w\s']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def compute_distance(self, original: str, transcription: str) -> float:
        """Compute normalized Levenshtein distance between texts."""
        orig_norm = self.normalize_text(original)
        trans_norm = self.normalize_text(transcription)

        if not orig_norm:
            return 0.0 if not trans_norm else 1.0

        distance = Levenshtein.distance(orig_norm, trans_norm)
        return distance / max(len(orig_norm), len(trans_norm))

    def verify(self, audio_path: Path, original_text: str) -> VerificationResult:
        """Verify a single audio file against original text."""
        transcription = self.transcribe(audio_path)
        distance = self.compute_distance(original_text, transcription)
        passed = distance <= self.error_threshold

        return VerificationResult(
            original_text=original_text,
            transcription=transcription,
            distance=distance,
            passed=passed,
            attempt=1,
        )

    def should_retry(self, distance: float, attempt: int) -> bool:
        """Decide whether to retry based on error and attempt number.

        Uses exponential threshold increase - more lenient on later attempts.
        Weighted to retry more on worse errors.
        """
        if attempt >= self.max_retries:
            return False

        # Base threshold increases with attempts (more lenient later)
        # attempt 1: 0.15, attempt 2: 0.18, attempt 3: 0.22, attempt 4: 0.27, attempt 5: 0.33
        adjusted_threshold = self.error_threshold * (1.2 ** (attempt - 1))

        # Weight retry probability by how bad the error is
        # Very bad errors (>0.5) always retry, medium errors sometimes retry
        if distance > 0.5:
            return True
        elif distance > adjusted_threshold:
            return True

        return False


def generate_with_verification(
    generator,
    text: str,
    voice,
    output_path: Path,
    verifier: STTVerifier,
) -> tuple[Path, VerificationResult]:
    """Generate audio with STT verification and retry on failure.

    Args:
        generator: TTS generator instance
        text: Text to synthesize
        voice: Voice configuration
        output_path: Where to save final audio
        verifier: STT verifier instance

    Returns:
        Tuple of (final_path, best_verification_result)
    """
    results = []

    for attempt in range(1, verifier.max_retries + 1):
        # Generate audio
        attempt_path = output_path.with_stem(f"{output_path.stem}_attempt{attempt}")
        segment = generator.generate(text, voice, attempt_path)

        if segment.audio_path is None:
            logger.warning(f"Generation failed on attempt {attempt}")
            continue

        # Verify with STT
        result = verifier.verify(attempt_path, text)
        result.attempt = attempt
        results.append((attempt_path, result))

        if result.passed:
            logger.debug(f"Verification passed on attempt {attempt} (distance={result.distance:.3f})")
            # Rename to final path
            attempt_path.rename(output_path)
            # Clean up other attempts
            for other_path, _ in results[:-1]:
                if other_path.exists():
                    other_path.unlink()
            return output_path, result

        logger.warning(
            f"Verification failed on attempt {attempt}: "
            f"distance={result.distance:.3f} > threshold={verifier.error_threshold}"
        )

        if not verifier.should_retry(result.distance, attempt):
            break

    # All attempts failed - use best result
    if results:
        best_path, best_result = min(results, key=lambda x: x[1].distance)
        logger.info(f"Using best attempt {best_result.attempt} (distance={best_result.distance:.3f})")
        best_path.rename(output_path)
        # Clean up other attempts
        for other_path, _ in results:
            if other_path.exists() and other_path != output_path:
                other_path.unlink()
        return output_path, best_result

    raise RuntimeError(f"All {verifier.max_retries} generation attempts failed")
