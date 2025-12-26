# Remaining Tasks for Audiobook Pipeline

## Status: In Progress

### 1. STT Verification Loop ⏳
**Goal**: Detect and retry bad generations using speech-to-text comparison.

**Approach**:
- Use Whisper (fast, accurate for French) to transcribe generated audio
- Compare transcription with original text using normalized Levenshtein distance
- Retry if distance > threshold (e.g., 0.15 = 15% different)
- Max 5 attempts, use best result if all fail
- Weight toward retrying more incorrect outputs (exponential backoff based on error)

**Algorithm**:
```
error_threshold = 0.15  # 15% edit distance
max_retries = 5

for attempt in range(max_retries):
    audio = generate(text)
    transcription = whisper(audio)
    distance = levenshtein(text, transcription) / len(text)

    if distance <= error_threshold:
        return audio  # Good enough

    results.append((audio, distance))

# Return best attempt if all failed
return min(results, key=lambda x: x[1])[0]
```

---

### 2. Audio Post-Processing ⏳
**Goal**: Clean up trailing silence, breathing, and artifacts at segment ends.

**Approach**:
- Trim trailing silence (energy below threshold for >200ms)
- Optionally trim leading silence
- Use pydub's `detect_silence` or energy-based detection
- Apply at segment level before combining

**Implementation**:
```python
def cleanup_audio(audio_path):
    audio = AudioSegment.from_file(audio_path)
    # Trim silence from end (keep last 100ms of real audio)
    trimmed = audio.strip_silence(
        silence_thresh=-40,  # dB
        padding=100  # ms to keep
    )
    return trimmed
```

---

### 3. Testing Checklist
- [ ] STT verification catches bad generations
- [ ] Retry rate is reasonable (<20%)
- [ ] Audio cleanup removes trailing artifacts
- [ ] Final output sounds clean and continuous
- [ ] Full chapter test passes

---

## Implementation Order
1. ✅ Text chunking (300 char max)
2. ✅ Voice cloning with Nadine sample
3. ✅ Audio post-processing (trim trailing silence)
4. ✅ STT verification loop (--verify flag)
5. ⏳ Final GPU test
6. ⏳ Full book generation

---

## Notes
- GPU instance still running: 29160873
- Voice sample: `assets/voices/nadine_french.wav` (Nadine Eckert-Boulet, LibriVox)
- Test output: `test_output/test_nadine_chunked.mp3`
