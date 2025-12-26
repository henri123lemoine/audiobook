# Future Improvements

## Implemented ✓

### ✓ Chapter-Level Parallelization
Generate multiple chapters simultaneously across multiple GPUs.
**Status: IMPLEMENTED** - Use `uv run audiobook generate-parallel --book absalon`

### ✓ Dry-Run Mode
`--dry-run` shows segment counts and missing ranges without renting instances.
**Status: IMPLEMENTED** - Use `uv run audiobook generate-parallel --dry-run`

### ✓ Progress Bar
Added tqdm progress bars for segment generation within chapters.
**Status: IMPLEMENTED** - Shows progress during generation.

## Medium Priority

### 4. Audio Metadata (ID3 Tags)
Add metadata to output MP3s:
- Title, Author, Chapter number
- Cover art (if available)
- Better audiobook player support

### 5. Different Voices for Characters
The `VoiceCasting` system already supports this. Would need:
- Voice samples for each character
- Character detection in text (dialogue attribution)
- Map characters to voice samples

### 6. Notification on Completion
Send webhook/email when generation finishes.
Useful for 10+ hour runs where you don't want to keep checking.

## Low Priority

### 7. Web UI
Simple web interface to:
- Upload book text
- Configure voices
- Monitor progress
- Download results

### 8. Streaming Output
Generate and stream audio in real-time instead of waiting for full completion.
