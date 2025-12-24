# Future Improvements

## Implemented ✓

### ✓ Chapter-Level Parallelization
Generate multiple chapters simultaneously across multiple GPUs.
**Status: IMPLEMENTED** - Use `uv run audiobook generate-parallel --book absalon`

### ✓ Progress Bar
Added tqdm progress bars for segment generation within chapters.
**Status: IMPLEMENTED** - Shows progress during generation.

## High Priority

### 3. Dry-Run Mode
`--dry-run` flag to show:
- Total segment count after chunking
- Estimated generation time
- Estimated cost
Without actually generating anything.

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

---

# Parallelization Analysis

## Current Single-GPU Performance
- Full book (Absalon): ~957k chars, 9 chapters
- Generation time: ~10 hours on RTX 4090
- Cost: 10h × $0.25/hr = **$2.50**

## Multi-GPU Parallelization

The book has 9 chapters. With N GPUs, we distribute chapters across GPUs.
Time is limited by the slowest GPU (the one with the most chapters).

| GPUs | Distribution | Wall Time | Total GPU-Hours | Cost | Speedup |
|------|--------------|-----------|-----------------|------|---------|
| 1 | 9 chapters | 10.0h | 10.0h | $2.50 | 1.0x |
| 2 | 5+4 chapters | 5.5h | 11.0h | $2.75 | 1.8x |
| 3 | 3+3+3 chapters | 3.3h | 10.0h | $2.50 | 3.0x |
| 4 | 3+2+2+2 chapters | 3.3h | 10.0h* | $2.50* | 3.0x |
| 5 | 2+2+2+2+1 chapters | 2.2h | 9.0h* | $2.25* | 4.5x |
| 8 | 2+1+1+1+1+1+1+1 | 2.2h | 9.9h* | $2.50* | 4.5x |
| 9 | 1 each | 1.1h | 10.0h | $2.50 | 9.0x |

*Assumes chapters are roughly equal size. Actual distribution depends on chapter lengths.

## Key Insights

1. **Total GPU-hours is roughly constant** - Parallelization doesn't increase compute cost much, just wall-clock time.

2. **Overhead exists** - Each GPU needs ~30 sec model loading. With 8 GPUs, that's 4 min overhead total (negligible for 10h job).

3. **Sweet spots**:
   - **3 GPUs**: Perfect 3x speedup (3.3h), same cost ($2.50)
   - **9 GPUs**: Perfect 9x speedup (1.1h), same cost ($2.50)

4. **Diminishing returns**: Going from 3→8 GPUs only gives 1.5x additional speedup.

## Recommended Approach

For Absalon (9 chapters):
- **Budget priority**: 1 GPU, 10 hours, $2.50
- **Balanced**: 3 GPUs, 3.3 hours, $2.50 (same cost, 3x faster!)
- **Speed priority**: 9 GPUs, 1.1 hours, $2.50 (same cost, 9x faster!)

The 9-GPU approach is actually optimal if you can find 9 cheap instances, since total cost is the same but you're done in ~1 hour.

## Implementation Notes

To implement parallelization:
1. Split book into chapters (already supported)
2. Launch N instances, each assigned specific chapters
3. Each instance runs: `uv run audiobook generate --book absalon --chapter X`
4. Collect outputs and combine

Could be orchestrated with a simple bash script:
```bash
# Launch in parallel (each in separate tmux/screen or background)
for chapter in 1 2 3 4 5 6 7 8 9; do
  ssh instance-$chapter "cd audiobook && uv run audiobook generate --book absalon --chapter $chapter --verify" &
done
wait
# Then combine the chapter audio files
```
