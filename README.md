# Audiobook Generator

Generate high-quality French audiobooks using Chatterbox Multilingual TTS. Designed for literary works with long, complex sentences.

## Features

- **Chatterbox TTS** - High-quality multilingual text-to-speech with voice cloning
- **Smart text chunking** - Splits at natural break points (sentences > semicolons > commas)
- **STT verification** - Uses Whisper to detect and retry bad generations
- **Resumable** - Skips already-generated segments on restart
- **GPU optimized** - Runs efficiently on Vast.ai (~$0.25/hr for RTX 4090)

## Quick Start (Local)

```bash
# Clone and install
git clone https://github.com/henri123lemoine/audiobook.git
cd audiobook
uv sync

# List available books
uv run audiobook list-books

# Quick test (5 segments, ~2-3 min on GPU)
uv run audiobook generate --book absalon --test

# Generate specific chapter
uv run audiobook generate --book absalon --chapter 1

# Full generation with verification
uv run audiobook generate --book absalon --verify
```

## Parallel Generation (Multi-GPU)

Generate a full audiobook in minutes instead of hours by using multiple GPUs in parallel. Scales linearly up to 100+ GPUs!

```bash
# See what would happen (dry run)
uv run audiobook generate-parallel --book absalon --gpus 27 --dry-run

# Fast: 27 GPUs (~22 min, ~$4)
uv run audiobook generate-parallel --book absalon --gpus 27

# Faster: 45 GPUs (~13 min, ~$4)
uv run audiobook generate-parallel --book absalon --gpus 45

# Even faster: 90 GPUs (~7 min, ~$4)
uv run audiobook generate-parallel --book absalon --gpus 90
```

### Scaling Analysis

| GPUs | Wall Time | Speedup | Cost |
|------|-----------|---------|------|
| 1    | 9.9h      | 1x      | ~$3  |
| 9    | 1.1h      | 9x      | ~$4  |
| 27   | 22 min    | 27x     | ~$4  |
| 45   | 13 min    | 45x     | ~$4  |
| 90   | 7 min     | 89x     | ~$4  |

**How it works (segment mode, default):**
1. Flattens all 1425 segments across 9 chapters
2. Distributes segments **evenly** across N GPUs (perfect load balancing)
3. Rents N GPU instances from Vast.ai
4. Runs generation in parallel
5. Downloads and recombines segments into chapters
6. Combines chapters into final audiobook
7. Automatically destroys instances

**Cost stays roughly constant** regardless of GPU count - you're paying for the same total GPU-hours, just distributed across more machines. The slight increase from $3 to $4 is due to fixed overhead (model loading, setup).

## Running on GPU (Vast.ai) - Single Instance

For full book generation, a GPU is highly recommended. Here's how to set up on Vast.ai:

### 1. Rent a GPU Instance

1. Go to [vast.ai](https://vast.ai) and create an account
2. Add your SSH public key in Account → SSH Keys
3. Search for instances with:
   - GPU: RTX 4090 or RTX 3090 (best price/performance)
   - Disk: 30GB+
   - Image: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` or similar
4. Rent the instance (~$0.25-0.40/hr for RTX 4090)

### 2. Set Up the Instance

```bash
# SSH into your instance (use the SSH command from Vast.ai dashboard)
ssh -p <PORT> root@<HOST>

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# Clone the repo
git clone https://github.com/henri123lemoine/audiobook.git
cd audiobook

# Install dependencies (first run downloads models, takes a few minutes)
uv sync
```

### 3. Generate the Audiobook

```bash
# Quick test first to verify everything works
uv run audiobook generate --book absalon --test --verify

# Full chapter 1 (~55k chars, ~30-45 min)
uv run audiobook generate --book absalon --chapter 1 --verify

# Full book generation (~957k chars, ~8-12 hours)
# Use tmux/screen so it survives SSH disconnection
tmux new -s audiobook
uv run audiobook generate --book absalon --verify

# Detach with Ctrl+B then D, reattach with: tmux attach -t audiobook
```

### 4. Download the Result

```bash
# From your local machine
scp -P <PORT> root@<HOST>:/workspace/audiobook/books/absalon/audio/audiobook_complete.mp3 ./
```

## CLI Reference

```
audiobook generate [OPTIONS]       Generate audiobook (single GPU)

Options:
  -b, --book TEXT          Book identifier (required)
  -c, --chapter INT        Specific chapter number (1-indexed)
  -p, --part INT           Specific part number (1-indexed)
  -o, --output-dir PATH    Output directory
  -d, --device [cuda|cpu|auto]  Device for inference
  -r, --reference-audio PATH    Voice sample for cloning (~10-30 sec)
  -l, --language TEXT      Language code (default: fr)
  --silence-ms INT         Silence between segments in ms (default: 500)
  -n, --limit INT          Limit to first N segments
  -t, --test               Quick test mode (5 segments only)
  -v, --verify             Enable STT verification for quality
  --whisper-model TEXT     Whisper model size for verification

audiobook generate-parallel [OPTIONS]  Generate using multiple GPUs

Options:
  -b, --book TEXT          Book identifier (required)
  -g, --gpus INT           Number of GPU instances (default: 9)
  --gpu-type TEXT          GPU model to rent (default: RTX_4090)
  --max-cost FLOAT         Max cost per hour per GPU (default: $0.40)
  -v, --verify/--no-verify Enable STT verification (default: enabled)
  --keep-instances         Keep instances running after completion
  --dry-run                Show plan without executing

audiobook estimate-parallel [OPTIONS]  Estimate parallel generation time/cost
audiobook instances                    List/manage Vast.ai instances
audiobook info --book TEXT             Show book structure and estimates
audiobook validate --book TEXT         Check for problematic segments
audiobook list-books                   List available books
```

## Voice Customization

The default voice is a French narrator from LibriVox. To use a custom voice:

1. Record or find a 10-30 second clean audio sample
2. Save as WAV in `assets/voices/`
3. Pass with `-r assets/voices/your_voice.wav`

## Cost Estimates

| Scope | Characters | Audio Length | GPU Time | Cost (RTX 4090) |
|-------|------------|--------------|----------|-----------------|
| Test (5 segments) | ~2k | ~2 min | ~3 min | ~$0.01 |
| Chapter 1 | ~55k | ~60 min | ~45 min | ~$0.20 |
| Full book | ~957k | ~17 hours | ~10 hours | ~$3-4 |

## Troubleshooting

**"CUDA out of memory"**
- Use a GPU with more VRAM (16GB+ recommended)
- Or reduce segment length in `src/audio/pipeline.py`

**Words being cut off at segment ends**
- This was fixed by making audio cleanup very conservative
- If still happening, check the `cleanup_audio` function

**High retry rate (>50%)**
- Chatterbox may struggle with certain text patterns
- Try shorter MAX_SEGMENT_LENGTH (currently 250)

**SSH disconnects during long runs**
- Use `tmux` or `screen` to keep the process running
- The pipeline is resumable - just restart and it skips completed segments

## Architecture

```
src/
├── audio/
│   ├── generators/
│   │   ├── base.py          # AudioGenerator ABC
│   │   └── chatterbox.py    # Chatterbox TTS implementation
│   ├── pipeline.py          # Main generation pipeline + chunking
│   └── verification.py      # STT verification with Whisper
├── book/
│   ├── base.py              # Book ABC
│   └── books/
│       └── absolon.py       # Absalom, Absalom! book loader
└── cli.py                   # Click CLI
```

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Lint
uv run ruff check .
```
