# Audiobook Generator

Generate high-quality French audiobooks using Chatterbox Multilingual TTS. Designed for literary works with long, complex sentences.

## Features

- **Chatterbox TTS** - High-quality multilingual text-to-speech with voice cloning
- **Smart text chunking** - Splits at natural break points (sentences > semicolons > commas)
- **STT verification** - Uses Whisper to detect and retry bad generations
- **Resumable** - Skips already-generated segments on restart
- **GPU optimized** - Runs efficiently on Vast.ai

## Quick Start (Local)

```bash
# Clone and install
git clone https://github.com/henri123lemoine/audiobook.git
cd audiobook
uv sync

# List available books
uv run audiobook list-books

# Quick test (5 segments)
uv run audiobook generate --book absalon --test

# Generate specific chapter
uv run audiobook generate --book absalon --chapter 1

# If you don't have a reference voice yet
uv run audiobook generate --book absalon --test --no-reference-audio

# Full generation with verification
uv run audiobook generate --book absalon --verify
```

## Parallel Generation (Multi-GPU)

Generate across multiple GPUs; wall-clock time depends on measured throughput.

```bash
uv run audiobook generate-parallel --book absalon --gpus 20 --dry-run  # preview ranges
uv run audiobook generate-parallel --book absalon --gpus 20            # run
```

Distributes missing ranges, rents Vast.ai instances, runs in parallel, syncs segments back locally, and auto-combines when all segments are present.

## Running on GPU (Vast.ai) - Single Instance

For full book generation, a GPU is highly recommended. Here's how to set up on Vast.ai:

### 1. Rent a GPU Instance

1. Go to [vast.ai](https://vast.ai) and create an account
2. Add your SSH public key in Account → SSH Keys
3. Search for instances with:
   - GPU: RTX 4090 or RTX 3090 (best price/performance)
   - Disk: 30GB+
   - Image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` or similar
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

# Install dependencies (first run downloads models)
uv sync
```

### 3. Generate the Audiobook

```bash
# Quick test first to verify everything works
uv run audiobook generate --book absalon --test --verify

# Full chapter 1 (~55k chars)
uv run audiobook generate --book absalon --chapter 1 --verify

# Full book generation (~957k chars)
# Run in background and tail the log
nohup uv run audiobook generate --book absalon --verify > generation.log 2>&1 &
tail -f generation.log
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
  --no-reference-audio          Disable voice cloning and use default voice
  -l, --language TEXT      Language code (default: fr)
  --silence-ms INT         Silence between segments in ms (default: 500)
  -n, --limit INT          Limit to first N segments
  -t, --test               Quick test mode (5 segments only)
  -v, --verify             Enable STT verification for quality
  --whisper-model TEXT     Whisper model size for verification

audiobook generate-parallel [OPTIONS]  Generate using multiple GPUs

Options:
  -b, --book TEXT          Book identifier (required)
  -g, --gpus INT           Number of GPU instances (default: 10)
  --gpu-type TEXT          GPU model to rent (default: RTX_3090)
  --max-cost FLOAT         Max cost per hour per GPU (default: $0.15)
  -v, --verify/--no-verify Enable STT verification (default: enabled)
  -n, --limit INT          Limit to first N segments (for testing)
  --keep-instances         Keep instances running after completion
  --dry-run                Show plan without executing

audiobook estimate-parallel [OPTIONS]  Show current Vast.ai price snapshot
audiobook combine --book TEXT          Combine segments into final audiobook
audiobook instances                    List/manage Vast.ai instances
audiobook info --book TEXT             Show book structure and counts
audiobook validate --book TEXT         Check for problematic segments
audiobook list-books                   List available books
```

## Voice Customization

The default voice is `assets/voices/default_french.wav`. To use a custom voice:

1. Record or find a 10-30 second clean audio sample
2. Save as WAV in `assets/voices/`
3. Pass with `-r assets/voices/your_voice.wav`
4. To disable voice cloning, pass `--no-reference-audio`

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
- Use `nohup` and tail the log to keep the process running
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
