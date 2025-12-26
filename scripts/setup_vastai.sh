#!/bin/bash
# Setup script for Vast.ai GPU deployment
#
# REQUIREMENTS:
#   - Python 3.11 (recommended) - chatterbox has compatibility issues with 3.12+
#   - NVIDIA GPU with CUDA support
#
# Usage (on a fresh Vast.ai instance):
#   git clone https://github.com/henri123lemoine/audiobook.git
#   cd audiobook
#   ./scripts/setup_vastai.sh
#
# Or one-liner:
#   git clone https://github.com/henri123lemoine/audiobook.git && cd audiobook && ./scripts/setup_vastai.sh

set -e

echo "=== Audiobook TTS Setup for Vast.ai ==="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"
if [[ "$PYTHON_VERSION" == "3.12" ]] || [[ "$PYTHON_VERSION" == "3.13" ]]; then
    echo "WARNING: Python $PYTHON_VERSION detected. Chatterbox works best with Python 3.11."
    echo "         Consider using a Vast.ai template with Python 3.11."
    echo ""
fi

# Install system dependencies
echo "[1/5] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq ffmpeg git curl rsync > /dev/null 2>&1
echo "      Done."

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "[2/5] Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    echo "      Done."
else
    echo "[2/5] uv already installed."
fi

# Make sure we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Please run from the audiobook directory."
    exit 1
fi

# Install Python dependencies
echo "[3/5] Installing Python dependencies (this may take a few minutes)..."
uv sync --quiet
echo "      Done."

# Create assets directory for reference audio
echo "[4/5] Setting up assets directory..."
mkdir -p assets/voices
echo "      Done."

# Check for reference audio
if [ ! -f "assets/voices/default_french.wav" ]; then
    echo ""
    echo "      NOTE: No default French voice reference found."
    echo "      To use voice cloning, add a French audio sample (~10-30 sec) to:"
    echo "        assets/voices/default_french.wav"
    echo ""
    echo "      You can generate one using TTSFree.com or similar services."
    echo "      Without reference audio, pass --no-reference-audio to use default voice."
fi

# Verify CUDA is available
echo "[5/5] Checking GPU availability..."
CUDA_CHECK=$(uv run python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>/dev/null || echo "ERROR")

if [ "$CUDA_CHECK" = "CUDA" ]; then
    GPU_NAME=$(uv run python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    echo "      GPU detected: $GPU_NAME"
elif [ "$CUDA_CHECK" = "CPU" ]; then
    echo "      WARNING: No GPU detected. Generation will be slow on CPU."
else
    echo "      WARNING: Could not verify GPU. PyTorch may need to be reinstalled."
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Usage examples:"
echo ""
echo "  # Show book info and cost estimate:"
echo "  uv run audiobook info --book absalon"
echo ""
echo "  # Generate chapter 1 only (for testing):"
echo "  uv run audiobook generate --book absalon --chapter 1"
echo ""
echo "  # Generate full book:"
echo "  uv run audiobook generate --book absalon"
echo ""
echo "  # If you haven't added a reference voice yet:"
echo "  uv run audiobook generate --book absalon --no-reference-audio"
echo ""
echo "  # With custom reference audio:"
echo "  uv run audiobook generate --book absalon -r path/to/voice.wav"
echo ""
echo "  # List available commands:"
echo "  uv run audiobook --help"
echo ""
