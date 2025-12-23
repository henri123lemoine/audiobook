#!/bin/bash
# Deploy audiobook to Vast.ai and optionally start generation
#
# Prerequisites:
#   - vastai CLI configured (uvx vastai set api-key YOUR_KEY)
#   - Positive Vast.ai balance
#
# Usage:
#   ./scripts/deploy_vastai.sh              # Just create instance
#   ./scripts/deploy_vastai.sh --generate   # Create instance and start generation

set -e

GENERATE=false
BOOK="absalon"
CHAPTER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --generate) GENERATE=true; shift ;;
        --book) BOOK="$2"; shift 2 ;;
        --chapter) CHAPTER="--chapter $2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Vast.ai Audiobook Deployment ==="
echo ""

# Check vastai is available
if ! command -v vastai &> /dev/null && ! uvx vastai --version &> /dev/null 2>&1; then
    echo "Error: vastai CLI not found. Install with: pip install vastai"
    exit 1
fi

# Use uvx if vastai not directly available
VASTAI="uvx vastai"
if command -v vastai &> /dev/null; then
    VASTAI="vastai"
fi

# Check balance
echo "[1/4] Checking account balance..."
BALANCE=$($VASTAI show user 2>/dev/null | tail -1 | awk '{print $1}')
echo "      Balance: \$$BALANCE"

if (( $(echo "$BALANCE < 1" | bc -l) )); then
    echo ""
    echo "WARNING: Low balance. Add funds at https://cloud.vast.ai/billing/"
    echo "         Estimated cost for full Absalon book: ~\$3-4"
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Find a good RTX 4090 instance
echo ""
echo "[2/4] Finding best available RTX 4090..."
OFFER_ID=$($VASTAI search offers 'gpu_name=RTX_4090 num_gpus=1 inet_down>100 reliability>0.95 cuda_vers>=12.0 disk_space>50' --order 'dph' 2>/dev/null | head -2 | tail -1 | awk '{print $1}')

if [ -z "$OFFER_ID" ]; then
    echo "Error: No suitable GPU found. Try again later."
    exit 1
fi

OFFER_INFO=$($VASTAI search offers 'gpu_name=RTX_4090 num_gpus=1 inet_down>100 reliability>0.95 cuda_vers>=12.0' --order 'dph' 2>/dev/null | head -2 | tail -1)
PRICE=$(echo "$OFFER_INFO" | awk '{print $10}')
COUNTRY=$(echo "$OFFER_INFO" | awk '{print $NF}')
echo "      Found offer $OFFER_ID at \$$PRICE/hr in $COUNTRY"

# Create instance
echo ""
echo "[3/4] Creating instance..."
INSTANCE_ID=$($VASTAI create instance $OFFER_ID \
    --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
    --disk 50 \
    --onstart-cmd "apt-get update && apt-get install -y git curl ffmpeg && curl -LsSf https://astral.sh/uv/install.sh | sh" \
    2>/dev/null | grep -oP 'new contract id: \K\d+' || echo "")

if [ -z "$INSTANCE_ID" ]; then
    # Try alternative parsing
    RESULT=$($VASTAI create instance $OFFER_ID \
        --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
        --disk 50 \
        --onstart-cmd "apt-get update && apt-get install -y git curl ffmpeg && curl -LsSf https://astral.sh/uv/install.sh | sh" \
        2>&1)
    echo "      Result: $RESULT"
    INSTANCE_ID=$(echo "$RESULT" | grep -oE '[0-9]+' | head -1)
fi

echo "      Instance ID: $INSTANCE_ID"

# Wait for instance to be ready
echo ""
echo "[4/4] Waiting for instance to start (this may take 1-2 minutes)..."
for i in {1..60}; do
    STATUS=$($VASTAI show instances --raw 2>/dev/null | grep "\"id\": $INSTANCE_ID" -A 50 | grep -oP '"actual_status": "\K[^"]+' | head -1 || echo "")
    if [ "$STATUS" = "running" ]; then
        echo "      Instance is running!"
        break
    fi
    sleep 5
    echo -n "."
done
echo ""

# Get SSH command
SSH_CMD=$($VASTAI ssh-url $INSTANCE_ID 2>/dev/null || echo "")
echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "SSH command: $SSH_CMD"
echo ""
echo "To set up and run:"
echo "  1. SSH into the instance:"
echo "     $SSH_CMD"
echo ""
echo "  2. Clone and set up:"
echo "     git clone https://github.com/henri123lemoine/audiobook.git"
echo "     cd audiobook"
echo "     ./scripts/setup_vastai.sh"
echo ""
echo "  3. Generate audiobook:"
echo "     uv run audiobook generate --book $BOOK $CHAPTER"
echo ""
echo "  4. When done, destroy instance:"
echo "     uvx vastai destroy instance $INSTANCE_ID"
echo ""
