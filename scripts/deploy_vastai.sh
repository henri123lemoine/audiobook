#!/bin/bash
# Vast.ai audiobook generation - full workflow
#
# Usage:
#   ./scripts/deploy_vastai.sh create                    # Create instance
#   ./scripts/deploy_vastai.sh setup <instance_id>       # Run setup on instance
#   ./scripts/deploy_vastai.sh generate <instance_id> [--chapter N]  # Generate
#   ./scripts/deploy_vastai.sh status <instance_id>      # Check generation status
#   ./scripts/deploy_vastai.sh download <instance_id>    # Download audio files
#   ./scripts/deploy_vastai.sh destroy <instance_id>     # Destroy instance
#   ./scripts/deploy_vastai.sh full [--chapter N]        # Do everything

set -e

BOOK="absalon"
CHAPTER_ARG=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

VASTAI="vastai"
if ! command -v vastai &> /dev/null; then
    echo "Error: vastai CLI not found. Install with: pip install vastai"
    exit 1
fi

# Save/load instance ID for convenience
INSTANCE_FILE="$PROJECT_DIR/.vastai_instance"

save_instance_id() {
    echo "$1" > "$INSTANCE_FILE"
}

load_instance_id() {
    if [ -f "$INSTANCE_FILE" ]; then
        cat "$INSTANCE_FILE"
    fi
}

get_ssh_info() {
    local instance_id=$1
    $VASTAI show instances --raw 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
for inst in data:
    if inst.get('id') == $instance_id:
        port = inst.get('ssh_port', inst.get('ports', {}).get('22/tcp', [{}])[0].get('HostPort', ''))
        host = inst.get('ssh_host', inst.get('public_ipaddr', ''))
        print(f'{host}:{port}')
        break
" 2>/dev/null || echo ""
}

wait_for_instance() {
    local instance_id=$1
    echo "Waiting for instance $instance_id to be ready..."
    for i in {1..60}; do
        STATUS=$($VASTAI show instances --raw 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
for inst in data:
    if inst.get('id') == $instance_id:
        print(inst.get('actual_status', 'unknown'))
        break
" 2>/dev/null || echo "unknown")
        if [ "$STATUS" = "running" ]; then
            echo "Instance is running!"
            return 0
        fi
        sleep 5
        echo -n "."
    done
    echo ""
    echo "Timeout waiting for instance"
    return 1
}

ssh_exec() {
    local instance_id=$1
    shift
    local cmd="$@"

    local ssh_info=$(get_ssh_info $instance_id)
    local host=$(echo "$ssh_info" | cut -d: -f1)
    local port=$(echo "$ssh_info" | cut -d: -f2)

    if [ -z "$host" ] || [ -z "$port" ]; then
        echo "Error: Could not get SSH info for instance $instance_id"
        return 1
    fi

    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$port" "root@$host" "$cmd"
}

cmd_create() {
    echo "=== Creating Vast.ai Instance ==="

    # Check balance
    BALANCE=$($VASTAI show user 2>/dev/null | tail -1 | awk '{print $1}')
    echo "Balance: \$$BALANCE"

    # Find best RTX 4090
    echo "Finding best RTX 4090..."
    OFFER=$($VASTAI search offers 'gpu_name=RTX_4090 num_gpus=1 inet_down>100 reliability>0.95 cuda_vers>=12.0 disk_space>50' --order 'dph' 2>/dev/null | head -2 | tail -1)
    OFFER_ID=$(echo "$OFFER" | awk '{print $1}')
    PRICE=$(echo "$OFFER" | awk '{print $10}')

    if [ -z "$OFFER_ID" ]; then
        echo "Error: No suitable GPU found"
        exit 1
    fi

    echo "Selected offer $OFFER_ID at \$$PRICE/hr"

    # Create instance
    echo "Creating instance..."
    RESULT=$($VASTAI create instance $OFFER_ID \
        --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
        --disk 50 \
        2>&1)

    INSTANCE_ID=$(echo "$RESULT" | grep -oE '[0-9]+' | head -1)

    if [ -z "$INSTANCE_ID" ]; then
        echo "Error creating instance: $RESULT"
        exit 1
    fi

    save_instance_id "$INSTANCE_ID"
    echo "Instance created: $INSTANCE_ID"
    echo "Saved to $INSTANCE_FILE"

    wait_for_instance "$INSTANCE_ID"

    echo ""
    echo "Next steps:"
    echo "  ./scripts/deploy_vastai.sh setup $INSTANCE_ID"
    echo "  ./scripts/deploy_vastai.sh generate $INSTANCE_ID --chapter 1"
}

cmd_setup() {
    local instance_id=${1:-$(load_instance_id)}
    if [ -z "$instance_id" ]; then
        echo "Usage: deploy_vastai.sh setup <instance_id>"
        exit 1
    fi

    echo "=== Setting up instance $instance_id ==="

    echo "Cloning repository..."
    ssh_exec $instance_id "rm -rf /workspace/audiobook && git clone https://github.com/henri123lemoine/audiobook.git /workspace/audiobook"

    echo "Running setup script..."
    ssh_exec $instance_id "cd /workspace/audiobook && bash ./scripts/setup_vastai.sh"

    echo ""
    echo "Setup complete! Next:"
    echo "  ./scripts/deploy_vastai.sh generate $instance_id --chapter 1"
}

cmd_generate() {
    local instance_id=${1:-$(load_instance_id)}
    shift 2>/dev/null || true

    if [ -z "$instance_id" ]; then
        echo "Usage: deploy_vastai.sh generate <instance_id> [--chapter N]"
        exit 1
    fi

    # Parse remaining args
    local chapter_arg=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --chapter) chapter_arg="--chapter $2"; shift 2 ;;
            *) shift ;;
        esac
    done

    echo "=== Starting generation on instance $instance_id ==="
    echo "Book: $BOOK"
    echo "Chapter: ${chapter_arg:-all}"
    echo ""

    # Run in background
    ssh_exec $instance_id "cd /workspace/audiobook && nohup /root/.local/bin/uv run audiobook generate --book $BOOK $chapter_arg > /workspace/audiobook/generation.log 2>&1 &"

    echo "Generation started in background."
    echo ""
    echo "To check status:"
    echo "  ./scripts/deploy_vastai.sh status $instance_id"
    echo ""
    echo "To watch live:"
    local ssh_info=$(get_ssh_info $instance_id)
    local host=$(echo "$ssh_info" | cut -d: -f1)
    local port=$(echo "$ssh_info" | cut -d: -f2)
    echo "  ssh -p $port root@$host 'tail -f /workspace/audiobook/generation.log'"
}

cmd_status() {
    local instance_id=${1:-$(load_instance_id)}
    if [ -z "$instance_id" ]; then
        echo "Usage: deploy_vastai.sh status <instance_id>"
        exit 1
    fi

    echo "=== Generation status ==="
    echo ""
    echo "Last 20 lines of log:"
    ssh_exec $instance_id "tail -20 /workspace/audiobook/generation.log 2>/dev/null || echo 'No log yet'"
    echo ""
    echo "Generated files:"
    ssh_exec $instance_id "ls -la /workspace/audiobook/books/absalon/audio/ 2>/dev/null || echo 'No audio files yet'"
}

cmd_download() {
    local instance_id=${1:-$(load_instance_id)}
    if [ -z "$instance_id" ]; then
        echo "Usage: deploy_vastai.sh download <instance_id>"
        exit 1
    fi

    echo "=== Downloading audio files ==="

    local ssh_info=$(get_ssh_info $instance_id)
    local host=$(echo "$ssh_info" | cut -d: -f1)
    local port=$(echo "$ssh_info" | cut -d: -f2)

    local dest="$PROJECT_DIR/books/absalon/audio"
    mkdir -p "$dest"

    echo "Downloading to: $dest"
    rsync -avz --progress -e "ssh -p $port -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
        "root@$host:/workspace/audiobook/books/absalon/audio/" "$dest/"

    echo ""
    echo "Download complete!"
    ls -la "$dest"
}

cmd_destroy() {
    local instance_id=${1:-$(load_instance_id)}
    if [ -z "$instance_id" ]; then
        echo "Usage: deploy_vastai.sh destroy <instance_id>"
        exit 1
    fi

    echo "Destroying instance $instance_id..."
    $VASTAI destroy instance $instance_id
    rm -f "$INSTANCE_FILE"
    echo "Done."
}

cmd_full() {
    # Parse args
    local chapter_arg=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --chapter) chapter_arg="--chapter $2"; shift 2 ;;
            *) shift ;;
        esac
    done

    cmd_create
    instance_id=$(load_instance_id)

    echo ""
    echo "Waiting 30s for instance to fully initialize..."
    sleep 30

    cmd_setup "$instance_id"
    cmd_generate "$instance_id" $chapter_arg

    echo ""
    echo "=== Generation running ==="
    echo "Check status: ./scripts/deploy_vastai.sh status"
    echo "Download:     ./scripts/deploy_vastai.sh download"
    echo "Destroy:      ./scripts/deploy_vastai.sh destroy"
}

cmd_ssh() {
    local instance_id=${1:-$(load_instance_id)}
    if [ -z "$instance_id" ]; then
        echo "Usage: deploy_vastai.sh ssh <instance_id>"
        exit 1
    fi

    local ssh_info=$(get_ssh_info $instance_id)
    local host=$(echo "$ssh_info" | cut -d: -f1)
    local port=$(echo "$ssh_info" | cut -d: -f2)

    echo "ssh -p $port root@$host"
}

# Main
case "${1:-}" in
    create)   cmd_create ;;
    setup)    cmd_setup "$2" ;;
    generate) shift; cmd_generate "$@" ;;
    status)   cmd_status "$2" ;;
    download) cmd_download "$2" ;;
    destroy)  cmd_destroy "$2" ;;
    full)     shift; cmd_full "$@" ;;
    ssh)      cmd_ssh "$2" ;;
    *)
        echo "Vast.ai Audiobook Deployment"
        echo ""
        echo "Usage:"
        echo "  ./scripts/deploy_vastai.sh create                     # Create GPU instance"
        echo "  ./scripts/deploy_vastai.sh setup [instance_id]        # Install deps"
        echo "  ./scripts/deploy_vastai.sh generate [instance_id] [--chapter N]"
        echo "  ./scripts/deploy_vastai.sh status [instance_id]       # Check progress"
        echo "  ./scripts/deploy_vastai.sh download [instance_id]     # Get audio files"
        echo "  ./scripts/deploy_vastai.sh destroy [instance_id]      # Cleanup"
        echo "  ./scripts/deploy_vastai.sh ssh [instance_id]          # Print SSH command"
        echo ""
        echo "  ./scripts/deploy_vastai.sh full --chapter 1           # Do everything"
        echo ""
        echo "Instance ID is saved to .vastai_instance and reused if not specified."
        ;;
esac
