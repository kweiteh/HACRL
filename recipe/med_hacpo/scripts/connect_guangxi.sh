#!/usr/bin/env bash
# Run from your Mac (must be on VPN / campus network to reach 10.208.x.x)
set -euo pipefail

HOST="${1:-guangxi}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Connecting to $HOST ..."
echo "Tip: ensure ~/.ssh/config contains Host $HOST (see ssh_config.cluster)"

ssh -t "$HOST" "cd ~/HACRL 2>/dev/null || cd ~; exec bash -l"
