#!/usr/bin/env bash
# Start tmux infrastructure for pseudo-federated medical HACPO (Phase 1+).
set -euo pipefail

TMUX_CONF="${TMUX_CONF:-/exec-daemon/tmux.portal.conf}"
SESSION="${SESSION:-med-hacrl-infra}"

tmux_cmd() {
  if [[ -f "$TMUX_CONF" ]]; then
    tmux -f "$TMUX_CONF" "$@"
  else
    tmux "$@"
  fi
}

if tmux_cmd has-session -t "$SESSION" 2>/dev/null; then
  echo "Session $SESSION already exists"
  tmux_cmd ls
  exit 0
fi

tmux_cmd new-session -d -s "$SESSION" -n redis
tmux_cmd send-keys -t "$SESSION:redis" 'redis-server --port 6379 --save ""' C-m

tmux_cmd new-window -t "$SESSION" -n ray-a
tmux_cmd send-keys -t "$SESSION:ray-a" 'export CUDA_VISIBLE_DEVICES=0,1,2,3' C-m
tmux_cmd send-keys -t "$SESSION:ray-a" 'ray start --head --port=6380 --num-gpus=4 || true' C-m

tmux_cmd new-window -t "$SESSION" -n ray-b
tmux_cmd send-keys -t "$SESSION:ray-b" 'export CUDA_VISIBLE_DEVICES=4,5,6,7' C-m
tmux_cmd send-keys -t "$SESSION:ray-b" 'ray start --head --port=6390 --num-gpus=4 || true' C-m

tmux_cmd new-window -t "$SESSION" -n monitor
tmux_cmd send-keys -t "$SESSION:monitor" 'watch -n5 nvidia-smi' C-m

echo "Started tmux session: $SESSION"
tmux_cmd ls
