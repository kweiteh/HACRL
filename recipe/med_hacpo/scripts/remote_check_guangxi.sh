#!/usr/bin/env bash
# Run on guangxi server (or via: ssh guangxi 'bash -s' < recipe/med_hacpo/scripts/remote_check_guangxi.sh)
set -euo pipefail

echo "========== Host =========="
hostname
hostname -I 2>/dev/null || true
whoami
pwd

echo
echo "========== GPU =========="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
else
  echo "nvidia-smi not found"
fi

echo
echo "========== HACRL =========="
if [[ -d "$HOME/HACRL" ]]; then
  ls -la "$HOME/HACRL" | head -25
  echo "--- git ---"
  cd "$HOME/HACRL" && git status -sb 2>/dev/null | head -5 || true
  echo "--- branches ---"
  git branch -a 2>/dev/null | head -10 || true
else
  echo "MISSING: $HOME/HACRL"
fi

echo
echo "========== Python Env =========="
for p in \
  "$HOME/venv" "$HOME/.venv" "$HOME/miniconda3/envs/verl" \
  "$HOME/anaconda3/envs/verl" "/opt/conda/envs/verl"; do
  [[ -f "$p/bin/activate" ]] && echo "FOUND venv: $p"
done
command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | head -10 || true
python3 -c "import sys; print('python:', sys.executable)" 2>/dev/null || true
for pkg in torch ray vllm verl transformers; do
  python3 -c "import $pkg; print('$pkg:', getattr($pkg,'__version__','ok'))" 2>/dev/null || echo "$pkg: MISSING"
done

echo
echo "========== Data =========="
find "$HOME/data" "$HOME/HACRL/data" -maxdepth 3 \
  \( -name '*.parquet' -o -name '*.jsonl' \) 2>/dev/null | head -20

echo
echo "========== Disk / Mem =========="
free -h 2>/dev/null | sed -n '1,2p' || true
df -h "$HOME" 2>/dev/null | sed -n '1,2p' || true
