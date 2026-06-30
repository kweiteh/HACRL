#!/usr/bin/env bash
# Quick environment and data inventory for medical HACPO runs.
set -euo pipefail

echo "========== System =========="
date
uname -a
echo "CPUs: $(nproc)"
free -h | sed -n '1,2p'
df -h / /workspace 2>/dev/null | sed -n '1,3p'

echo
echo "========== GPU =========="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
else
  echo "nvidia-smi not found"
fi

echo
echo "========== Python / Core Packages =========="
python3 --version
for pkg in torch verl ray vllm datasets pandas pyarrow transformers; do
  python3 - <<PY 2>/dev/null || echo "$pkg: NOT INSTALLED"
import importlib
m = importlib.import_module("${pkg}")
print("${pkg}:", getattr(m, "__version__", "ok"))
PY
done

echo
echo "========== Data Inventory =========="
DATA_ROOT="${DATA_ROOT:-$HOME/data}"
echo "DATA_ROOT=$DATA_ROOT"
for d in medical_mcq medical_mcq/medqa medical_mcq/medmcqa; do
  path="$DATA_ROOT/$d"
  if [[ -d "$path" ]]; then
    echo "[FOUND] $path"
    ls -lh "$path"/*.parquet 2>/dev/null || true
    [[ -f "$path/meta.json" ]] && cat "$path/meta.json"
  else
    echo "[MISSING] $path"
  fi
done

echo
echo "========== Custom Data Search =========="
find "$HOME" /workspace /data -maxdepth 4 \
  \( -iname '*med*' -o -iname '*health*' -o -iname '*clinical*' \) \
  \( -name '*.parquet' -o -name '*.jsonl' -o -name '*.csv' \) 2>/dev/null | head -20
