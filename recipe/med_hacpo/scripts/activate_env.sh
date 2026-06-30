#!/usr/bin/env bash
# Auto-detect and activate a preconfigured Python environment for HACRL/verl.
# Override manually with: ENV_PATH=/path/to/env source recipe/med_hacpo/scripts/activate_env.sh
set -euo pipefail

if [[ -n "${HACRL_ENV_ACTIVATED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi

_candidates=()

if [[ -n "${ENV_PATH:-}" ]]; then
  _candidates+=("$ENV_PATH")
fi

# Explicit user overrides
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  _candidates+=("$CONDA_ENV_PATH")
fi
if [[ -n "${VENV_PATH:-}" ]]; then
  _candidates+=("$VENV_PATH")
fi

# Common conda env names used in verl recipes
_home="${HOME:-/home/ubuntu}"
for _conda_root in \
  "${CONDA_ROOT:-}" \
  "$_home/miniconda3" \
  "$_home/anaconda3" \
  "$_home/miniforge3" \
  "$_home/mambaforge" \
  "/opt/conda" \
  "/opt/miniconda3" \
  "/usr/local/miniconda3" \
  "/root/miniconda3" \
  "/root/anaconda3"; do
  [[ -n "$_conda_root" && -d "$_conda_root" ]] || continue
  for _env in verl hacrl hacpo rl; do
    _candidates+=("$_conda_root/envs/$_env")
  done
done

# Common venv locations
for _venv in \
  "$_home/.venv" \
  "$_home/venv" \
  "$_home/.python/verl_env" \
  "$_home/.python/spin_env" \
  "/workspace/.venv" \
  "/workspace/venv"; do
  _candidates+=("$_venv")
done

_activate() {
  local _path="$1"
  if [[ -f "$_path/bin/activate" ]]; then
  # shellcheck disable=SC1091
    source "$_path/bin/activate"
    export HACRL_ENV_ACTIVATED=1
    export HACRL_ENV_PATH="$_path"
    echo "[activate_env] venv: $_path"
    return 0
  fi
  if [[ -f "$_path/bin/python" ]]; then
    export PATH="$_path/bin:$PATH"
    export HACRL_ENV_ACTIVATED=1
    export HACRL_ENV_PATH="$_path"
    echo "[activate_env] python prefix: $_path"
    return 0
  fi
  return 1
}

for _cand in "${_candidates[@]}"; do
  [[ -n "$_cand" ]] || continue
  _cand="${_cand/#\~/$HOME}"
  if _activate "$_cand"; then
    break
  fi
done

# Fallback: conda activate verl if conda is on PATH
if [[ -z "${HACRL_ENV_ACTIVATED:-}" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  for _env in verl hacrl hacpo; do
    if conda activate "$_env" 2>/dev/null; then
      export HACRL_ENV_ACTIVATED=1
      export HACRL_ENV_PATH="conda:${_env}"
      echo "[activate_env] conda env: $_env"
      break
    fi
  done
fi

# Project-local editable install fallback
if [[ -z "${HACRL_ENV_ACTIVATED:-}" && -d "/workspace/verl" ]]; then
  export PYTHONPATH="/workspace:${PYTHONPATH:-}"
  export HACRL_ENV_ACTIVATED=1
  export HACRL_ENV_PATH="PYTHONPATH:/workspace"
  echo "[activate_env] fallback PYTHONPATH=/workspace (no venv found)"
fi

# Disable wandb by default
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

echo "[activate_env] python: $(command -v python3 || command -v python)"
python3 - <<'PY' || true
import importlib
for pkg in ("torch", "ray", "vllm", "verl", "transformers"):
    try:
        m = importlib.import_module(pkg)
        print(f"[activate_env] {pkg}: {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        print(f"[activate_env] {pkg}: MISSING ({e.__class__.__name__})")
PY

if [[ -z "${HACRL_ENV_ACTIVATED:-}" ]]; then
  echo "[activate_env] WARNING: no virtual environment activated." >&2
  echo "[activate_env] Set ENV_PATH=/your/env and re-run." >&2
  return 1 2>/dev/null || exit 1
fi
