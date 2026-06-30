#!/usr/bin/env bash
# Prepare medical MCQ parquet data for verl / HACPO.
set -euo pipefail

SOURCE="${SOURCE:-medqa}"
LOCAL_DIR="${LOCAL_DIR:-$HOME/data/medical_mcq/${SOURCE}}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-}"
INPUT_PATH="${INPUT_PATH:-}"

ARGS=(--source "$SOURCE" --local_dir "$LOCAL_DIR")
if [[ -n "$MAX_TRAIN_SAMPLES" ]]; then
  ARGS+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
fi
if [[ -n "$MAX_TEST_SAMPLES" ]]; then
  ARGS+=(--max_test_samples "$MAX_TEST_SAMPLES")
fi
if [[ -n "$INPUT_PATH" ]]; then
  ARGS+=(--input_path "$INPUT_PATH")
fi

echo "Preparing medical data: source=$SOURCE -> $LOCAL_DIR"
python3 /workspace/examples/data_preprocess/medical_mcq.py "${ARGS[@]}"
