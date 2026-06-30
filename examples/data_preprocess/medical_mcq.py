# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess medical multiple-choice datasets into verl parquet format.

Supported sources:
  - medqa: GBaker/MedQA-USMLE-4-options
  - medmcqa: openlifescienceai/medmcqa
  - custom_jsonl: local JSONL with question/options/answer fields
  - custom_parquet: local parquet with the same logical fields
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from typing import Any

import datasets
import pandas as pd

INSTRUCTION = (
  "You are a medical expert. Read the clinical question carefully, reason step by step, "
  "and output the final option letter within \\boxed{}."
)

DATA_SOURCE_BY_NAME = {
  "medqa": "GBaker/MedQA-USMLE-4-options",
  "medmcqa": "openlifescienceai/medmcqa",
  "custom_jsonl": "custom/medical_mcq_jsonl",
  "custom_parquet": "custom/medical_mcq_parquet",
}


def normalize_choice(value: str | int | None) -> str | None:
  if value is None:
    return None
  if isinstance(value, int):
    if 0 <= value <= 3:
      return chr(ord("A") + value)
    return None
  text = str(value).strip().upper()
  if text in {"A", "B", "C", "D"}:
    return text
  if text.isdigit():
    idx = int(text)
    if 0 <= idx <= 3:
      return chr(ord("A") + idx)
  return None


def format_options(options: dict[str, str]) -> str:
  lines = [f"{key}. {value}" for key, value in sorted(options.items())]
  return "\n".join(lines)


def build_prompt(question: str, options: dict[str, str]) -> list[dict[str, str]]:
  content = (
    f"{question.strip()}\n\n"
    f"Options:\n{format_options(options)}\n\n"
    f"{INSTRUCTION}"
  )
  return [{"role": "user", "content": content}]


def to_verl_row(
  *,
  data_source: str,
  question: str,
  options: dict[str, str],
  answer: str,
  split: str,
  index: int,
  extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
  answer_letter = normalize_choice(answer)
  if answer_letter is None:
    raise ValueError(f"Invalid answer for index={index}: {answer!r}")
  return {
    "data_source": data_source,
    "prompt": build_prompt(question, options),
    "ability": "medical",
    "reward_model": {"style": "rule", "ground_truth": answer_letter},
    "extra_info": {
      "split": split,
      "index": index,
      "question": question,
      "options": options,
      "answer": answer_letter,
      **(extra or {}),
    },
  }


def parse_options_field(raw: Any) -> dict[str, str]:
  if isinstance(raw, dict):
    return {str(k).strip().upper(): str(v).strip() for k, v in raw.items()}
  if isinstance(raw, str):
    try:
      parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
      parsed = json.loads(raw)
    if isinstance(parsed, dict):
      return {str(k).strip().upper(): str(v).strip() for k, v in parsed.items()}
  raise ValueError(f"Unsupported options field: {raw!r}")


def load_medqa() -> datasets.DatasetDict:
  dataset = datasets.load_dataset(DATA_SOURCE_BY_NAME["medqa"])
  return datasets.DatasetDict(
    {
      "train": dataset["train"],
      "test": dataset["test"],
    }
  )


def load_medmcqa() -> datasets.DatasetDict:
  dataset = datasets.load_dataset(DATA_SOURCE_BY_NAME["medmcqa"])
  return datasets.DatasetDict(
    {
      "train": dataset["train"],
      "test": dataset["validation"],
    }
  )


def load_custom_jsonl(path: str) -> datasets.DatasetDict:
  rows = []
  with open(path, encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  if not rows:
    raise ValueError(f"No rows found in {path}")
  ds = datasets.Dataset.from_list(rows)
  if "split" in ds.column_names:
    train = ds.filter(lambda x: x["split"] != "test")
    test = ds.filter(lambda x: x["split"] == "test")
    if len(test) == 0:
      if len(ds) == 1:
        train, test = ds, ds
      else:
        split = ds.train_test_split(test_size=max(1, int(len(ds) * 0.1)), seed=42)
        train, test = split["train"], split["test"]
  else:
    if len(ds) == 1:
      train, test = ds, ds
    else:
      split = ds.train_test_split(test_size=max(0.1, 1 / len(ds)), seed=42)
      train, test = split["train"], split["test"]
  return datasets.DatasetDict({"train": train, "test": test})


def load_custom_parquet(path: str) -> datasets.DatasetDict:
  df = pd.read_parquet(path)
  ds = datasets.Dataset.from_pandas(df)
  if "split" in ds.column_names:
    train = ds.filter(lambda x: x["split"] != "test")
    test = ds.filter(lambda x: x["split"] == "test")
    if len(test) == 0:
      if len(ds) == 1:
        train, test = ds, ds
      else:
        split = ds.train_test_split(test_size=max(1, int(len(ds) * 0.1)), seed=42)
        train, test = split["train"], split["test"]
  else:
    if len(ds) == 1:
      train, test = ds, ds
    else:
      split = ds.train_test_split(test_size=max(0.1, 1 / len(ds)), seed=42)
      train, test = split["train"], split["test"]
  return datasets.DatasetDict({"train": train, "test": test})


def map_medqa(example: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
  options = parse_options_field(example["options"])
  return to_verl_row(
    data_source=DATA_SOURCE_BY_NAME["medqa"],
    question=example["question"],
    options=options,
    answer=example["answer_idx"],
    split=split,
    index=idx,
    extra={"meta_info": example.get("meta_info")},
  )


def map_medmcqa(example: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
  options = {
    "A": example["opa"],
    "B": example["opb"],
    "C": example["opc"],
    "D": example["opd"],
  }
  return to_verl_row(
    data_source=DATA_SOURCE_BY_NAME["medmcqa"],
    question=example["question"],
    options=options,
    answer=example["cop"],
    split=split,
    index=idx,
    extra={
      "subject_name": example.get("subject_name"),
      "topic_name": example.get("topic_name"),
    },
  )


def map_custom(example: dict[str, Any], idx: int, split: str, data_source: str) -> dict[str, Any]:
  question = example.get("question") or example.get("stem") or example.get("prompt_text")
  if question is None:
    raise KeyError("custom row must contain question/stem/prompt_text")
  options = example.get("options")
  if options is None and all(k in example for k in ["A", "B", "C", "D"]):
    options = {k: example[k] for k in ["A", "B", "C", "D"]}
  if options is None:
    raise KeyError("custom row must contain options or A/B/C/D fields")
  options = parse_options_field(options)
  answer = example.get("answer_idx") or example.get("answer") or example.get("label")
  if answer is None:
    raise KeyError("custom row must contain answer_idx/answer/label")
  return to_verl_row(
    data_source=data_source,
    question=question,
    options=options,
    answer=answer,
    split=split,
    index=idx,
    extra={k: v for k, v in example.items() if k not in {"question", "stem", "options", "answer", "answer_idx", "label"}},
  )


def make_map_fn(source: str, split: str):
  def _fn(example: dict[str, Any], idx: int) -> dict[str, Any]:
    if source == "medqa":
      return map_medqa(example, idx, split)
    if source == "medmcqa":
      return map_medmcqa(example, idx, split)
  return _fn


def make_custom_map_fn(data_source: str, split: str):
  def _fn(example: dict[str, Any], idx: int) -> dict[str, Any]:
    return map_custom(example, idx, split, data_source)
  return _fn


def maybe_subsample(ds: datasets.Dataset, max_samples: int | None) -> datasets.Dataset:
  if max_samples is None or max_samples <= 0 or len(ds) <= max_samples:
    return ds
  return ds.select(range(max_samples))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--source",
    choices=["medqa", "medmcqa", "custom_jsonl", "custom_parquet"],
    default="medqa",
    help="Dataset source",
  )
  parser.add_argument("--input_path", default=None, help="Required for custom_jsonl/custom_parquet")
  parser.add_argument("--local_dir", default="~/data/medical_mcq", help="Output directory")
  parser.add_argument("--max_train_samples", type=int, default=None)
  parser.add_argument("--max_test_samples", type=int, default=None)
  args = parser.parse_args()

  if args.source == "medqa":
    raw = load_medqa()
    data_source = DATA_SOURCE_BY_NAME["medqa"]
    train_map = make_map_fn("medqa", "train")
    test_map = make_map_fn("medqa", "test")
  elif args.source == "medmcqa":
    raw = load_medmcqa()
    data_source = DATA_SOURCE_BY_NAME["medmcqa"]
    train_map = make_map_fn("medmcqa", "train")
    test_map = make_map_fn("medmcqa", "test")
  elif args.source == "custom_jsonl":
    if not args.input_path:
      raise ValueError("--input_path is required for custom_jsonl")
    raw = load_custom_jsonl(os.path.expanduser(args.input_path))
    data_source = DATA_SOURCE_BY_NAME["custom_jsonl"]
    train_map = make_custom_map_fn(data_source, "train")
    test_map = make_custom_map_fn(data_source, "test")
  else:
    if not args.input_path:
      raise ValueError("--input_path is required for custom_parquet")
    raw = load_custom_parquet(os.path.expanduser(args.input_path))
    data_source = DATA_SOURCE_BY_NAME["custom_parquet"]
    train_map = make_custom_map_fn(data_source, "train")
    test_map = make_custom_map_fn(data_source, "test")

  train_dataset = maybe_subsample(raw["train"], args.max_train_samples).map(
    function=train_map, with_indices=True, remove_columns=raw["train"].column_names
  )
  test_dataset = maybe_subsample(raw["test"], args.max_test_samples).map(
    function=test_map, with_indices=True, remove_columns=raw["test"].column_names
  )

  local_dir = os.path.expanduser(args.local_dir)
  os.makedirs(local_dir, exist_ok=True)
  train_path = os.path.join(local_dir, "train.parquet")
  test_path = os.path.join(local_dir, "test.parquet")
  train_dataset.to_parquet(train_path)
  test_dataset.to_parquet(test_path)

  example = train_dataset[0]
  with open(os.path.join(local_dir, "train_example.json"), "w", encoding="utf-8") as f:
    json.dump(example, f, ensure_ascii=False, indent=2)

  meta = {
    "source": args.source,
    "data_source": data_source,
    "train_rows": len(train_dataset),
    "test_rows": len(test_dataset),
    "train_path": train_path,
    "test_path": test_path,
  }
  with open(os.path.join(local_dir, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

  print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
