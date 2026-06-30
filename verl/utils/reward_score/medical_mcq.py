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

"""Rule-based reward for medical multiple-choice QA (RLVR)."""

import re

_VALID_CHOICES = set("ABCD")
_SOLUTION_CLIP_CHARS = 800

# Patterns ordered from strict to flexible.
_ANSWER_PATTERNS = [
  r"\\boxed\{([ABCD])\}",
  r"<answer>\s*([ABCD])\s*</answer>",
  r"(?:final\s+answer|correct\s+answer|answer)\s*(?:is|:)?\s*([ABCD])\b",
  r"\b(?:option|choice)\s*([ABCD])\b",
  r"(?:^|\n)\s*([ABCD])\s*[\.\):]",
  r"\b([ABCD])\b\s*$",
]


def normalize_choice(value: str | int | None) -> str | None:
  """Normalize ground-truth / prediction to a single uppercase letter A-D."""
  if value is None:
    return None
  if isinstance(value, int):
    if 0 <= value <= 3:
      return chr(ord("A") + value)
    return None
  text = str(value).strip().upper()
  if text in _VALID_CHOICES:
    return text
  if text.isdigit():
    idx = int(text)
    if 0 <= idx <= 3:
      return chr(ord("A") + idx)
  return None


def extract_choice(solution_str: str, method: str = "flexible") -> str | None:
  """Extract the predicted option letter from model output."""
  assert method in ["strict", "flexible"]
  text = solution_str
  if len(text) > _SOLUTION_CLIP_CHARS:
    text = text[-_SOLUTION_CLIP_CHARS:]

  for pattern in _ANSWER_PATTERNS:
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if matches:
      choice = normalize_choice(matches[-1])
      if choice is not None:
        return choice

  if method == "flexible":
    tail = text.strip().split()[-1] if text.strip() else ""
    return normalize_choice(tail)
  return None


def compute_score(
  solution_str: str,
  ground_truth: str,
  method: str = "flexible",
  format_score: float = 0.0,
  score: float = 1.0,
) -> float:
  """Score a medical MCQ response against the reference option letter."""
  pred = extract_choice(solution_str=solution_str, method=method)
  ref = normalize_choice(ground_truth)
  if pred is None or ref is None:
    return format_score
  return score if pred == ref else format_score
