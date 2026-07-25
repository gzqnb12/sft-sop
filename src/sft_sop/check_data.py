"""Validate SFT JSONL files before spending GPU time."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sft_sop.constants import INTENTS, URGENCIES
from sft_sop.metrics import is_valid_schema

EXPECTED_SPLITS = ("train", "validation", "test")
EXPECTED_ROLES = ["system", "user", "assistant"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each row must be an object")
            rows.append(value)
    return rows


def validate_row(row: dict[str, Any], location: str) -> tuple[str, str, str]:
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError(f"{location}: missing non-empty string id")

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"{location}: messages must contain system, user and assistant turns")

    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if roles != EXPECTED_ROLES:
        raise ValueError(f"{location}: roles must be {EXPECTED_ROLES}, got {roles}")

    for message in messages:
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"{location}: every message needs non-empty content")

    try:
        answer = json.loads(messages[2]["content"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{location}: assistant content must be strict JSON") from exc
    if not is_valid_schema(answer):
        raise ValueError(
            f"{location}: label must use intent={INTENTS} and urgency={URGENCIES}"
        )
    return row["id"], messages[1]["content"], messages[2]["content"]


def validate_dataset(data_dir: Path) -> dict[str, int]:
    seen_ids: set[str] = set()
    user_text_to_split: dict[str, str] = {}
    counts: dict[str, int] = {}
    label_counts: Counter[tuple[str, str, str]] = Counter()

    for split in EXPECTED_SPLITS:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            raise ValueError(f"Missing split: {path}. Run `make data` first.")
        rows = load_jsonl(path)
        if not rows:
            raise ValueError(f"{path} is empty")
        counts[split] = len(rows)

        for index, row in enumerate(rows, start=1):
            row_id, user_text, answer_text = validate_row(row, f"{path}:{index}")
            if row_id in seen_ids:
                raise ValueError(f"Duplicate id across dataset: {row_id}")
            seen_ids.add(row_id)

            previous_split = user_text_to_split.get(user_text)
            if previous_split is not None and previous_split != split:
                raise ValueError(
                    f"Data leakage: identical user text appears in {previous_split} and {split}"
                )
            user_text_to_split[user_text] = split

            answer = json.loads(answer_text)
            label_counts[(split, answer["intent"], answer["urgency"])] += 1

    missing_labels = [
        (split, intent, urgency)
        for split in EXPECTED_SPLITS
        for intent in INTENTS
        for urgency in URGENCIES
        if label_counts[(split, intent, urgency)] == 0
    ]
    if missing_labels:
        raise ValueError(f"Some splits do not cover the full label grid: {missing_labels}")

    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    counts = validate_dataset(args.data_dir)
    print("Data check passed.")
    for split, count in counts.items():
        print(f"{split:>10}: {count:>3} examples")
    print("No duplicate IDs, cross-split text leakage, schema errors or missing labels found.")


if __name__ == "__main__":
    main()
