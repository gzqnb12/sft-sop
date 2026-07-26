"""在占用 GPU 之前校验 SFT JSONL 文件。"""

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
                raise ValueError(f"{path}:{line_number}：JSON 无效：{exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}：每行必须是一个对象")
            rows.append(value)
    return rows


def validate_row(row: dict[str, Any], location: str) -> tuple[str, str, str]:
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError(f"{location}：缺少非空字符串 id")

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"{location}：messages 必须包含 system、user 和 assistant 三轮消息")

    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if roles != EXPECTED_ROLES:
        raise ValueError(f"{location}：角色必须是 {EXPECTED_ROLES}，实际为 {roles}")

    for message in messages:
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"{location}：每条消息都必须包含非空 content")

    try:
        answer = json.loads(messages[2]["content"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{location}：assistant content 必须是严格 JSON") from exc
    if not is_valid_schema(answer):
        raise ValueError(
            f"{location}：标签必须使用 intent={INTENTS} 和 urgency={URGENCIES}"
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
            raise ValueError(f"缺少数据集：{path}。请先运行 `make data`。")
        rows = load_jsonl(path)
        if not rows:
            raise ValueError(f"{path} 为空")
        counts[split] = len(rows)

        for index, row in enumerate(rows, start=1):
            row_id, user_text, answer_text = validate_row(row, f"{path}:{index}")
            if row_id in seen_ids:
                raise ValueError(f"数据集中存在重复 id：{row_id}")
            seen_ids.add(row_id)

            previous_split = user_text_to_split.get(user_text)
            if previous_split is not None and previous_split != split:
                raise ValueError(
                    f"数据泄漏：相同用户文本同时出现在 {previous_split} 和 {split}"
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
        raise ValueError(f"部分数据集未覆盖完整标签组合：{missing_labels}")

    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    counts = validate_dataset(args.data_dir)
    print("数据校验通过。")
    for split, count in counts.items():
        print(f"{split:>10}: {count:>3} 条样本")
    print("未发现重复 ID、跨数据集文本泄漏、结构错误或标签缺失。")


if __name__ == "__main__":
    main()
