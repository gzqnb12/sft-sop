"""仅使用 Python 的输出解析与任务指标。"""

from __future__ import annotations

import json
from typing import Any

from sft_sop.constants import INTENTS, URGENCIES


def parse_json_output(text: str) -> dict[str, Any] | None:
    """解析模型回复，同时容忍常见的外层包装。

    评测器会单独报告严格的 schema 有效性，因此这里有意采用宽松提取：借此区分模型
    不知道答案，还是知道答案但在外层添加了 Markdown 或推理。
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end < start:
        return None

    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def is_valid_schema(value: dict[str, Any] | None) -> bool:
    """返回对象是否严格符合任务 schema。"""

    if value is None or set(value) != {"intent", "urgency"}:
        return False
    return value["intent"] in INTENTS and value["urgency"] in URGENCIES


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """汇总生成预测相对于参考答案的指标。"""

    total = len(rows)
    if total == 0:
        raise ValueError("不能对空预测集评分。")

    json_valid = 0
    schema_valid = 0
    intent_correct = 0
    urgency_correct = 0
    joint_correct = 0

    for row in rows:
        parsed = row.get("parsed")
        reference = row["reference"]
        json_valid += parsed is not None
        schema_valid += is_valid_schema(parsed)
        if parsed is not None:
            intent_match = parsed.get("intent") == reference["intent"]
            urgency_match = parsed.get("urgency") == reference["urgency"]
            intent_correct += intent_match
            urgency_correct += urgency_match
            joint_correct += intent_match and urgency_match

    return {
        "examples": total,
        "json_valid_rate": round(json_valid / total, 4),
        "schema_valid_rate": round(schema_valid / total, 4),
        "intent_accuracy": round(intent_correct / total, 4),
        "urgency_accuracy": round(urgency_correct / total, 4),
        "joint_accuracy": round(joint_correct / total, 4),
    }
