"""Pure-Python output parsing and task metrics."""

from __future__ import annotations

import json
from typing import Any

from sft_sop.constants import INTENTS, URGENCIES


def parse_json_output(text: str) -> dict[str, Any] | None:
    """Parse a model response while tolerating common wrappers.

    The evaluator reports strict schema validity separately, so extraction here
    is deliberately forgiving: it lets us see whether a model knew the answer
    but added Markdown or reasoning around it.
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
    """Return whether an object exactly matches the task schema."""

    if value is None or set(value) != {"intent", "urgency"}:
        return False
    return value["intent"] in INTENTS and value["urgency"] in URGENCIES


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Aggregate generated predictions against references."""

    total = len(rows)
    if total == 0:
        raise ValueError("Cannot score an empty prediction set.")

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
