"""Validate SDD artifact structure and requirement-to-task traceability."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

FEATURE_NAME = re.compile(r"^\d{3}-[a-z0-9]+(?:-[a-z0-9]+)*$")
REQUIREMENT_ID = re.compile(r"\bFR-\d{3}\b")
ACCEPTANCE_ID = re.compile(r"\bAC-\d{3}\b")
TASK_ID = re.compile(r"\bT-\d{3}\b")
STATUS = re.compile(
    r"^Status: (Draft|Accepted|In Progress|Implemented|Superseded)$",
    re.MULTILINE,
)
REQUIRED_ARTIFACTS = ("spec.md", "plan.md", "tasks.md")


def _read(path: Path, errors: list[str]) -> str:
    if not path.exists():
        errors.append(f"{path}: missing required artifact")
        return ""
    text = path.read_text(encoding="utf-8")
    if not STATUS.search(text):
        errors.append(f"{path}: missing or invalid Status line")
    return text


def validate_feature(feature_dir: Path) -> list[str]:
    errors: list[str] = []
    if not FEATURE_NAME.fullmatch(feature_dir.name):
        return [f"{feature_dir}: directory name must match NNN-lowercase-kebab-case"]

    artifacts = {
        name: _read(feature_dir / name, errors) for name in REQUIRED_ARTIFACTS
    }
    spec_text = artifacts["spec.md"]
    plan_text = artifacts["plan.md"]
    tasks_text = artifacts["tasks.md"]

    requirement_ids = set(REQUIREMENT_ID.findall(spec_text))
    acceptance_ids = set(ACCEPTANCE_ID.findall(spec_text))
    task_ids = TASK_ID.findall(tasks_text)

    if not requirement_ids:
        errors.append(f"{feature_dir / 'spec.md'}: no FR-NNN requirement IDs found")
    if not acceptance_ids:
        errors.append(f"{feature_dir / 'spec.md'}: no AC-NNN acceptance IDs found")
    if not task_ids:
        errors.append(f"{feature_dir / 'tasks.md'}: no T-NNN task IDs found")
    if len(task_ids) != len(set(task_ids)):
        errors.append(f"{feature_dir / 'tasks.md'}: duplicate task IDs found")

    for requirement_id in sorted(requirement_ids):
        if requirement_id not in tasks_text:
            errors.append(
                f"{feature_dir / 'tasks.md'}: {requirement_id} is not traced to a task"
            )
    for acceptance_id in sorted(acceptance_ids):
        if acceptance_id not in tasks_text:
            errors.append(
                f"{feature_dir / 'tasks.md'}: {acceptance_id} is not traced to a task"
            )

    if "## Verification" not in plan_text:
        errors.append(f"{feature_dir / 'plan.md'}: missing ## Verification section")

    spec_status = STATUS.search(spec_text)
    tasks_status = STATUS.search(tasks_text)
    if (
        spec_status
        and tasks_status
        and spec_status.group(1) == "Implemented"
        and tasks_status.group(1) != "Implemented"
    ):
        errors.append(f"{feature_dir}: implemented spec requires implemented task status")
    if spec_status and spec_status.group(1) == "Implemented" and "- [ ]" in tasks_text:
        errors.append(f"{feature_dir / 'tasks.md'}: implemented spec has open tasks")

    return errors


def validate_specs(specs_dir: Path) -> list[str]:
    errors: list[str] = []
    if not specs_dir.exists():
        return [f"{specs_dir}: directory does not exist"]
    if not (specs_dir / "README.md").exists():
        errors.append(f"{specs_dir / 'README.md'}: missing SDD workflow documentation")

    feature_dirs = sorted(
        path
        for path in specs_dir.iterdir()
        if path.is_dir() and not path.name.startswith(("_", "."))
    )
    if not feature_dirs:
        errors.append(f"{specs_dir}: no numbered feature specifications found")
    for feature_dir in feature_dirs:
        errors.extend(validate_feature(feature_dir))
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs-dir", type=Path, default=Path("specs"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    errors = validate_specs(args.specs_dir)
    if errors:
        print("SDD check failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    feature_count = sum(
        path.is_dir() and FEATURE_NAME.fullmatch(path.name) is not None
        for path in args.specs_dir.iterdir()
    )
    print(f"SDD check passed: {feature_count} feature specification(s) validated.")


if __name__ == "__main__":
    main()
