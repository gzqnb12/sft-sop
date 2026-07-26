"""校验 SDD 文档结构，以及需求到任务的追踪关系。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

FEATURE_NAME = re.compile(r"^\d{3}-[a-z0-9]+(?:-[a-z0-9]+)*$")
REQUIREMENT_ID = re.compile(r"\bFR-\d{3}\b")
ACCEPTANCE_ID = re.compile(r"\bAC-\d{3}\b")
TASK_ID = re.compile(r"\bT-\d{3}\b")
STATUS = re.compile(
    r"^(?:"
    r"状态：(?P<zh>草案|已接受|进行中|已实现|已取代)"
    r"|Status: (?P<en>Draft|Accepted|In Progress|Implemented|Superseded)"
    r")$",
    re.MULTILINE,
)
STATUS_ALIASES = {
    "草案": "Draft",
    "已接受": "Accepted",
    "进行中": "In Progress",
    "已实现": "Implemented",
    "已取代": "Superseded",
    "Draft": "Draft",
    "Accepted": "Accepted",
    "In Progress": "In Progress",
    "Implemented": "Implemented",
    "Superseded": "Superseded",
}
REQUIRED_ARTIFACTS = ("spec.md", "plan.md", "tasks.md")


def _status(text: str) -> str | None:
    match = STATUS.search(text)
    if not match:
        return None
    value = match.group("zh") or match.group("en")
    return STATUS_ALIASES[value]


def _read(path: Path, errors: list[str]) -> str:
    if not path.exists():
        errors.append(f"{path}：缺少必需文档")
        return ""
    text = path.read_text(encoding="utf-8")
    if not STATUS.search(text):
        errors.append(f"{path}：缺少状态行或状态值无效")
    return text


def validate_feature(feature_dir: Path) -> list[str]:
    errors: list[str] = []
    if not FEATURE_NAME.fullmatch(feature_dir.name):
        return [f"{feature_dir}：目录名必须符合 NNN-lowercase-kebab-case"]

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
        errors.append(f"{feature_dir / 'spec.md'}：未找到 FR-NNN 功能需求 ID")
    if not acceptance_ids:
        errors.append(f"{feature_dir / 'spec.md'}：未找到 AC-NNN 验收标准 ID")
    if not task_ids:
        errors.append(f"{feature_dir / 'tasks.md'}：未找到 T-NNN 任务 ID")
    if len(task_ids) != len(set(task_ids)):
        errors.append(f"{feature_dir / 'tasks.md'}：发现重复任务 ID")

    for requirement_id in sorted(requirement_ids):
        if requirement_id not in tasks_text:
            errors.append(
                f"{feature_dir / 'tasks.md'}：{requirement_id} 未追踪到任何任务"
            )
    for acceptance_id in sorted(acceptance_ids):
        if acceptance_id not in tasks_text:
            errors.append(
                f"{feature_dir / 'tasks.md'}：{acceptance_id} 未追踪到任何任务"
            )

    if "## 验证方式" not in plan_text and "## Verification" not in plan_text:
        errors.append(f"{feature_dir / 'plan.md'}：缺少“验证方式”章节")

    spec_status = _status(spec_text)
    tasks_status = _status(tasks_text)
    if (
        spec_status
        and tasks_status
        and spec_status == "Implemented"
        and tasks_status != "Implemented"
    ):
        errors.append(f"{feature_dir}：已实现规格要求任务文档状态同为已实现")
    if spec_status == "Implemented" and "- [ ]" in tasks_text:
        errors.append(f"{feature_dir / 'tasks.md'}：已实现规格仍有未完成任务")

    return errors


def validate_specs(specs_dir: Path) -> list[str]:
    errors: list[str] = []
    if not specs_dir.exists():
        return [f"{specs_dir}：目录不存在"]
    if not (specs_dir / "README.md").exists():
        errors.append(f"{specs_dir / 'README.md'}：缺少 SDD 工作流说明")

    feature_dirs = sorted(
        path
        for path in specs_dir.iterdir()
        if path.is_dir() and not path.name.startswith(("_", "."))
    )
    if not feature_dirs:
        errors.append(f"{specs_dir}：未找到带编号的功能规格")
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
        print("SDD 校验失败：")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    feature_count = sum(
        path.is_dir() and FEATURE_NAME.fullmatch(path.name) is not None
        for path in args.specs_dir.iterdir()
    )
    print(f"SDD 校验通过：已验证 {feature_count} 份功能规格。")


if __name__ == "__main__":
    main()
