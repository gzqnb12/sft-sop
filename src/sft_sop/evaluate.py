"""在独立数据集上评测基础模型或训练后的 LoRA 适配器。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from sft_sop.check_data import load_jsonl, validate_dataset
from sft_sop.metrics import compute_metrics, parse_json_output
from sft_sop.modeling import generate_response, load_for_inference


def evaluate(
    model_name: str,
    adapter: Path | None,
    data_dir: Path,
    split: str,
    max_new_tokens: int,
    limit: int | None,
) -> dict[str, Any]:
    validate_dataset(data_dir)
    examples = load_jsonl(data_dir / f"{split}.jsonl")
    if limit is not None:
        examples = examples[:limit]
    if not examples:
        raise ValueError("没有选中任何评测样本。")

    model, tokenizer, device = load_for_inference(model_name, adapter)
    predictions: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    for index, example in enumerate(examples, start=1):
        prompt_messages = example["messages"][:-1]
        reference = json.loads(example["messages"][-1]["content"])
        raw_prediction = generate_response(
            model,
            tokenizer,
            prompt_messages,
            device,
            max_new_tokens=max_new_tokens,
        )
        predictions.append(
            {
                "id": example["id"],
                "input": prompt_messages[-1]["content"],
                "reference": reference,
                "prediction": raw_prediction,
                "parsed": parse_json_output(raw_prediction),
            }
        )
        print(f"[{index:>2}/{len(examples)}] {example['id']}: {raw_prediction}")

    elapsed = time.perf_counter() - started_at
    return {
        "model": model_name,
        "adapter": str(adapter) if adapter else None,
        "device": device,
        "split": split,
        "elapsed_seconds": round(elapsed, 2),
        "metrics": compute_metrics(predictions),
        "predictions": predictions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report", type=Path, default=Path("reports/evaluation.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate(
        model_name=args.model,
        adapter=args.adapter,
        data_dir=args.data_dir,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n评测指标")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"完整报告 -> {args.report}")


if __name__ == "__main__":
    main()
