"""使用基础模型或 LoRA 适配器执行一次交互式预测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sft_sop.constants import SYSTEM_PROMPT
from sft_sop.metrics import is_valid_schema, parse_json_output
from sft_sop.modeling import generate_response, load_for_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--text", required=True, help="原始客户消息。")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model, tokenizer, device = load_for_inference(args.model, args.adapter)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"判断这条客户诉求的意图和紧急程度：{args.text}"},
    ]
    response = generate_response(
        model,
        tokenizer,
        messages,
        device,
        max_new_tokens=args.max_new_tokens,
    )
    parsed = parse_json_output(response)
    print(
        json.dumps(
            {
                "device": device,
                "raw_output": response,
                "parsed": parsed,
                "schema_valid": is_valid_schema(parsed),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
