"""使用 TRL 的 SFTTrainer 训练 LoRA 适配器。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from sft_sop.check_data import validate_dataset
from sft_sop.modeling import select_device, select_dtype


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    train_file: str
    validation_file: str
    output_dir: str
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_length: int
    seed: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]

    @classmethod
    def from_yaml(cls, path: Path) -> TrainConfig:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"{path} 必须包含一个 YAML 对象")

        known = {field.name for field in fields(cls)}
        missing = known - set(raw)
        unknown = set(raw) - known
        if missing:
            raise ValueError(f"缺少配置项：{sorted(missing)}")
        if unknown:
            raise ValueError(f"存在未知配置项：{sorted(unknown)}")
        return cls(**raw)


def load_model(model_name: str, device: str) -> Any:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=select_dtype(device),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    return model


def train(config: TrainConfig) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    train_path = Path(config.train_file)
    validation_path = Path(config.validation_file)
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("缺少训练数据。请先运行 `make data`。")

    # 在分配模型前尽早发现 schema 错误和 train/test 数据泄漏。
    if train_path.parent == validation_path.parent:
        validate_dataset(train_path.parent)

    set_seed(config.seed)
    device = select_device()
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    fp16 = device == "cuda" and not bf16
    print(f"运行设备={device}，dtype={select_dtype(device)}")

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(validation_path),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = load_model(config.model_name, device)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )

    output_dir = Path(config.output_dir)
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_length=config.max_length,
        seed=config.seed,
        data_seed=config.seed,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16,
        fp16=fp16,
        use_cpu=device == "cpu",
        assistant_only_loss=True,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    trainer.save_metrics("train", result.metrics)

    print(f"LoRA 适配器已保存到 {output_dir}")
    print("下一步：运行 `make evaluate`，比较 reports/baseline.json 与 finetuned.json。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sft_lora.yaml"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(TrainConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
