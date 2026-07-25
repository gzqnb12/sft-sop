"""Model loading and deterministic generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_dtype(device: str) -> Any:
    import torch

    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def load_for_inference(model_name: str, adapter: str | Path | None = None) -> tuple[Any, Any, str]:
    """Load a base model and, optionally, a LoRA adapter."""

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=select_dtype(device),
        low_cpu_mem_usage=True,
    )
    if adapter is not None:
        adapter_path = Path(adapter)
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter not found: {adapter_path}. Run `make train` before evaluating it."
            )
        model = PeftModel.from_pretrained(model, adapter_path)

    model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_response(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    device: str,
    max_new_tokens: int = 64,
) -> str:
    """Generate one deterministic response from a conversational prompt."""

    import torch

    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
    prompt_length = model_inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0, prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
