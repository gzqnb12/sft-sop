# 001 End-to-end SFT learning SOP — implementation plan

Status: Implemented

## Approach

Use a deterministic synthetic customer-ticket task with two categorical output
fields. Keep data construction, validation, training, generation, parsing, and
metrics as separate modules so each stage can be read and tested independently.

Use a Qwen3 0.6B base model to make instruction-following improvement visible.
Apply LoRA through PEFT and TRL, with assistant-only loss. Use the same chat
template options during training and inference. Detect CUDA, MPS, or CPU at
runtime and avoid quantization in the first learning project.

## Architecture

```text
build_data → JSONL splits → check_data
                              ↓
base model → evaluate → baseline report
     ↓
SFTTrainer + LoRA → adapter → evaluate → fine-tuned report
                                  ↓
                                infer
```

## Affected files

- `src/sft_sop/build_data.py`: deterministic examples and splits.
- `src/sft_sop/check_data.py`: pre-training data gate.
- `src/sft_sop/train.py`: configuration loading and LoRA SFT.
- `src/sft_sop/modeling.py`: device selection and deterministic generation.
- `src/sft_sop/evaluate.py`: held-out evaluation and report generation.
- `src/sft_sop/infer.py`: single-message inference.
- `src/sft_sop/metrics.py`: parsing and objective metrics.
- `configs/sft_lora.yaml`: reproducible experiment configuration.
- `README.md`: learner-facing SOP.

## Decisions

- Use a base model instead of an instruct model to expose the effect of SFT.
- Use LoRA instead of full fine-tuning to reduce memory and checkpoint size.
- Avoid QLoRA in the first project so CUDA, MPS, and CPU share one path.
- Generate small synthetic data so no private source data enters the repository.
- Keep baseline and adapter evaluation on the same held-out test set.
- Do not run GPU training as part of ordinary automated verification.

## Risks and mitigations

- Risk: The synthetic task is too easy to represent real-world data work.
  - Mitigation: Document that it teaches mechanics, not production quality.
- Risk: Template leakage inflates metrics.
  - Mitigation: Reserve utterances and prompt wrappers by split and check exact
    cross-split duplicates.
- Risk: TRL chat-template behavior changes across versions.
  - Mitigation: Lock dependencies and test current configuration construction.
- Risk: Learners mistake training loss for task quality.
  - Mitigation: Require objective test metrics and per-example error inspection.

## Verification

- `make data && make check`: FR-001, FR-002, AC-001.
- `make test`: NFR-001, AC-002.
- `make lint`: AC-003.
- `make baseline`: FR-003, FR-005, AC-004.
- `make train`: FR-004, AC-005.
- `make evaluate`: FR-005, AC-006.
- `make infer`: FR-006, AC-007.
- README review: FR-007 and teaching clarity.
