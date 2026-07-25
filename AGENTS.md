# AGENTS.md

## Mission

This repository is a small, reproducible learning project for closing the full
supervised fine-tuning loop: define a task, build and validate data, measure a
base-model baseline, train a LoRA adapter, evaluate it, and run inference.

Keep the project educational, local-first, inexpensive, and easy to inspect.

## Source map

- `README.md`: learner-facing SOP and commands.
- `specs/`: source of truth for intended behavior and accepted changes.
- `src/sft_sop/`: data, training, evaluation, metrics, and inference code.
- `configs/sft_lora.yaml`: reproducible training configuration.
- `data/`: deterministic generated teaching data.
- `tests/`: CPU-only fast tests.

## Spec-driven development

For every material behavior change:

1. Read `specs/README.md` and the relevant existing specification.
2. Create or update `specs/NNN-short-name/spec.md`.
3. Resolve requirements and acceptance criteria before changing code.
4. Write `plan.md`, including affected files, risks, and verification.
5. Write `tasks.md`; every task must reference requirement and acceptance IDs.
6. Implement tasks in order and keep the checkboxes current.
7. Run verification and update the artifacts if implementation decisions changed.

A typo, wording-only documentation edit, or mechanical formatting change may
update the closest existing spec instead of creating a new feature directory.
Never change observable behavior first and retrofit a spec afterward.

## Supported commands

```bash
make data       # regenerate deterministic JSONL splits
make check      # validate data schema, labels, and leakage
make sdd-check  # validate SDD artifact structure and traceability
make test       # run fast CPU-only tests
make lint       # run Ruff
```

Commands that download a model or use substantial compute are opt-in:

```bash
make baseline
make train
make evaluate
make infer
```

Do not run GPU training merely to validate a code or documentation change.

## Engineering constraints

- Use Python 3.11 unless a spec explicitly changes the supported range.
- Keep `Qwen/Qwen3-0.6B-Base` as the default teaching model.
- Keep training compatible with CUDA, Apple MPS, and CPU.
- Treat `src/sft_sop/build_data.py` as the source of truth for generated data;
  regenerate JSONL instead of editing generated rows by hand.
- Do not commit checkpoints, reports, caches, local paths, credentials, personal
  email addresses, device identifiers, or machine-specific metadata.
- Preserve assistant-only loss and train/evaluation chat-template consistency.
- Prefer objective evaluation metrics and held-out test data over anecdotal output.

## Definition of done

A material change is complete only when:

- the spec, plan, and tasks agree with the implementation;
- `make sdd-check`, `make check`, `make test`, and `make lint` pass;
- learner-facing behavior is reflected in `README.md`;
- generated data is regenerated when its source changes;
- no ignored model artifacts or private information are staged;
- compute-heavy validation is either completed when explicitly requested or
  clearly recorded as not run.
