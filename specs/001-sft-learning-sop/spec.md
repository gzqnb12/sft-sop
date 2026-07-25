# 001 End-to-end SFT learning SOP

Status: Implemented

## Context

A learner needs a small project that demonstrates the complete supervised
fine-tuning lifecycle without requiring a production dataset, a large model, or
expensive compute. The project must make the effect of SFT measurable instead
of relying on subjective chat examples.

## Goals

- Demonstrate a complete baseline → SFT → evaluation → inference loop.
- Keep every data and model decision visible in a small codebase.
- Use objective metrics on a held-out test split.
- Run on NVIDIA CUDA, Apple MPS, or CPU.
- Keep normal validation independent of model downloads and GPU training.

## Non-goals

- Producing a production-quality customer-support classifier.
- Benchmarking different foundation models or tuning for maximum accuracy.
- Serving the adapter as an online API.
- Teaching preference optimization or reinforcement learning.

## User stories

- As a learner, I want deterministic example data so that I can inspect every
  training target.
- As a learner, I want a base-model baseline so that I can attribute changes to
  SFT instead of prompting.
- As a learner, I want LoRA training so that I can finish the loop on modest
  local hardware.
- As a learner, I want objective reports so that I can compare models and inspect
  failures.

## Functional requirements

- FR-001: The project must deterministically generate 60 training, 15 validation,
  and 15 test examples in conversational JSONL format.
- FR-002: The project must reject malformed rows, invalid labels, duplicate IDs,
  cross-split duplicate prompts, and incomplete label coverage before training.
- FR-003: The evaluator must run against the base model without an adapter and
  save per-example predictions plus aggregate metrics.
- FR-004: The trainer must apply LoRA to `Qwen/Qwen3-0.6B-Base`, compute loss only
  on assistant tokens, validate each epoch, and save a reusable adapter.
- FR-005: The evaluator must load the trained adapter and compute JSON validity,
  schema validity, intent accuracy, urgency accuracy, and joint accuracy.
- FR-006: The inference command must load the base model plus optional adapter
  and return raw output, parsed output, and schema validity for one message.
- FR-007: The README must explain every lifecycle stage, expected artifacts,
  metrics, resource constraints, and suggested experiments.

## Non-functional requirements

- NFR-001: CPU-only data checks, unit tests, and linting must finish without
  downloading a model.
- NFR-002: The default experiment must remain teaching-scale: a 0.6B model,
  256-token maximum length, LoRA, and a small deterministic dataset.
- NFR-003: Checkpoints, reports, environments, caches, credentials, personal
  identifiers, and machine metadata must not be committed.
- NFR-004: Dependency versions must be reproducible through `uv.lock`.

## Acceptance criteria

- AC-001: `make data` produces exactly 60/15/15 examples and `make check` passes.
- AC-002: `make test` passes all CPU-only data and metric tests.
- AC-003: `make lint` completes with no Ruff errors.
- AC-004: `make baseline` writes a baseline report containing all five documented
  metrics and individual predictions.
- AC-005: `make train` saves a LoRA adapter without merging or overwriting the
  base model.
- AC-006: `make evaluate` loads that adapter and writes a comparable test report.
- AC-007: `make infer` produces a parseable result object for a supplied message.

## Open questions

None for the implemented baseline.
