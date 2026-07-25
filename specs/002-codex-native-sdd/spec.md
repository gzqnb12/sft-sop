# 002 Codex-native spec-driven development

Status: Implemented

## Context

The repository has a working educational implementation but no durable contract
for how future AI-assisted changes should move from intent to verified code.
Codex `/init` provides a repository-level `AGENTS.md` scaffold, but a complete
lightweight SDD loop also needs versioned specifications, plans, tasks, and
traceability checks.

## Goals

- Make the repository's development process spec-driven by default.
- Keep the workflow understandable without installing another SDD framework.
- Capture the existing SFT project as a brownfield baseline specification.
- Mechanically detect missing artifacts and broken requirement traceability.

## Non-goals

- Installing or emulating GitHub Spec Kit, OpenSpec, or another external CLI.
- Generating implementation code from specifications automatically.
- Requiring GPU training for ordinary specification validation.
- Adding heavyweight policy or CI infrastructure.

## User stories

- As a maintainer, I want Codex to load the SDD rules automatically so that new
  sessions follow the same workflow.
- As a learner, I want templates and a real baseline example so that I know how
  to write the next specification.
- As a reviewer, I want mechanical traceability checks so that requirements are
  not silently omitted from implementation tasks.

## Functional requirements

- FR-001: The repository must contain a concise root `AGENTS.md` with its mission,
  source map, SDD lifecycle, commands, constraints, and definition of done.
- FR-002: The `specs/` directory must document naming, statuses, ID conventions,
  lifecycle stages, and reusable `spec.md`, `plan.md`, and `tasks.md` templates.
- FR-003: The existing SFT learning behavior must be represented by a numbered,
  implemented baseline specification with requirements, acceptance criteria,
  implementation decisions, verification, and completed tasks.
- FR-004: A CPU-only validator must reject missing artifacts, invalid statuses,
  absent IDs, duplicate task IDs, untraced requirements or acceptance criteria,
  missing verification plans, and open tasks on implemented specifications.
- FR-005: The validator must be available through `make sdd-check` and an installed
  `sft-check-sdd` command, and the README must explain the SDD workflow.

## Non-functional requirements

- NFR-001: SDD validation must use only the Python standard library.
- NFR-002: SDD validation must not download models or access a GPU.
- NFR-003: Instructions must remain short enough for Codex project guidance and
  link to detailed artifacts instead of duplicating them.
- NFR-004: The workflow must remain usable by humans and tools other than Codex.

## Acceptance criteria

- AC-001: A new Codex session opened at the Git root discovers `AGENTS.md`.
- AC-002: `make sdd-check` validates every numbered specification successfully.
- AC-003: Automated tests prove the validator reports missing FR/AC task links.
- AC-004: `make check`, `make test`, and `make lint` continue to pass without GPU
  or model downloads.
- AC-005: The README links to `AGENTS.md`, `specs/README.md`, templates, and the
  SDD validation command.

## Open questions

None for the lightweight Codex-native workflow.
