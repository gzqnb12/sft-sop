# 002 Codex-native spec-driven development — implementation plan

Status: Implemented

## Approach

Use Codex's supported repository guidance surface, `AGENTS.md`, for concise and
automatically loaded rules. Store detailed, versioned SDD artifacts in `specs/`
so the root guidance remains practical. Capture the existing code as specification
001, and capture this initialization itself as specification 002.

Add a standard-library validator in the existing Python package. It discovers
numbered specification directories, checks their three artifacts, and enforces
basic requirement-to-task traceability. Expose it through the same Makefile and
package entry-point patterns as the existing data validator.

## Affected files

- `AGENTS.md`: persistent repository instructions loaded by Codex.
- `specs/README.md`: detailed SDD lifecycle and conventions.
- `specs/_template/`: reusable artifact templates.
- `specs/001-sft-learning-sop/`: brownfield baseline contract.
- `specs/002-codex-native-sdd/`: contract for this initialization.
- `src/sft_sop/check_sdd.py`: mechanical structure and traceability checks.
- `tests/test_sdd.py`: positive and negative validator tests.
- `Makefile`: `sdd-check` target.
- `pyproject.toml`: installed `sft-check-sdd` command.
- `README.md`: learner-facing explanation and links.

## Decisions

- Use Codex-native `AGENTS.md` because the request explicitly targets `/init`.
- Do not install a third-party SDD CLI when no such framework is selected.
- Keep specifications as Markdown so they remain reviewable in GitHub.
- Validate a small set of high-value invariants rather than inventing a complex
  schema or parser.
- Treat the current implementation as a brownfield baseline, not as a new feature.

## Risks and mitigations

- Risk: Process documentation drifts from actual behavior.
  - Mitigation: Keep the baseline specification versioned and require updates in
    the definition of done.
- Risk: SDD becomes ceremony for trivial changes.
  - Mitigation: Allow wording-only and mechanical edits to update the closest
    existing specification instead of creating a new directory.
- Risk: Markdown conventions are ignored.
  - Mitigation: Validate structure, status, IDs, traceability, and completed tasks.
- Risk: Agent guidance grows too large.
  - Mitigation: Keep root rules concise and link to `specs/README.md`.

## Verification

- `make sdd-check`: FR-002, FR-003, FR-004, FR-005, AC-002.
- `make test`: FR-004, AC-003, AC-004.
- `make check`: regression coverage for the existing data workflow, AC-004.
- `make lint`: code-quality regression coverage, AC-004.
- `sft-check-sdd`: installed CLI smoke test, FR-005.
- Manual file review: FR-001, AC-001, AC-005.
