# SDD workflow

`specs/` is the source of truth for what this repository is intended to do.
Code implements an accepted specification; code is not the specification.

## Lifecycle

```text
Spec → Plan → Tasks → Implement → Verify → Update spec if reality changed
```

1. **Spec** defines the problem, goals, requirements, constraints, and measurable
   acceptance criteria without prescribing implementation details.
2. **Plan** records architecture, file-level changes, risks, decisions, and the
   verification strategy.
3. **Tasks** decomposes the plan into ordered, checkable work. Each task traces
   to at least one `FR-NNN` requirement and one `AC-NNN` acceptance criterion.
4. **Implement** changes code and keeps task checkboxes accurate.
5. **Verify** runs the commands named by the plan and records intentional gaps.

## Directory convention

```text
specs/
├── README.md
├── _template/
│   ├── spec.md
│   ├── plan.md
│   └── tasks.md
└── NNN-short-feature-name/
    ├── spec.md
    ├── plan.md
    └── tasks.md
```

- Use the next unused three-digit number.
- Use lowercase kebab-case after the number.
- One directory represents one coherent behavior change.
- Update an existing specification when refining its existing contract.
- Create a new specification when introducing a new capability or changing a
  previously accepted contract.

## Status values

Use one status at the top of each artifact:

- `Draft`: still being clarified.
- `Accepted`: ready to implement.
- `In Progress`: implementation has started.
- `Implemented`: acceptance criteria have been verified.
- `Superseded`: replaced by another explicitly linked specification.

## IDs and traceability

- Functional requirements: `FR-001`, `FR-002`, …
- Non-functional requirements: `NFR-001`, `NFR-002`, …
- Acceptance criteria: `AC-001`, `AC-002`, …
- Tasks: `T-001`, `T-002`, …

Run `make sdd-check` before implementation and before committing.
