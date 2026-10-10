## What changed

<!-- One short paragraph. What is different in the codebase after this PR? -->

## Why

<!-- The motivation: the bug, the failed experiment, the reviewer comment, the issue. -->

Closes #

## Type of change

- [ ] `feat` — new capability
- [ ] `fix` — bug fix
- [ ] `exp` — experiment, config or result change
- [ ] `refactor` — no behaviour change
- [ ] `docs` — documentation only
- [ ] `chore` — tooling, dependencies, CI

## How it was verified

<!-- Paste the exact commands you ran and what you observed. -->

```bash
pytest -q
```

## Checklist

- [ ] Branch is rebased on the latest `main` (rebased, not merged)
- [ ] Commit messages follow Conventional Commits
- [ ] Tests added or updated for any behaviour change
- [ ] `ruff check` passes
- [ ] No data files, checkpoints, `__pycache__`, zips or secrets included
- [ ] If a config changed: the run is reproducible from the committed YAML + seed
- [ ] Docs updated when a public interface or CLI flag changed

## Reviewer notes

<!-- Anything you specifically want a second pair of eyes on. Delete if not needed. -->
