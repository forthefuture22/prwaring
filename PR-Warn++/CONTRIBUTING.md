# Contributing to PR-Warn++

> **Why this document exists.**
> This project used to track versions by copying files
> (`PR-Warn++_v3.1_...(1).docx`), by parallel folders (`3.0/`, `3.2/`) and by zip
> snapshots (`prwarn_plus.zip`). Those mechanisms are now retired. **Git owns the
> history; file names describe content, not version.**
> If you are about to write a version number into a filename, stop — make a
> commit or a tag instead.

---

## 0. Quick start

```bash
git switch main
git pull --rebase
git switch -c feat/cfm-ot-coupling

# ... work ...

git add -p                            # stage in reviewable chunks
git commit -m "feat(cfm): add minibatch OT coupling for A5"
git fetch origin
git rebase origin/main                # keep history linear
git push -u origin HEAD               # then open a Pull Request
```

After the PR is squash-merged:

```bash
git switch main
git pull --rebase
git branch -d feat/cfm-ot-coupling
```

---

## 1. Branch model

**Trunk-based development with short-lived branches** (the GitHub Flow model).
With one or two committers, Git Flow's `develop` / `release` / `hotfix`
machinery buys coordination we do not need and costs us merge conflicts.

```
main    ──●──────────●──────────────────●──────────●──►   always green, protected
           \        /                    \        /
feat/…      ●──●───┘                      ●──●──●─┘
          (≤ 3 days)                   (≤ 3 days)
```

The rules:

1. `main` is the **only** long-lived branch. It must always install, lint and pass tests.
2. Never commit directly to `main` — see §5.4 for the single documented exception.
3. Every change reaches `main` through a **Pull Request**.
4. Branch lifetime is measured in **days, not weeks**. Past three days, split the work.
5. **Rebase** onto `main`. Do not merge `main` into your feature branch.

---

## 2. Branch naming

`<type>/<short-slug>` — lowercase, hyphen-separated, no version numbers.

| Prefix | Use for | Example |
|---|---|---|
| `feat/` | new capability | `feat/adaptive-graph-builder` |
| `fix/` | bug fix | `fix/conformal-coverage-gap` |
| `exp/` | experiment run or config sweep | `exp/a5-seed-sweep` |
| `refactor/` | no behaviour change | `refactor/cache-contract` |
| `docs/` | documentation only | `docs/refresh-runbook` |
| `chore/` | tooling, dependencies, CI | `chore/pin-torch-version` |

Avoid `mybranch`, `test`, `final`, `final2`, `v3.2-new`.

---

## 3. Commit messages — Conventional Commits

```
<type>(<scope>)!: <subject>      ← ≤ 72 chars, imperative mood, no trailing period

<body: why this change is needed, not a restatement of the diff>

<footer: Closes #12 / BREAKING CHANGE: ...>
```

**Types**

| Type | Meaning |
|---|---|
| `feat` | new capability |
| `fix` | bug fix |
| `docs` | documentation only |
| `style` | formatting, no logic change |
| `refactor` | restructuring, no behaviour change |
| `perf` | performance improvement |
| `test` | tests only |
| `build` | packaging, dependencies |
| `ci` | CI configuration |
| `chore` | tooling and housekeeping |
| `exp` | experiment config, sweep or result |

**Scopes for this project**

`data` · `preprocess` · `graphs` · `models` · `cfm` · `calib` · `eval` · `risk` ·
`baselines` · `cli` · `experiments` · `docs` · `deps` · `repo`

**Good**

```
feat(cfm): add minibatch OT coupling for A5
fix(calib): correct conformal quantile on odd sample counts
exp(risk): add seed sweep config for the CVaR proxy
docs(repo): document the release tagging procedure
refactor(data): extract rigid-grid builder from preprocess CLI
```

**Rejected**

```
update                     ← says nothing
fixed bug                  ← which bug, where?
wip                        ← do not commit work in progress to a shared branch
feat: a lot of changes     ← split it
CFM stuff (again)          ← no type, no scope, conversational
```

The header is enforced by the `commit-msg` hook, so a malformed message will be
rejected before it is created.

**Body — explain the *why*.** The diff already shows the *what*. Record the
reasoning, the trade-off you rejected, and the experiment config that produced
the numbers. This is what future-you reads in six months.

---

## 4. Pull requests

Open a PR **as soon as the branch is pushed**, not when it is finished. Mark it
draft while you work. The PR is a discussion record, and for a small team it is
also the checkpoint where you review your own diff with fresh eyes.

Fill in every section of the template. A PR description that only says "fixes
stuff" is worse than no description.

**Size.** Aim for **≤ 400 changed lines** excluding generated files. Reviewers
(including you, tomorrow) cannot meaningfully review a 3 000-line PR. Split by
concern: preprocessing change in one PR, model change in the next.

---

## 5. Review and merge

### 5.1 What a reviewer looks for

1. **Correctness** — does it do what the description claims?
2. **Leakage** — does anything touch validation or test data outside the
   documented split? This is the highest-risk class of bug in this codebase.
3. **Reproducibility** — is the run reproducible from the committed config plus seed?
4. **Tests** — is new behaviour covered?
5. **Scope** — is anything unrelated riding along?

### 5.2 Two-person team

* Author writes the PR, the other person reviews.
* The reviewer approves, then the **author** merges. The author owns the cleanup.

### 5.3 One-person team

You are still required to open the PR. Before merging, self-review with:

```bash
git diff main...HEAD            # read your own diff, top to bottom
git log main..HEAD --oneline    # do the commits read as a coherent story?
```

Then wait until the next working session before merging anything
non-trivial. A night of distance catches more than an extra cup of coffee.

### 5.4 Merge strategy

* **Squash and merge** is the default — one branch becomes one commit on `main`.
* The squash commit message must still follow Conventional Commits.
* Rebase-merge only when the branch is a sequence of genuinely independent,
  individually valuable commits.
* **Never** create a merge commit on `main`.

### 5.5 The escape hatch

Direct pushes to `main` are reserved for:

* fixing a broken build on `main` (then open a retroactive PR describing it), and
* the very first bootstrap commit.

Anything else goes through a PR. If you find yourself using the escape hatch
weekly, the review process is too heavy — tell the team and we will lighten it.

---

## 6. Releases and tags

Versions live in **tags**, never in filenames or folders. We follow SemVer:
`MAJOR.MINOR.PATCH`.

| Bump | When |
|---|---|
| `MAJOR` | a published result is invalidated, or a config format breaks |
| `MINOR` | a new experiment, model or baseline is added |
| `PATCH` | bug fix that does not change results |

```bash
git switch main
git pull --rebase
git tag -a v3.2.0 -m "PR-Warn++ v3.2 — joint probabilistic forecasting baseline"
git push origin v3.2.0
```

Then write release notes on GitHub, and attach large artifacts (benchmark
result archives, trained checkpoints, compiled PDFs) as **Release assets** —
this is exactly what replaces the old `prwarn_benchmark_results.zip` habit.

The current codebase is tagged **v3.2.0**. Note that `pyproject.toml` still
reads `version = "0.1.0"`; align it at the next release.

---

## 7. What never goes into Git

| Never commit | Why | Where it goes instead |
|---|---|---|
| `__pycache__/`, `*.pyc` | regenerated on every run | ignored |
| `.venv/`, `venv/` | machine-specific | `pyproject.toml` + lockfile |
| `data/raw/`, `data/interim/`, `data/processed/` | large, licensed, regenerable | a documented download script |
| checkpoints (`*.pt`, `*.ckpt`) | hundreds of MB | Release assets or object storage |
| `outputs/`, `output/`, `tmp/` | generated | regenerated by the pipeline |
| `.env`, keys, tokens | **security incident** | secret manager; commit `.env.example` |
| zip snapshots | unreadable history | **git tags + Release assets** |
| `..._final_v2(1).docx` | duplicate-by-copy | one file, tracked in Git |

The `pre-commit` hook enforces the first four mechanically.

---

## 8. Repository layout

```
PR-Warn++
├── src/prwarn/          # installable package — the only place logic lives
│   ├── data/ preprocess/ graphs/ models/ cfm/ calibration/ eval/ risk/
│   ├── baselines/ experiments/ cli/
├── tests/               # mirrors src/ structure; pytest
├── configs/             # experiment YAML — tracked, versioned
├── docs/                # prose, figures, paper sources
├── .githooks/           # commit-msg, pre-commit
├── .github/             # PR and issue templates, CI workflows
└── CONTRIBUTING.md      # this file
```

Conventions:

* Every source file lives under `src/prwarn/`. Do not add top-level scripts.
* Every experiment is a YAML file in `configs/`. Never hardcode parameters in Python.
* Every seed is explicit in the config. Never rely on an implicit default seed.
* Tests mirror the module they cover: `src/prwarn/eval/x.py` → `tests/test_x.py`.

**Cleanup still pending** (tracked, not yet done): the legacy `3.0/` and `3.2/`
folders, the `(1)` duplicate documents, and the two root-level zip archives.
Once `v3.2.0` is tagged, these are redundant — they can be removed in a
dedicated `chore(repo): retire legacy snapshot folders` commit.

---

## 9. Tooling

Enabled in this repository:

```bash
git config core.hooksPath .githooks      # commit-msg + pre-commit guards
git config commit.template .gitmessage   # guided commit messages
```

| File | Purpose |
|---|---|
| `.gitattributes` | LF everywhere; binary assets never diffed |
| `.editorconfig` | consistent indentation across editors |
| `.gitignore` | keeps regenerable files out of history |
| `.githooks/commit-msg` | rejects non-Conventional commit messages |
| `.githooks/pre-commit` | blocks caches, oversized files, credentials; lints Python |

Hooks are per-clone configuration. **Every teammate runs the two `git config`
commands above once after cloning.**

Still recommended, not yet added: a GitHub Actions workflow running
`ruff check` + `pytest` on every PR.

---

## 10. Migrating from the old habits

| Old habit | Replace with |
|---|---|
| `PR-Warn++_v3.1_...(1).docx` | one tracked file; history in `git log -p` |
| `3.0/`, `3.2/` folders | tags `v3.0.0`, `v3.2.0` + `git switch -c` |
| `prwarn_plus.zip` | tag, or a Release asset |
| `backup_2026-09-01/` | a commit; branches are free |
| "who changed this line?" | `git log -L 40,60:src/prwarn/eval/x.py` |
| "which version produced Fig. 1?" | the commit hash recorded in the run config |

---

## 11. Cheat sheet

```bash
# start work
git switch main && git pull --rebase
git switch -c feat/short-slug

# inspect before committing
git status
git diff
git add -p

# commit
git commit                      # template opens, hook validates the header

# keep up to date
git fetch origin && git rebase origin/main

# publish
git push -u origin HEAD

# undo the last commit, keep the changes staged
git reset --soft HEAD~1

# find which commit broke something
git bisect start && git bisect bad && git bisect good v3.2.0

# who last touched these lines, and why
git blame -L 40,60 src/prwarn/eval/metrics.py

# recover something you thought was lost
git reflog
```
