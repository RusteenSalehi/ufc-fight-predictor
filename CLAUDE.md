# CLAUDE.md

Instructions for Claude Code working in this repository. Read this file and everything in `docs/` before starting any task.

## What this project is

A public web app that predicts upcoming UFC fights with a machine learning model, explains each prediction, tracks the model's real track record, and includes a grounded AI analyst. It is a portfolio project. Credibility matters more than flash.

Key documents:
- `docs/SPEC.md`: what V1 includes and what is deferred
- `docs/ARCHITECTURE.md`: data flow, schema, storage, AWS design, cost limits
- `docs/ML_REQUIREMENTS.md`: leakage rules, features, evaluation protocol (non-negotiable)
- `docs/DECISIONS.md`: log of design decisions; append to it (see below)
- `docs/ROADMAP.md`: milestones and their order
- `docs/MODEL_REPORT.md`: selection, ablation, test-period and backtest results (generated block; see below)

## Environment

- WSL2 Ubuntu on an ARM64 (aarch64) laptop. The repo lives in the Linux home directory. Never move work to `/mnt/c`.
- Python 3.12, managed by `uv`. Run everything through `uv run` (for example `uv run pytest`, `uv run python -m ufc.ingest load`).
- Add dependencies with `uv add` and state why in your summary. Check aarch64 Linux support before adding anything with compiled code.

## Data sources

- **Completed fights:** the Greco1899/scrape_ufc_stats CSV snapshot, pinned to an upstream commit and downloaded into `data/external/greco/` by `uv run python -m ufc.ingest fetch`. `uv run python -m ufc.ingest load` validates it and loads `data/ufc.db`. Everything in `src/ufc/ingest/` exists to serve this (DECISIONS D13 to D22).
- **Upcoming cards (planned, M5):** English Wikipedia through the MediaWiki API (DECISIONS D16). Treat what it returns as untrusted input.
- **ufcstats.com is never requested.** UFCStats URLs inside the dataset are parsed for their IDs only.

## Repository layout

```
src/ufc/
  ingest/     snapshot download, parsing, name-to-ID resolution, validation gate, CLI
  db/         schema.sql and idempotent loaders (SQLite)
  features/   as-of feature computation, Elo, opponent adjustment, matchup features
  models/     splits with a test-period lock, baselines, XGBoost and LightGBM, calibration,
              selection, evaluation, walk-forward backtest, model registry, report, CLI
  predict/    upcoming-card predictions and artifact publishing (later)
  analyst/    AI analyst backend (later)
scripts/      one-off and operational scripts (fixture generator, reports)
tests/        pytest; synthetic dataset fixtures in tests/fixtures/greco/, saved
              Wikipedia API responses in tests/fixtures/wikipedia/
web/          frontend (later, React + TypeScript + Vite)
infra/        AWS CDK app in Python (later)
data/         gitignored: external/greco snapshot, ufc.db, features/, model artifacts
```

Do not create `web/`, `infra/`, `predict/`, or `analyst/` until the roadmap reaches them.

## Models

- `uv run python -m ufc.models train` selects on validation and saves a candidate under `data/models/`. It also rewrites `data/features/` with the feature table it chose.
- `uv run python -m ufc.models evaluate` is the only command that scores a candidate on test-period rows. Each run is recorded in `data/models/test_evaluations.jsonl`. Do not run it to compare ideas; change something, select on validation, and evaluate a finalized candidate once.
- `uv run python -m ufc.models.diagnostics` is analysis only (DECISIONS D45). It also reads test-period rows, records each run in `data/models/test_diagnostics.jsonl`, and writes `data/reports/diagnostics/`. Nothing it reports may be used to change a selection decision.
- `uv run python -m ufc.models report` rewrites the generated block of `docs/MODEL_REPORT.md`. Numbers in that file's hand-written part must come from the block.
- `data/models/approved.json` is edited by the user only. No code and no session writes it.

## Hard rules

1. **No data leakage.** Follow `docs/ML_REQUIREMENTS.md` exactly. Any feature for a bout on date D uses only bouts on events strictly before D. Fighter-profile career stats (SLpM, Str. Acc., TD Def., and similar) are never used for training. If a requested change would risk leakage, stop and explain.
2. **No invented numbers.** Never write metrics, accuracy figures, or results into docs, READMEs, or the UI unless they were produced by code in this repo and can be reproduced with a documented command.
3. **Never commit data or secrets.** Nothing under `data/`, no `.env`, no API keys, no model binaries. The pre-commit secret scan must pass. No rows from the dataset, not even excerpts, in tests or docs: test fixtures are synthetic (`scripts/make_synthetic_fixtures.py`), and a test that needs real rows reads them from `data/external/greco/` and skips when absent (DECISIONS D15).
4. **No AWS changes without explicit approval.** Do not run `cdk deploy`, create resources, or change IAM unless the user approves the specific action in this session after seeing a cost estimate.
5. **No code requests ufcstats.com, and the bot challenge is never worked around.** UFCStats serves a JavaScript proof-of-work challenge on every page (DECISIONS D13). Do not write code that fetches from ufcstats.com, solve or bypass the challenge, drive a headless browser at it, or spoof a browser User-Agent. If a task seems to need UFCStats pages, stop and ask. Network code that does exist (the snapshot download, later the Wikipedia client) uses timeouts on every request, retries with backoff, and a descriptive User-Agent, and no fixture or test touches the network.
6. **Tests must pass** before you report a task as done. If a test fails, report it honestly; do not weaken or delete tests to make them pass.

## How to work

- Before writing code, give a short plan: what you will build, files you will touch, and assumptions. Then proceed unless something is ambiguous in a way that affects the database schema or the leakage rules; in that case ask.
- Work in small increments. Commit after each logical step with a clear message (imperative mood, under 72 characters for the subject). Do not push; the user reviews and pushes.
- Prefer pure functions for parsing and feature logic so they are easy to test.
- Use type hints and dataclasses (or pydantic where validation matters). Use `logging`, not `print`, in library code.
- Ruff handles lint and format; run `uv run ruff check --fix` and `uv run ruff format` before committing.
- When you make a non-obvious design choice, append an entry to `docs/DECISIONS.md` in the existing format: the decision, alternatives considered, and why. Keep entries short. The user will use this file to explain the project in interviews, so write it so a person can defend the choice.
- End every session with a summary: what was done, test status, commits made, open issues, and the suggested next step.
