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

## Environment

- WSL2 Ubuntu on an ARM64 (aarch64) laptop. The repo lives in the Linux home directory. Never move work to `/mnt/c`.
- Python 3.12, managed by `uv`. Run everything through `uv run` (for example `uv run pytest`, `uv run python -m ufc.scrape`).
- Add dependencies with `uv add` and state why in your summary. Check aarch64 Linux support before adding anything with compiled code.

## Repository layout

```
src/ufc/
  scrape/     HTTP client (rate limit, cache, retries) and pure HTML parsers
  db/         schema.sql and idempotent loaders (SQLite)
  features/   as-of feature computation, Elo, opponent adjustment, matchup features
  models/     baselines, training, calibration, evaluation, model registry
  predict/    upcoming-card predictions and artifact publishing (later)
  analyst/    AI analyst backend (later)
scripts/      one-off and operational scripts
tests/        pytest; HTML fixtures in tests/fixtures/html/
web/          frontend (later, React + TypeScript + Vite)
infra/        AWS CDK app in Python (later)
data/         gitignored: raw HTML cache, ufc.db, model artifacts
```

Do not create `web/`, `infra/`, `predict/`, or `analyst/` until the roadmap reaches them.

## Hard rules

1. **No data leakage.** Follow `docs/ML_REQUIREMENTS.md` exactly. Any feature for a bout on date D uses only bouts on events strictly before D. Fighter-profile career stats (SLpM, Str. Acc., TD Def., and similar) are never used for training. If a requested change would risk leakage, stop and explain.
2. **No invented numbers.** Never write metrics, accuracy figures, or results into docs, READMEs, or the UI unless they were produced by code in this repo and can be reproduced with a documented command.
3. **Never commit data or secrets.** Nothing under `data/`, no `.env`, no API keys, no model binaries. The pre-commit secret scan must pass.
4. **No AWS changes without explicit approval.** Do not run `cdk deploy`, create resources, or change IAM unless the user approves the specific action in this session after seeing a cost estimate.
5. **Polite scraping.** At most 1 request per second to ufcstats.com, a descriptive User-Agent, retries with exponential backoff, timeouts on every request, and a raw HTML cache so completed-event pages are never fetched twice.
6. **Tests must pass** before you report a task as done. If a test fails, report it honestly; do not weaken or delete tests to make them pass.

## How to work

- Before writing code, give a short plan: what you will build, files you will touch, and assumptions. Then proceed unless something is ambiguous in a way that affects the database schema or the leakage rules; in that case ask.
- Work in small increments. Commit after each logical step with a clear message (imperative mood, under 72 characters for the subject). Do not push; the user reviews and pushes.
- Prefer pure functions for parsing and feature logic so they are easy to test.
- Use type hints and dataclasses (or pydantic where validation matters). Use `logging`, not `print`, in library code.
- Ruff handles lint and format; run `uv run ruff check --fix` and `uv run ruff format` before committing.
- When you make a non-obvious design choice, append an entry to `docs/DECISIONS.md` in the existing format: the decision, alternatives considered, and why. Keep entries short. The user will use this file to explain the project in interviews, so write it so a person can defend the choice.
- End every session with a summary: what was done, test status, commits made, open issues, and the suggested next step.
