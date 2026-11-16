# Roadmap

Each milestone lists the recommended Claude Code model and effort. Milestones run in order; each ends with passing tests and small commits.

| # | Milestone | Model | Effort |
|---|---|---|---|
| M0 | Environment setup and verification | Sonnet 5.5 | Medium |
| M1 | UFCStats scraper, parsers, schema, tests | Fable 5.2 | High |
| M2 | Full historical scrape and data quality report | Sonnet 5.5 | Medium |
| M3 | As-of feature pipeline, Elo, leakage tests | Fable 5.2 | High |
| M4 | Baselines, training, calibration, evaluation report | Fable 5.2 (or Opus 5.5 after the Fable window) | High |
| M5 | Upcoming-card ingestion, card-change handling, batch prediction | Opus 5.5 | High |
| M6 | SHAP explanations and JSON artifact publishing with validation gate | Opus 5.5 | Medium |
| M7 | Frontend: upcoming card, matchup detail, performance, methodology | Sonnet 5.5 (Opus 5.5 for design pass) | Medium |
| M8 | AWS infrastructure with CDK (cost estimate and approval first) | Opus 5.5 | High |
| M9 | AI analyst: pre-generated analysis, grounded chat, caps, eval set | Opus 5.5 | High |
| M10 | CI with GitHub Actions (tests, lint, leakage and symmetry tests) | Sonnet 5.5 | Medium |
| M11 | V1.1: style archetypes | Opus 5.5 | High |
| M12 | V1.1: odds sourcing, market-blended model, betting simulation | Opus 5.5 | High |
| M13 | V1.1: news research in the analyst | Opus 5.5 | High |

M10 can move earlier at any time; it is small.

---

## M1 prompt (Fable 5.2, high effort)

```
Read CLAUDE.md and every file in docs/ before starting. This session is Milestone M1: the UFCStats scraper, parsers, database schema, and tests. Do not start M2 or later.

1. Check http://ufcstats.com/robots.txt and look for any terms-of-use page. Report what you find. If automated access to the pages we need is disallowed, stop and tell me before continuing.
2. Recon: fetch one page of each type (completed events list, upcoming events list, one completed event, one fight details page, one fighter page, plus one early-era event from the 1990s and one bout that ended in a draw or no contest). Save them to tests/fixtures/html/. Describe the structure and the ID scheme in your summary.
3. Implement src/ufc/scrape/client.py: descriptive User-Agent, max 1 request per second, timeouts, retries with exponential backoff, and a raw HTML cache in data/raw/ indexed in the raw_pages table. Pages for completed events and fights are fetched once and then served from cache; listing pages and upcoming events are re-fetchable.
4. Implement pure parser functions (HTML in, dataclasses out) for each page type. Fight details: method, round, time, referee, and per-fighter stats (totals plus significant strikes by target and position). Fighter pages: physical attributes only (height, reach, stance, date of birth); do not store the career aggregate stats.
5. Write src/ufc/db/schema.sql following docs/ARCHITECTURE.md (adjust columns to what recon shows; record any deviation in DECISIONS.md). Write idempotent upsert loaders keyed on UFCStats IDs.
6. Tests: parser tests against every fixture; edge cases (draw, no contest, DQ, missing reach, missing date of birth, early events with no per-fight stats, split decisions, title bouts, women's bouts); an idempotency test (loading the same data twice gives identical tables).
7. A CLI: `uv run python -m ufc.scrape completed --limit N` scrapes the N most recent completed events with all their fights and fighters. It must be resumable: rerunning after an interruption continues without refetching cached pages. Run it with --limit 3 and report row counts per table.
8. Add a `--all` mode for the full history, but do not run it.
9. Append decisions to docs/DECISIONS.md. Commit in small steps. Do not push.
```

## M2 prompt (Sonnet 5.5, medium effort)

```
Read CLAUDE.md and docs/ARCHITECTURE.md. This session is Milestone M2: run the full historical scrape and produce a data quality report.

1. Estimate the number of pages and the run time at 1 request per second, and tell me before starting.
2. Start `uv run python -m ufc.scrape completed --all` inside a tmux session (install tmux if missing) with logging to data/logs/, so it survives this session ending. Tell me the exact command to reattach and check progress.
3. While it runs, write scripts/data_quality.py that reports: events and bouts per year, results breakdown (win, draw, NC), method breakdown, bouts missing per-fight stats by year, fighters missing reach, height, or date of birth, duplicate checks, and any bouts whose fighters are missing from the fighters table.
4. When the scrape finishes (or now, on partial data if it is still running), run the report and write the results to docs/DATA_QUALITY.md. Only include numbers the script produced.
5. Commit the script and report. Do not push.
```

## M3 prompt (Fable 5.2, high effort)

```
Read CLAUDE.md and every file in docs/ before starting, especially ML_REQUIREMENTS.md, which is binding. This session is Milestone M3: the as-of feature pipeline. The historical scrape may still be running; develop and test against whatever data exists, and make the pipeline rerunnable.

1. Propose the module design for src/ufc/features/ first: how you will process bouts chronologically, how state per fighter is accumulated, and how same-event bouts are excluded. Then implement.
2. Implement an Elo rating system (as-of, updated after each event; parameters fit on the training period only) in features/elo.py.
3. Implement every feature family in ML_REQUIREMENTS.md section "Feature families" except style archetypes. Opponent-adjusted stats must respect the cutoff for the opponents' baselines too; explain your method in DECISIONS.md.
4. Missing stats (debuts, early events without stats) must be NaN with indicator flags, never zero.
5. Output: one row per bout in A-minus-B difference form, plus the bout date, weight class, debut flags, and label, written to data/features/features.parquet. Provide a function that builds the same features for a hypothetical upcoming bout given two fighter IDs and a date.
6. Required tests: the perturbation test, the same-event test, a hand-calculated spot check for one known fighter, and a test that building features for (A, B) and (B, A) gives exactly negated difference features.
7. Commit in small steps. Do not push. Summarize the final feature list with counts of missing values.
```

## M4 prompt (Fable 5.2 or Opus 5.5, high effort)

```
Read CLAUDE.md and every file in docs/ before starting; ML_REQUIREMENTS.md is binding. This session is Milestone M4: baselines, models, calibration, and the evaluation report.

1. Implement the splits exactly as specified. Assert in code that no test-period bout is used before final evaluation.
2. Baselines: Elo probability and higher-Elo accuracy; logistic regression.
3. Candidates: XGBoost and LightGBM trained on both orientations with recency sample weights. Tune on validation (log loss) with a modest search; choose the half-life on validation.
4. Calibrate the selected model on validation. Inference averages both orientations.
5. Symmetry test as specified.
6. Feature-family ablation on validation.
7. Evaluate the final candidate and all baselines once on the test period. Produce data/reports/eval/ with metrics JSON and plots (reliability diagrams, metrics by year, by confidence bucket, debut bouts separately), and a yearly walk-forward backtest.
8. Implement the model registry described in ML_REQUIREMENTS.md and save the model. Do not edit approved.json; I will promote it.
9. Write docs/MODEL_REPORT.md with only numbers produced by the code, including ablations that did not help. Append decisions. Commit in small steps. Do not push.
```

Prompts for M5 onward are written when the project reaches them, based on what M1 to M4 produced.
