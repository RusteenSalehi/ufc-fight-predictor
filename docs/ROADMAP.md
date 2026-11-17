# Roadmap

Each milestone lists the recommended Claude Code model and effort. Milestones run in order; each ends with passing tests and small commits.

| # | Milestone | Model | Effort |
|---|---|---|---|
| M0 | Environment setup and verification | Sonnet 5.5 | Medium |
| M1 | Dataset ingestion (published UFCStats-derived CSVs), schema, validation, tests | Fable 5.2 | High |
| M2 | Data quality report on the loaded dataset | Sonnet 5.5 | Medium |
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

## M1 (Fable 5.2, high effort): done, re-scoped

M1 was planned as a UFCStats scraper. On the first request UFCStats returned a JavaScript proof-of-work bot challenge on every page, and the project does not work around it (DECISIONS D13). M1 was re-scoped to ingest a published dataset instead (D14). What it delivered:

- `uv run python -m ufc.ingest fetch`: downloads the six CSVs from `Greco1899/scrape_ufc_stats`, pinned to a commit, into `data/external/greco/`.
- `uv run python -m ufc.ingest load`: parses, joins, validates and loads them into `data/ufc.db`. A validation gate stops a bad snapshot from overwriting good data. Loads are idempotent.
- `src/ufc/db/schema.sql` and upsert loaders keyed on UFCStats IDs. Per-round stats are stored and per-fight totals derived.
- Fighter names on bouts are resolved to fighter IDs by exact rules and an explicit override file (D17).
- Tests run against a small synthetic fixture (`tests/fixtures/greco/`, written by `scripts/make_synthetic_fixtures.py`, no real rows) covering draws, no contests, DQs, missing attributes, early events without stats, split decisions, title bouts, women's bouts, and the dataset's own defects.
- Saved Wikipedia API responses (`tests/fixtures/wikipedia/`) confirming that upcoming cards can be parsed in M5.

## M3 (Fable 5.2, high effort): done

- `uv run python -m ufc.features build`: fits Elo parameters and league baselines on bouts through 2022-12-31, builds one row per completed bout, and writes `data/features/features.parquet` and `features_meta.json`. Rerunning replaces both with identical content.
- `uv run python -m ufc.features summary`: lists every feature by family with its missing-value counts.
- `src/ufc/features/`: as-of store, Elo, per-fighter features for families 1 to 12 (style archetypes excluded), opponent adjustment, matchup assembly. `build_matchup` gives the same features for a hypothetical bout from two fighter IDs and a date.
- Tests: perturbation, same-event, symmetry, and a hand-calculated spot check on four real bouts (read from `data/external/greco/` at test time; skipped if absent). Design choices are in DECISIONS D23 to D29.

## M4 (Fable 5.2, high effort): done

- M3 review changes applied first: own Elo is family 13 (D31), one training-fit Elo for features and baseline (D32), per-minute rates shrunk toward weight-class means with the strength chosen on validation (D33), swap-invariant level features as candidates (D34).
- `uv run python -m ufc.models train`: selection on validation in fixed stages, then the candidate is saved under `data/models/`. `evaluate`: the one test-period evaluation plus a yearly walk-forward backtest. `report`: the generated block of `docs/MODEL_REPORT.md`.
- `src/ufc/models/`: splits with a test-period lock, exact-symmetry inference, Elo and logistic baselines, XGBoost and LightGBM, calibration, ablation, registry, plots. Design choices are in DECISIONS D35 to D44.
- A leak through missing profile attributes was found on the training period and removed before the test period was opened (D44).
- Results, including the choices validation could not separate and the families that did not help, are in `docs/MODEL_REPORT.md`. The candidate is not approved; `data/models/approved.json` is written by hand.
- Open for M5: bouts with an attribute gap (mostly recent debutants) need the "limited data" label; prediction must build features with the settings and fitted parameters stored beside the model (`feature_pipeline.json`); the recipe has been refitted on data through 2026-11-14 as `xgboost-ffc3ca8c-ee729619` (D47; not approved), and M5 predicts with whichever version `approved.json` names.

## M4.5 (Opus 5.5, high effort): model revision, analysis done, adoption open

- Triggered by age dominating SHAP when an older, higher-rated fighter meets a much younger one. `uv run python -m ufc.models.revision run` (slices and variants) and `... calibration-window`; results in `docs/MODEL_REPORT.md`, "Model revision M4.5"; decisions D54 to D57.
- Finding: the model does not underrate elite veterans; it slightly overrates older fighters. Of four pre-registered variants, `age_shape` passes the rule (weakly) and is the recommended recipe; nothing has been refitted or approved.
- Open, the user's call: adopt `age_shape` (save a candidate with the fixed recipe, refit, approve) or keep the current recipe; the calibration window for the next refit (rolling 2023 to 2026 recommended).

## M5 (Opus 5.5, high effort): done

- `uv run python -m ufc.predict refresh`: the scheduled events and their cards from English Wikipedia (MediaWiki API, two to four requests, cached), fighter resolution with the ingest's rules, card changes applied whole or not at all, staleness against completed events the dataset lacks, missing attributes from fighters' infoboxes, predictions with the approved model, and a JSON dump in `data/publish/` behind the validation gate. Design choices are in DECISIONS D48 to D53.
- The first live run found five veterans whose names Wikipedia spells differently; they are overrides now, and a guard leaves such a bout unresolved when the fighter's article cites a UFCStats profile in the dataset (D49).
- Open for M6: SHAP factors (`shap_json` is empty), the S3 artifact layout and immutable archive snapshots, scoring live predictions once their events reach the dataset, and the "limited data" label on the site.

## M2 (Sonnet 5.5, medium effort): done

- `scripts/data_quality.py` reads `data/ufc.db` and writes the generated block of `docs/DATA_QUALITY.md`: attribute completeness by activity and debut date, freshness against today, per-year coverage, duplicates, `ingest_issues`, dropped-bout evidence and event gaps. Tested in `tests/test_data_quality.py`.
- Findings and the review of dropped bouts and name resolutions are in `docs/DATA_QUALITY.md`; no override was added (DECISIONS D30).

## M2 prompt (Sonnet 5.5, medium effort)

```
Read CLAUDE.md and docs/ARCHITECTURE.md. This session is Milestone M2: a data quality report on the loaded dataset. There is no long scrape any more; a full load takes seconds.

1. Run `uv run python -m ufc.ingest fetch` and `uv run python -m ufc.ingest load`. Report the upstream commit, the row counts per table, and every error and warning in the validation report.
2. Write scripts/data_quality.py that reads data/ufc.db and reports: events and bouts per year, results breakdown (win, draw, NC), method breakdown, bouts missing per-fight stats by year, rounds missing control time by year, fighters missing reach, height, stance, or date of birth (overall, and separately for fighters with a bout in the last two years), duplicate checks, the contents of ingest_issues grouped by check, and the gap in days between consecutive events (to spot missing events).
3. Review the dropped bouts and the name resolutions listed by `load --verbose`. For each dropped bout, say whether an entry in src/ufc/ingest/identity_overrides.csv could resolve it and what evidence in the data supports it. Do not add an override without evidence from the dataset itself.
4. Run the report and write the results to docs/DATA_QUALITY.md. Only include numbers the script produced, and name the upstream commit they came from.
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
6. Feature-family ablation on validation, over all twelve families in data/features/features_meta.json (family 12 is per-round pace and cardio).
7. Evaluate the final candidate and all baselines once on the test period. Produce data/reports/eval/ with metrics JSON and plots (reliability diagrams, metrics by year, by confidence bucket, debut bouts separately), and a yearly walk-forward backtest.
8. Implement the model registry described in ML_REQUIREMENTS.md and save the model. Do not edit approved.json; I will promote it.
9. Write docs/MODEL_REPORT.md with only numbers produced by the code, including ablations that did not help. Append decisions. Commit in small steps. Do not push.
```

Prompts for M5 onward are written when the project reaches them, based on what M1 to M4 produced.
