# Architecture

## Overview

```
published dataset on GitHub (CSV, derived from ufcstats.com)
    |  fetch, pinned to a commit (7 requests)
    v
snapshot (data/external/greco)  ->  transform + validation gate  ->  SQLite (ufc.db)
                                               |
                                               v
                                as-of feature builder
                                   |                 \
                                   v                  v
                      training + evaluation     upcoming-bout features
                      (local, manual)                 |
                                   |                  v
                         approved model  ----->  batch prediction + SHAP
                                                      |
                                                      v
                                     validation gate (publish or keep last good)
                                                      |
                                                      v
                                     S3 JSON artifacts  ->  CloudFront  ->  React site
                                                                               |
                                                     analyst API (Lambda)  <---+
                                                           |
                                                     Anthropic API (prepaid credits)
```

Core principle: predictions are precomputed in batch. Page views read static JSON. No always-on servers.

## Data source

### Completed fights: a published dataset, not a scraper

UFCStats.com serves every page behind a JavaScript proof-of-work "Checking your browser" challenge, and this project does not solve or work around it (see DECISIONS D13). Historical data therefore comes from the CSV files published at https://github.com/Greco1899/scrape_ufc_stats, which that project collects from UFCStats and refreshes daily. Terms-of-use caveats are in DECISIONS D15.

| File | Grain | Key it carries |
|---|---|---|
| `ufc_event_details.csv` | one row per event: name, date, location | event URL |
| `ufc_fight_details.csv` | one row per fight: event name, bout text | fight URL |
| `ufc_fight_results.csv` | one row per fight: outcome, weight class, method, round, time, format, referee | fight URL |
| `ufc_fight_stats.csv` | one row per fighter per round: totals and significant strikes by target and position | none (event name, bout text, fighter name) |
| `ufc_fighter_details.csv` | one row per fighter: current name, nickname | fighter URL |
| `ufc_fighter_tott.csv` | one row per fighter: height, weight, reach, stance, date of birth | fighter URL |

Stable IDs are the 16-character hex IDs in those UFCStats URLs. URLs are parsed for their IDs and never requested. Names are display data, with one exception forced by the dataset: a fight row names its two fighters as text only, so fighter IDs on bouts are resolved from names (below).

The dataset holds no career aggregates, and none are stored. Only physical attributes (height, reach, stance, date of birth) are taken from fighter profiles.

### Ingestion (`src/ufc/ingest/`)

```
uv run python -m ufc.ingest fetch     # download the snapshot, pinned to a commit, with SHA-256 per file
uv run python -m ufc.ingest load      # validate, then load data/ufc.db
```

1. **Fetch** resolves the upstream branch to a commit, downloads the six files to a staging directory, checks their columns, and only then replaces the previous snapshot. `SOURCE.json` records the commit.
2. **Transform** (pure functions) parses every cell and performs the joins the files leave to text:
   - fight to event, by event name;
   - stats to fight, by event name and bout text;
   - fighter name to fighter ID, by rule: an explicit override in `identity_overrides.csv`; else a name belonging to exactly one profile; else, for a shared name, the one profile whose listed weight fits the bout's weight class. No fuzzy matching.
   - Anything that cannot be joined with certainty is dropped and reported, never guessed.
3. **Validate** checks row counts, duplicate IDs, references, fighters without bouts, date and age sanity, results breakdown, and stat arithmetic.
4. **Gate**: if validation finds structural errors or more than 1% of fights were dropped, nothing is written and the database keeps its last good contents.
5. **Load** upserts everything in one transaction.

Known limits of the dataset, all surfaced by the validation report rather than hidden:
- It starts at UFC 2 (1994-03-11). UFC 1 is absent.
- Fighter profiles are captured when a fighter first appears and only rarely refreshed upstream (once, in autumn 2025, as far as its history from December 2024 shows; DECISIONS D46). Height and reach are often empty for recent debutants, and names can go stale.
- A few fights cannot be placed or attributed and are dropped (3 of 9,003 in the snapshot at commit `96adace`; run `uv run python -m ufc.ingest load --dry-run` to list them).

### Upcoming cards and predictions (`src/ufc/predict/`, M5)

```
uv run python -m ufc.predict refresh   # cards from Wikipedia, predictions, data/publish/predictions.json
```

The dataset covers completed events only. Upcoming cards come from English Wikipedia through the MediaWiki API (D16, D48); ufcstats.com is never requested. One refresh:

1. **Schedule** (`wikipedia.py`, `cards.py`): "List of UFC events", read with `query`+`revisions`. Its "Scheduled events" table gives each event's page and date; its "Past events" table is used for staleness. If it cannot be read, the refresh stops and nothing changes.
2. **Cards**: every scheduled event page in one request; bouts from `{{MMAevent bout|class|fighter|vs.|fighter|...}}` templates in page order, with the card segment, weight class mapped to the dataset's labels, linked article and title-bout note. "Announced bouts" lists are not predicted. Wikipedia text is untrusted: anything of unexpected shape is a parse error, not a guess.
3. **Resolution** (`resolve.py`, D49): the ingest's resolver and overrides, plus `wikipedia_overrides.csv`. Unknown names become provisional fighters (presumed debutants); ambiguous names, placeholders and veterans spelled differently (detected when their article cites a UFCStats profile in the dataset) make the bout unresolved.
4. **Card changes** (`cards_db.py`, D50): each card is applied in one transaction (new pairings added, missing ones cancelled with their predictions superseded) or, if unreadable or implausible, not at all.
5. **Staleness** (`staleness.py`, D52): completed events after the dataset's latest; bouts with a fighter on one of them are labelled "stale stats".
6. **Backfill** (`backfill.py`, `infobox.py`, D51): missing height, reach and date of birth from the linked article's infobox, for predictions only.
7. **Predictions** (`predict.py`, D53): the approved model, the feature settings and fitted parameters stored with it, `build_matchup` at the event date. New rows only when inputs change.
8. **Publish** (`publish.py`): the validation gate, then a plain JSON dump of current predictions in `data/publish/predictions.json` (the site's artifact layout is M6).

Requests: two to four per refresh (list, event pages, provisional fighters' articles, articles for backfill), one at a time, spaced, cached; see D48. Test fixtures are saved API responses (`tests/fixtures/wikipedia/`, written by `scripts/save_wikipedia_fixtures.py`); no test touches the network.

## Database schema (SQLite)

The schema is `src/ufc/db/schema.sql`. Deviations from the original plan are explained in DECISIONS D20.

- **events**: event_id (PK, UFCStats ID), name, date, location, status (`completed` / `upcoming`), first_seen_at, last_seen_at
- **fighters**: fighter_id (PK), name, nickname, height_cm, reach_cm, stance, dob
- **bouts**: bout_id (PK), event_id, fighter_a_id, fighter_b_id, weight_class, bout_type_raw, is_womens, is_title_bout, is_interim_title, is_tournament_final, scheduled_rounds, time_format, status (`scheduled` / `completed` / `cancelled`), result (`win` / `draw` / `nc`), winner_id (nullable), method, method_details, referee, end_round, end_time_s, fight_time_s, has_stats, first_seen_at, last_seen_at
  - fighter_a/b follow the order the source lists them in, for storage only. Corner is never a model feature.
  - `has_stats = 0` marks bouts with no recorded stats; they have no stats rows, never rows of zeros.
- **bout_round_stats**: (bout_id, fighter_id, round) PK; knockdowns, sig strikes landed/attempted, total strikes landed/attempted, takedowns landed/attempted, submission attempts, reversals, control time (s, NULL where not recorded), sig strikes landed/attempted by target (head, body, leg) and position (distance, clinch, ground)
- **bout_fighter_stats**: (bout_id, fighter_id) PK; the same columns, derived at load time by summing the fighter's rounds
- **source_files**: file_name (PK), sha256, row_count, upstream_repo, upstream_commit, snapshot_at. Which snapshot the tables were last loaded from.
- **ingest_issues**: seq (PK), severity (`error` = data dropped, `warning`, `info`), check_name, entity_type, entity_id, message. The validation report of the last load.
- **upcoming_events** (M5): event_key (PK, `wp:<page ID>`), pageid, page_title, name, date, revid, status (`scheduled` / `past` / `unlisted`), first_seen_at, last_seen_at, card_applied_at
- **upcoming_bouts** (M5): bout_id (PK, `wb` + 16 hex from page ID and sorted name keys), event_key, card_order, segment, weight_class_raw, weight_class, is_title_bout, name_a/page_a, name_b/page_b, fighter_a_ref/fighter_b_ref (UFCStats ID or provisional ID; NULL when unresolved), method_a/method_b, unresolved_reason, status (`scheduled` / `cancelled`), first_seen_at, last_seen_at, cancelled_at
- **provisional_fighters** (M5): provisional_id (PK, `wp:<article>` or `wpname:<name>`), page_title, name, fighter_id (set once linked), first_seen_at, last_seen_at, linked_at
- **attribute_backfill** (M5): (fighter_ref, field) PK; value, source (`wikipedia`), page_title, revid, fetched_at. Never copied into `fighters`.
- **predictions** (M5): prediction_id (PK), bout_id (upcoming_bouts), model_version, created_at, event_date, fighter_a_ref, fighter_b_ref, p_fighter_a, p_fighter_b, inputs_sha256, features_json (model inputs, flags, per-fighter values, attribute sources), shap_json (M6), limited_data, limited_reasons_json, is_live, superseded_at, superseded_reason. At most one current (not superseded) row per bout.
- **refresh_log** (M5): seq, run_at, level, event_key, message. Append-only.

Loaders are idempotent upserts keyed on UFCStats IDs: running the same load twice produces identical tables. Timestamps come from the snapshot (the upstream commit date), not the clock.

## Feature pipeline (`src/ufc/features/`)

```
uv run python -m ufc.features build      # fit on the training period, build, write data/features/
uv run python -m ufc.features summary    # every feature by family, with missing-value counts
```

`build` takes `--shrink-minutes` (default 0). The model pipeline chooses that value on validation and writes the feature file itself, so after `python -m ufc.models train` the file in `data/features/` is the one the model was trained on.

1. **Read** completed bouts, per-fight and per-round stats, and fighter attributes from SQLite (`data.py`).
2. **Fit** on bouts through 2022-12-31 only: Elo parameters by log loss of pre-fight predictions (`elo.py`), the league-average stat line used as a prior for opponent baselines (`adjust.py`), and the weight-class mean rates that per-minute rates are shrunk toward (`shrink.py`).
3. **Build** one date at a time (`build.py`). Features for every bout on a date are computed from a store that holds earlier dates only; the date's bouts are added afterwards. Each bout enters the store as one vector of additive counts per fighter, and each fighter keeps running sums of those vectors (`components.py`, `store.py`), so career, last-3, last-5 and time-decayed rates are lookups.
4. **Assemble** each bout's row (`fighter.py`, `matchup.py`): every `d_` feature is fighter A's value minus fighter B's, or a pair value minus its mirror image, so swapping the fighters negates it exactly. Three `lvl_` columns are the mean of the two fighters' age, UFC fights and Elo, which swapping leaves exactly unchanged.
5. **Write** `features.parquet` (identity, bout context, `label`, six per-fighter flags, the `d_` features, then the `lvl_` columns) and `features_meta.json` (snapshot commit, settings, fitted parameters, ordered feature list, feature-to-family map for the thirteen families, and the level columns).

`build_matchup(store, fighter_a_id, fighter_b_id, date, weight_class)` returns the same features for any pair and date, real or hypothetical, plus each fighter's own as-of values for display. It reads histories through one lookup that returns only bouts strictly before the date. The batch build calls the same code, and a test checks that both give identical rows for every bout.

Design choices are in DECISIONS D23 to D29 and D31 to D34.

## Models (`src/ufc/models/`)

```
uv run python -m ufc.models train       # select on validation, save the candidate to the registry
uv run python -m ufc.models evaluate    # score the saved candidate on the test period; walk-forward backtest
uv run python -m ufc.models report      # write the generated part of docs/MODEL_REPORT.md
uv run python -m ufc.models.diagnostics # analysis only: era-shift checks (D45); reads the test period
uv run python -m ufc.models refit       # production refit of the candidate's recipe on every bout (D47)
```

1. **Splits** (`splits.py`): train through 2022-12-31, validation 2023 to 2024, test from 2025. All rows reach model code through `Splits`, which refuses test-period rows until `evaluate` unlocks them for a saved candidate. Fitting uses bouts with a winner and no attribute gap (D44).
2. **Inference** (`design.py`, `symmetric.py`): a model is asked about (A, B) and (B, A); the two answers are averaged and calibrated in a way that makes P(A) + P(B) exactly 1 and swaps exactly when the fighters swap (D36).
3. **Baselines** (`baselines.py`): Elo probability from the stored rating difference, and a logistic regression on the standardized difference features.
4. **Candidates** (`gbm.py`, `recipe.py`): XGBoost and LightGBM on both orientations of every training bout, with optional recency weights; trees chosen by early stopping on validation; calibrator fitted on validation (`calibrate.py`).
5. **Selection** (`tune.py`, `select.py`), all on validation: rate-shrinkage strength, level features, hyperparameter search, half-life and library, calibration method, then the feature-family ablation (D38). `train` writes the chosen feature table to `data/features/` and the record of every comparison to `data/reports/eval/selection.json`.
6. **Registry** (`registry.py`): `data/models/{version}/` holds the booster, calibrator, feature list, training configuration, feature settings and fitted parameters, validation metrics, and a manifest with the git commit and data hashes. `data/models/approved.json` (`{"version": "..."}`) is written by hand; no code writes it (D4, D42).
7. **Evaluation** (`evaluate.py`, `walkforward.py`, `plots.py`): `evaluate` loads the saved candidate, scores it and the baselines once on the test period, runs the yearly walk-forward backtest, and writes `test_metrics.json`, `walk_forward.json` and plots under `data/reports/eval/`.
8. **Report** (`report.py`): `docs/MODEL_REPORT.md` has a generated block built only from those JSON files.

Design choices are in DECISIONS D35 to D44.

## Handling card changes

Each refresh reads upcoming events (from Wikipedia, see above) and compares with the database:
- New bout: insert, predict
- Bout no longer on the card: mark `cancelled`, set `superseded_at` on its predictions
- Replacement opponent: this is a new bout_id; old one is cancelled, new one predicted
- Every bout tracks `first_seen_at` and `last_seen_at`
- A page that cannot be read, or a card that loses all or most of its bouts at once, changes nothing (D50)

## Validation gate

Before publishing, check: event exists with a date in the future, at least one bout, every fighter ID resolved, feature vectors match the model's feature schema, probabilities in [0, 1] and summing to 1 per bout. If any check fails, log the failure, do not publish, and leave the last good artifacts in place.

## Published artifacts (S3)

- `public/upcoming.json`: next event(s) with bouts and probabilities
- `public/bouts/{bout_id}.json`: matchup detail, SHAP factors, analysis text
- `public/performance.json`: backtest and live track record metrics
- `archive/predictions/{event_id}/{timestamp}.json`: immutable snapshots, never overwritten. This is the evidence for the live track record.

## Storage choices

- SQLite: a few thousand bouts, written by one batch job at a time. No server, no monthly cost.
- In production, `ufc.db` lives in S3. The refresh job downloads it, updates it, and uploads it. Lambda reserved concurrency of 1 prevents two writers.
- No managed relational database: it would cost money every month to hold a few megabytes.

## AWS (V1, deployed later with approval)

| Service | Role |
|---|---|
| S3 | Database file, published JSON, frontend build, model artifacts |
| CloudFront | Serves the frontend and public JSON |
| Lambda (container image) | Weekly refresh: fetch the dataset snapshot and upcoming cards, predict, validate, publish. Container image because XGBoost and SHAP exceed zip size limits |
| EventBridge Scheduler | Triggers the refresh (more often in fight week) |
| Lambda + function URL or HTTP API | Analyst endpoint |
| DynamoDB (on-demand) | Analyst rate limits and daily spend counter |
| CloudWatch | Logs (14-day retention) and alarms on failed refreshes |
| AWS Budgets | Alerts at $5 and $10 |
| IAM | One least-privilege role per function |

Infrastructure is defined with AWS CDK in Python (`infra/`). All costs must be estimated and approved before the first deploy.

## AI analyst

- Provider: Anthropic API with prepaid credits, auto-reload off. Accessed through a small provider interface so Bedrock can be swapped in later.
- Model: a small, inexpensive model; chosen when the analyst milestone starts.
- Context: retrieved via tools from the bout JSON (prediction, features, SHAP, fighter stats). No free-form database access.
- Guardrails: per-IP rate limit, request size limit, daily dollar cap tracked in DynamoDB, timeout, system prompt forbidding new probabilities or unsupported stats.
- An evaluation set of test questions checks grounding before launch.

## Cost target

About $5/month total at portfolio traffic, excluding analyst spend, which is bounded by prepaid credits and the daily cap. Every number here must be verified against current AWS pricing before deployment.
