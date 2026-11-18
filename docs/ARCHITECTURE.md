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
uv run python -m ufc.predict refresh   # cards from Wikipedia, predictions, the site's artifacts in data/publish/
```

The dataset covers completed events only. Upcoming cards come from English Wikipedia through the MediaWiki API (D16, D48); ufcstats.com is never requested. One refresh:

1. **Schedule** (`wikipedia.py`, `cards.py`): "List of UFC events", read with `query`+`revisions`. Its "Scheduled events" table gives each event's page and date; its "Past events" table is used for staleness. If it cannot be read, the refresh stops and nothing changes.
2. **Cards**: every scheduled event page in one request; bouts from `{{MMAevent bout|class|fighter|vs.|fighter|...}}` templates in page order, with the card segment, weight class mapped to the dataset's labels, linked article and title-bout note. "Announced bouts" lists are not predicted. Wikipedia text is untrusted: anything of unexpected shape is a parse error, not a guess.
3. **Resolution** (`resolve.py`, D49): the ingest's resolver and overrides, plus `wikipedia_overrides.csv`. Unknown names become provisional fighters (presumed debutants); ambiguous names, placeholders and veterans spelled differently (detected when their article cites a UFCStats profile in the dataset) make the bout unresolved.
4. **Card changes** (`cards_db.py`, D50): each card is applied in one transaction (new pairings added, missing ones cancelled with their predictions superseded) or, if unreadable or implausible, not at all.
5. **Staleness** (`staleness.py`, D52): completed events after the dataset's latest; bouts with a fighter on one of them are labelled "stale stats".
6. **Backfill** (`backfill.py`, `infobox.py`, D51): missing height, reach and date of birth from the linked article's infobox, for predictions only.
7. **Predictions** (`predict.py`, D53): the approved model, the feature settings and fitted parameters stored with it, `build_matchup` at the event date. New rows only when inputs change.
8. **Link** (`cards_db.py`, D67): provisional fighters on cards dated today or earlier whose name now resolves to a dataset profile, so their bouts can be scored.
9. **Publish** (`src/ufc/publish/`, M6): the site's artifacts, built from the current predictions (`publish.snapshot`), validated whole, and written with archive snapshots, or not at all; see "Published artifacts" below.

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
- **predictions** (M5): prediction_id (PK), bout_id (upcoming_bouts), model_version, created_at, event_date, fighter_a_ref, fighter_b_ref, p_fighter_a, p_fighter_b, inputs_sha256, features_json (model inputs, flags, per-fighter values, attribute sources), shap_json (symmetrised SHAP contributions toward fighter A and the `why` explanation built from them, `explain.py`, D59; NULL for rows made before it existed; the published `why` is re-rendered from the stored contributions and values with the current templates, D65), limited_data, limited_reasons_json, is_live (made before the event date in UTC; the live record uses the stricter rule of D67), superseded_at, superseded_reason. At most one current (not superseded) row per bout.
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
4. **Assemble** each bout's row (`fighter.py`, `matchup.py`): every `d_` feature is fighter A's value minus fighter B's, or a pair value minus its mirror image, so swapping the fighters negates it exactly. Three `lvl_` columns are the mean of the two fighters' age, UFC fights and Elo, which swapping leaves exactly unchanged; a fourth, `lvl_age_rel_class`, belongs to family 15 (age shape) and goes wherever that family goes (D58).
5. **Write** `features.parquet` (identity, bout context, `label`, six per-fighter flags, the `d_` features with family 15's two after every M4 column, the `lvl_` columns, then the candidate columns of family 14, which no model reads unless its feature list names them, D55) and `features_meta.json` (snapshot commit, settings, fitted parameters, ordered feature list, feature-to-family map for the fourteen model families, and the level columns).

`build_matchup(store, fighter_a_id, fighter_b_id, date, weight_class)` returns the same features for any pair and date, real or hypothetical, plus each fighter's own as-of values for display. It reads histories through one lookup that returns only bouts strictly before the date. The batch build calls the same code, and a test checks that both give identical rows for every bout.

Design choices are in DECISIONS D23 to D29 and D31 to D34.

## Models (`src/ufc/models/`)

```
uv run python -m ufc.models train       # select on validation, save the candidate to the registry
uv run python -m ufc.models train --recipe-of VERSION   # VERSION's recipe with today's families, nothing re-tuned (D58)
uv run python -m ufc.models evaluate    # score the saved candidate on the test period; walk-forward backtest
uv run python -m ufc.models report      # write the generated part of docs/MODEL_REPORT.md
uv run python -m ufc.models.diagnostics # analysis only: era-shift checks (D45); reads the test period
uv run python -m ufc.models refit       # production refit of the candidate's recipe on every bout (D47);
                                        # Platt slope on the rolling 2023-onward window by default (D58)
uv run python -m ufc.models.revision run --variants ...   # M4.5 variant analysis (D54 to D57); analysis only
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

Before publishing, check: event exists with a date in the future, at least one bout, every fighter ID resolved, feature vectors match the model's feature schema, probabilities in [0, 1] and summing to 1 per bout (`ufc.predict.publish.validate`). Then every artifact is validated against its pydantic model (`ufc.publish.schemas`), and `upcoming.json` and the bout pages are checked against each other. If any check fails, log the failure, write nothing at all (no archive snapshot either), and leave the last good artifacts in place (D66).

## Published artifacts (`src/ufc/publish/`, M6)

Written locally under `data/publish/` by the refresh; in production the same tree goes to S3 (M8).

| Path | Contents |
|---|---|
| `public/upcoming.json` | scheduled events in date order; each predicted bout's probabilities, winner, model version, limited-data labels and a link to its page; unresolved bouts with the reason; the global staleness status; attribution |
| `public/bouts/{bout_id}.json` | one predicted bout: fighters, event, weight class, calibrated probabilities and fair moneyline, model version and time, limited-data labels with reasons, the `why`, up to five factors toward each fighter, a display set of as-of stats, each fighter's last five UFC bouts with the M4.6 deserved-win probability as context, null `market` fields for M12 |
| `public/performance.json` | the backtest (copied from the evaluation record of the approved model's candidate), the era-shift caveat, and the live record, in separate sections |
| `archive/predictions/{event}/{timestamp}.json` | what an event's card showed at one publish; written once, read-only, never overwritten, and only when the card's content changed. The evidence for the live record |

Modules: `schemas.py` (every artifact's model, `schema_version`), `artifacts.py` (bout pages, `upcoming.json`, snapshots), `history.py` (recent bouts and the deserved-win probability), `performance.py` (backtest numbers from `data/reports/`), `live.py` (live scoring), `gate.py` (build, validate, write in the order archive, bout pages, `performance.json`, `upcoming.json`), `jsonschema.py` (the public models as one JSON Schema for the frontend's types, `uv run python -m ufc.publish schema`, D68).

### Live record (`live.py`, D67)

For each bout with a prediction whose event date has come: the latest prediction created before 10:00 UTC on the day before the event date (the event's date begins at UTC+14 then, so no bout can have started) is the live one. A bout off the card at that cut-off is `cancelled`. Once the dataset reaches the event date, the bout is matched to the completed bout between the same two dataset fighter IDs on that date; a win is scored (log loss, Brier, accuracy, a calibration table by the favourite's probability), a draw or no contest is `no result`, and `pending`, `no_live_prediction`, `not_found` and `unlinked` are counted. The record is recomputed from the database on each publish and never pooled with backtest numbers.

## Frontend (`web/`, M7)

A static single-page site: React, TypeScript, Vite, Tailwind CSS, React Router and Recharts, tested with Vitest. It reads the three public artifacts and nothing else, has no backend, and computes no prediction (D68).

```
cd web && npm install      # once
npm run dev                # http://localhost:5173, data served from ../data/publish/public at /data/
npm run build              # tsc, then web/dist/ (no data inside)
npm test                   # Vitest
npm run check              # typecheck, lint, tests, and the type-drift check
npm run types:gen          # after changing ufc.publish.schemas: schema, then TypeScript
```

| Route | Page | Reads |
|---|---|---|
| `/` | next event and its card (two-sided probability bar, pick, fair moneyline, limited-data and stale-stats labels with reasons), later events below | `upcoming.json` |
| `/bout/{bout_id}` | probability with model version and time, the "why" and its factor chart, side-by-side as-of stats, each fighter's last five bouts with the deserved-win probability as description | `bouts/{bout_id}.json` |
| `/performance` | backtest with the era-shift and "already seen" caveats, calibration plot, walk-forward chart, then the live record (empty state until a bout is scored) | `performance.json` |
| `/methodology` | data sources, leakage prevention, features, models, evaluation, limitations, link to the repository | nothing |

**Types.** `uv run python -m ufc.publish schema` writes `web/schema/artifacts.schema.json` from `ufc.publish.schemas`; `web/scripts/gen-types.mjs` writes `web/src/types/artifacts.ts` from it. Both are committed. `tests/test_publish_jsonschema.py` fails if the schema drifts from the models, and `web/scripts/artifact-types.test.mjs` fails if the TypeScript drifts from the schema.

**Data.** `web/src/api/client.ts` fetches from `VITE_DATA_BASE_URL` (build-time, default `/data`), caches each file for the session, and refuses a file whose `kind` or `schema_version` it does not expect. In `vite` and `vite preview`, `web/vite.publish-data.ts` serves `../data/publish/public/` at `/data/` (`UFC_PUBLISH_DIR` to point elsewhere), JSON only, nothing outside that directory (D69).

**Code layout.** `src/pages/` (one per route), `src/components/`, `src/api/` (fetching), `src/probability.ts`, `src/format.ts` and `src/events.ts` (pure display logic), `src/config.ts` (`SITE_NAME`, repository URL, data base URL). Names that `.gitignore` ignores anywhere in the tree (`data/`, `lib/`, `models/`, `artifacts/`, `dist/`) are not used for source directories.

### Hosting requirements for M8 (S3 + CloudFront)

The build is static files; the routes above exist only in the browser (D71). The distribution needs:

1. **Origins.** The site build (`web/dist/`) at the root of a private S3 bucket behind Origin Access Control. The public artifacts at `/data/` on the same distribution (an S3 prefix or a second origin), so the default `VITE_DATA_BASE_URL=/data` works and no CORS is needed. If the JSON is served from another host instead, build with `VITE_DATA_BASE_URL=https://...` and allow `GET` from the site's origin in that host's CORS rules.
2. **Deep links.** A CloudFront Function on viewer request for the default behaviour: if the URI's last segment has no file extension (`/`, `/bout/wb...`, `/performance`, `/methodology`, an unknown path), rewrite the URI to `/index.html`. Leave `/assets/*`, `/data/*` and `/favicon.svg` untouched. Do **not** use distribution-wide custom error responses that turn 403 or 404 into `/index.html` with 200: a missing `/data/bouts/{id}.json` must stay a 404 so the site can say "bout not found".
3. **Caching.** `/assets/*` file names carry content hashes: cache for a year, `immutable`. `/index.html`: `no-cache` (revalidate every time), so a deploy is visible at once. `/data/*.json`: a short TTL (a few minutes) or an invalidation of `/data/*` after each publish, since the refresh rewrites them in place; `Content-Type: application/json`.
4. **Order of uploads.** Data and site deploy independently. For data, keep the gate's order (D66): bout pages before `upcoming.json`, so a link on the home page never points at a missing page. For the site, upload `assets/` before `index.html`.

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
