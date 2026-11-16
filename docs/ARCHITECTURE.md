# Architecture

## Overview

```
ufcstats.com
    |  polite scraper (1 req/s, raw HTML cache)
    v
raw HTML cache (data/raw)  ->  parsers  ->  SQLite (ufc.db)
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

UFCStats.com is the primary source. Page types:
- Completed events list and upcoming events list
- Event page: bouts on the card
- Fight details page: per-fight, per-fighter stats (totals and significant strikes by target and position), method, round, time
- Fighter page: physical attributes and career aggregates

Only physical attributes (height, reach, stance, date of birth) are taken from fighter pages. Career aggregates on fighter pages reflect today's values and are never used for training.

Stable IDs come from UFCStats URLs (each event, fight, and fighter has a hex ID). Names are display data only.

## Database schema (SQLite)

Exact columns are finalized in `src/ufc/db/schema.sql` after parser recon; this is the target shape.

- **events**: event_id (PK, UFCStats ID), name, date, location, status (`completed` / `upcoming`), first_seen_at, last_seen_at
- **fighters**: fighter_id (PK), name, nickname, height_cm, reach_cm, stance, dob
- **bouts**: bout_id (PK), event_id, fighter_a_id, fighter_b_id, weight_class, is_womens, is_title_bout, scheduled_rounds, status (`scheduled` / `completed` / `cancelled`), winner_id (nullable), result (`win` / `draw` / `nc`), method, end_round, end_time_s, first_seen_at, last_seen_at
  - fighter_a/b follow UFCStats red/blue order for storage only. Corner is never a model feature.
- **bout_fighter_stats**: (bout_id, fighter_id) PK; knockdowns, sig strikes landed/attempted, total strikes landed/attempted, takedowns landed/attempted, submission attempts, reversals, control time (s), sig strikes landed/attempted by target (head, body, leg) and position (distance, clinch, ground)
- **raw_pages**: url (PK), fetched_at, sha256, cache_path, http_status
- **predictions**: prediction_id, bout_id, model_version, created_at, p_fighter_a, p_fighter_b, features_json, shap_json, superseded_at (nullable), is_live (true if published before the event)

Loaders are idempotent upserts keyed on UFCStats IDs: running the same load twice produces identical tables.

## Handling card changes

Each refresh scrapes upcoming events and compares with the database:
- New bout: insert, predict
- Bout no longer on the card: mark `cancelled`, set `superseded_at` on its predictions
- Replacement opponent: this is a new bout_id; old one is cancelled, new one predicted
- Every bout tracks `first_seen_at` and `last_seen_at`

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
| Lambda (container image) | Weekly refresh: scrape upcoming cards, predict, validate, publish. Container image because XGBoost and SHAP exceed zip size limits |
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
