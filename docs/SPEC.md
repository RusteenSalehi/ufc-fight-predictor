# Project Spec: UFC Fight Predictor

## Goal

A publicly accessible website that automatically publishes win probabilities for upcoming UFC fights, explains them, reports the model's honest track record, and offers a grounded AI analyst. Built as a portfolio project: every claim on the site must be backed by reproducible code.

## V1 scope

### 1. Upcoming events (home page)
- Next UFC event: name, date, location, last-updated timestamp
- Every bout on the card: fighters, weight class, model win probability for each fighter, predicted winner
- Bouts involving a fighter with zero prior UFC fights are predicted but labeled "limited data"
- Cancelled bouts disappear from the current card; their old predictions are archived, never shown as current

### 2. Matchup detail page
- Fighter comparison: physical attributes, record, key as-of stats
- Win probability with model version and generation timestamp
- Top factors behind the prediction (SHAP), worded as "what moved the model," never as causes
- Pre-generated written analysis of the matchup

### 3. AI analyst (hybrid design)
- Pre-generated analysis per bout, created once when predictions are published
- Free-typed questions answered only from stored stats, prediction, and SHAP data
- Never changes or invents probabilities; never invents stats
- Hard daily spend cap; when reached, chat falls back to the pre-generated analysis
- Prepaid Anthropic API credits with auto-reload off (true hard ceiling)

### 4. Model performance page
- Backtest results on the held-out test period: log loss, Brier score, accuracy, calibration plot, results by year, debut bouts reported separately
- Comparison against baselines (Elo, logistic regression; market odds once available)
- Live track record: predictions published before events, scored after, kept separate from backtests
- Feature ablation results, including fan-intuition features that did not help

### 5. Methodology page
- Data source, leakage prevention, features, models, evaluation, limitations, architecture, links to the repo

## Decisions already made

- Primary product: automatic predictions for upcoming fights
- Model updates are manual: scheduled jobs predict with the approved model; retraining and promotion are deliberate and reversible
- Training data: all UFC history for feature construction; recent fights weighted more heavily in training
- Draws and no contests are excluded from training and shown as "no result" in evaluation
- All weight classes, men's and women's
- Odds: the headline model is stats-only; a separate market-blended model is evaluated against it once historical odds data is sourced
- Budget: about $5/month, designed so caps can be raised later
- Hosting: frontend on AWS (S3 + CloudFront), linked from a GitHub Pages personal site; custom domain later
- Repo is public from day one; data never committed

## Deferred (not V1)

- Live news and injury research in the analyst (V1.1)
- Style archetypes via soft clustering (V1.1)
- Market-blended model and betting simulation (after odds data is sourced)
- Pre-UFC records from other promotions for debuting fighters
- Short-notice booking features (not reconstructable from current data)
- Automated retraining and promotion
- Fighter search and historical exploration UI
- Custom domain and branding

## Non-goals

- Taking bets or giving betting advice
- Claims of guaranteed profit
- Commercial use of scraped data without resolving licensing

## V1 acceptance criteria

1. A public URL shows the next UFC card with predictions, refreshed automatically on a schedule
2. Card changes (cancellations, replacements) are reflected within one refresh cycle
3. A failed scrape or failed validation never overwrites the last good published predictions
4. The performance page shows test-period metrics reproducible with one documented command
5. Leakage tests and symmetry tests pass in CI
6. The analyst answers grounded questions and refuses or flags questions it cannot answer from stored data
7. Monthly AWS cost stays under budget, with alerts configured
8. `docs/DECISIONS.md` explains every major design choice
