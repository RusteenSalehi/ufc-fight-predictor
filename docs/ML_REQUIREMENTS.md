# ML Requirements

These rules are non-negotiable. Code that violates them is wrong even if metrics improve.

## Prediction task

For a bout between fighters A and B, estimate P(A wins). P(B wins) = 1 - P(A wins).

- Training rows: completed bouts with a winner. Draws and no contests are excluded from training and reported as "no result" in evaluation.
- Bouts with no recorded per-fight stats (common in early events) still count for records and Elo, but their missing stats are handled explicitly, never silently filled with zeros.

## Leakage rules

1. **As-of cutoff.** Every feature for a bout on date D uses only bouts from events with date strictly before D. Other bouts on the same event never count.
2. **No profile aggregates.** Career stats shown on UFCStats fighter pages are never used. Every stat is recomputed from per-fight data as of the cutoff.
3. **Static attributes.** Height, reach, stance, and date of birth come from current profiles. Assumed effectively static; document this assumption. Age is computed as of the bout date.
4. **Fit on training data only.** Scalers, imputers, Elo parameters, opponent-adjustment baselines, clustering, calibration, and any other fitted transformation are fit on the training period only (calibration on validation).
5. **Tests prove it.** Required tests:
   - Perturbation test: change or delete bouts after date D; features for bouts on or before D must not change.
   - Same-event test: a bout's features are unaffected by other bouts on the same card.
   - A spot check that a known fighter's as-of stats before a known fight match a hand calculation.

## Symmetry

Corner (red/blue) is never a feature; red corner correlates with being the favorite or champion.

- Features are expressed as A-minus-B differences (plus a few non-differenced fields like weight class).
- Training uses both orientations of every bout: (A, B, label) and (B, A, 1 - label).
- Inference averages P(A beats B) and 1 - P(B beats A).
- Symmetry test: for any bout, the final P(A) + P(B) = 1 exactly, and swapping inputs swaps outputs.

## Feature families

All computed as of the cutoff. Where useful, both career and rolling (last 3 and last 5 fights) versions, plus exponentially time-decayed averages.

1. **Physical:** age, height, reach, stance matchup (for example orthodox vs southpaw)
2. **Record and experience:** UFC fights, wins, losses, win rate, finish rate, KO/TKO and submission win counts, title fight experience
3. **Striking:** sig strikes landed and absorbed per minute, accuracy, defense, knockdowns per 15 minutes, target and position distribution
4. **Grappling:** takedowns landed per 15 minutes, takedown accuracy, takedown defense, submission attempts per 15 minutes, control time share
5. **Durability:** KO/TKO losses (career and recent), knockdowns absorbed, head strikes absorbed per minute
6. **Activity:** days since last fight, fights in the last 2 years
7. **Weight change:** moving up or down from the previous bout's weight class, fights at the current weight
8. **Opponent quality:** opponents' Elo at the time they fought (mean and recent), from the Elo system below
9. **Opponent-adjusted stats:** each stat compared with what that fighter's opponents typically allowed or achieved against others, as of the cutoff. Example: if a fighter's opponents usually land 40% of takedown attempts against others but only 20% against him, his adjusted takedown defense is strong. Computed with careful as-of handling, since opponents' baselines must also respect the cutoff.
10. **Style matchups:** A's offense against B's defense (takedown offense vs takedown defense, strike volume vs striking defense, and so on)
11. **Debut flag:** fighter has zero prior UFC bouts; their stats are missing, not zero

## Recency weighting

Use all history for feature construction. During training, weight samples by recency with exponential decay. Candidate half-lives: none, 2, 4, 8 years. Choose on validation log loss.

## Data splits

- Train: bouts through 2022-12-31
- Validation: 2023-01-01 through 2024-12-31 (model selection, tuning, calibration, half-life choice)
- Test: 2025-01-01 onward. Evaluated once per finalized candidate. Do not iterate on test results.
- Walk-forward backtest (yearly expanding window) for the performance page's "results over time" chart.

## Baselines

1. Elo rating system (fit on training data; K-factor tuned on validation); prediction from rating difference. Also pick-the-higher-rated-fighter accuracy.
2. Logistic regression on the feature set (standardized)
3. Market odds (normalized implied probability, vig removed) once historical odds are sourced

## Candidate models

XGBoost and LightGBM. A model is better only if it beats the baselines on validation log loss. No neural networks without a specific experimental reason.

## Calibration

Calibrate the chosen model on the validation set (isotonic or Platt; pick by validation log loss using cross-validation within validation, or a documented split). Report reliability diagrams before and after.

## Metrics

- Primary selection metric: log loss
- Also report: Brier score, accuracy, expected calibration error, reliability diagram
- Breakdowns: by year, by confidence bucket, debut bouts separately, by weight class only where sample size allows
- Feature ablation: remove each feature family, report change in validation log loss

## Model registry

Each trained model is saved under `data/models/{version}/` with: model file, ordered feature list, metrics JSON, git commit hash, hash of the training dataset, and training config. `data/models/approved.json` points to the approved version. Promotion is a manual edit of that pointer; rollback is changing it back.

## Live vs backtest

Predictions published before an event are "live" and stored as immutable snapshots. Backtest predictions are generated after the fact. They are reported separately everywhere.

## V1.1: style archetypes

- Soft clustering (Gaussian mixture) on as-of style rate stats, never outcomes; fit on training period only
- Clusters named after inspection; fighters get archetype probabilities
- Attribute strength shown as within-weight-class percentiles, not distance to a cluster center (a center represents the typical member, not the best)
- Archetype matchup features go through the same ablation test
