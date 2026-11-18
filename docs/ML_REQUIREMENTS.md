# ML Requirements

These rules are non-negotiable. Code that violates them is wrong even if metrics improve.

## Prediction task

For a bout between fighters A and B, estimate P(A wins). P(B wins) = 1 - P(A wins).

- Training rows: completed bouts with a winner. Draws and no contests are excluded from training and reported as "no result" in evaluation.
- Bouts with no recorded per-fight stats (common in early events) still count for records and Elo, but their missing stats are handled explicitly, never silently filled with zeros.

## Leakage rules

1. **As-of cutoff.** Every feature for a bout on date D uses only bouts from events with date strictly before D. Other bouts on the same event never count.
2. **No profile aggregates.** Career stats shown on UFCStats fighter pages are never used. Every stat is recomputed from per-fight data as of the cutoff.
3. **Static attributes.** Height, reach, stance, and date of birth come from current profiles. Assumed effectively static; document this assumption. Age is computed as of the bout date. The values are treated as static; whether a value is *present* is not. A profile is as it stands today, and an incomplete one mostly belongs to a fighter whose UFC career turned out short, so a bout in which one fighter's profile lacks an attribute the other's has is never used to fit a model. Such bouts are still predicted and scored, and reported separately.
4. **Fit on training data only.** Scalers, imputers, Elo parameters, opponent-adjustment baselines, the weight-class means used for rate shrinkage, clustering, calibration, and any other fitted transformation are fit on the training period only (calibration on validation).
5. **Tests prove it.** Required tests:
   - Perturbation test: change or delete bouts after date D; features for bouts on or before D must not change.
   - Same-event test: a bout's features are unaffected by other bouts on the same card.
   - A spot check that a known fighter's as-of stats before a known fight match a hand calculation.

## Symmetry

Corner (red/blue) is never a feature; red corner correlates with being the favorite or champion.

- Features are expressed as A-minus-B differences (plus a few non-differenced fields like weight class).
- Swap-invariant "level" features (the mean of the two fighters' age, UFC fights, and Elo) are allowed because swapping the fighters leaves them exactly unchanged. They are kept in separate `lvl_` columns and enter a model only if they improve validation log loss; the result is reported either way.
- A family may also own a swap-invariant column (family 15's `lvl_age_rel_class`). It is not part of the optional level group: it enters and leaves the model with its family.
- Training uses both orientations of every bout: (A, B, label) and (B, A, 1 - label).
- Inference averages P(A beats B) and 1 - P(B beats A).
- Symmetry test: for any bout, the final P(A) + P(B) = 1 exactly, and swapping inputs swaps outputs.

## Feature families

All computed as of the cutoff. Where useful, both career and rolling (last 3 and last 5 fights) versions, plus exponentially time-decayed averages.

Per-minute and per-15-minute rates are shrunk toward the weight-class mean: `(count + m * class mean per minute) / (minutes + m)`, using the fighter's counts and minutes as of the cutoff. The class means are fit on the training period only. The prior strength `m` (in cage minutes) is chosen on validation from a small grid that includes 0 (no shrinkage). A fighter with no recorded cage time still has missing rates, not the class mean.

1. **Physical:** age, height, reach, stance matchup (for example orthodox vs southpaw)
2. **Record and experience:** UFC fights, wins, losses, win rate, finish rate, KO/TKO and submission win counts, title fight experience
3. **Striking:** sig strikes landed and absorbed per minute, accuracy, defense, knockdowns per 15 minutes, target and position distribution
4. **Grappling:** takedowns landed per 15 minutes, takedown accuracy, takedown defense, submission attempts per 15 minutes, control time share
5. **Durability:** KO/TKO losses (career and recent), knockdowns absorbed, head strikes absorbed per minute
6. **Activity:** days since last fight, fights in the last 2 years
7. **Weight change:** moving up or down from the previous bout's weight class, fights at the current weight
8. **Opponent quality:** opponents' Elo at the time they fought (mean and recent), from the Elo system below. The fighter's own rating is family 13, not part of this family.
9. **Opponent-adjusted stats:** each stat compared with what that fighter's opponents typically allowed or achieved against others, as of the cutoff. Example: if a fighter's opponents usually land 40% of takedown attempts against others but only 20% against him, his adjusted takedown defense is strong. Computed with careful as-of handling, since opponents' baselines must also respect the cutoff.
10. **Style matchups:** A's offense against B's defense (takedown offense vs takedown defense, strike volume vs striking defense, and so on)
11. **Debut flag:** fighter has zero prior UFC bouts; their stats are missing, not zero
12. **Per-round pace and cardio:** from per-round stats, how a fighter's output and defense in rounds 3 and later compare with round 1: significant strikes landed, attempted, and absorbed per minute in rounds 3+ divided by the round-1 rate, and the change in accuracy and defense. Only bouts that went past round 2 are used; the features are NaN for a fighter with no such bout.
13. **Rating:** the fighter's own Elo rating as of the cutoff. Kept apart from family 8 so that the ablation can tell "how highly is he rated" from "whom has he fought".
14. **Strength of schedule (candidate, not adopted):** best win and worst loss by the opponent's Elo at the time, and wins and losses against opponents then rated above the fighter. Computed and written to the feature table, but not a model input: it failed the M4.5 adoption rule (DECISIONS D54 to D56). A model reads it only if its feature list names it.
15. **Age shape:** years past 32 and years past the weight class's median age, each `max(0, age - threshold)`, as A-minus-B differences; and the bout's mean age minus the class median as a swap-invariant column of this family. The class medians are fitted on the training period only (a class needs 50 training bouts and a fixed weight limit; otherwise the median over all training bouts is used). Adopted in M4.5 (DECISIONS D56, D58). Age comes from today's profiles (leakage rule 3).
16. **Performance (candidate, M4.6):** built on a "deserved win probability" per fighter per bout, from a performance model: a logistic regression without intercept on in-fight stat differences, fitted on training-period decisions with the judged winner as label (leakage rule 4) and applied to every bout with stats, finishes (stats up to the stoppage) and draws included. As-of differences: deserved share (career, last 3, decayed), luck (actual minus deserved, career and last 3), a performance Elo updated on the deserved share, and deserved share weighted by the opponent's Elo above 1500. Computed and written to the feature table, not a model input unless adopted under the rule in DECISIONS D63. A model reads it only if its feature list names it.

## Recency weighting

Use all history for feature construction. During training, weight samples by recency with exponential decay. Candidate half-lives: none, 2, 4, 8 years. Choose on validation log loss.

## Data splits

- Train: bouts through 2022-12-31
- Validation: 2023-01-01 through 2024-12-31 (model selection, tuning, calibration, half-life choice)
- Test: 2025-01-01 onward. Evaluated once per finalized candidate. Do not iterate on test results.
- Walk-forward backtest (yearly expanding window) for the performance page's "results over time" chart.

## Baselines

1. Elo rating system; prediction from rating difference. Also pick-the-higher-rated-fighter accuracy. The Elo parameters (K and its multipliers) are fit on the training period only, and the baseline uses the same as-of ratings that feed the features. Nothing about Elo is tuned on validation. Ratings keep updating after the training period as bouts are completed; only the parameters are frozen.
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
- Feature ablation: remove each family the model reads in turn (families 1 to 13 and 15; family 14 is not a model input), including family 12 (per-round pace and cardio), family 13 (rating) and family 15 (age shape, all three of its columns together), and report the change in validation log loss. The feature-to-family map is the `families` entry of `data/features/features_meta.json`. If the optional level features are in the model they are ablated as one more group; a family's own swap-invariant column is removed with its family, not with that group.

## Model registry

Each trained model is saved under `data/models/{version}/` with: model file, ordered feature list, metrics JSON, git commit hash, hash of the training dataset, and training config. `data/models/approved.json` points to the approved version. Promotion is a manual edit of that pointer; rollback is changing it back.

## Live vs backtest

Predictions published before an event are "live" and stored as immutable snapshots. Backtest predictions are generated after the fact. They are reported separately everywhere.

## Betting evaluation (pre-registered 2026-11-18, before any odds results exist; DECISIONS D62)

This section is fixed. Nothing below may change after odds results are seen. A change needs a new DECISIONS entry that says what was seen first, and the original definition's results are still reported. Odds are used for evaluation only; the headline model stays stats-only (D7). This is a simulation of a rule, not betting advice (SPEC).

**Reference price.** Pinnacle's `h2h` moneyline (`bookmaker.key == "pinnacle"` in The Odds API data). A bout with no Pinnacle price in its closing snapshot is excluded and counted, never replaced by another book. Other books and a consensus may be reported as separate, labelled sensitivity checks.

**Card start.** The earliest `commence_time` among the card's bouts in the odds feed (the feed's start time is per bout, `docs/ODDS_SOURCES.md` section 2a).

**Closing.** The last snapshot taken at or before the card start. Its price for a bout is the Pinnacle entry in that snapshot; a snapshot taken after the card start is never used. A bout whose last qualifying snapshot is more than 24 hours before the card start is excluded and counted. Bouts are matched to ours by date and name through an override file with evidence, no fuzzy matching (D17); unmatched bouts are excluded and counted.

**Vig removal.** Proportional. Each fighter's implied probability is `100 / (american + 100)` for a positive price and `|american| / (|american| + 100)` for a negative price; both are divided by their sum. This is the "market probability" everywhere below.

**Payout.** A $100 stake on a price `o` wins `o` if `o > 0` and `10000 / |o|` if `o < 0`, and loses $100 otherwise. A draw or no contest is a refunded stake (zero profit) and is counted separately.

**Model probability.** The calibrated probability for the fighter, out of time for the model that made it: the walk-forward and test-period predictions of the candidate `approved.json` names when the evaluation is run, and for live bouts the stored immutable live prediction. Backtest and live bouts are reported separately (see "Live vs backtest"). The primary backtest sample is every bout from 2025-01-01 onward with a valid closing price; the earlier folds with a valid price are reported as a secondary table.

**Strategies.** Each places a flat $100 and uses the closing price above. Each is reported on the same sample.

1. **Market favourite:** bet on the fighter with the higher market probability, on every bout in the sample. A bout whose two market probabilities are exactly equal is not bet and is counted.
2. **Model pick:** bet on the fighter the model gives more than 50%, on every bout in the sample, at that fighter's closing price.
3. **Model edge:** bet on a fighter only when `model probability - market probability >= 0.05` (five percentage points, inclusive), at that fighter's closing price. Both fighters cannot qualify. A bout with no qualifying fighter gets no bet.

**Reported for each strategy:** number of bets, total staked, total profit, ROI (`profit / staked`), hit rate, and a 95% percentile bootstrap interval of the ROI, resampling bouts with 10,000 draws and seed 0. Also reported, for the sample: the exclusion counts above, the market's own log loss and Brier score next to the model's, and the average vig removed. The threshold, stake, reference book, bootstrap settings and the headline sample are not tuned. A sweep of other thresholds or books may be shown only as a table labelled exploratory, after the headline. Results are published whatever their sign.

## V1.1: style archetypes

- Soft clustering (Gaussian mixture) on as-of style rate stats, never outcomes; fit on training period only
- Clusters named after inspection; fighters get archetype probabilities
- Attribute strength shown as within-weight-class percentiles, not distance to a cluster center (a center represents the typical member, not the best)
- Archetype matchup features go through the same ablation test
