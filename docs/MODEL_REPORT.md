# Model report (M4)

Candidate `xgboost-71fab189-44ddd3f0`, selected on validation (2023 to 2024) and scored once on the test
period (2025-01-11 to 2026-11-14). The candidate itself was never approved. Its production refit
`xgboost-ffc3ca8c-ee729619` is the version `data/models/approved.json` named when this was written
(2026-11-17); approval is a manual edit of that file. The section "Production model after M4.5" at the end
names the model proposed to replace it.

The block between the GENERATED markers is rebuilt from the metrics files in `data/reports/eval/` by the
last command below and is replaced on every run. It describes the M4 candidate. Everything above it is
written by hand and uses only numbers from that block. The "Era-shift diagnostic", "Production refit",
"Model revision M4.5", "Production model after M4.5" and "Model revision M4.6" sections after the block are
also written by hand, each from the output of its own commands.

    uv run python -m ufc.ingest fetch
    uv run python -m ufc.ingest load
    uv run python -m ufc.models train       # selection on validation (several hundred fits); saves the candidate
    uv run python -m ufc.models evaluate    # the only command that reads the test period
    uv run python -m ufc.models report

## Findings

**Test period.** On 1,013 bouts with a winner the calibrated model has log loss 0.6216 (95% interval 0.6013
to 0.6424), Brier score 0.2158 and accuracy 65.8%. Elo has 0.6788 and 56.2%; logistic regression has 0.6259
and 66.3%. The model is clearly better than Elo: the paired difference is -0.0572 (-0.0747 to -0.0393). It
is **not** shown to be better than logistic regression: the difference is -0.0043 with an interval of
-0.0135 to +0.0060, and logistic regression has the higher accuracy. Validation says the same (-0.0027,
-0.0123 to +0.0065). The tree model meets the rule in ML_REQUIREMENTS, lower validation log loss than both
baselines, but on this evidence a regularized linear model on the same features does about as well.

**The test period scored better than any earlier year, and that should not be read as the expected level.**
In the walk-forward backtest the same recipe has log loss 0.6619 pooled over 2013 to 2024, with single years
between 0.6308 and 0.6850; 2025 is 0.6118 and 2026 is 0.6261. Elo does not move (0.6814 before the test
period, 0.6789 in it), so whatever changed is in the features other than the rating, and logistic
regression shows the same improvement. I do not have a tested explanation. A leak that only affects recent
years would look like this, and so would a real change in how bouts are matched. The checks that exist
(the as-of tests, the attribute-gap exclusion below, the 2025 fold reproducing the registered model exactly)
do not point to a leak, but the honest summary is: expect something between the backtest and the test
figure, and treat the first live results as the real measurement. The "Era-shift diagnostic" section at
the end of this file tests this further: it finds no leak, and places the change at 2024, not at the start
of the test period.

**A leak was found and removed before the test period was opened.** In 895 training bouts one fighter's
profile lacks a physical attribute that the other's has, and the fighter with the complete profile won
91.5% of them. Profiles are as they stand today, so which side has the gap is information from after the
bout (DECISIONS D44). Those 900 bouts are no longer fitted on; models are fitted on the other 5,913.
Validation has no such bouts, so it could not have shown the problem. The test period has 102, and there
the model is worse than a coin flip in log loss (0.7251, accuracy 52.5%), as is Elo (0.7310). On the 911
bouts without a gap the model has 0.6100 and 67.3%. These are mostly bouts with a recent debutant whose
profile is incomplete; they need the "limited data" label on the site as much as debuts do.

**Debut bouts.** With any debutant (197 test bouts) the model has 0.6482 and 63.7%, against 0.6152 and
66.3% with none. Bouts between two debutants show how little 30 bouts can say: the model picked 26.7%
right on validation's 30 and 56.5% on the test period's 31.

**Calibration.** Platt scaling won the cross-validation inside validation (0.6353, against 0.6364
uncalibrated and 0.6502 for isotonic regression) with a slope of 1.216, which moves probabilities away from
50%. On the test period it lowered log loss from 0.6252 to 0.6216 (paired difference -0.0036, interval
-0.0070 to +0.0000) and calibration error from 0.0485 to 0.0412. The model is still underconfident in the
middle of the range: picks given 65% to 70% won 76.0% of the time (154 bouts). At the top it is about
right: picks given 80% or more averaged 83.8% and won 82.4% (68 bouts).

**Choices that validation could not separate.** Each was made by the stated rule, and each interval
contains zero:

- Rate shrinkage at 15 cage minutes: -0.0030 (-0.0076 to +0.0016) for the tree models. It matters more to
  logistic regression, whose validation log loss goes from 0.6446 without shrinkage to 0.6380 with it.
- Level features (mean age, UFC fights, Elo): kept because the criterion was lower with them, by -0.0003
  (-0.0011 to +0.0006). The ablation agrees on the size: +0.0007 (-0.0001 to +0.0014) when they are removed.
- Recency half-life of 8 years: -0.0004 (-0.0032 to +0.0024) against no weighting. A 2-year half-life was
  worse for both libraries.
- XGBoost over LightGBM: 0.6355 against 0.6361.

**Ablation.** Three families help with intervals that exclude zero: physical (+0.0115), opponent-adjusted
stats (+0.0043) and opponent quality (+0.0038). Removing durability (-0.0002), grappling (-0.0008) or
record (-0.0009) did not raise validation log loss at all. The rest are positive with intervals that
include zero: striking, rating, activity, cardio, level, debut, weight change and style matchups. Of the
features that follow fan intuition, style matchups add +0.0001, weight change +0.0002 and cardio +0.0010.
Removing one family at a time understates families that overlap: the rating's +0.0012 does not mean Elo is
unimportant, only that record and opponent quality carry most of the same information when it is gone.

**Limits.** There is no market-odds baseline yet. Every choice was made on one validation period of 1,017
bouts. The registered model was fitted on bouts through 2022 and has not been refitted on later data; the
2026 backtest fold, which is fitted through 2023, has 0.6261 on 2026 where the registered model has 0.6316.

<!-- BEGIN GENERATED: python -m ufc.models report -->

## Candidate

- Registry version: `xgboost-71fab189-44ddd3f0` (created 2026-11-17T06:32:10+00:00)
- Code: git commit `d5cd7504cb1ad06861bb584008844dd7a5139753`
- Data: upstream snapshot commit `96adace863c0a793579cc06d27d468797545607f`; training rows SHA-256 `71fab18949bc648d...`
- Model: XGBoost, 405 trees, 136 features, seed 0
- Settings: colsample_bytree 1, learning_rate 0.03, max_depth 3, min_child_weight 40, reg_lambda 1, subsample 0.6
- Recency half-life: 8 years
- Rate shrinkage: 15 cage minutes
- Calibration: platt
- Libraries: lightgbm 4.7.0, numpy 2.5.4, pandas 3.1.1, python 3.12.16, scikit-learn 1.9.1, xgboost 3.4.1

## Splits

| period | dates | bouts | with a winner |
|---|---:|---:|---:|
| train | through 2022-12-31 | 6,937 | 6,813 |
| validation | 2023-01-01 to 2024-12-31 | 1,037 | 1,017 |
| test | from 2025-01-01 | see below | see below |

Draws and no contests have no winner; they are not trained on and not scored.

## Features and fitted parameters

133 difference features in 13 families: activity 2, cardio 9, debut 1, durability 13, grappling 24, opponent_adjusted 20, opponent_quality 4, physical 6, rating 1, record 15, striking 30, style_matchup 5, weight_change 3.

Elo parameters, fit on the training period only: K = 56, finish multiplier 1.15, early-career multiplier 1.5 for the first 3 rated bouts (training log loss 0.6791 over 6,813 bouts).

## Bouts left out of fitting: attribute gaps

A bout has an attribute gap when one fighter's profile lacks a physical attribute (height, reach, stance or date of birth) that the other's has. Of the 6,813 training bouts with a winner, 900 have a gap. In the 895 where only one fighter has a gap, the fighter with the complete profile won **91.5%** of the time:

| training bouts dated | gap on one side only | complete profile won |
|---|---:|---:|
| through 2005 | 276 | 84.4% |
| 2006 to 2010 | 240 | 94.6% |
| 2011 to 2014 | 234 | 93.2% |
| 2015 to 2018 | 144 | 97.2% |
| from 2019 | 1 | 100.0% |

Profiles are as they stand today, so which fighter has the gap is information from after the bout (DECISIONS D44). No model or baseline is fitted on these bouts: fitting uses the other 5,913 training bouts. They are still predicted and scored. Validation has 0 bouts with a gap.

## Selection on validation

Every comparison below is validation log loss, averaged over 5 seeds unless it says otherwise. Intervals are 95% paired bootstrap intervals over validation bouts: the bouts are resampled and both sides of a comparison keep their loss on each resampled bout. An interval that contains zero means the comparison did not separate the two options.

### 1. Rate shrinkage strength

Criterion: validation log loss, mean of XGBoost and LightGBM at reference settings, mean over seeds. Delta: criterion minus the criterion at 0 minutes, paired over bouts. The last two columns describe the shrunk features on the training period: the largest absolute value of any of them in standard deviations, and their median excess kurtosis. The logistic regression column is that baseline's validation log loss at its best C; it is shown for information and is not part of the criterion.

| cage minutes | XGBoost | LightGBM | criterion | delta | 95% interval | logistic regression | largest SD | median kurtosis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6436 | 0.6426 | 0.6431 | +0.0000 | +0.0000 to +0.0000 | 0.6446 | 39.2 | 37.4 |
| 5 | 0.6410 | 0.6403 | 0.6407 | -0.0024 | -0.0060 to +0.0013 | 0.6386 | 16.2 | 2.0 |
| 15 | 0.6404 | 0.6399 | 0.6401 | -0.0030 | -0.0076 to +0.0016 | 0.6380 | 16.4 | 1.7 |
| 30 | 0.6424 | 0.6426 | 0.6425 | -0.0006 | -0.0055 to +0.0044 | 0.6384 | 14.3 | 1.7 |
| 60 | 0.6429 | 0.6429 | 0.6429 | -0.0002 | -0.0054 to +0.0049 | 0.6396 | 11.3 | 1.8 |

Chosen: **15 cage minutes** (lowest criterion; a tie goes to the smaller value).

Weight classes with their own training mean (11): Bantamweight (581 bouts), Featherweight (652 bouts), Flyweight (278 bouts), Heavyweight (649 bouts), Light Heavyweight (636 bouts), Lightweight (1,209 bouts), Middleweight (928 bouts), Welterweight (1,193 bouts), Women's Bantamweight (177 bouts), Women's Flyweight (179 bouts), Women's Strawweight (251 bouts). Other bouts use the mean over all training bouts.

### 2. Swap-invariant level features

Candidates: `lvl_age`, `lvl_n_fights`, `lvl_elo`. Same criterion as above.

|  | XGBoost | LightGBM | criterion |
|---|---:|---:|---:|
| without level features | 0.6404 | 0.6399 | 0.6401 |
| with level features | 0.6394 | 0.6403 | 0.6399 |

Delta (with minus without): -0.0003, 95% interval -0.0011 to +0.0006. **Kept**: the criterion is lower with them.

### 3. Hyperparameter search

40 settings per library (the reference settings plus random draws from the search space in `src/ufc/models/tune.py`), one seed each, no recency weighting. The best 5 were scored again on all 5 seeds and the lowest mean chosen. The number of trees is chosen by early stopping on validation in every fit.

**LightGBM.** Reference settings: 0.6391 (one seed). Across the 40 settings, one-seed log loss ranged from 0.6358 to 0.6423 (median 0.6391). Finalists:

| trial | settings | one seed | trees | mean over seeds |
|---|---:|---:|---:|---:|
| 6 | colsample_bytree 0.6, learning_rate 0.02, min_child_samples 160, num_leaves 6, reg_lambda 3, subsample 0.6 | 0.6358 | 963 | 0.6361 |
| 18 | colsample_bytree 0.8, learning_rate 0.05, min_child_samples 160, num_leaves 12, reg_lambda 30, subsample 0.6 | 0.6364 | 269 | 0.6378 |
| 24 | colsample_bytree 0.4, learning_rate 0.05, min_child_samples 160, num_leaves 16, reg_lambda 3, subsample 0.6 | 0.6367 | 269 | 0.6374 |
| 19 | colsample_bytree 0.4, learning_rate 0.03, min_child_samples 40, num_leaves 4, reg_lambda 30, subsample 0.6 | 0.6368 | 939 | 0.6374 |
| 15 | colsample_bytree 0.6, learning_rate 0.02, min_child_samples 160, num_leaves 8, reg_lambda 30, subsample 0.7 | 0.6370 | 894 | 0.6372 |

Chosen: trial 6, 0.6361.

**XGBoost.** Reference settings: 0.6388 (one seed). Across the 40 settings, one-seed log loss ranged from 0.6362 to 0.6445 (median 0.6397). Finalists:

| trial | settings | one seed | trees | mean over seeds |
|---|---:|---:|---:|---:|
| 21 | colsample_bytree 1, learning_rate 0.03, max_depth 3, min_child_weight 40, reg_lambda 1, subsample 0.6 | 0.6362 | 393 | 0.6360 |
| 12 | colsample_bytree 0.6, learning_rate 0.03, max_depth 4, min_child_weight 40, reg_lambda 3, subsample 0.8 | 0.6374 | 528 | 0.6374 |
| 8 | colsample_bytree 0.8, learning_rate 0.02, max_depth 2, min_child_weight 1, reg_lambda 10, subsample 0.9 | 0.6378 | 1,391 | 0.6385 |
| 22 | colsample_bytree 1, learning_rate 0.03, max_depth 4, min_child_weight 40, reg_lambda 10, subsample 0.7 | 0.6379 | 488 | 0.6374 |
| 28 | colsample_bytree 0.4, learning_rate 0.03, max_depth 3, min_child_weight 10, reg_lambda 30, subsample 0.8 | 0.6379 | 809 | 0.6387 |

Chosen: trial 21, 0.6360.

### 4. Recency half-life and library

Each library's chosen settings at each half-life. Delta: log loss minus the same library with no weighting, paired over bouts.

| library | half-life | log loss | delta | 95% interval | trees |
|---|---:|---:|---:|---:|---:|
| XGBoost | none | 0.6360 | +0.0000 | +0.0000 to +0.0000 | 393 to 582 |
| XGBoost | 2 years | 0.6411 | +0.0051 | -0.0010 to +0.0116 | 182 to 406 |
| XGBoost | 4 years | 0.6380 | +0.0021 | -0.0024 to +0.0065 | 268 to 474 |
| XGBoost | 8 years | 0.6355 | -0.0004 | -0.0032 to +0.0024 | 405 to 617 |
| LightGBM | none | 0.6361 | +0.0000 | +0.0000 to +0.0000 | 656 to 963 |
| LightGBM | 2 years | 0.6400 | +0.0039 | -0.0022 to +0.0102 | 463 to 656 |
| LightGBM | 4 years | 0.6378 | +0.0017 | -0.0025 to +0.0058 | 528 to 695 |
| LightGBM | 8 years | 0.6367 | +0.0006 | -0.0020 to +0.0031 | 528 to 979 |

Selected: **XGBoost, half-life 8 years** (0.6355), the lowest of the eight.

### 5. Against the baselines

Candidate (selected library and half-life, mean over seeds, uncalibrated): **0.6355**. Delta: candidate log loss minus the baseline's, paired over bouts.

| baseline | log loss | delta | 95% interval |
|---|---:|---:|---:|
| Elo | 0.6780 | -0.0425 | -0.0573 to -0.0276 |
| Logistic regression | 0.6380 | -0.0024 | -0.0114 to +0.0060 |

The candidate has lower validation log loss than both baselines.

Logistic regression: C = 0.1, chosen on validation from 0.0003 (0.6553), 0.001 (0.6450), 0.003 (0.6401), 0.01 (0.6383), 0.03 (0.6380), 0.1 (0.6380), 0.3 (0.6380), 1 (0.6382). Elo has nothing to tune: its parameters were fit on the training period.

### 6. Calibration

The kept model is one fit (seed 0) with 405 trees. Calibration methods were compared by 5-fold cross-validation inside validation (1,017 bouts, folds made of whole event dates):

| method | cross-validated log loss |
|---|---:|
| none (uncalibrated) | 0.6364 |
| Platt scaling (slope only) | 0.6353 |
| isotonic regression | 0.6502 |

Chosen: **platt** (the lower of the two methods). Fitted on all of validation, the slope is 1.216 (below 1 pulls probabilities toward 50%, above 1 pushes them away).

### 7. Feature-family ablation

The selected recipe refitted without each family in turn, 5 seeds each. With every family: 0.6355. Delta: validation log loss without the family minus with it; positive means the family helps.

| family removed | features | log loss without | delta | 95% interval | helped |
|---|---:|---:|---:|---:|---:|
| physical | 6 | 0.6471 | +0.0115 | +0.0035 to +0.0189 | yes |
| opponent_adjusted | 20 | 0.6399 | +0.0043 | +0.0002 to +0.0083 | yes |
| opponent_quality | 4 | 0.6393 | +0.0038 | +0.0011 to +0.0066 | yes |
| striking | 30 | 0.6368 | +0.0013 | -0.0013 to +0.0038 | yes |
| rating | 1 | 0.6368 | +0.0012 | -0.0020 to +0.0045 | yes |
| activity | 2 | 0.6366 | +0.0010 | -0.0008 to +0.0028 | yes |
| cardio | 9 | 0.6365 | +0.0010 | -0.0001 to +0.0020 | yes |
| level | 3 | 0.6362 | +0.0007 | -0.0001 to +0.0014 | yes |
| debut | 1 | 0.6359 | +0.0003 | -0.0004 to +0.0011 | yes |
| weight_change | 3 | 0.6358 | +0.0002 | -0.0004 to +0.0009 | yes |
| style_matchup | 5 | 0.6356 | +0.0001 | -0.0013 to +0.0015 | yes |
| durability | 13 | 0.6354 | -0.0002 | -0.0026 to +0.0022 | no |
| grappling | 24 | 0.6347 | -0.0008 | -0.0031 to +0.0015 | no |
| record | 15 | 0.6347 | -0.0009 | -0.0036 to +0.0020 | no |

Families whose removal did not raise validation log loss: durability, grappling, record.

## Validation period

1,037 bouts from 2023-01-14 to 2024-12-14: 1,017 with a winner, 20 no result (6 draw, 14 nc). These bouts were used for every choice above, so these numbers are optimistic. The calibrated row uses calibrators fitted by cross-validation inside validation; the final calibrator is fitted on all of it.

|  | bouts | log loss | 95% interval | Brier | accuracy | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Model (calibrated) | 1,017 | 0.6353 | 0.6155 to 0.6545 | 0.2226 | 63.4% | 0.0234 |
| Model before calibration | 1,017 | 0.6364 | 0.6202 to 0.6523 | 0.2230 | 63.4% | 0.0268 |
| Logistic regression | 1,017 | 0.6380 | 0.6198 to 0.6561 | 0.2237 | 63.4% | 0.0273 |
| Elo | 1,017 | 0.6780 | 0.6662 to 0.6900 | 0.2428 | 55.4% | 0.0306 |

Accuracy counts a bout where a model gives exactly 50% as half right. Elo's accuracy is the higher-rated-fighter baseline; 31 of these bouts have equal ratings. ECE is over ten equal-width bins of the favoured fighter's probability.

Paired differences in log loss, calibrated model minus the other (negative: the model is better):

| model minus | difference | 95% interval |
|---|---:|---:|
| Elo | -0.0428 | -0.0596 to -0.0254 |
| Logistic regression | -0.0027 | -0.0123 to +0.0065 |
| Model before calibration | -0.0011 | -0.0046 to +0.0023 |

**By year**

| year | bouts | Model (calibrated) log loss | accuracy | Logistic regression log loss | accuracy | Elo log loss | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 504 | 0.6454 | 62.9% | 0.6506 | 60.9% | 0.6818 | 55.3% |
| 2024 | 513 | 0.6253 | 63.9% | 0.6256 | 65.9% | 0.6744 | 55.6% |

**By confidence of the calibrated model**

| favoured fighter's probability | bouts | mean probability | pick won | 95% interval |
|---|---:|---:|---:|---:|
| 50.0% to 55.0% | 206 | 52.2% | 50.5% | 43.7% to 57.2% |
| 55.0% to 60.0% | 201 | 57.6% | 59.7% | 52.8% to 66.2% |
| 60.0% to 65.0% | 175 | 62.4% | 57.7% | 50.3% to 64.8% |
| 65.0% to 70.0% | 166 | 67.6% | 68.7% | 61.3% to 75.2% |
| 70.0% to 80.0% | 210 | 74.0% | 75.2% | 69.0% to 80.6% |
| 80.0% to 100.0% | 59 | 83.6% | 81.4% | 69.6% to 89.3% |

**Reliability, before and after calibration** (favoured fighter's probability, ten bins; bins with no bouts omitted)

| bin | before: bouts | mean probability | pick won | after: bouts | mean probability | pick won |
|---|---:|---:|---:|---:|---:|---:|
| 50.0% to 55.0% | 241 | 52.2% | 53.1% | 206 | 52.2% | 50.5% |
| 55.0% to 60.0% | 234 | 57.4% | 56.0% | 201 | 57.6% | 59.7% |
| 60.0% to 65.0% | 200 | 62.3% | 61.5% | 175 | 62.4% | 57.7% |
| 65.0% to 70.0% | 175 | 67.2% | 74.3% | 166 | 67.6% | 68.7% |
| 70.0% to 75.0% | 98 | 72.1% | 77.6% | 135 | 72.3% | 73.3% |
| 75.0% to 80.0% | 50 | 77.2% | 78.0% | 75 | 77.2% | 78.7% |
| 80.0% to 85.0% | 16 | 82.1% | 93.8% | 43 | 82.3% | 76.7% |
| 85.0% to 90.0% | 3 | 86.8% | 100.0% | 15 | 87.0% | 93.3% |
| 90.0% to 95.0% | 0 | n/a | n/a | 1 | 92.5% | 100.0% |

**Debut bouts** (a debutant has no earlier UFC bout; "any" is one or two)

| bout has | bouts | Model (calibrated) log loss | accuracy | Logistic regression log loss | accuracy | Elo log loss | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| any debutant | 198 | 0.6677 | 57.1% | 0.6659 | 55.6% | 0.6656 | 58.6% |
| no debutant | 819 | 0.6274 | 65.0% | 0.6312 | 65.3% | 0.6811 | 54.6% |
| one debutant | 168 | 0.6505 | 62.5% | 0.6471 | 61.3% | 0.6607 | 60.1% |
| two debutants | 30 | 0.7639 | 26.7% | 0.7714 | 23.3% | 0.6931 | 50.0% |

**By weight class** (classes with at least 100 scored bouts in the period)

| weight class | bouts | Model (calibrated) log loss | accuracy | Logistic regression log loss | accuracy | Elo log loss | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bantamweight | 115 | 0.5998 | 66.1% | 0.6412 | 60.0% | 0.6735 | 49.6% |
| Featherweight | 122 | 0.6790 | 56.6% | 0.6846 | 58.2% | 0.6933 | 52.5% |
| Lightweight | 139 | 0.6410 | 61.2% | 0.6736 | 58.3% | 0.6905 | 52.2% |
| Middleweight | 113 | 0.6249 | 62.8% | 0.5957 | 66.4% | 0.6847 | 54.9% |
| Welterweight | 122 | 0.6119 | 69.7% | 0.6211 | 70.5% | 0.6715 | 59.4% |

Not shown, too few bouts: Catch Weight (15), Flyweight (80), Heavyweight (67), Light Heavyweight (66), Women's Bantamweight (39), Women's Featherweight (4), Women's Flyweight (62), Women's Strawweight (73).

**By completeness of physical attributes**

| bouts where | bouts | Model (calibrated) log loss | accuracy | Logistic regression log loss | accuracy | Elo log loss | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| height and reach known for both | 1,017 | 0.6353 | 63.4% | 0.6380 | 63.4% | 0.6780 | 55.4% |
| no attribute gap | 1,017 | 0.6353 | 63.4% | 0.6380 | 63.4% | 0.6780 | 55.4% |

## Test period

1,026 bouts from 2025-01-11 to 2026-11-14: 1,013 with a winner, 13 no result (9 draw, 4 nc). No choice above used these bouts.

|  | bouts | log loss | 95% interval | Brier | accuracy | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Model (calibrated) | 1,013 | 0.6216 | 0.6013 to 0.6424 | 0.2158 | 65.8% | 0.0412 |
| Model before calibration | 1,013 | 0.6252 | 0.6084 to 0.6424 | 0.2174 | 65.8% | 0.0485 |
| Logistic regression | 1,013 | 0.6259 | 0.6070 to 0.6454 | 0.2178 | 66.3% | 0.0418 |
| Elo | 1,013 | 0.6788 | 0.6662 to 0.6916 | 0.2429 | 56.2% | 0.0376 |

Accuracy counts a bout where a model gives exactly 50% as half right. Elo's accuracy is the higher-rated-fighter baseline; 31 of these bouts have equal ratings. ECE is over ten equal-width bins of the favoured fighter's probability.

Paired differences in log loss, calibrated model minus the other (negative: the model is better):

| model minus | difference | 95% interval |
|---|---:|---:|
| Elo | -0.0572 | -0.0747 to -0.0393 |
| Logistic regression | -0.0043 | -0.0135 to +0.0060 |
| Model before calibration | -0.0036 | -0.0070 to +0.0000 |

Symmetry test on all 1,026 test-period bouts: P(A) + P(B) is exactly 1 for every bout: yes. Presenting every bout with the fighters swapped swaps the two probabilities exactly: yes.

Test-period evaluations recorded in the registry ledger: 1, of 1 version(s): `xgboost-71fab189-44ddd3f0`.

**By year**

| year | bouts | Model (calibrated) log loss | accuracy | Logistic regression log loss | accuracy | Elo log loss | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025 | 513 | 0.6118 | 66.7% | 0.6198 | 67.3% | 0.6742 | 56.8% |
| 2026 | 500 | 0.6316 | 64.9% | 0.6320 | 65.3% | 0.6835 | 55.6% |

**By confidence of the calibrated model**

| favoured fighter's probability | bouts | mean probability | pick won | 95% interval |
|---|---:|---:|---:|---:|
| 50.0% to 55.0% | 223 | 52.6% | 50.9% | 44.4% to 57.4% |
| 55.0% to 60.0% | 200 | 57.6% | 63.0% | 56.1% to 69.4% |
| 60.0% to 65.0% | 157 | 62.8% | 59.2% | 51.4% to 66.6% |
| 65.0% to 70.0% | 154 | 67.3% | 76.0% | 68.6% to 82.0% |
| 70.0% to 80.0% | 211 | 74.4% | 76.3% | 70.1% to 81.5% |
| 80.0% to 100.0% | 68 | 83.8% | 82.4% | 71.6% to 89.6% |

**Reliability, before and after calibration** (favoured fighter's probability, ten bins; bins with no bouts omitted)

| bin | before: bouts | mean probability | pick won | after: bouts | mean probability | pick won |
|---|---:|---:|---:|---:|---:|---:|
| 50.0% to 55.0% | 260 | 52.5% | 51.0% | 223 | 52.6% | 50.9% |
| 55.0% to 60.0% | 210 | 57.2% | 67.1% | 200 | 57.6% | 63.0% |
| 60.0% to 65.0% | 208 | 62.4% | 63.5% | 157 | 62.8% | 59.2% |
| 65.0% to 70.0% | 149 | 67.3% | 76.5% | 154 | 67.3% | 76.0% |
| 70.0% to 75.0% | 107 | 72.3% | 77.6% | 125 | 72.3% | 76.8% |
| 75.0% to 80.0% | 51 | 77.0% | 80.4% | 86 | 77.4% | 75.6% |
| 80.0% to 85.0% | 25 | 82.1% | 80.0% | 45 | 82.0% | 80.0% |
| 85.0% to 90.0% | 3 | 85.2% | 100.0% | 23 | 87.2% | 87.0% |

**Debut bouts** (a debutant has no earlier UFC bout; "any" is one or two)

| bout has | bouts | Model (calibrated) log loss | accuracy | Logistic regression log loss | accuracy | Elo log loss | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| any debutant | 197 | 0.6482 | 63.7% | 0.6466 | 62.2% | 0.6657 | 60.2% |
| no debutant | 816 | 0.6152 | 66.3% | 0.6209 | 67.3% | 0.6820 | 55.3% |
| one debutant | 166 | 0.6450 | 65.1% | 0.6441 | 63.3% | 0.6605 | 62.0% |
| two debutants | 31 | 0.6654 | 56.5% | 0.6596 | 56.5% | 0.6931 | 50.0% |

**By weight class** (classes with at least 100 scored bouts in the period)

| weight class | bouts | Model (calibrated) log loss | accuracy | Logistic regression log loss | accuracy | Elo log loss | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bantamweight | 113 | 0.6337 | 63.7% | 0.6422 | 65.5% | 0.6717 | 52.7% |
| Featherweight | 125 | 0.6094 | 65.6% | 0.6115 | 66.4% | 0.6774 | 57.2% |
| Lightweight | 139 | 0.6448 | 62.9% | 0.6325 | 68.7% | 0.6950 | 51.8% |
| Middleweight | 137 | 0.6298 | 65.7% | 0.6316 | 63.5% | 0.6858 | 57.3% |
| Welterweight | 111 | 0.6357 | 64.9% | 0.6497 | 62.2% | 0.7082 | 51.4% |

Not shown, too few bouts: Catch Weight (15), Flyweight (76), Heavyweight (76), Light Heavyweight (74), Women's Bantamweight (41), Women's Featherweight (1), Women's Flyweight (44), Women's Strawweight (61).

**By completeness of physical attributes**

| bouts where | bouts | Model (calibrated) log loss | accuracy | Logistic regression log loss | accuracy | Elo log loss | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| an attribute known for one fighter only | 102 | 0.7251 | 52.5% | 0.7086 | 51.5% | 0.7310 | 49.0% |
| height and reach known for both | 903 | 0.6094 | 67.3% | 0.6156 | 68.1% | 0.6733 | 56.9% |
| height or reach missing for either | 110 | 0.7213 | 53.2% | 0.7102 | 51.4% | 0.7242 | 50.9% |
| no attribute gap | 911 | 0.6100 | 67.3% | 0.6166 | 67.9% | 0.6730 | 57.0% |

## Walk-forward backtest

Schedule for each year Y: fit through Y-3; trees and calibrator on Y-2 and Y-1; scored on Y. Refitted in every fold: Elo parameters, league baseline and class means; feature table; model; number of trees; calibrator; logistic baseline. Fixed from the selection above: library and settings; half-life; shrinkage strength; calibration method; feature list; logistic regression C. Years marked * are in the test period; the others were scored with a recipe chosen on 2023 to 2024 and are a backtest, not a held-out result.

| year | training bouts | scored | trees | model log loss | logistic | Elo | model accuracy | logistic | Elo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2013 | 980 | 376 | 119 | 0.6565 | 0.6897 | 0.6830 | 60.9% | 60.1% | 56.1% |
| 2014 | 1,235 | 494 | 181 | 0.6799 | 0.7016 | 0.6719 | 56.9% | 59.9% | 58.2% |
| 2015 | 1,511 | 464 | 227 | 0.6779 | 0.7135 | 0.6864 | 57.5% | 58.6% | 56.7% |
| 2016 | 1,837 | 483 | 85 | 0.6672 | 0.6696 | 0.6730 | 57.6% | 60.5% | 58.5% |
| 2017 | 2,243 | 446 | 229 | 0.6609 | 0.6784 | 0.6864 | 59.0% | 59.2% | 54.7% |
| 2018 | 2,631 | 469 | 500 | 0.6651 | 0.6532 | 0.6827 | 62.9% | 61.6% | 54.5% |
| 2019 | 3,069 | 506 | 287 | 0.6850 | 0.6751 | 0.7048 | 54.5% | 57.7% | 50.1% |
| 2020 | 3,497 | 444 | 398 | 0.6572 | 0.6719 | 0.6777 | 62.2% | 62.2% | 57.4% |
| 2021 | 3,961 | 497 | 154 | 0.6609 | 0.6996 | 0.6819 | 61.0% | 57.9% | 58.5% |
| 2022 | 4,466 | 506 | 199 | 0.6533 | 0.6778 | 0.6728 | 60.5% | 57.1% | 56.7% |
| 2023 | 4,910 | 504 | 146 | 0.6490 | 0.6550 | 0.6818 | 61.0% | 57.9% | 55.3% |
| 2024 | 5,407 | 513 | 277 | 0.6308 | 0.6229 | 0.6744 | 64.7% | 65.1% | 55.6% |
| 2025 * | 5,913 | 513 | 405 | 0.6118 | 0.6198 | 0.6742 | 66.7% | 67.3% | 56.8% |
| 2026 * | 6,417 | 500 | 556 | 0.6261 | 0.6312 | 0.6837 | 65.1% | 65.7% | 55.6% |

Pooled over folds:

| years |  | bouts | log loss | Brier | accuracy | ECE |
|---|---:|---:|---:|---:|---:|---:|
| before the test period (2013 to 2024) | Model (calibrated) | 5,702 | 0.6619 | 0.2348 | 59.9% | 0.0116 |
| before the test period (2013 to 2024) | Logistic regression | 5,702 | 0.6751 | 0.2396 | 59.8% | 0.0374 |
| before the test period (2013 to 2024) | Elo | 5,702 | 0.6814 | 0.2442 | 56.0% | 0.0195 |
| test period (2025 to 2026) | Model (calibrated) | 1,013 | 0.6188 | 0.2142 | 65.9% | 0.0342 |
| test period (2025 to 2026) | Logistic regression | 1,013 | 0.6254 | 0.2176 | 66.5% | 0.0356 |
| test period (2025 to 2026) | Elo | 1,013 | 0.6789 | 0.2429 | 56.2% | 0.0349 |

Check: the 2025 fold repeats the registered model's own schedule. Its probabilities are identical to the registered model's, bout for bout.

## Files

Written by the commands above, under `data/` (not committed):

- `data/reports/eval/selection.json`: every comparison in the selection sections
- `data/reports/eval/test_metrics.json`, `walk_forward.json`: the test-period and backtest sections
- `data/reports/eval/plots/validation/` and `plots/test/`: `reliability.png`, `by_confidence.png`, `by_year.png`, `debut.png`; `plots/validation/ablation.png`; `plots/walk_forward.png`
- `data/models/xgboost-71fab189-44ddd3f0/`: the model, calibrator, feature list, configuration, metrics and manifest

<!-- END GENERATED -->

## Era-shift diagnostic

Written by hand on 2026-11-17 from `data/reports/diagnostics/era_shift.json` and, for check 6,
`profile_history.json` beside it. Two commands write them, in this order:

    uv run python -m ufc.models.diagnostics
    uv run python scripts/profile_history.py    # downloads earlier commits of upstream's profile file

Question: why do 2025 and 2026 score better than every earlier backtest year, and is it a leak? The command
is analysis only. It does not select, save or change a model, and nothing below was used to change the
candidate or any selection decision. It reads test-period bouts, so each run is recorded in
`data/models/test_diagnostics.jsonl` (DECISIONS D45). Its "nothing removed" run repeats the saved
walk-forward backtest exactly (largest difference in any fold's log loss: 0), so the numbers below sit on
the same folds as the backtest table above. "Gap" means log loss pooled over 2013 to 2024 minus log loss
pooled over 2025 to 2026: +0.0431 for the model.

**Verdict: no leak found; the cause is not established.** Every check that could have exposed a leak came
back clean. What the checks do show is that the better scores are a property of the bouts from 2024 onward,
not of the model or of the test boundary: a model fitted through 2016 and never touched again shows the
same drop, and so do one-line rules with no model at all. The checks cannot tell a real change in the sport
from a defect in the source's recent rows that I have not thought of. Closing odds would (see "Not checked").

**One documented assumption turned out to be false, and it was tested directly.** Upstream does refresh
profiles: one refresh between 2025-09-30 and 2025-11-14 changed 698 of them (check 6), so most of 2025 was
scored with heights, reaches, stances and dates of birth that upstream did not hold on the day. Giving the
saved model only the profiles upstream held before the refresh moves its log loss on the 385 bouts of
January to September 2025 from 0.6161 to 0.6205 (+0.0044, 95% interval -0.0028 to +0.0115). Profile edits
made after those bouts are therefore worth at most a few thousandths, against a gap of +0.0431.

### 1. Single-feature rules by year: no leak found; the shift is visible without any model

Each feature as a rule ("pick the younger fighter"), scored over bouts where the two fighters differ on it.
The direction of each rule was set on bouts through 2022 and held fixed. The twelve rules were listed in the
script before any was scored.

| pick the | 2013 to 2018 | 2019 to 2022 | 2023 | 2024 | 2025 | 2026 | 2013 to 2024 | 2025 to 2026 | bouts, 2025 to 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| younger fighter | 56.7% | 57.3% | 58.9% | 59.1% | 60.5% | 61.3% | 57.3% | 60.9% | 999 |
| taller fighter | 50.5% | 53.3% | 54.9% | 53.3% | 56.0% | 55.1% | 52.1% | 55.6% | 741 |
| longer reach | 52.2% | 52.4% | 48.8% | 51.2% | 51.1% | 55.1% | 51.9% | 52.9% | 779 |
| higher Elo | 57.1% | 55.8% | 55.4% | 55.8% | 56.9% | 55.9% | 56.4% | 56.4% | 982 |
| more UFC fights | 52.2% | 50.2% | 46.6% | 43.8% | 44.5% | 43.8% | 50.2% | 44.2% | 921 |
| higher career win rate | 55.7% | 56.2% | 57.7% | 62.7% | 60.0% | 59.7% | 56.7% | 59.9% | 745 |
| longer win streak | 57.7% | 57.1% | 61.0% | 61.1% | 58.0% | 55.8% | 58.1% | 56.9% | 571 |
| shorter loss streak | 51.5% | 54.1% | 58.4% | 57.4% | 61.9% | 59.5% | 53.7% | 60.7% | 516 |
| shorter layoff | 50.8% | 51.6% | 49.6% | 52.8% | 53.6% | 53.1% | 51.2% | 53.3% | 780 |
| more fights in the last 2 years | 54.3% | 54.6% | 54.2% | 50.0% | 54.7% | 46.8% | 54.0% | 50.8% | 773 |
| stronger past opponents (mean Elo) | 51.8% | 50.2% | 50.0% | 51.3% | 49.4% | 47.7% | 51.0% | 48.6% | 815 |
| non-debutant, when one is a debutant | 56.1% | 61.0% | 50.0% | 57.6% | 48.1% | 44.7% | 57.2% | 46.4% | 166 |

The higher-rated fighter wins as often as ever (56.4% in both periods). The younger fighter, the fighter on
the shorter losing streak and the fighter with the better win rate win more often than they used to, and the
more experienced fighter now loses more often than not (44.2%, against 50.2%). These move from 2023 or
2024, not from 2025. None of the rules that moved is a profile field except age, and a leak through one
feature would not move several unrelated ones together. A yearly cell rests on a few hundred bouts (fewer
for the debut rule), so single cells are within a few points of noise; the pattern across rows is the
evidence, not any one cell.

### 2. Matchup difficulty by year: not more lopsided by rating; more lopsided as the model sees them

| bouts dated | bouts | median Elo gap | 90th percentile | Elo favourite, mean | mean age gap (years) | mean age | mean UFC fights | model favourite, mean | before calibration | given 70% or more | pick won | log loss the model expects | log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2013 to 2018 | 2,732 | 48.1 | 122.7 | 58.0% | 4.16 | 30.2 | 5.4 | 59.5% | 59.6% | 11.1% | 59.0% | 0.6627 | 0.6685 |
| 2019 to 2022 | 1,953 | 46.8 | 120.2 | 57.9% | 4.21 | 31.0 | 6.6 | 60.5% | 60.1% | 14.8% | 59.4% | 0.6568 | 0.6643 |
| 2023 | 504 | 48.3 | 113.6 | 57.8% | 4.28 | 31.3 | 6.9 | 60.8% | 59.0% | 12.1% | 61.0% | 0.6578 | 0.6490 |
| 2024 | 513 | 45.6 | 112.2 | 57.5% | 4.18 | 31.7 | 7.6 | 62.4% | 61.1% | 17.5% | 64.7% | 0.6477 | 0.6308 |
| 2025 | 513 | 45.8 | 115.2 | 57.8% | 4.39 | 31.9 | 7.5 | 63.7% | 61.6% | 27.1% | 66.7% | 0.6345 | 0.6118 |
| 2026 | 500 | 47.4 | 135.7 | 58.4% | 4.56 | 31.8 | 7.4 | 66.2% | 62.6% | 34.8% | 65.1% | 0.6101 | 0.6261 |
| 2013 to 2024 | 5,702 | 47.9 | 120.6 | 57.9% | 4.19 | 30.7 | 6.1 | 60.2% | 59.9% | 13.1% | 59.9% | 0.6589 | 0.6619 |
| 2025 to 2026 | 1,013 | 47.1 | 127.1 | 58.1% | 4.47 | 31.9 | 7.4 | 65.0% | 62.1% | 30.9% | 65.9% | 0.6225 | 0.6188 |

"Log loss the model expects" is the mean entropy of its own probabilities: what it would score if it were
exactly calibrated.

By rating the cards have not changed: the median Elo gap is 47.9 before and 47.1 in the test period, and
Elo's favourite averages 57.9% and 58.1%. So "recent cards are more lopsided" is false in the sense of
rating mismatches. The model, though, is more confident on recent bouts (its favourite averages 65.0%
against 60.2%; it gives 70% or more in 30.9% of bouts against 13.1%) and it is right about that: its picks
won 65.9%, and the log loss it expected of itself (0.6225) is what it got (0.6188). Rosters are older and
more experienced than they were (mean age 31.9 against 30.7, mean UFC fights 7.4 against 6.1) and age gaps
are slightly wider, which is the kind of difference the model reads and Elo does not. Part of the rise in
confidence is the calibrator, not the trees: the Platt slopes of the 2025 and 2026 folds are 1.216 and
1.347, against 0.833 to 1.138 for the folds through 2022, and in 2026 the model was overconfident (expected
0.6101, got 0.6261). As a leak test this is **inconclusive**: a model that expects to do well and does is
what both a real signal and a leaked one look like.

### 3. Attribution by removing inputs: no leak suspect

The backtest rerun with inputs removed, one fit per fold (seed 0). "Change" is the rerun's log loss minus
the full model's on the same bouts, with a 95% paired bootstrap interval.

| removed | 2013 to 2024 | 2025 to 2026 | gap | change, 2013 to 2024 | 95% interval | change, 2025 to 2026 | 95% interval |
|---|---:|---:|---:|---:|---:|---:|---:|
| nothing | 0.6619 | 0.6188 | +0.0431 | | | | |
| physical family (6 features) | 0.6697 | 0.6378 | +0.0319 | +0.0078 | +0.0047 to +0.0107 | +0.0190 | +0.0080 to +0.0293 |
| height | 0.6616 | 0.6185 | +0.0432 | -0.0003 | -0.0009 to +0.0003 | -0.0004 | -0.0022 to +0.0014 |
| reach | 0.6619 | 0.6196 | +0.0423 | -0.0001 | -0.0010 to +0.0009 | +0.0007 | -0.0024 to +0.0038 |
| stance (3 features) | 0.6619 | 0.6178 | +0.0442 | -0.0000 | -0.0008 to +0.0007 | -0.0011 | -0.0028 to +0.0005 |
| date of birth (age difference and mean age) | 0.6690 | 0.6350 | +0.0339 | +0.0070 | +0.0042 to +0.0098 | +0.0162 | +0.0058 to +0.0261 |
| every profile-sourced input (7) | 0.6698 | 0.6376 | +0.0322 | +0.0079 | +0.0048 to +0.0109 | +0.0188 | +0.0078 to +0.0285 |

No removal shrinks the gap sharply. With every input that comes from a profile removed, the gap is +0.0322
of the original +0.0431, and that model still goes from 0.6609 in 2023 to 0.6400 in 2024 and 0.6310 in
2025. Height, reach and stance make no measurable difference in either period. Age is the one profile
attribute that matters, and it is worth about twice as much in the test period (+0.0162 against +0.0070;
the intervals overlap). I do not flag it as a leak suspect: a date of birth does not change after a bout,
whether it is present is handled by the attribute-gap rule (D44), +0.0339 of the +0.0431 remains without
it, and "pick the younger fighter" rises in check 1 with no model involved. It is the attribute to look at
first if an older copy of the profiles becomes available. Removing each of the other twelve families in
turn, and the level features, leaves a gap between +0.0403 (opponent-adjusted) and +0.0459 (striking): the gap does not live in any
one family, which is what removing one overlapping family at a time would show either way.

### 4. Profile capture: no leak found

The database has no capture time for a profile. This check takes 22 Nov 2025 as upstream's last full
refresh of existing profiles (the debut cut-off in `docs/DATA_QUALITY.md`): a fighter who debuted before it
has a profile from that refresh or earlier, and a later debutant was captured when first listed. On that
reading, for a bout after the refresh between two fighters who debuted before it, both profiles were
captured before the bout and cannot hold anything from after it. Check 6 tests the reading against
upstream's history: the refresh happened a few weeks earlier (between 2025-09-30 and 2025-11-14), and no
existing profile has changed since 2025-11-14, so the groups below are what they claim to be.

| test-period bouts | bouts | model log loss | accuracy | without profile inputs | logistic regression | Elo |
|---|---:|---:|---:|---:|---:|---:|
| all | 1,013 | 0.6188 | 65.9% | 0.6376 | 0.6254 | 0.6789 |
| bout before the refresh (profiles captured after the bout) | 475 | 0.6147 | 66.3% | 0.6319 | 0.6213 | 0.6722 |
| bout after the refresh, both debuted before it (captured before the bout) | 358 | 0.5988 | 68.2% | 0.6214 | 0.6128 | 0.6813 |
| bout after the refresh, either debuted after it (captured when first listed) | 180 | 0.6697 | 60.3% | 0.6852 | 0.6613 | 0.6917 |
| the same, without attribute-gap bouts | 89 | 0.5931 | 69.7% | 0.6153 | 0.6047 | 0.6436 |
| both debuted before 2024 | 448 | 0.5912 | 68.5% | 0.6199 | 0.6096 | 0.6702 |
| either debuted in 2024 or later | 565 | 0.6407 | 63.8% | 0.6517 | 0.6380 | 0.6858 |
| either debuted in 2024 or later, without attribute-gap bouts | 463 | 0.6174 | 66.5% | 0.6291 | 0.6221 | 0.6751 |

If profiles captured after a bout were leaking its result, the 475 bouts before the refresh would score
best. They do not: the 358 bouts whose profiles were captured beforehand score better (0.5988 against
0.6147), with and without the profile inputs. The model is best where both fighters are established (0.5912
for two fighters who debuted before 2024) and worst where a recent debutant has an incomplete profile, which
is the attribute-gap group already reported above. In the test period the gap itself says nothing useful:
the fighter with the complete profile won 44.1% of the 102 one-sided bouts, against 94.0% of 282 in 2013 to
2018, and no model is fitted on such bouts in any case.

### 5. Other recency-specific paths

- **Feature build on the real data, later bouts deleted: no leak found.** Rebuilding the table with every
  bout after 2024-12-31 deleted leaves all 152 columns of the 7,974 earlier bouts identical; the same with
  a cutoff of 2025-11-21 (8,454 bouts). The existing perturbation test does this on synthetic data; this is
  the real table. Profiles cannot be truncated this way, which is why checks 3 and 4 exist.
- **Same card, real data: no leak found.** For six event dates spread from 2025-01-11 to 2026-11-14,
  deleting the other 10 to 13 bouts held that day leaves the remaining bout's features identical.
- **An old model on new bouts: the shift is in the bouts.** Each fold's model scored on every later year,
  unchanged:

  | fitted through | training bouts | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 2013 | 1,837 | 0.6780 | 0.6645 | 0.6692 | 0.6701 | 0.6649 | 0.6520 | 0.6459 | 0.6576 |
  | 2016 | 3,069 | 0.6850 | 0.6530 | 0.6741 | 0.6649 | 0.6619 | 0.6349 | 0.6182 | 0.6394 |
  | 2019 | 4,466 | | | | 0.6533 | 0.6469 | 0.6315 | 0.6179 | 0.6360 |
  | 2021 | 5,407 | | | | | | 0.6308 | 0.6170 | 0.6338 |
  | 2022 (the registered model's schedule) | 5,913 | | | | | | | 0.6118 | 0.6316 |

  A model fitted through 2016 scores 0.6182 on 2025, close to the 0.6118 of the model fitted through 2022.
  More training data explains little of the gap; the step is at 2024, inside the validation period, and
  every model sees it.
- **How large the gap is against recent years.** The test period is better than 2013 to 2024 by +0.0431
  (95% interval +0.0201 to +0.0644, each period resampled on its own), but better than 2023 to 2024 by
  +0.0209 (-0.0079 to +0.0478) and than 2024 alone by +0.0119 (-0.0202 to +0.0442). Elo's difference from
  2013 to 2024 is +0.0025 (-0.0111 to +0.0164). The test period is not distinguishable from 2024; what
  needs explaining is 2024 onward against the years before.
- **Fighter order in the source, Elo ties:** not a path. Inputs are differences and every probability is
  the average of both orientations, which the symmetry test on all test-period bouts confirms exactly.

### 6. Profile history: profiles were edited after the fact; no sign that it helps the model

`scripts/profile_history.py` downloads upstream's profile file as it was at the last commit before six
dates and compares the parsed attributes with the pinned snapshot.

| upstream commits compared | profiles in both | new profiles | profiles with any attribute changed |
|---|---:|---:|---:|
| 2024-12-11 to 2025-02-14 | 4,268 | 9 | 0 |
| 2025-02-14 to 2025-06-20 | 4,277 | 102 | 0 |
| 2025-06-20 to 2025-09-30 | 4,379 | 70 | 0 |
| 2025-09-30 to 2025-11-14 | 4,444 | 4 | 698 |
| 2025-11-14 to 2025-11-23 | 4,448 | 1 | 0 |
| 2025-11-23 to 2026-11-15 (pinned) | 4,449 | 180 | 0 |

So "captured once and never refreshed" (D14, D22, ARCHITECTURE) is wrong: there was one refresh, between
2025-09-30 and 2025-11-14, and nothing before it in 2025 or after it. Of the 791 fighters with a
test-period bout, 622 already had a profile on 2024-12-11, and 281 of those profiles differ today:

| attribute, 622 test-period fighters listed on 2024-12-11 | unchanged | empty then, filled since | value changed | empty in both |
|---|---:|---:|---:|---:|
| height | 396 | 206 | 18 | 2 |
| reach | 352 | 258 | 9 | 3 |
| stance | 367 | 251 | 1 | 3 |
| date of birth | 550 | 47 | 25 | 0 |

No value went from present to empty. Over all profiles, 37 dates of birth changed; the median change is
0.21 years, 29 are under one year and the largest is 4.9 years. Test-period results, grouped by what
happened to the two fighters' profiles after 2024-12-11:

| test-period bouts | bouts | model | without profile inputs | without date of birth | logistic regression | Elo | model accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 1,013 | 0.6188 | 0.6376 | 0.6350 | 0.6254 | 0.6789 | 65.9% |
| both profiles unchanged | 247 | 0.6162 | 0.6295 | 0.6296 | 0.6407 | 0.6740 | 65.6% |
| both listed, either profile changed | 508 | 0.6007 | 0.6242 | 0.6214 | 0.6029 | 0.6773 | 67.9% |
| either fighter not listed yet | 258 | 0.6570 | 0.6719 | 0.6672 | 0.6552 | 0.6868 | 62.2% |
| before 2025-09-30, both profiles unchanged | 113 | 0.5983 | 0.6055 | 0.6071 | 0.6076 | 0.6611 | 68.1% |
| before 2025-09-30, both listed, either changed | 241 | 0.6267 | 0.6421 | 0.6349 | 0.6301 | 0.6780 | 68.0% |
| either date of birth filled in later | 122 | 0.6242 | 0.6607 | 0.6499 | 0.6238 | 0.6923 | 65.6% |
| either date of birth changed | 74 | 0.5764 | 0.6025 | 0.5862 | 0.5944 | 0.6564 | 71.6% |

**No leak found.** The 247 bouts between fighters whose profiles have not changed since before the test
period score 0.6162, the level of the whole period and far from the 0.6619 of 2013 to 2024. The cleanest
comparison is the bouts fought before the refresh: where a profile was edited after the bout (241) the
model scores 0.6267, worse than where neither was (113 bouts, 0.5983). The 74 bouts with a changed date of
birth do score well (0.5764), but so does the model that never sees a date of birth (0.5862) and so does
Elo (0.6564, against 0.6789 overall): those are easier bouts, not bouts that the change made easier. These
groups are small and differ in who is in them, so this rules out a large effect, not a small one.

**Scored with the profiles upstream held at the time.** No existing profile changed between 2024-12-11 and
2025-09-30, so the 2025-09-30 file is what upstream held for every bout from January to September 2025 (a
fighter with no profile in it gets no attributes). The saved model, not refitted, on those 385 bouts:

| profiles | bouts | log loss | accuracy | attribute-gap bouts | age difference missing | height or reach difference missing |
|---|---:|---:|---:|---:|---:|---:|
| today's (as in the test evaluation) | 385 | 0.6161 | 67.8% | 0 | 0 | 0 |
| as upstream held them on 2025-09-30 | 385 | 0.6205 | 66.6% | 208 | 51 | 258 |

The difference is +0.0044 (95% paired interval -0.0028 to +0.0115). With profiles that cannot contain
anything from after the bouts, and with 208 of the 385 turned into attribute-gap bouts, the model scores
about the same. This is the direct test of "profiles are as-of-today data": for these nine months the
as-of-today part is worth 0.004 at most, and it includes honest loss of information, not only hindsight.
It also says a live run in 2025 would have scored close to the reported figure. Two limits: the comparison
covers 2025 only up to the refresh, and the model itself was still fitted on today's profiles for bouts
through 2022; what upstream held in those years is not known from this history.

### Not checked

- **Closing odds.** If bookmakers' favourites also won more often from 2024, the change is in the sport;
  if not, it is in this data. Historical odds are not sourced yet (baseline 3 in ML_REQUIREMENTS). This is
  the check most likely to settle the question.
- **Profile history before December 2024.** Whether earlier refreshes happened, and so whether the
  training and backtest years were fitted and scored with profiles more complete than upstream held at the
  time, was not looked at. Upstream's history goes back further. Check 6 covers January to September 2025.
- **A change in how the source records recent bouts** (stat definitions, corrections applied only to recent
  events). Nothing above would catch it.
- **Why 2024.** Older, more experienced rosters and wider age gaps are described in check 2, not shown to be
  the cause. No test here separates matchmaking from chance over three years.
- **Seed noise in check 3.** Each rerun is one seed. Changes under about 0.002 should not be read at all;
  the conclusions above rest on differences of 0.007 or more, or on their absence.
- **Families that overlap.** Removing one at a time understates each. Fitting on one family alone, by era,
  would say more about where the gap lives.

## Production refit

Written by hand on 2026-11-17 from `data/reports/eval/refit.json`, which one command writes:

    uv run python -m ufc.models refit

**From now on, the only out-of-sample evidence for the live model is predictions published before events.**
The model that goes live is `xgboost-ffc3ca8c-ee729619`. It is the recipe of the candidate above, refitted
on every completed bout through 2026-11-14 (DECISIONS D47). It has no held-out test: the validation and
test periods are now part of its training data. Every number earlier in this report describes the
candidate `xgboost-71fab189-44ddd3f0` and the recipe, not this model. (When this section was written
nothing had been approved; the user later approved this refit by hand.)

- **Fitted on:** 7,841 bouts with a winner and no attribute gap; 405 trees. Elo parameters refitted on all
  bouts (K 56, finish multiplier 1.3, against 1.15 for the candidate).
- **Calibration:** Platt slope 1.393, fitted on the out-of-time predictions of a model fitted through
  2024-12-31 (405 trees, 6,930 bouts) for the 911 bouts from 2025-01-11 to 2026-11-14 without an attribute
  gap. With the 102 attribute-gap bouts included the slope would be 1.284. The candidate's slope, fitted on
  2023 to 2024, was 1.216.
- **The mismatch the calibration carries:** on the 6,930 bouts both models were fitted on, the final
  booster's mean absolute logit is 0.531 against the calibration model's 0.510, a ratio of 1.042. Early
  stopping inside the final window would have chosen 786 trees and a ratio of 1.161; it is not used (D47).
- **Checks:** P(A) + P(B) is exactly 1 and swapping the fighters swaps the probabilities exactly, on all 560
  bouts from 2025-11-15 to 2026-11-14. The feature list (136 inputs), the feature settings apart from the
  training cut-off, and the feature table's columns and types are identical to the candidate's.
- **Against the candidate, on the 506 bouts of 2026:** correlation of P(A wins) 0.941, mean absolute
  difference 0.061 (median 0.053, 90th percentile 0.127, largest 0.219), and the two pick different
  winners in 52. The refit is more confident: its favourite averages 67.9% against the candidate's
  64.3%. The refit was fitted on these bouts, so their outcomes say nothing about which model is better,
  and no log loss is reported for it.

The refit is more confident than the candidate for two reasons: the slope fitted on 2025 to 2026, and a
booster fitted on more data. 2025 to 2026 is the period the era-shift diagnostic could not explain. If
recent cards stop being as predictable as they were there, live predictions will be overconfident, and the
calibration table of published predictions is where that will show first.


## Model revision M4.5

Written by hand on 2026-11-17 from `data/reports/revision/revision_base-schedule-age_shape-catch_weight-combined.json`
and `calibration_window.json` beside it, which two commands write:

    uv run python -m ufc.models.revision run --variants base,schedule,age_shape,catch_weight,combined
    uv run python -m ufc.models.revision calibration-window

**Trigger.** In upcoming predictions, age was the largest SHAP contribution in bouts where an older,
higher-rated fighter meets a much younger one (`scripts/explain_bouts.py`). The hypothesis: age-gap bouts
are confounded with matchmaking (declining veterans booked against prospects), so the model over-penalises
age when the older fighter is still elite. The question came from upcoming bouts; none of them is used
below. The variants and the adoption rule were registered in DECISIONS D54 after step 1 and before any
variant was run.

**Evidence and its status.** *Validation* is 2023 to 2024 with the candidate's pipeline (fitted through
2022), five seeds, Platt calibration cross-validated inside validation; variants are compared here.
*Backtest* is the walk-forward folds 2013 to 2024, seed 0, on the registered schedule; these years were
scored with a recipe chosen on 2023 to 2024, so they are a backtest, not a held-out result, but they do not
read the test period. *Secondary* is the 2025 and 2026 folds: test-period bouts that have been looked at
before (D45). They are reported for completeness, logged in `data/models/test_diagnostics.jsonl`, and used
for no decision. The unchanged recipe reproduces the registered candidate's validation probabilities and
the saved backtest exactly.

**The slice.** One fighter is at least 6 years older than the other, and his Elo rating before the bout is
higher by at least 50 (or 100) points. The mirror slice has the older fighter rated lower by as much.
Everything is from the older fighter's side: predicted is the mean probability the model gave him, won is
how often he won (95% Wilson interval), gap is predicted minus won (95% bootstrap interval; positive means
the model **over**rates the older fighter).

### 1. Is the model overconfident against elite veterans? No

The unchanged recipe (`base`):

| period | slice | bouts | predicted | won | gap | log loss |
|---|---|---:|---:|---:|---:|---:|
| validation | older, Elo +50 | 54 | 44.3% | 33.3% (22.2 to 46.6) | +0.109 (-0.009 to +0.227) | 0.6298 |
| validation | older, Elo +100 | 20 | 48.6% | 35.0% (18.1 to 56.7) | +0.136 (-0.061 to +0.335) | 0.6465 |
| validation | older, Elo -50 (mirror) | 71 | 27.2% | 14.1% (7.8 to 24.0) | +0.131 (+0.047 to +0.211) | 0.4517 |
| validation | older, Elo -100 (mirror) | 27 | 23.4% | 11.1% (3.9 to 28.1) | +0.122 (+0.008 to +0.215) | 0.3467 |
| validation | every 6-year age gap | 274 | 35.0% | 29.2% (24.1 to 34.8) | +0.058 (+0.003 to +0.111) | 0.5825 |
| backtest 2013-2024 | older, Elo +50 | 356 | 48.6% | 43.8% (38.8 to 49.0) | +0.048 (-0.004 to +0.096) | 0.6756 |
| backtest 2013-2024 | older, Elo +100 | 130 | 51.0% | 53.1% (44.5 to 61.4) | -0.021 (-0.107 to +0.059) | 0.6826 |
| backtest 2013-2024 | older, Elo -50 (mirror) | 368 | 34.7% | 27.2% (22.9 to 31.9) | +0.075 (+0.029 to +0.121) | 0.5960 |
| backtest 2013-2024 | older, Elo -100 (mirror) | 123 | 31.5% | 20.3% (14.2 to 28.3) | +0.111 (+0.042 to +0.178) | 0.5234 |
| backtest 2013-2024 | every 6-year age gap | 1,455 | 41.6% | 36.4% (34.0 to 38.9) | +0.051 (+0.028 to +0.077) | 0.6510 |
| secondary (already seen) | older, Elo +50 | 73 | 45.1% | 42.5% (31.8 to 53.9) | +0.027 (-0.075 to +0.135) | 0.6090 |
| secondary (already seen) | older, Elo +100 | 25 | 50.0% | 52.0% (33.5 to 70.0) | -0.020 (-0.219 to +0.171) | 0.7063 |

For comparison, Elo alone gives the older, higher-rated fighter (+50) 62.8% on validation and 63.1% on the
backtest, against 33.3% and 43.8% won.

**Verdict.** The hypothesis is not supported. In no period does the model underrate the older fighter
when he is rated higher. At +50 it overrates him by about 5 points on the backtest and 11 on validation,
with intervals that just include zero; at +100 the backtest is close to calibrated (-2.1 points, interval
-10.7 to +5.9). Across every bout with a 6-year age gap, and in the mirror slices, the model gives the older
fighter **too much**, with intervals that exclude zero. The large age contribution in SHAP is the model
discounting Elo, which overrates older fighters by about 19 points in this slice; if anything it discounts
too little. The slices are small (54 and 356 bouts at +50), so the data cannot rule out a modest
underrating of genuinely elite veterans at +100, but nothing points that way.

### 2. Variants

Each variant adds inputs to the candidate's recipe and changes nothing else: no re-tuning, the same
library, settings, half-life, shrinkage, level features and calibration method (DECISIONS D54, D55).

- `schedule` (proposed family 14): best win and worst loss by the opponent's rating at the time, and wins
  and losses against opponents then rated above the fighter. 4 inputs.
- `age_shape`: years past 32 and years past the class's median age (fitted on training) as differences,
  and the bout's mean age relative to the class median as a level column. 3 inputs. (Age relative to the
  median *as a difference* would equal `d_age`; D55.)
- `catch_weight`: catch-weight bouts counted as the division of the fighter's adjacent bouts in "fights at
  this weight" and the weight-change features. No new inputs.
- `combined`: all three.

Validation log loss is uncalibrated, mean over five seeds (the selection convention); deltas are variant
minus base, paired over bouts, with 95% intervals.

| variant | validation log loss | delta | backtest 2013-2024 | delta | secondary (already seen) delta |
|---|---:|---:|---:|---:|---:|
| base | 0.6355 | | 0.6619 | | |
| schedule | 0.6345 | -0.0010 (-0.0029 to +0.0007) | 0.6613 | -0.0006 (-0.0016 to +0.0003) | -0.0013 (-0.0047 to +0.0019) |
| age_shape | 0.6367 | +0.0011 (-0.0013 to +0.0036) | 0.6608 | -0.0012 (-0.0022 to -0.0002) | +0.0021 (-0.0016 to +0.0058) |
| catch_weight | 0.6362 | +0.0006 (-0.0001 to +0.0012) | 0.6618 | -0.0001 (-0.0006 to +0.0003) | -0.0007 (-0.0024 to +0.0011) |
| combined | 0.6353 | -0.0002 (-0.0029 to +0.0024) | 0.6609 | -0.0010 (-0.0022 to +0.0001) | +0.0014 (-0.0032 to +0.0060) |

The elite-veteran slice (older by 6 years, Elo +50) for each variant. Slice delta is variant minus base,
paired over the slice's bouts; "moved" is how far the variant moved the older fighter's probability.

| variant | backtest gap | backtest slice log loss | delta | moved | validation slice log loss | delta |
|---|---:|---:|---:|---:|---:|---:|
| base | +0.0480 | 0.6756 | | | 0.6298 | |
| schedule | +0.0490 | 0.6739 | -0.0018 (-0.0060 to +0.0024) | +0.0010 | 0.6273 | -0.0024 (-0.0116 to +0.0063) |
| age_shape | +0.0366 | 0.6742 | -0.0014 (-0.0065 to +0.0042) | -0.0114 | 0.6248 | -0.0050 (-0.0207 to +0.0121) |
| catch_weight | +0.0482 | 0.6739 | -0.0018 (-0.0041 to +0.0005) | +0.0003 | 0.6320 | +0.0022 (-0.0000 to +0.0046) |
| combined | +0.0369 | 0.6732 | -0.0025 (-0.0082 to +0.0035) | -0.0110 | 0.6209 | -0.0088 (-0.0239 to +0.0072) |

### 3. The pre-registered rule

A variant passes if (1) the validation interval does not lie entirely above zero, (2) on the backtest the
slice's absolute gap is smaller and its log loss lower than the base's, and (3) the slice's validation log
loss is not higher (D54).

| variant | (1) | (2) | (3) | result |
|---|---|---|---|---|
| schedule | yes | no: gap 0.0490 against 0.0480 | yes | fails |
| age_shape | yes | yes | yes | **passes** |
| catch_weight | yes | no: gap 0.0482 against 0.0480 | no: 0.6320 against 0.6298 | fails |
| combined | yes | yes | yes | passes |

Two pass. `combined` has the lower validation log loss, but against `age_shape` the difference is -0.0014
(-0.0034 to +0.0004), inside the noise, so the rule's tie-break goes to the variant with fewer inputs:
**`age_shape`**.

**How strong the pass is.** Weak. `age_shape` passes the rule as written, and the rule was written before
the result. But its validation log loss is higher than the base's by 0.0011 (inside the noise, which is
what criterion 1 allows); the slice improvements are inside their intervals; and on the already-seen 2025
to 2026 folds it is 0.0021 worse, also inside the noise. Its one clear result is on the backtest overall:
-0.0012 (-0.0022 to -0.0002) over 5,702 bouts. It also moves predictions in the opposite direction from the
hypothesis: the older fighter's probability falls by about one point in the slice, because the model now
weighs age past the class's typical age more, not less.

`schedule` is the closest miss: the lowest validation log loss of the four and lower slice log loss
everywhere before the test period, but the slice's gap grows by 0.001, so it fails criterion 2. It did not
change the slice's calibration in either direction. `catch_weight` has no measurable effect: 85 bouts in
the whole dataset are catch weights.

### 4. (d) Calibration window for the production refit

The refit's Platt slope (D47) is fitted on out-of-time predictions for 2025 to 2026. Fitting it on 2023 to
2026 instead cannot be scored on validation (there is no held-out period left for a refit), so only the
slopes and what they do to confidence are reported. All fits leave out attribute-gap bouts, as D47 does.
Spread is measured on the production booster's 500 bouts of 2026 with a winner; it was fitted on them, so
these are confidence levels, not scores. Uncalibrated, its favourite averages 63.7%.

| slope fitted on | bouts | slope | mean favourite | median | 90th percentile | at 70% or more |
|---|---:|---:|---:|---:|---:|---:|
| 2025 to 2026, model fitted through 2024 (current) | 911 | 1.393 | 68.0% | 67.2% | 83.8% | 41.2% |
| 2023 to 2026, model fitted through 2022 | 1,928 | 1.333 | 67.4% | 66.5% | 82.8% | 40.2% |
| 2023 to 2026, 2023-2024 from the 2022 model and 2025-2026 from the 2024 model | 1,928 | 1.303 | 67.1% | 66.2% | 82.3% | 39.2% |
| 2023 to 2024 only, model fitted through 2022 (the candidate's own) | 1,017 | 1.216 | 66.1% | 65.2% | 80.8% | 35.6% |

The two slopes reproduce the recorded ones (1.393 in "Production refit", 1.216 for the candidate). Widening
the window lowers the slope by 0.06 to 0.09 and the average favourite by 0.6 to 0.9 points. Every window
gives a slope above 1, so the model is underconfident before calibration in every period; the question is
only by how much, and 2025 to 2026 says most. The rolling window keeps every prediction out of time for the
model that made it and halves the weight of the two years the era-shift diagnostic could not explain.

### Recommendation (not applied)

- **Recipe:** the candidate's recipe plus the three `age_shape` inputs, everything else unchanged, as the
  pre-registered rule selects. Expect a small change: it is not shown to be better on validation, it is
  better on the 2013 to 2024 backtest by about 0.001, and it makes the model slightly *less* generous to
  older fighters. Keeping the current recipe is the alternative; the evidence for either is thin, and the
  main finding of M4.5 is that the problem it set out to fix is not in the data.
- **Calibration window, if the recipe is refitted:** the rolling 2023 to 2026 window (slope 1.303 on the
  current recipe). This is a judgment, not a measured improvement.
- **Not adopted:** `schedule`, `catch_weight`, `combined`. Their columns stay in the feature table as
  candidates and no model reads them (D55).
- Nothing has been refitted, no model has been saved, and `data/models/approved.json` is unchanged.

**Limits.** The elite slice has 54 validation bouts and 356 backtest bouts, so slice differences of a
point or two are noise, and four variants gave four chances to pass by luck. The backtest years were scored
with a recipe chosen on 2023 to 2024. Age comes from today's profiles (leakage rule 3). The hypothesis was
prompted by upcoming bouts, which the analysis does not use but which shaped the question.


## Production model after M4.5

Written by hand on 2026-11-17 from `data/reports/m45/` (`selection.json`, `test_metrics.json`,
`walk_forward.json`, `refit.json`), which three commands write:

    uv run python -m ufc.models train --recipe-of xgboost-71fab189-44ddd3f0 \
        --reports data/reports/m45 --parent-reports data/reports/eval
    uv run python -m ufc.models evaluate --reports data/reports/m45
    uv run python -m ufc.models refit --candidate xgboost-ed7a4d35-852c380d \
        --reports data/reports/m45 --calibration-window rolling

**Proposed production model: `xgboost-a586af7b-609ecab1`.** It is the production refit of candidate
`xgboost-ed7a4d35-852c380d`, which is the M4 recipe plus family 15 (age shape), with the Platt slope fitted
on the rolling 2023 to 2026 window. It is not approved; the live model is still `xgboost-ffc3ca8c-ee729619`
until `approved.json` is edited by hand (DECISIONS D58).

**Why.** The M4.5 rule, fixed before any variant was run, selects `age_shape` (D54, D56), and the user
chose to adopt it. The pass is weak, and nothing below makes it stronger: the case for this model is that
the decision was made by a pre-registered rule, not that it scores better. The rolling calibration window is
a judgment (D57): every bout in it is out of time for the model that predicted it, it has twice the bouts,
and it halves the weight of 2025 to 2026, which the era-shift diagnostic could not explain.

**The candidate `xgboost-ed7a4d35-852c380d`.** Library, settings, half-life (8 years), shrinkage (15 cage
minutes), level features and Platt calibration are the M4 candidate's; nothing was re-tuned. Fitted through
2022, 307 trees by early stopping on 2023 to 2024 (the M4 candidate has 405), 139 inputs. Its validation
log loss, uncalibrated and averaged over five seeds, is 0.6367, which is the M4.5 `age_shape` figure to
every digit recorded, so the saved candidate is the variant the rule judged. Cross-validated Platt
calibration chose Platt again (0.6360, against 0.6353 for the M4 candidate; slope 1.232). Against the logistic baseline on the same inputs:
-0.0005 (-0.0085 to +0.0074); against Elo -0.0414 (-0.0564 to -0.0265).

**Ablation, rerun with family 15.** Only opponent quality helps with an interval that excludes zero,
+0.0029 (+0.0003 to +0.0055); opponent-adjusted stats are just short of it, +0.0038 (-0.0000 to +0.0075).
Removing family 15 changes validation log loss by -0.0011 (-0.0036 to +0.0013); that is the M4.5 result
seen from the other side. **Physical drops from +0.0115 in M4 to -0.0004 (-0.0034 to +0.0026)**: with
the age-shape columns present, removing the physical family no longer removes the information about age,
so its separate contribution vanishes. This is the overlap problem the M4 ablation already warned about,
not evidence that height, reach and age stopped mattering.

**Test period, already seen.** `evaluate` was run once on this candidate. **The test period had been
opened before, for `xgboost-71fab189-44ddd3f0`, and the record says so; these numbers are not a held-out
result.** On 1,013 bouts: log loss 0.6228, Brier 0.2165, accuracy 65.1% (the M4 candidate: 0.6216, 65.8%).
Logistic regression 0.6257, Elo 0.6788. Symmetry exact on all 1,026 bouts. Walk-forward backtest, same
schedule as M4: 2013 to 2024 pooled 0.6608 (M4 recipe 0.6619); the 2025 to 2026 folds 0.6210 (M4 recipe
0.6188). The 2025 fold reproduces the candidate's probabilities exactly.

**The refit `xgboost-a586af7b-609ecab1`.**

- **Fitted on:** 7,841 bouts through 2026-11-14 with a winner and no attribute gap; 307 trees, the count
  the candidate's schedule chose. Elo parameters refitted on all bouts (K 56, finish multiplier 1.3).
- **Calibration (rolling window):** Platt slope 1.302 on 1,928 bouts: 1,017 from 2023-01-14 to 2024-12-14
  predicted by the model fitted through 2022, and 911 from 2025-01-11 to 2026-11-14 predicted by the model
  fitted through 2024. 102 attribute-gap bouts are left out; with them the slope would be 1.256.
- **Mismatch the calibration carries:** on the 6,930 bouts both were fitted on, the final booster's mean
  absolute logit is 0.513 against the 2024 calibration model's 0.491, a ratio of 1.044 (the current refit:
  1.042). Early stopping inside the final window would have chosen 663 trees and a ratio of 1.169; not used
  (D47).
- **Checks:** P(A) + P(B) is exactly 1 and swapping the fighters swaps the probabilities exactly on all 560
  bouts from 2025-11-15 to 2026-11-14. Feature list (139 inputs), feature settings apart from the training
  cut-off, and the feature table's columns and types are identical to the candidate's.
- **Against the live model `xgboost-ffc3ca8c-ee729619`, on the 506 bouts of 2026:** correlation 0.994, mean
  absolute difference 0.021 (median 0.017, 90th percentile 0.044, largest 0.090); the two pick different
  winners in 16. The new model is a little less confident: its favourite averages 66.6% against 67.9%.
  Both were fitted on these bouts, so their outcomes cannot say which is better, and no log loss is
  reported.
- **Against its own candidate, same bouts:** correlation 0.952, mean absolute difference 0.050, different
  picks in 54; favourite 66.6% against 64.2%.

**What changes for the site.** Little: on 2026 cards one bout in about 32 changes its pick, the average
probability moves by about two points, and the average favourite is 1.3 points less confident. As with the current refit, the
only out-of-sample evidence from here on is predictions published before events.


## Model revision M4.6

Written by hand on 2026-11-17 from `data/reports/m46/` (`transform.json`, `revision.json`, `ablation.json`),
which three commands write:

    uv run python -m ufc.models.revision_m46 transform
    uv run python -m ufc.models.revision_m46 run
    uv run python -m ufc.models.revision_m46 ablation

**Question.** Results are a noisy measure of performance: a fighter can lose every exchange and land one
knockout, a decision can be close or wrong, a draw can come from a point deduction, and opponent-quality
features credit a fighter for facing strong opponents even when he was beaten up. Do as-of features built
on how a fighter *performed* help the production recipe? The design, the seven columns and the adoption
rule were registered in DECISIONS D63 and committed before any variant was run.

### 0. What the data supports

- **Judges' scorecards:** yes, but only as bout totals. The details text of the results file holds each
  judge's total (for example `29 - 28`) for 4,113 of 4,169 decisions, every one from 2001 on and none
  before. There are no per-round cards. For a decision win the totals are written loser first, so they give
  the margin; for a draw they cannot be tied to a fighter.
- **Point deductions:** yes, in the same text (`Point Deducted: <foul> by <surname>`), in 34 bouts, all
  decisions, 6 of them draws.
- **Decision type** (unanimous, split, majority): yes, as the method.
- **Per-round stats:** yes, for every bout with stats; control time is missing before 2000.

So a performance model can be fitted on decisions with the judged winner as label, and scorecards can
check it, but not supply a per-round label.

### 1. The performance model

A logistic regression without intercept on A-minus-B in-fight differences, fitted on the 3,115 training
decisions with a winner (through 2022-12-31; 3,087 with control time), both orientations, L2 with C = 1,
not tuned (D63). Inputs are per 5 minutes with the fight counted as at least 5 minutes, plus per-round
dominance. Weights per unit of each input (full fit, with control time):

| input | weight per unit | standardised coefficient |
|---|---:|---:|
| significant strikes landed, per 5 min | 0.188 | 1.863 |
| knockdowns, per 5 min | 2.094 | 0.378 |
| takedowns landed, per 5 min | 0.447 | 0.467 |
| submission attempts, per 5 min | 0.750 | 0.293 |
| control minutes, per 5 min | 0.751 | 1.306 |
| (rounds led on strikes - rounds trailed) / rounds | 0.488 | 0.357 |
| (rounds led on control - rounds trailed) / rounds | -0.153 | -0.114 |

The negative last weight is the control-minutes input taking the credit for control (the two are
correlated); it is not a claim that leading rounds on control hurts. The performance Elo chose K 96 and a
provisional multiplier of 1.5, inside the grid (D63).

**Sanity check: does it pick the judged winner?**

| decisions | bouts | accuracy | log loss |
|---|---:|---:|---:|
| training (in sample) | 3,115 | 84.7% | 0.349 |
| validation 2023-2024 | 523 | 82.4% | 0.364 |
| validation, unanimous | 404 | 88.6% | 0.258 |
| validation, split | 112 | 58.9% | 0.758 |
| validation, majority | 7 | 100% | 0.175 |
| test period 2025-2026 (already seen) | 465 | 82.8% | 0.357 |

Against the judges' margin on the 523 validation decisions with three cards (mean of the three judges'
winner-minus-loser points):

| mean judges' margin | bouts | accuracy | mean deserved share of the winner |
|---|---:|---:|---:|
| 1 point or less | 217 | 65.4% | 0.616 |
| more than 1, up to 2 | 92 | 88.0% | 0.807 |
| more than 2, up to 3 | 162 | 96.3% | 0.898 |
| more than 3 | 52 | 100% | 0.969 |

Correlation of the winner's deserved share with the margin: 0.524. The model is sure where the judges were
sure and close to a coin flip where they split, which is what a measure of "who was winning" should do.

**Finishes** (stats up to the stoppage; validation): the KO/TKO winner was ahead on the judges' criteria in
94.4% of 306 bouts (mean deserved share 0.902), the submission winner in 84.9% of 185 (0.813). A finishing
knockdown is itself a recorded knockdown, so these shares include the finishing blow.

**Draws with a point deduction:** 6. In three the fighter who lost the point was clearly ahead on stats
(deserved share 0.82, 0.94 and 0.96), the case of a draw caused by a deduction; in the other three the
deducted fighter was behind (0.16, 0.27, 0.16).

### 2. Example bouts

The transform fitted through 2022; bouts from 2025 on are in the test period, already seen. These are
descriptions of the transform for the reader to judge, not evidence for any decision.

**Requested.** Both are in the data, both from UFC 319 (2025-08-16), both first-round KO/TKO wins by a
spinning back elbow:

| bout | result | stats to the stoppage | deserved share |
|---|---|---|---|
| Lerone Murphy vs Aaron Pico | Murphy, KO/TKO R1 3:21 | Murphy 8 significant, 1 knockdown, 1 sub attempt; Pico 13 significant, 2 takedowns, 2:01 control | Murphy **0.302**, Pico 0.698 |
| Carlos Prates vs Geoff Neal | Prates, KO/TKO R1 4:59 | Prates 32 significant, 1 knockdown; Neal 25 significant, 0:15 control | Prates **0.979**, Neal 0.021 |

Murphy's knockout came while Pico was winning the round on takedowns and control, so the model calls it a
finish against the run of play. Prates was ahead on volume before the knockdown that ended it.

**The 10 judged wins with the lowest deserved share** (likely robberies, or bouts the stats misread):

| date | winner | loser | method | judges (loser - winner) | deserved share of winner |
|---|---|---|---|---|---:|
| 2022-02-26 | Priscila Cachoeira | Ji Yeon Kim | unanimous | 28-29 x3 | 0.010 |
| 2020-08-01 | Nate Maness | Johnny Munoz | unanimous | 27-29 x3 | 0.011 |
| 2013-05-25 | Mike Pyle | Rick Story | split | 29-28, 28-29, 28-29 | 0.012 |
| 2023-12-16 | Irene Aldana | Karol Rosa | unanimous | 28-29 x3 | 0.015 |
| 2018-02-03 | Sergio Moraes | Tim Means | split | 29-28, 28-29, 28-29 | 0.017 |
| 2022-08-13 | Martin Buday | Lukasz Brzeski | split | 28-29, 29-28, 28-29 | 0.023 |
| 2013-06-15 | Sean Pierson | Kenny Robertson | majority | 28-29, 28-29, 28-28 | 0.025 |
| 2018-12-15 | Drakkar Klose | King Green | unanimous | 28-29 x3 | 0.025 |
| 2021-05-01 | Felipe Colares | Luke Sanders | unanimous | 28-29 x3 | 0.027 |
| 2014-06-28 | Gian Villante | Sean O'Connell | split | 28-29, 29-28, 28-29 | 0.028 |

Each has the loser ahead on significant strikes, control or both (for example Story: 57 to 33 significant
strikes and 9:29 of control against Pyle). Some will be real robberies; others will be bouts where damage
the stats do not record (strikes that hurt, near-finishes) decided rounds. All but one are in the
transform's own training period.

**The 10 KO/TKO wins with the lowest deserved share** (likely finishes against the run of play):

| date | winner | loser | end | stats to the stoppage | deserved share of winner |
|---|---|---|---|---|---:|
| 2013-08-17 | Travis Browne | Alistair Overeem | R1 4:08 | 14 to 41 significant; 1 knockdown to 2 | 0.000 |
| 2012-07-11 | Andrew Craig | Rafael Natal | R2 4:52 | 23 to 73; 1 knockdown to 2 | 0.001 |
| 2021-03-06 | Kennedy Nzechukwu | Carlos Ulberg | R2 3:19 | 82 to 146 | 0.002 |
| 2022-03-12 | Drew Dober | Terrance McKinney | R1 3:17 | 17 to 36; 1 knockdown each | 0.003 |
| 2018-10-06 | Derrick Lewis | Alexander Volkov | R3 4:49 | 39 to 121 | 0.005 |
| 2010-08-07 | Stefan Struve | Christian Morecraft | R2 0:22 | 11 to 36; 0:07 to 4:44 control | 0.005 |
| 2019-09-21 | Steven Peterson | Martin Bravo | R2 1:31 | 49 to 86 | 0.006 |
| 2026-10-17 | Joel Faglier | Javad Mahjoub | R2 2:11 | 51 to 60; 0 knockdowns to 2 | 0.007 |
| 2016-06-04 | Dan Henderson | Hector Lombard | R2 1:27 | 13 to 23; 1 knockdown each | 0.010 |
| 2014-12-20 | Daniel Sarafian | Antonio Dos Santos | R2 1:01 | 24 to 37; 0 knockdowns to 1 | 0.011 |

Most are well-known comebacks. The last is not a knockout: the dataset records a finger injury as a TKO,
and the deserved share correctly says Sarafian was behind when it ended.

### 3. Family 16 under the pre-registered rule

`base` is the recipe of `xgboost-ed7a4d35-852c380d` (M4 recipe plus family 15, the recipe of the approved
refit); `performance` adds the seven family 16 columns and changes nothing else. The base reproduces the
registered candidate's validation probabilities and the saved backtest exactly (every fold, 2013 to 2026).

| | base | performance | difference (95% interval) |
|---|---:|---:|---:|
| validation log loss, uncalibrated, mean of seeds 0-4 (**rule**) | 0.6367 | 0.6369 | +0.0002 (-0.0033 to +0.0035) |
| validation log loss, Platt cross-validated | 0.6361 | 0.6365 | +0.0004 (-0.0036 to +0.0042) |
| backtest 2013-2024, 5,702 bouts (**rule**) | 0.6608 | 0.6613 | +0.0005 (-0.0022 to +0.0033) |
| secondary 2025-2026, 1,013 bouts (already seen) | 0.6210 | 0.6203 | -0.0007 (-0.0053 to +0.0041) |

Per seed, the uncalibrated validation difference is -0.0010, +0.0009, -0.0004, +0.0019 and -0.0007: the
sign depends on the seed. By backtest year the difference ranges from -0.0108 (2017) to +0.0080 (2021):
better in 2013, 2015 to 2017 and 2024, worse in 2014 and 2018 to 2023.

| criterion (D63) | result |
|---|---|
| 1. validation difference negative, interval's upper end below +0.001 | **no**: +0.0002, upper end +0.0035 |
| 2. backtest 2013-2024 improves | **no**: +0.0005 |

**Family 16 fails both criteria and is not recommended for adoption.**

### 4. Ablation with family 16 in the model

Validation log loss without the family minus with it (positive: the family helps), seeds 0 to 4, full
model 0.6369. The M4.5 candidate's own ablation (without family 16, from `data/reports/m45/selection.json`)
is beside it.

| family | inputs | with family 16 | M4.5 candidate |
|---|---:|---:|---:|
| physical | 6 | -0.0000 (-0.0030 to +0.0029) | -0.0004 |
| record | 15 | -0.0001 (-0.0019 to +0.0019) | -0.0012 |
| striking | 30 | +0.0010 (-0.0015 to +0.0032) | +0.0007 |
| grappling | 24 | -0.0011 (-0.0032 to +0.0010) | -0.0015 |
| durability | 13 | -0.0002 (-0.0023 to +0.0020) | -0.0009 |
| activity | 2 | +0.0012 (-0.0007 to +0.0030) | +0.0003 |
| weight change | 3 | +0.0000 (-0.0006 to +0.0007) | -0.0004 |
| opponent quality | 4 | **+0.0024 (+0.0001 to +0.0048)** | +0.0029 |
| opponent-adjusted stats | 20 | **+0.0037 (+0.0004 to +0.0068)** | +0.0038 |
| style matchup | 5 | +0.0000 (-0.0012 to +0.0013) | -0.0007 |
| debut | 1 | +0.0000 (-0.0008 to +0.0009) | -0.0002 |
| cardio | 9 | +0.0001 (-0.0008 to +0.0010) | -0.0001 |
| rating | 1 | +0.0015 (-0.0004 to +0.0033) | +0.0009 |
| age shape | 3 | -0.0003 (-0.0029 to +0.0021) | -0.0011 |
| **performance (16)** | 7 | -0.0002 (-0.0035 to +0.0033) | |
| level | 3 | +0.0002 (-0.0006 to +0.0010) | +0.0001 |

Two families help with intervals that exclude zero, both with and without family 16: opponent quality and
opponent-adjusted stats. Family 16 did not take their place. In particular, the concern that opponent
quality credits a fighter for strong opponents who beat him up is not borne out as a weakness the model
can fix: the weighted deserved share against strong opponents was available and opponent quality kept its
value.

### Recommendation (not applied)

**Reject family 16.** It fails both criteria of the rule fixed before it was run, and nothing else points
the other way: the calibrated validation score, the ablation and the per-seed results all put it at zero
within noise. The columns stay in the feature table as candidates that no model reads, like family 14
(DECISIONS D55, D64). No model was saved, nothing was refitted, and `data/models/approved.json` is
unchanged.

**Why it may not help, as a hypothesis.** The family 16 columns are not copies of existing ones (on
training bouts the correlation of deserved share with win rate is 0.63 to 0.65 across the three windows,
and of the performance Elo with Elo 0.63; `transform.json`, `training_correlations`). But the model already reads the stats the performance model is made of,
per minute and opponent-adjusted, in families 3, 4 and 9. A weighted sum of those same stats adds little a
tree model cannot build itself. This was not tested.

**What the transform is good for anyway.** The deserved share is a readable, checked description of a past
bout: it picks the judged winner in 82% of validation decisions, is confident when the judges were, and
names the comebacks above. It could serve the site's fighter pages or the analyst as context. It is not a
reason to change the model.

**Limits.** Validation has 1,017 bouts, so differences under about 0.003 cannot be told from
noise; the rule asks for a negative point estimate with an upper end below +0.001, which a family with a
real but small benefit could also fail. The performance model is fitted on decisions and applied to
finishes, where the stats cover a shorter, different kind of fight; a finishing blow counts as a
knockdown. Damage that the stats do not record (a hurt opponent, a near-finish) is invisible to it. The
2025 to 2026 folds and the example bouts read test-period outcomes that had been looked at before; they
were logged and used for no decision.
