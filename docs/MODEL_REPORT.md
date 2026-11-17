# Model report (M4)

Candidate `xgboost-71fab189-44ddd3f0`, selected on validation (2023 to 2024) and scored once on the test
period (2025-01-11 to 2026-11-14). It has **not** been approved: `data/models/approved.json` does not exist,
and creating it is a manual step (`{"version": "xgboost-71fab189-44ddd3f0"}`).

The block between the GENERATED markers is rebuilt from the metrics files by the last command below and is
replaced on every run. Everything above it is written by hand and uses only numbers from that block. The
"Era-shift diagnostic" section after the block is also written by hand, from the output of its own command.

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
