# Model report (M4)

Candidate `xgboost-71fab189-44ddd3f0`, selected on validation (2023 to 2024) and scored once on the test
period (2025-01-11 to 2026-11-14). It has **not** been approved: `data/models/approved.json` does not exist,
and creating it is a manual step (`{"version": "xgboost-71fab189-44ddd3f0"}`).

The block between the GENERATED markers is rebuilt from the metrics files by the last command below and is
replaced on every run. Everything above it is written by hand and uses only numbers from that block.

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
figure, and treat the first live results as the real measurement.

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
