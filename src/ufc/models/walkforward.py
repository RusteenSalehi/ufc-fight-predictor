"""Yearly walk-forward backtest of the selected recipe.

For each year Y the whole procedure is rerun as it would have been at the
start of Y, on the same schedule as the registered model:

- fitted feature parameters (Elo, baselines, class means): bouts through Y-3
- model fitted on bouts through Y-3
- number of trees and calibrator: bouts in Y-2 and Y-1
- scored on bouts in Y

The training window expands by one year per fold. The fold for the first
test-period year is, by construction, the registered model.

What is *not* refit: the recipe itself. The library, its settings, the
half-life, the shrinkage strength and the calibration method were chosen
once, on the 2023 to 2024 validation period. Folds before the test period are
therefore a backtest of a recipe that was chosen with knowledge of later
years, and are reported as such, apart from the test-period folds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import date

import numpy as np
from joblib import Parallel, delayed

from ufc.features.build import build_features
from ufc.features.config import FeatureConfig
from ufc.features.data import FeatureData
from ufc.models.metrics import summary
from ufc.models.recipe import Recipe, fit_recipe
from ufc.models.select import baseline_predictions
from ufc.models.splits import SPLIT_DATES, SplitDates, Splits, fit_rows, labelled

FIRST_YEAR = 2013
"""The first year scored. Its model is fitted on bouts through 2010, the
first training window with at least 1,500 bouts; earlier folds would be
fitted on too little to say anything."""

VALIDATION_YEARS = 2
MODELS: tuple[str, ...] = ("model", "model_uncalibrated", "elo", "logistic")


def fold_dates(year: int) -> SplitDates:
    """Training and validation periods for the fold that scores ``year``."""
    return SplitDates(
        train_end=date(year - VALIDATION_YEARS - 1, 12, 31),
        valid_end=date(year - 1, 12, 31),
    )


def run_fold(
    data: FeatureData,
    recipe: Recipe,
    feature_config: FeatureConfig,
    logistic_c: float,
    year: int,
    unlocked_for: str | None,
    test_dates: SplitDates = SPLIT_DATES,
) -> dict:
    """Refit everything for one year and score it.

    Rows come through ``Splits`` with the project's test period locked
    unless ``unlocked_for`` names the candidate under final evaluation, so a
    fold that would touch the test period cannot run during selection.
    """
    dates = fold_dates(year)
    build = build_features(data, replace(feature_config, train_end=dates.train_end))
    splits = Splits(build.frame, dates, test_dates=test_dates)
    if unlocked_for is not None:
        splits.unlock_test(unlocked_for)
    train, valid = fit_rows(splits.train()), labelled(splits.valid())
    scored = labelled(
        splits.between(date(year, 1, 1), date(year, 12, 31), f"walk-forward {year}")
    )
    candidate, _ = fit_recipe(recipe, train, valid, dates.train_end)
    predictions = {
        "model": candidate.predict(scored),
        "model_uncalibrated": candidate.predict(scored, calibrated=False),
        **baseline_predictions(train, scored, recipe.spec, logistic_c),
    }
    y = scored["label"].to_numpy(dtype=np.float64)
    return {
        "year": year,
        "train_end": dates.train_end.isoformat(),
        "validation": [dates.valid_start.isoformat(), dates.valid_end.isoformat()],
        "n_train": len(train),
        "n_validation": len(valid),
        "n_scored": len(scored),
        "n_trees": candidate.model.n_trees,
        "calibrator_slope": candidate.calibrator.slope,
        "elo_params": build.params.elo.to_dict(),
        "in_test_period": year >= test_dates.test_start.year,
        "metrics": {name: summary(p, y) for name, p in predictions.items()},
        "_bout_ids": list(scored["bout_id"]),
        "_y": y,
        "_predictions": predictions,
    }


def walk_forward(
    data: FeatureData,
    recipe: Recipe,
    feature_config: FeatureConfig,
    logistic_c: float,
    years: Sequence[int],
    *,
    unlocked_for: str | None = None,
    test_dates: SplitDates = SPLIT_DATES,
    n_jobs: int = 1,
) -> tuple[dict, dict[int, dict]]:
    """Run every fold. Returns the JSON-ready record and, per year, the bout
    IDs and probabilities (for checking a fold against the registered model)."""
    folds = Parallel(n_jobs=n_jobs)(
        delayed(run_fold)(
            data, recipe, feature_config, logistic_c, year, unlocked_for, test_dates
        )
        for year in years
    )
    raw = {
        fold["year"]: {
            "bout_ids": fold.pop("_bout_ids"),
            "y": fold.pop("_y"),
            "predictions": fold.pop("_predictions"),
        }
        for fold in folds
    }

    def pooled(selected: list[int]) -> dict | None:
        if not selected:
            return None
        y = np.concatenate([raw[year]["y"] for year in selected])
        return {
            "years": [selected[0], selected[-1]],
            "metrics": {
                name: summary(
                    np.concatenate(
                        [raw[year]["predictions"][name] for year in selected]
                    ),
                    y,
                )
                for name in MODELS
            },
        }

    before = [f["year"] for f in folds if not f["in_test_period"]]
    inside = [f["year"] for f in folds if f["in_test_period"]]
    record = {
        "folds": folds,
        "pooled_before_test_period": pooled(before),
        "pooled_test_period": pooled(inside),
        "schedule": "fit through Y-3; trees and calibrator on Y-2 and Y-1; scored on Y",
        "refit_per_fold": [
            "Elo parameters, league baseline and class means",
            "feature table",
            "model",
            "number of trees",
            "calibrator",
            "logistic baseline",
        ],
        "fixed_from_selection": [
            "library and settings",
            "half-life",
            "shrinkage strength",
            "calibration method",
            "feature list",
            "logistic regression C",
        ],
    }
    return record, raw
