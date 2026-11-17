"""Production refit: the candidate's recipe, fitted on every completed bout.

    uv run python -m ufc.models refit

The candidate was fitted on bouts through 2022 and scored once on the test
period. The model that goes live is the same recipe fitted on everything up
to the latest event (DECISIONS D47). Nothing about the recipe changes: the
library and settings, half-life, shrinkage strength, feature list (level
features included) and calibration method are read from the candidate.

What is refitted, and on what:

- feature parameters (Elo, league baseline, weight-class means for rate
  shrinkage): every completed bout through the latest event
- the booster: every bout through the latest event with a winner and no
  attribute gap (D44)
- the Platt slope: out-of-time predictions of a second model, fitted with the
  same recipe on bouts through 2024-12-31 (its feature parameters too), on
  the bouts from 2025-01-01 on. The final booster is never calibrated on
  bouts it was fitted on.
- the number of trees, for both models: the count the candidate's own
  schedule chooses (fitted through 2022, early stopping on 2023 and 2024).
  The calibrator is fitted for a booster of that size, so the final booster
  has the same size. What early stopping inside the final window would have
  chosen, and how much sharper that booster would be, is recorded and not
  used (DECISIONS D47).

After this there is no held-out period left. The only out-of-sample evidence
for the refitted model is predictions published before events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ufc.features.build import FeatureBuild, build_features, mirror
from ufc.features.config import FeatureConfig
from ufc.features.data import FeatureData
from ufc.models.calibrate import IDENTITY_CALIBRATOR, PLATT, fit_calibrator
from ufc.models.design import both_orientations, labels, matrix
from ufc.models.gbm import GbmModel, fit_gbm
from ufc.models.recipe import Candidate, Recipe, fit_recipe
from ufc.models.splits import (
    SPLIT_DATES,
    SplitDates,
    Splits,
    attribute_gap,
    fit_rows,
    labelled,
    recency_weights,
)

logger = logging.getLogger(__name__)

STATUS = "production refit; no held-out test; recipe validated by the candidate"
CALIBRATION_END = SPLIT_DATES.valid_end
"""The calibration model is fitted on bouts through this date and its
predictions for later bouts are what the Platt slope is fitted on."""
STOPPING_YEARS = 2
"""Years at the end of a window held out to choose the number of trees, as in
the candidate's own schedule (two validation years after training)."""


def years_before(day: date, years: int) -> date:
    """The same calendar day ``years`` earlier (28 Feb for 29 Feb)."""
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(year=day.year - years, day=28)


def _build(data: FeatureData, config: FeatureConfig, train_end: date) -> FeatureBuild:
    """The feature table with every fitted parameter fitted through
    ``train_end``. Rows for later bouts are still computed, as of their own
    dates, with those parameters."""
    return build_features(data, replace(config, train_end=train_end))


def _splits(frame: pd.DataFrame, train_end: date, reason: str) -> Splits:
    """Periods of a refit table. The project's test period is open: the
    refit is fitted on it, which is why it has no held-out test."""
    last = frame["date"].max().date()
    end = max(last, train_end + timedelta(days=1))
    splits = Splits(frame, SplitDates(train_end, end), test_dates=SPLIT_DATES)
    splits.unlock_test(reason)
    return splits


@dataclass(frozen=True, slots=True)
class TreeCount:
    """The number of trees for a window, and how it was chosen."""

    n_trees: int
    window_end: date
    fitted_through: date
    stopped_on: tuple[date, date]
    n_fitted: int
    n_stopping: int

    def to_dict(self) -> dict:
        return {
            "n_trees": self.n_trees,
            "window_end": self.window_end.isoformat(),
            "fitted_through": self.fitted_through.isoformat(),
            "stopped_on": [d.isoformat() for d in self.stopped_on],
            "n_fitted_bouts": self.n_fitted,
            "n_stopping_bouts": self.n_stopping,
        }


def tree_count(
    data: FeatureData,
    recipe: Recipe,
    config: FeatureConfig,
    window_end: date,
    reason: str,
) -> tuple[TreeCount, Candidate]:
    """Early stopping inside a window: fit through ``window_end`` minus two
    years (feature parameters too), stop on the two years after. Returns the
    count and the fitted model. For a window ending 2024-12-31 this is the
    candidate's own schedule, so the count is the candidate's."""
    inner_end = years_before(window_end, STOPPING_YEARS)
    frame = _build(data, config, inner_end).frame
    splits = _splits(frame, inner_end, reason)
    train = fit_rows(splits.train())
    stop = labelled(
        splits.between(
            inner_end + timedelta(days=1),
            window_end,
            "stopping rows",
        )
    )
    model, _ = fit_recipe(recipe, train, stop, inner_end)
    count = TreeCount(
        n_trees=model.model.n_trees,
        window_end=window_end,
        fitted_through=inner_end,
        stopped_on=(
            stop["date"].min().date(),
            stop["date"].max().date(),
        ),
        n_fitted=len(train),
        n_stopping=len(stop),
    )
    return count, model


def fit_fixed(
    recipe: Recipe, rows: pd.DataFrame, n_trees: int, train_end: date
) -> GbmModel:
    """The recipe's booster on ``rows`` (a winner, no attribute gap) in both
    orientations, recency-weighted back from ``train_end``, with exactly
    ``n_trees`` trees."""
    weights = recency_weights(rows["date"], train_end, recipe.half_life_years)
    x, y, w = both_orientations(
        matrix(rows, recipe.spec), labels(rows), weights, recipe.spec
    )
    return fit_gbm(recipe.gbm, x, y, w, n_trees=n_trees)


@dataclass(frozen=True, slots=True)
class Refit:
    candidate: Candidate  # the final booster with the out-of-time calibrator
    final: FeatureBuild  # feature table and parameters fitted on everything
    final_rows: pd.DataFrame  # the bouts the final booster was fitted on
    calibration: dict  # how the calibrator was fitted, as JSON
    trees: dict  # how each model's number of trees was chosen, as JSON
    sharpness: dict  # final booster against the calibration model


def _mean_abs_logit(model: GbmModel, frame: pd.DataFrame, recipe: Recipe) -> float:
    """Mean |logit| of the favoured fighter's uncalibrated probability."""
    p = Candidate(recipe, model, IDENTITY_CALIBRATOR).predict(frame, calibrated=False)
    q = np.clip(np.maximum(p, 1 - p), 1e-12, 1 - 1e-12)
    return float(np.mean(np.log(q) - np.log1p(-q)))


def refit(
    data: FeatureData,
    recipe: Recipe,
    config: FeatureConfig,
    reason: str,
    calibration_end: date = CALIBRATION_END,
) -> Refit:
    """Fit the production model. ``recipe`` and ``config`` come from the
    candidate; only ``config.train_end`` is replaced."""
    if recipe.calibration != PLATT:
        raise ValueError("the production refit implements Platt calibration only")
    last = max(b.date for b in data.bouts)
    if last <= calibration_end:
        raise ValueError("no completed bouts after the calibration model's window")

    # 1. The calibration model: everything fitted through calibration_end.
    cal_count, _ = tree_count(data, recipe, config, calibration_end, reason)
    cal_build = _build(data, config, calibration_end)
    cal_splits = _splits(cal_build.frame, calibration_end, reason)
    cal_train = fit_rows(cal_splits.train())
    cal_model = fit_fixed(recipe, cal_train, cal_count.n_trees, calibration_end)
    later = labelled(cal_splits.valid())  # every labelled bout after the window
    p_later = Candidate(recipe, cal_model, IDENTITY_CALIBRATOR).predict(
        later, calibrated=False
    )
    y_later = later["label"].to_numpy(dtype=np.float64)
    gap = attribute_gap(later).to_numpy()
    calibrator = fit_calibrator(PLATT, p_later[~gap], y_later[~gap])
    with_gaps = fit_calibrator(PLATT, p_later, y_later)

    # 2. The final booster: everything fitted through the latest event, with
    # as many trees as the model the calibrator was fitted for.
    final = _build(data, config, last)
    final_rows = fit_rows(_splits(final.frame, last, reason).train())
    final_model = fit_fixed(recipe, final_rows, cal_count.n_trees, last)
    # Recorded, not used: early stopping inside the final window.
    stopped_count, _ = tree_count(data, recipe, config, last, reason)
    stopped_model = fit_fixed(recipe, final_rows, stopped_count.n_trees, last)

    # How much sharper the final booster is than the one the slope was fitted
    # for, on the bouts both were fitted on, each from its own feature table.
    shared = final_rows[final_rows["bout_id"].isin(cal_train["bout_id"])]
    assert len(shared) == len(cal_train)
    sharpness = {
        "n_bouts": len(shared),
        "calibration_model_mean_abs_logit": _mean_abs_logit(
            cal_model, cal_train, recipe
        ),
        "final_model_mean_abs_logit": _mean_abs_logit(final_model, shared, recipe),
        "not_used_stopped_model_mean_abs_logit": _mean_abs_logit(
            stopped_model, shared, recipe
        ),
    }
    base = sharpness["calibration_model_mean_abs_logit"]
    sharpness["ratio"] = sharpness["final_model_mean_abs_logit"] / base
    sharpness["not_used_stopped_model_ratio"] = (
        sharpness["not_used_stopped_model_mean_abs_logit"] / base
    )

    calibration = {
        "method": PLATT,
        "slope": calibrator.slope,
        "model_fitted_through": calibration_end.isoformat(),
        "model_n_trees": cal_count.n_trees,
        "model_n_fitted_bouts": len(cal_train),
        "fitted_on": [
            later["date"].min().date().isoformat(),
            later["date"].max().date().isoformat(),
        ],
        "n_bouts": int((~gap).sum()),
        "n_attribute_gap_left_out": int(gap.sum()),
        "slope_if_attribute_gap_bouts_were_included": with_gaps.slope,
        "elo_params": cal_build.params.elo.to_dict(),
    }
    logger.info(
        "Platt slope %.4f on %d out-of-time bouts; final booster %d trees on %d bouts",
        calibrator.slope,
        calibration["n_bouts"],
        final_model.n_trees,
        len(final_rows),
    )
    return Refit(
        candidate=Candidate(recipe, final_model, calibrator),
        final=final,
        final_rows=final_rows,
        calibration=calibration,
        trees={
            "rule": (
                "the candidate's schedule chooses the count (fit through 2022, "
                "early stopping on 2023 and 2024); the calibration model and the "
                "final booster both have that many trees"
            ),
            "calibration_model": cal_count.to_dict(),
            "final_model": {"n_trees": final_model.n_trees},
            "not_used_stopping_in_final_window": stopped_count.to_dict(),
        },
        sharpness=sharpness,
    )


# --- checks ----------------------------------------------------------------------


def symmetry_check(candidate: Candidate, frame: pd.DataFrame) -> dict:
    """P(A) + P(B) is exactly 1, and presenting every bout the other way
    round swaps the two probabilities exactly."""
    p_a, p_b = candidate.predict_pair(frame)
    q_a, q_b = candidate.predict_pair(mirror(frame))
    return {
        "n_bouts": len(frame),
        "first_date": frame["date"].min().date().isoformat(),
        "last_date": frame["date"].max().date().isoformat(),
        "sum_is_exactly_one": bool(np.all(p_a + p_b == 1.0)),
        "swap_is_exact": bool(np.array_equal(p_a, q_b) and np.array_equal(p_b, q_a)),
    }


def schema_check(
    refit_spec: dict,
    candidate_spec: dict,
    refit_pipeline: dict,
    candidate_pipeline: dict,
    refit_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
) -> dict:
    """The refit reads the same inputs, built the same way, as the candidate.
    Only ``train_end`` (and so the fitted parameters) may differ."""
    keys = ("context_columns", "flag_columns", "features", "families")
    differing_keys = [k for k in keys if refit_pipeline[k] != candidate_pipeline[k]]
    config_r = {k: v for k, v in refit_pipeline["config"].items() if k != "train_end"}
    config_c = {
        k: v for k, v in candidate_pipeline["config"].items() if k != "train_end"
    }
    same_columns = list(refit_frame.columns) == list(candidate_frame.columns)
    same_dtypes = same_columns and bool(
        (refit_frame.dtypes == candidate_frame.dtypes).all()
    )
    result = {
        "feature_list_identical": refit_spec == candidate_spec,
        "n_features": len(refit_spec["features"]),
        "pipeline_settings_identical_except_train_end": config_r == config_c,
        "pipeline_layout_identical": not differing_keys,
        "table_columns_identical": same_columns,
        "table_dtypes_identical": same_dtypes,
    }
    result["all"] = all(v for k, v in result.items() if k != "n_features")
    return result


def compare_predictions(
    refit_candidate: Candidate,
    refit_frame: pd.DataFrame,
    old: Candidate,
    old_frame: pd.DataFrame,
    year: int,
) -> dict:
    """Both models' calibrated P(A wins) on every bout of ``year``, each from
    its own feature table. Outcomes are not used; the refit was fitted on
    these bouts, so comparing their log loss would mean nothing."""
    mine = refit_frame[refit_frame["date"].dt.year == year]
    theirs = old_frame[old_frame["date"].dt.year == year]
    if list(mine["bout_id"]) != list(theirs["bout_id"]):
        theirs = theirs.set_index("bout_id").loc[mine["bout_id"]].reset_index()
    p_new = refit_candidate.predict(mine)
    p_old = old.predict(theirs)
    diff = np.abs(p_new - p_old)
    return {
        "year": year,
        "n_bouts": len(mine),
        "correlation": float(np.corrcoef(p_new, p_old)[0, 1]),
        "mean_abs_difference": float(diff.mean()),
        "median_abs_difference": float(np.median(diff)),
        "p90_abs_difference": float(np.quantile(diff, 0.9)),
        "largest_abs_difference": float(diff.max()),
        "pick_differs": int(np.sum((p_new > 0.5) != (p_old > 0.5))),
        "mean_favoured_refit": float(np.maximum(p_new, 1 - p_new).mean()),
        "mean_favoured_candidate": float(np.maximum(p_old, 1 - p_old).mean()),
    }
