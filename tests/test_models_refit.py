"""Production refit on an invented league: what is fitted on which bouts.

The league's winners are random, so nothing here checks that the refit is a
good model; it checks the schedule (D47).
"""

from datetime import date

import numpy as np
import pytest

from tests.feature_helpers import random_league
from tests.model_helpers import FAST
from ufc.features.build import build_features, fit_params
from ufc.features.config import FeatureConfig
from ufc.features.matchup import FEATURE_NAMES, LEVEL_NAMES
from ufc.models.calibrate import IDENTITY_CALIBRATOR, ISOTONIC, PLATT, fit_calibrator
from ufc.models.design import FeatureSpec
from ufc.models.recipe import Candidate, Recipe, fit_recipe
from ufc.models.refit import (
    CALIBRATION_END,
    STATUS,
    compare_predictions,
    fit_fixed,
    refit,
    schema_check,
    symmetry_check,
    years_before,
)
from ufc.models.splits import SPLIT_DATES, Splits, attribute_gap, fit_rows, labelled

RECIPE = Recipe(FAST["xgboost"], 8.0, FeatureSpec(FEATURE_NAMES, LEVEL_NAMES), PLATT)
CONFIG = FeatureConfig(shrink_minutes=15.0)


@pytest.fixture(scope="module")
def data():
    league = random_league(seed=5, n_fighters=60, n_events=150, start=date(2018, 1, 6))
    return league.data()


@pytest.fixture(scope="module")
def result(data):
    return refit(data, RECIPE, CONFIG, "test refit")


def test_years_before_keeps_the_calendar_day():
    assert years_before(date(2026, 11, 14), 2) == date(2024, 11, 14)
    assert years_before(date(2024, 12, 31), 2) == date(2022, 12, 31)
    assert years_before(date(2028, 2, 29), 2) == date(2026, 2, 28)
    assert CALIBRATION_END == date(2024, 12, 31)
    assert "no held-out test" in STATUS


def test_final_booster_is_fitted_on_every_bout_without_a_gap(data, result):
    last = max(b.date for b in data.bouts)
    frame = result.final.frame
    assert frame["date"].max().date() == last
    assert result.final.store.config.train_end == last
    expected = frame[frame["label"].notna() & ~attribute_gap(frame)]
    assert list(result.final_rows["bout_id"]) == list(expected["bout_id"])
    assert result.final_rows["date"].max() >= np.datetime64("2026-01-01")
    own, _ = fit_params(
        list(data.bouts), FeatureConfig(train_end=last, shrink_minutes=15.0)
    )
    assert result.final.params.elo == own.elo


def test_calibrator_is_fitted_out_of_time_by_a_model_fitted_through_2024(data, result):
    cal = result.calibration
    assert cal["model_fitted_through"] == "2024-12-31"
    assert cal["fitted_on"][0] >= "2025-01-01"
    own, _ = fit_params(
        list(data.bouts), FeatureConfig(train_end=CALIBRATION_END, shrink_minutes=15)
    )
    assert cal["elo_params"] == own.elo.to_dict()

    # Recomputed by hand: same table, same model, same bouts, same slope.
    frame = build_features(
        data, FeatureConfig(train_end=CALIBRATION_END, shrink_minutes=15)
    ).frame
    train = fit_rows(frame[frame["date"] <= np.datetime64("2024-12-31")])
    later = labelled(frame[frame["date"] > np.datetime64("2024-12-31")])
    later = later[~attribute_gap(later)]
    assert cal["n_bouts"] == len(later) > 0
    model = fit_fixed(RECIPE, train, cal["model_n_trees"], CALIBRATION_END)
    p = Candidate(RECIPE, model, IDENTITY_CALIBRATOR).predict(later, calibrated=False)
    assert fit_calibrator(PLATT, p, later["label"].to_numpy()).slope == pytest.approx(
        cal["slope"], abs=0
    )
    assert result.candidate.calibrator.slope == cal["slope"]


def test_calibration_models_tree_count_is_the_candidates_own_schedule(data, result):
    """Window ending 2024-12-31: fit through 2022, stop on 2023 and 2024."""
    frame = build_features(data, CONFIG).frame
    splits = Splits(frame)
    candidate, _ = fit_recipe(
        RECIPE,
        fit_rows(splits.train()),
        labelled(splits.valid()),
        SPLIT_DATES.train_end,
    )
    trees = result.trees["calibration_model"]
    assert trees["fitted_through"] == "2022-12-31"
    assert trees["n_trees"] == candidate.model.n_trees
    # The final booster has as many trees as the model the slope was fitted for.
    assert result.trees["final_model"]["n_trees"] == trees["n_trees"]
    assert result.candidate.model.n_trees == trees["n_trees"]
    unused = result.trees["not_used_stopping_in_final_window"]
    assert unused["window_end"] == result.final.frame["date"].max().date().isoformat()
    assert set(result.sharpness) >= {"ratio", "not_used_stopped_model_ratio"}


def test_refit_only_implements_platt():
    with pytest.raises(ValueError, match="Platt"):
        refit(None, Recipe(FAST["xgboost"], 8.0, RECIPE.spec, ISOTONIC), CONFIG, "x")


def test_checks_on_the_refit(data, result):
    frame = result.final.frame
    recent = frame[frame["date"] >= np.datetime64("2026-01-01")]
    sym = symmetry_check(result.candidate, recent)
    assert sym["sum_is_exactly_one"] and sym["swap_is_exact"]
    assert sym["n_bouts"] == len(recent) > 0

    old_frame = build_features(data, CONFIG).frame
    spec = result.candidate.spec.to_dict()
    meta = {
        "config": {"train_end": "2026-01-01", "shrink_minutes": 15.0},
        "context_columns": [1],
        "flag_columns": [2],
        "features": [3],
        "families": {},
    }
    old_meta = {**meta, "config": {"train_end": "2022-12-31", "shrink_minutes": 15.0}}
    assert schema_check(spec, spec, meta, old_meta, frame, old_frame)["all"]
    other = {**old_meta, "config": {"train_end": "x", "shrink_minutes": 5.0}}
    assert not schema_check(spec, spec, meta, other, frame, old_frame)["all"]
    fewer = FeatureSpec(FEATURE_NAMES[:-1]).to_dict()
    assert not schema_check(fewer, spec, meta, old_meta, frame, old_frame)["all"]

    same = compare_predictions(result.candidate, frame, result.candidate, frame, 2026)
    assert same["correlation"] == pytest.approx(1.0)
    assert same["mean_abs_difference"] == 0.0 and same["pick_differs"] == 0
