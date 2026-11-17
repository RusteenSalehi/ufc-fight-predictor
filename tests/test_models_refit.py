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
    RECENT,
    ROLLING,
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
        list(data.bouts),
        FeatureConfig(train_end=last, shrink_minutes=15.0),
        data.fighters,
    )
    assert result.final.params.elo == own.elo


def test_calibrator_is_fitted_out_of_time_by_a_model_fitted_through_2024(data, result):
    cal = result.calibration
    assert cal["model_fitted_through"] == "2024-12-31"
    assert cal["fitted_on"][0] >= "2025-01-01"
    own, _ = fit_params(
        list(data.bouts),
        FeatureConfig(train_end=CALIBRATION_END, shrink_minutes=15),
        data.fighters,
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


# --- rolling calibration window (D57, D58) -----------------------------------------


@pytest.fixture(scope="module")
def rolling(data):
    return refit(data, RECIPE, CONFIG, "test refit", window=ROLLING)


def test_the_function_keeps_the_recent_window_by_default(result):
    assert result.calibration["window"] == RECENT
    assert len(result.calibration["segments"]) == 1
    assert result.calibration["segments"][0]["n_bouts"] == result.calibration["n_bouts"]


def test_rolling_window_adds_two_years_predicted_by_the_model_through_2022(
    data, result, rolling
):
    cal = rolling.calibration
    assert cal["window"] == ROLLING
    early, late = cal["segments"]
    assert early["predicted_by_model_fitted_through"] == "2022-12-31"
    assert "2023-01-01" <= early["dates"][0] <= early["dates"][1] <= "2024-12-31"
    assert late == result.calibration["segments"][0]
    assert cal["n_bouts"] == early["n_bouts"] + late["n_bouts"]
    assert cal["fitted_on"] == [early["dates"][0], late["dates"][1]]

    # Recomputed by hand: the 2022 model is the candidate's own schedule, on
    # its own table; the 2024 model is the one the recent window uses.
    t2022 = build_features(data, CONFIG).frame
    splits = Splits(t2022)
    m2022, _ = fit_recipe(
        RECIPE,
        fit_rows(splits.train()),
        labelled(splits.valid()),
        SPLIT_DATES.train_end,
    )
    rows22 = labelled(splits.valid())
    rows22 = rows22[~attribute_gap(rows22)]
    p22 = m2022.predict(rows22, calibrated=False)
    t2024 = build_features(
        data, FeatureConfig(train_end=CALIBRATION_END, shrink_minutes=15)
    ).frame
    train24 = fit_rows(t2024[t2024["date"] <= np.datetime64("2024-12-31")])
    rows24 = labelled(t2024[t2024["date"] > np.datetime64("2024-12-31")])
    rows24 = rows24[~attribute_gap(rows24)]
    m2024 = fit_fixed(RECIPE, train24, cal["model_n_trees"], CALIBRATION_END)
    p24 = Candidate(RECIPE, m2024, IDENTITY_CALIBRATOR).predict(
        rows24, calibrated=False
    )
    assert len(p22) == early["n_bouts"] and len(p24) == late["n_bouts"]
    slope = fit_calibrator(
        PLATT,
        np.concatenate((p22, p24)),
        np.concatenate((rows22["label"].to_numpy(), rows24["label"].to_numpy())),
    ).slope
    assert slope == pytest.approx(cal["slope"], abs=0)
    assert rolling.candidate.calibrator.slope == cal["slope"]


def test_the_window_changes_the_calibrator_and_nothing_else(result, rolling):
    frame = rolling.final.frame
    np.testing.assert_array_equal(
        rolling.candidate.predict(frame, calibrated=False),
        result.candidate.predict(frame, calibrated=False),
    )
    assert rolling.trees == result.trees
    assert rolling.sharpness == result.sharpness


def test_an_unknown_window_is_refused():
    with pytest.raises(ValueError, match="window"):
        refit(None, RECIPE, CONFIG, "x", window="everything")
