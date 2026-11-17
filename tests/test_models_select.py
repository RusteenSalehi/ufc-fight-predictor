"""Selection stages on an invented table: structure, choices, and the lock."""

import json

import numpy as np
import pytest

from tests.model_helpers import (
    FAMILIES,
    FAST,
    FEATURES,
    LEVELS,
    SHRUNK,
    SMALL,
    SPEC,
    synthetic_frame,
)
from ufc.models.calibrate import ISOTONIC, PLATT
from ufc.models.design import labels
from ufc.models.evaluate import evaluate_period
from ufc.models.gbm import LIBRARIES
from ufc.models.metrics import log_loss
from ufc.models.recipe import Recipe, fit_data, fit_recipe
from ufc.models.select import (
    _rows,
    ablate,
    attribute_gap_record,
    choose_half_life,
    choose_shrinkage,
    search,
    select,
    try_level_features,
)
from ufc.models.splits import (
    SPLIT_DATES,
    Splits,
    TestPeriodLocked,
    attribute_gap,
    development_only,
    fit_rows,
    labelled,
)
from ufc.models.tune import (
    N_TRIALS,
    REFERENCE,
    SEARCH_SPACE,
    Variant,
    difference,
    sample_params,
    score_variants,
)


@pytest.fixture(scope="module")
def frames():
    return {m: synthetic_frame(shrink=m) for m in SMALL.shrink_grid}


@pytest.fixture(scope="module")
def splits_for(frames):
    """Splits over the full tables, test period locked. Any stage that asked
    for a test-period row would raise."""
    made = {m: Splits(frame) for m, frame in frames.items()}
    return made.__getitem__


@pytest.fixture(scope="module")
def selection(splits_for):
    return select(
        splits_for, FEATURES, FAMILIES, LEVELS, SHRUNK, settings=SMALL, n_jobs=2
    )


def test_search_space_always_includes_the_reference_and_is_reproducible():
    for library in LIBRARIES:
        params = sample_params(library)
        assert len(params) == N_TRIALS and params[0] == REFERENCE[library]
        assert params == sample_params(library)
        assert len({json.dumps(p, sort_keys=True) for p in params}) == N_TRIALS
        for draw in params[1:]:
            assert all(draw[name] in SEARCH_SPACE[library][name] for name in draw)
        assert all(REFERENCE[library][k] in v for k, v in SEARCH_SPACE[library].items())
    assert sample_params("xgboost") != sample_params("lightgbm")
    with pytest.raises(ValueError):
        sample_params("xgboost", n_trials=10**6)


def test_scores_are_seed_means_and_do_not_depend_on_parallelism(splits_for):
    splits = splits_for(0.0)
    data = fit_data(
        labelled(splits.train()),
        labelled(splits.valid()),
        SPEC,
        None,
        SPLIT_DATES.train_end,
    )
    variants = [Variant(library, FAST[library], data) for library in LIBRARIES]
    serial = score_variants(variants, seeds=(0, 1), n_jobs=1)
    parallel = score_variants(variants, seeds=(0, 1), n_jobs=2)
    for library in LIBRARIES:
        score = serial[library]
        assert score.log_loss == parallel[library].log_loss
        np.testing.assert_array_equal(score.bout_losses, parallel[library].bout_losses)
        assert len(score.seed_log_losses) == len(score.n_trees) == 2
        assert score.log_loss == pytest.approx(np.mean(score.seed_log_losses))
        assert score.bout_losses.mean() == pytest.approx(score.log_loss)
        assert len(score.bout_losses) == len(data.valid_y)
    with pytest.raises(ValueError, match="unique"):
        score_variants([variants[0], variants[0]])
    delta = difference(serial["xgboost"].bout_losses, serial["lightgbm"].bout_losses)
    assert delta["delta"] == pytest.approx(
        serial["xgboost"].log_loss - serial["lightgbm"].log_loss
    )
    assert delta["delta_low"] <= delta["delta"] <= delta["delta_high"]


def test_selection_runs_every_stage_without_touching_the_test_period(
    selection, splits_for
):
    report = selection.report
    assert set(report) >= {
        "shrinkage",
        "levels",
        "search",
        "half_life",
        "logistic",
        "versus_baselines",
        "calibration",
        "selected",
        "validation",
        "ablation",
        "splits",
    }
    json.dumps(report)  # the whole record is plain JSON
    # The lock was never opened, and the full tables (test rows included)
    # were available the whole time: nothing asked for them.
    assert all(splits_for(m).test_locked for m in SMALL.shrink_grid)
    assert report["splits"] == SPLIT_DATES.to_dict()
    assert report["n_validation_scored"] < report["n_validation_bouts"]


def test_shrinkage_choice_is_the_lowest_criterion(selection):
    stage = selection.report["shrinkage"]
    grid = stage["grid"]
    assert [row["shrink_minutes"] for row in grid] == sorted(SMALL.shrink_grid)
    for row in grid:
        assert row["criterion"] == pytest.approx((row["xgboost"] + row["lightgbm"]) / 2)
        assert {
            "logistic_log_loss",
            "tails",
            "delta",
            "delta_low",
            "delta_high",
        } <= set(row)
    assert grid[0]["delta"] == 0.0  # the reference against itself
    best = min(grid, key=lambda row: (row["criterion"], row["shrink_minutes"]))
    assert stage["chosen"] == best["shrink_minutes"] == selection.shrink_minutes
    assert grid[0]["tails"]["n_features"] == len(SHRUNK)


def test_shrinkage_grid_must_include_zero_and_the_same_bouts(splits_for, frames):
    with pytest.raises(ValueError, match="include 0"):
        choose_shrinkage(splits_for, SPEC, SHRUNK, grid=(5.0, 15.0))
    shorter = {0.0: Splits(frames[0.0]), 15.0: Splits(frames[15.0].iloc[:-400])}
    with pytest.raises(ValueError, match="differ"):
        choose_shrinkage(
            shorter.__getitem__, SPEC, SHRUNK, grid=(0.0, 15.0), seeds=(0,)
        )


def test_level_features_are_kept_only_if_the_criterion_improves(selection):
    stage = selection.report["levels"]
    assert stage["features"] == list(LEVELS)
    improved = stage["with"]["criterion"] < stage["without"]["criterion"]
    assert stage["kept"] == improved
    assert stage["delta"] == pytest.approx(
        stage["with"]["criterion"] - stage["without"]["criterion"]
    )
    assert selection.recipe.spec.symmetric == (LEVELS if improved else ())
    assert selection.recipe.spec.antisymmetric == FEATURES


def test_level_features_that_carry_signal_are_kept(frames):
    """Make the outcome depend on a level column through an interaction, and
    the test picks it up."""
    frame = frames[0.0].copy()
    rng = np.random.default_rng(5)
    strength = np.where(
        frame["lvl_age"] > 0, 3.0, 0.0
    )  # Elo only matters in some bouts
    won = rng.random(len(frame)) < 1 / (1 + np.exp(-strength * frame["d_elo"] / 120))
    frame["label"] = np.where(frame["result"] == "win", won.astype(float), np.nan)
    stage = try_level_features(Splits(frame), SPEC, LEVELS, seeds=(0,))
    assert stage["kept"] and stage["delta"] < 0
    assert stage["with"]["criterion"] < stage["without"]["criterion"]


def test_search_rescoring_and_half_life_choice(selection):
    report = selection.report
    for library in LIBRARIES:
        stage = report["search"][library]
        assert len(stage["trials"]) == SMALL.n_trials
        assert len(stage["finalists"]) == SMALL.top_k
        assert stage["trials"][0]["params"] == REFERENCE[library]
        cutoff = sorted(t["log_loss"] for t in stage["trials"])[SMALL.top_k - 1]
        assert all(t["log_loss"] <= cutoff for t in stage["finalists"])
        best = min(
            stage["finalists"], key=lambda t: (t["seed_mean_log_loss"], t["trial"])
        )
        assert stage["best_trial"] == best["trial"]
        assert stage["best_params"] == best["params"]

    stage = report["half_life"]
    assert len(stage["table"]) == len(LIBRARIES) * len(SMALL.half_lives)
    best = min(stage["table"], key=lambda row: row["log_loss"])
    assert (stage["library"], stage["half_life_years"]) == (
        best["library"],
        best["half_life_years"],
    )
    for row in stage["table"]:
        if row["half_life_years"] is None:
            assert row["delta"] == 0.0
    recipe = selection.recipe
    assert recipe.gbm.library == stage["library"]
    assert recipe.half_life_years == stage["half_life_years"]
    assert dict(recipe.gbm.params) == report["search"][stage["library"]]["best_params"]


def test_candidate_is_compared_with_both_baselines(selection, splits_for):
    versus = selection.report["versus_baselines"]
    assert set(versus["baselines"]) == {"elo", "logistic"}
    assert versus["candidate_log_loss"] == selection.report["half_life"]["log_loss"]
    assert versus["beats_all"] == all(
        versus["candidate_log_loss"] < row["log_loss"]
        for row in versus["baselines"].values()
    )
    for row in versus["baselines"].values():
        assert row["delta"] == pytest.approx(
            versus["candidate_log_loss"] - row["log_loss"]
        )
    assert selection.logistic_c == selection.report["logistic"]["c"]


def test_calibration_is_chosen_by_cross_validation_and_fitted_on_validation(
    selection, splits_for
):
    stage = selection.report["calibration"]
    assert stage["chosen"] in (PLATT, ISOTONIC)
    assert (
        stage["chosen"]
        == selection.recipe.calibration
        == selection.candidate.calibrator.kind
    )
    losses = stage["cv_log_loss"]
    assert losses[stage["chosen"]] == min(losses[PLATT], losses[ISOTONIC])
    assert "predictions" not in stage

    valid = splits_for(selection.shrink_minutes).valid()
    scored = labelled(valid)
    record = selection.report["validation"]
    assert record["n_scored"] == len(scored) == stage["n_bouts"]
    raw = selection.candidate.predict(scored, calibrated=False)
    assert record["models"]["model_uncalibrated"]["log_loss"] == pytest.approx(
        log_loss(raw, labels(scored))
    )
    # The validation row for the calibrated model is the cross-validated one.
    assert record["models"]["model"]["log_loss"] == pytest.approx(
        losses[stage["chosen"]]
    )
    assert selection.report["selected"]["n_trees"] == selection.candidate.model.n_trees


def test_ablation_removes_each_family_once(selection):
    stage = selection.report["ablation"]
    expected = set(FAMILIES) | ({"level"} if selection.recipe.spec.symmetric else set())
    assert {row["family"] for row in stage["families"]} == expected
    for row in stage["families"]:
        assert row["delta"] == pytest.approx(
            row["log_loss_without"] - stage["full_log_loss"]
        )
        assert row["helped"] == (row["delta"] > 0)
        assert row["delta_low"] <= row["delta"] <= row["delta_high"]
    # The label was generated from the rating: removing it must hurt.
    rating = next(row for row in stage["families"] if row["family"] == "rating")
    assert rating["helped"] and rating["delta_low"] > 0


def leaky_frame():
    """A table in which an incomplete profile gives the outcome away, as it
    does in the real training period: the fighter with the gap always loses,
    and he is the debutant, so a model can tell which one he is."""
    frame = synthetic_frame(seed=13)
    rng = np.random.default_rng(13)
    n = len(frame)
    gap = rng.random(n) < 0.25
    a_side = rng.random(n) < 0.5
    frame["a_attr_gap"], frame["b_attr_gap"] = gap & a_side, gap & ~a_side
    frame["a_debut"], frame["b_debut"] = frame["a_attr_gap"], frame["b_attr_gap"]
    frame["d_n_fights"] = np.where(
        frame["a_attr_gap"],
        -6.0,
        np.where(frame["b_attr_gap"], 6.0, frame["d_n_fights"]),
    )
    frame.loc[gap, ["d_height_cm", "d_reach_cm"]] = np.nan
    frame.loc[~gap, ["d_height_cm", "d_reach_cm"]] = rng.normal(
        size=(int((~gap).sum()), 2)
    )
    lost = frame["a_attr_gap"].to_numpy()
    won = frame["b_attr_gap"].to_numpy()
    label = np.where(lost, 0.0, np.where(won, 1.0, frame["label"]))
    frame["label"] = np.where(frame["result"] == "win", label, np.nan)
    return frame


def test_attribute_gap_bouts_are_measured_but_never_fitted_on():
    frame = leaky_frame()
    splits = Splits(frame)
    record = attribute_gap_record(splits)
    scored = labelled(splits.train())
    assert record["n_train_scored"] == len(scored)
    assert (
        record["n_train_gap"]
        == record["n_train_one_sided"]
        == int(attribute_gap(scored).sum())
    )
    assert record["complete_profile_won"] == 1.0  # how the table was built
    assert (
        sum(p["n"] for p in record["one_sided_by_period"].values())
        == record["n_train_gap"]
    )
    assert record["n_train_fitted"] == len(scored) - record["n_train_gap"]

    train, valid = _rows(splits, "a test")
    assert len(train) == record["n_train_fitted"] and not attribute_gap(train).any()
    assert attribute_gap(valid).any()  # validation keeps every bout with a winner


def test_a_model_fitted_without_gap_bouts_cannot_use_the_gap():
    """The leak has teeth: fitted on every bout, a model learns that the
    fighter with the incomplete profile loses. Fitted the way the pipeline
    fits, it has never seen a gap go with an outcome and stays unsure."""
    frame = leaky_frame()
    splits = Splits(frame)
    valid = labelled(splits.valid())
    gap_bouts = valid[attribute_gap(valid)]
    recipe = Recipe(FAST["xgboost"], None, SPEC, PLATT)
    confidence = {}
    for name, train in (
        ("every bout", labelled(splits.train())),
        ("pipeline", fit_rows(splits.train())),
    ):
        candidate, _ = fit_recipe(recipe, train, valid, SPLIT_DATES.train_end)
        p = candidate.predict(gap_bouts, calibrated=False)
        complete_wins = np.where(gap_bouts["a_attr_gap"], 1 - p, p)
        confidence[name] = float(complete_wins.mean())
    assert confidence["every bout"] > 0.85
    assert confidence["pipeline"] < 0.70


def test_every_stage_refuses_to_run_once_the_test_period_is_open(frames):
    def unlocked():
        splits = Splits(frames[0.0])
        splits.unlock_test("some-version")
        return splits

    recipe = Recipe(FAST["xgboost"], None, SPEC, PLATT)
    params = {library: FAST[library].params for library in LIBRARIES}
    stages = [
        lambda: choose_shrinkage(lambda m: unlocked(), SPEC, SHRUNK, grid=(0.0,)),
        lambda: try_level_features(unlocked(), SPEC, LEVELS),
        lambda: search(unlocked(), SPEC),
        lambda: choose_half_life(unlocked(), SPEC, params),
        lambda: ablate(unlocked(), recipe, FAMILIES),
        lambda: select(
            lambda m: unlocked(), FEATURES, FAMILIES, LEVELS, SHRUNK, settings=SMALL
        ),
    ]
    for stage in stages:
        with pytest.raises(TestPeriodLocked, match="selection must finish"):
            stage()


def test_selection_is_the_same_with_the_test_rows_removed(selection, frames):
    """Selecting on tables that do not contain the test period at all gives
    the same recipe and the same record: the test rows play no part."""
    trimmed = {m: Splits(development_only(frame)) for m, frame in frames.items()}
    again = select(
        trimmed.__getitem__,
        FEATURES,
        FAMILIES,
        LEVELS,
        SHRUNK,
        settings=SMALL,
        n_jobs=2,
    )
    assert again.recipe == selection.recipe
    assert again.report == selection.report
    np.testing.assert_array_equal(
        again.candidate.predict(frames[0.0]), selection.candidate.predict(frames[0.0])
    )


def test_evaluate_period_counts_no_results_and_breaks_down(frames):
    frame = Splits(frames[0.0]).valid()
    scored = labelled(frame)
    rng = np.random.default_rng(0)
    predictions = {
        "model": rng.uniform(0.2, 0.8, len(scored)),
        "elo": np.full(len(scored), 0.5),
    }
    record = evaluate_period(frame, predictions, primary="model")
    assert record["n_bouts"] == len(frame)
    assert record["n_scored"] == len(scored)
    assert record["n_no_result"] == len(frame) - len(scored) > 0
    assert sum(record["no_result"].values()) == record["n_no_result"]
    assert set(record["no_result"]) <= {"draw", "nc"}
    assert set(record["by_year"]) == {"2023", "2024"}
    assert sum(g["model"]["n"] for g in record["by_year"].values()) == len(scored)
    debut = record["by_debut"]
    assert debut["no debutant"]["model"]["n"] + debut["any debutant"]["model"][
        "n"
    ] == len(scored)
    assert (
        debut["one debutant"]["model"]["n"] + debut["two debutants"]["model"]["n"]
        == debut["any debutant"]["model"]["n"]
    )
    attributes = record["by_attributes"]
    known = attributes["height and reach known for both"]["model"]["n"]
    missing = attributes["height or reach missing for either"]["model"]["n"]
    gap = attributes["an attribute known for one fighter only"]["model"]["n"]
    assert known + missing == len(scored)
    assert attributes["no attribute gap"]["model"]["n"] + gap == len(scored)
    assert 0 < gap == int(attribute_gap(scored).sum()) <= missing
    assert record["models"]["elo"]["log_loss"] == pytest.approx(np.log(2))
    assert record["models"]["elo"]["accuracy"] == 0.5
    assert record["primary_minus"]["elo"]["mean"] == pytest.approx(
        record["models"]["model"]["log_loss"] - np.log(2)
    )
    assert record["by_weight_class"] == {}  # no class reaches the minimum here
    assert sum(record["weight_classes_too_small"].values()) == len(scored)
    assert record["equal_elo"] == int((scored["d_elo"] == 0).sum())
    with pytest.raises(ValueError, match="predictions for"):
        evaluate_period(frame, {"model": predictions["model"][:-1]}, primary="model")
