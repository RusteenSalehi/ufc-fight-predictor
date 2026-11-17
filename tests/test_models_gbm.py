"""XGBoost and LightGBM wrappers, calibration, and the recipe."""

import numpy as np
import pytest

from tests.model_helpers import FAST, FEATURES, LEVELS, SPEC, synthetic_frame
from ufc.models.calibrate import (
    IDENTITY,
    ISOTONIC,
    PLATT,
    Calibrator,
    choose_method,
    cross_validated,
    fit_calibrator,
    fold_ids,
)
from ufc.models.design import FeatureSpec, both_orientations, labels, matrix, swapped
from ufc.models.gbm import LIBRARIES, MODEL_FILES, GbmConfig, GbmModel, fit_gbm
from ufc.models.metrics import log_loss
from ufc.models.recipe import Recipe, fit_data, fit_early_stopped, fit_recipe
from ufc.models.splits import SPLIT_DATES, Splits, labelled


@pytest.fixture(scope="module")
def rows():
    splits = Splits(synthetic_frame())
    return labelled(splits.train()), labelled(splits.valid())


@pytest.fixture(scope="module")
def data(rows):
    return fit_data(*rows, SPEC, None, SPLIT_DATES.train_end)


def test_training_rows_hold_both_orientations(rows, data):
    train, valid = rows
    n = len(train)
    assert data.x.shape == (2 * n, len(FEATURES))
    np.testing.assert_array_equal(data.x[n:], -data.x[:n])
    np.testing.assert_array_equal(data.y[n:], 1 - data.y[:n])
    np.testing.assert_array_equal(data.w[n:], data.w[:n])
    assert data.y.mean() == 0.5  # no trace of who was listed first
    np.testing.assert_array_equal(data.valid_ba, -data.valid_ab)
    assert len(data.valid_y) == len(valid)


def test_symmetric_columns_are_not_negated_when_fighters_swap(rows):
    train, _ = rows
    spec = FeatureSpec(FEATURES, LEVELS)
    x = matrix(train, spec)
    other = swapped(x, spec)
    k = len(FEATURES)
    np.testing.assert_array_equal(other[:, :k], -x[:, :k])
    np.testing.assert_array_equal(other[:, k:], x[:, k:])
    doubled, y, _ = both_orientations(x, labels(train), np.ones(len(x)), spec)
    np.testing.assert_array_equal(doubled[len(x) :, k:], x[:, k:])
    assert y.mean() == 0.5


@pytest.mark.parametrize("library", LIBRARIES)
def test_fit_is_reproducible_and_survives_saving(library, data, tmp_path):
    model, p_valid = fit_early_stopped(FAST[library], data)
    assert 1 <= model.n_trees < 3000
    assert log_loss(p_valid, data.valid_y) < np.log(2)  # it learned something
    again, p_again = fit_early_stopped(FAST[library], data)
    assert again.n_trees == model.n_trees
    np.testing.assert_array_equal(p_again, p_valid)

    one_sided = model.one_sided(data.valid_ab)
    assert one_sided.dtype == np.float64
    # A row's prediction does not depend on the batch it is in.
    np.testing.assert_array_equal(model.one_sided(data.valid_ab[3:4]), one_sided[3:4])

    path = model.save(tmp_path)
    assert path.name == MODEL_FILES[library]
    loaded = GbmModel.load(library, path)
    assert loaded.n_trees == model.n_trees
    np.testing.assert_array_equal(loaded.one_sided(data.valid_ab), one_sided)

    # A fixed number of trees gives exactly that many.
    fixed = fit_gbm(FAST[library], data.x, data.y, data.w, n_trees=7)
    assert fixed.n_trees == 7
    other_seed = fit_early_stopped(FAST[library].with_seed(5), data)[1]
    assert not np.array_equal(other_seed, p_valid)


def test_fit_arguments_are_checked(data):
    config = FAST["xgboost"]
    with pytest.raises(ValueError, match="either"):
        fit_gbm(config, data.x, data.y, data.w)
    with pytest.raises(ValueError, match="either"):
        fit_gbm(
            config,
            data.x,
            data.y,
            data.w,
            n_trees=5,
            valid=(data.valid_ab, data.valid_y),
        )
    with pytest.raises(ValueError, match="at least 1"):
        fit_gbm(config, data.x, data.y, data.w, n_trees=0)
    with pytest.raises(ValueError, match="unknown library"):
        GbmConfig("catboost")
    assert GbmConfig.from_dict(config.to_dict()) == config


def test_platt_scaling_recovers_a_known_distortion():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.05, 0.95, 40_000)
    y = (rng.random(len(truth)) < truth).astype(float)
    logit = np.log(truth / (1 - truth))
    overconfident = 1 / (1 + np.exp(-2.0 * logit))
    calibrator = fit_calibrator(PLATT, overconfident, y)
    assert calibrator.slope == pytest.approx(0.5, abs=0.02)
    np.testing.assert_allclose(calibrator.apply(overconfident), truth, atol=0.02)
    assert log_loss(calibrator.apply(overconfident), y) < log_loss(overconfident, y)
    # Already calibrated: the slope stays at one.
    assert fit_calibrator(PLATT, truth, y).slope == pytest.approx(1.0, abs=0.03)


def test_isotonic_regression_is_monotone_and_symmetric():
    rng = np.random.default_rng(1)
    truth = rng.uniform(0.05, 0.95, 20_000)
    y = (rng.random(len(truth)) < truth).astype(float)
    distorted = truth**2 / (truth**2 + (1 - truth) ** 2)
    calibrator = fit_calibrator(ISOTONIC, distorted, y)
    grid = np.linspace(0.5, 1.0, 101)
    assert np.all(np.diff(calibrator(grid)) >= 0)
    assert calibrator(np.array([0.5]))[0] == pytest.approx(0.5, abs=0.03)
    calibrated = calibrator.apply(distorted)
    assert np.all(calibrated + calibrator.apply(1 - distorted) == 1.0)
    assert log_loss(calibrated, y) < log_loss(distorted, y)
    assert Calibrator.from_dict(calibrator.to_dict()) == calibrator
    np.testing.assert_array_equal(
        fit_calibrator(IDENTITY, distorted, y).apply(distorted), distorted
    )
    with pytest.raises(ValueError):
        fit_calibrator("beta", distorted, y)


def test_cross_validation_keeps_an_event_date_in_one_fold():
    groups = np.repeat(np.arange(40), 5)  # 40 dates, 5 bouts each
    folds = fold_ids(groups)
    assert set(folds) == {0, 1, 2, 3, 4}
    for group in range(40):
        assert len(set(folds[groups == group])) == 1
    np.testing.assert_array_equal(folds, fold_ids(groups))  # deterministic

    rng = np.random.default_rng(2)
    truth = rng.uniform(0.1, 0.9, 200)
    y = (rng.random(200) < truth).astype(float)
    result = choose_method(truth, y, groups)
    assert result["chosen"] in (PLATT, ISOTONIC)
    assert set(result["cv_log_loss"]) == {PLATT, ISOTONIC, IDENTITY}
    assert result["cv_log_loss"][result["chosen"]] == min(
        result["cv_log_loss"][PLATT], result["cv_log_loss"][ISOTONIC]
    )
    # Each bout's cross-validated value comes from a calibrator that did not see it.
    out = cross_validated(PLATT, truth, y, folds)
    held_out = folds == 0
    expected = fit_calibrator(PLATT, truth[~held_out], y[~held_out]).apply(
        truth[held_out]
    )
    np.testing.assert_array_equal(out[held_out], expected)


@pytest.mark.parametrize("library", LIBRARIES)
def test_fit_recipe_fits_calibrates_and_is_deterministic(library, rows):
    train, valid = rows
    recipe = Recipe(FAST[library], 4.0, FeatureSpec(FEATURES, LEVELS), PLATT)
    candidate, p_valid = fit_recipe(recipe, train, valid, SPLIT_DATES.train_end)
    assert candidate.calibrator.kind == PLATT and candidate.calibrator.slope > 0
    np.testing.assert_array_equal(candidate.predict(valid, calibrated=False), p_valid)
    np.testing.assert_array_equal(
        candidate.predict(valid), candidate.calibrator.apply(p_valid)
    )
    again, p_again = fit_recipe(recipe, train, valid, SPLIT_DATES.train_end)
    np.testing.assert_array_equal(p_again, p_valid)
    assert again.calibrator == candidate.calibrator
    assert Recipe.from_dict(recipe.to_dict()) == recipe
