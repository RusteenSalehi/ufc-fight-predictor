"""The symmetry test of docs/ML_REQUIREMENTS.md, on fitted models.

For any bout, the final P(A) + P(B) is exactly 1, and swapping the inputs
swaps the outputs. Checked with exact equality for both baselines and both
candidate libraries, with and without the swap-invariant level features, and
with each calibration method.
"""

import numpy as np
import pytest

from tests.model_helpers import FAST, FEATURES, LEVELS, SPEC, mirrored, synthetic_frame
from ufc.models.baselines import ELO_SPEC, EloModel, fit_logistic
from ufc.models.calibrate import IDENTITY, ISOTONIC, PLATT
from ufc.models.design import FeatureSpec, labels, matrix, predict_pair
from ufc.models.gbm import LIBRARIES
from ufc.models.recipe import Recipe, fit_recipe
from ufc.models.splits import SPLIT_DATES, Splits, labelled


@pytest.fixture(scope="module")
def frame():
    return synthetic_frame(seed=3)


@pytest.fixture(scope="module")
def rows(frame):
    splits = Splits(frame)
    return labelled(splits.train()), labelled(splits.valid())


def assert_symmetric(pair_of, frame):
    """``pair_of(frame)`` returns (P(A), P(B)) for every bout in a frame."""
    p_a, p_b = pair_of(frame)
    assert np.all(p_a + p_b == 1.0), "P(A) + P(B) must be exactly 1"
    assert np.all((p_a > 0) & (p_a < 1))
    q_a, q_b = pair_of(mirrored(frame))
    np.testing.assert_array_equal(q_a, p_b)
    np.testing.assert_array_equal(q_b, p_a)
    # The model has an opinion: this is not passing by predicting 0.5.
    assert p_a.std() > 0.01
    # One bout on its own gets the same answer as inside the batch.
    one_a, one_b = pair_of(frame.iloc[11:12])
    assert (one_a[0], one_b[0]) == (p_a[11], p_b[11])


def test_elo_baseline_is_symmetric(frame):
    assert_symmetric(lambda f: predict_pair(EloModel(), f, ELO_SPEC), frame)


def test_logistic_baseline_is_symmetric(frame, rows):
    train, _ = rows
    model = fit_logistic(matrix(train, SPEC), labels(train), 0.1, SPEC)
    assert_symmetric(lambda f: predict_pair(model, f, SPEC), frame)


@pytest.mark.parametrize("library", LIBRARIES)
@pytest.mark.parametrize(
    "with_levels", [False, True], ids=["differences", "with_levels"]
)
@pytest.mark.parametrize("calibration", [IDENTITY, PLATT, ISOTONIC])
def test_candidate_models_are_symmetric(frame, rows, library, with_levels, calibration):
    spec = FeatureSpec(FEATURES, LEVELS if with_levels else ())
    recipe = Recipe(FAST[library], None, spec, calibration)
    candidate, _ = fit_recipe(recipe, *rows, SPLIT_DATES.train_end)
    assert_symmetric(candidate.predict_pair, frame)
    assert_symmetric(lambda f: candidate.predict_pair(f, calibrated=False), frame)
    np.testing.assert_array_equal(
        candidate.predict(frame), candidate.predict_pair(frame)[0]
    )


@pytest.mark.parametrize("library", LIBRARIES)
def test_level_features_change_predictions_but_not_symmetry(frame, rows, library):
    """A model that uses the level columns really does read them, and is
    still symmetric: they cannot favour a side."""
    spec = FeatureSpec(FEATURES, LEVELS)
    candidate, _ = fit_recipe(
        Recipe(FAST[library], None, spec, PLATT), *rows, SPLIT_DATES.train_end
    )
    shifted = frame.copy()
    shifted[list(LEVELS)] = shifted[list(LEVELS)] + 5.0
    assert not np.array_equal(candidate.predict(shifted), candidate.predict(frame))
    assert_symmetric(candidate.predict_pair, shifted)


def test_one_sided_outputs_alone_are_not_symmetric(frame, rows):
    """Why the averaging exists: a tree model asked once does not give
    complementary answers for the two orientations."""
    candidate, _ = fit_recipe(
        Recipe(FAST["xgboost"], None, SPEC, IDENTITY), *rows, SPLIT_DATES.train_end
    )
    x = matrix(frame, SPEC)
    u, v = candidate.model.one_sided(x), candidate.model.one_sided(-x)
    assert not np.all(u + v == 1.0)
