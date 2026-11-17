"""Elo and logistic-regression baselines."""

import numpy as np
import pytest

from tests.model_helpers import FEATURES, SPEC, synthetic_frame
from ufc.features.elo import expected_score
from ufc.models.baselines import (
    CLIP_SD,
    ELO_SPEC,
    EloModel,
    fit_logistic,
    logistic_from_dict,
)
from ufc.models.design import FeatureSpec, labels, matrix, predict, swapped
from ufc.models.metrics import accuracy, log_loss
from ufc.models.select import baseline_predictions, tune_logistic
from ufc.models.splits import Splits, labelled


@pytest.fixture(scope="module")
def rows():
    splits = Splits(synthetic_frame())
    return labelled(splits.train()), labelled(splits.valid())


def test_elo_baseline_is_the_rating_formula_on_the_stored_difference(rows):
    _, valid = rows
    difference = valid["d_elo"].to_numpy()
    one_sided = EloModel().one_sided(matrix(valid, ELO_SPEC))
    np.testing.assert_allclose(
        one_sided, [expected_score(d, 0.0) for d in difference], rtol=1e-12
    )
    p = predict(EloModel(), valid, ELO_SPEC)
    # Equal ratings are a coin flip; the higher-rated fighter is the pick.
    assert np.all(p[difference == 0] == 0.5)
    assert np.all((p > 0.5) == (difference > 0))
    assert (difference == 0).any()
    higher_rated_won = np.where(
        difference == 0, 0.5, (difference > 0) == (valid["label"] == 1)
    ).mean()
    assert accuracy(p, labels(valid)) == pytest.approx(higher_rated_won)
    with pytest.raises(ValueError, match="rating difference only"):
        EloModel().one_sided(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="missing"):
        EloModel().one_sided(np.array([[np.nan]]))


def test_logistic_regression_has_no_intercept_and_is_exactly_odd(rows):
    train, valid = rows
    model = fit_logistic(matrix(train, SPEC), labels(train), 0.1, SPEC)
    x = matrix(valid, SPEC)
    np.testing.assert_array_equal(model.score(swapped(x, SPEC)), -model.score(x))
    assert model.score(np.zeros((1, len(FEATURES))))[0] == 0.0
    # All features missing: no known difference, so a coin flip.
    assert model.one_sided(np.full((1, len(FEATURES)), np.nan))[0] == 0.5
    # A bout's score does not depend on what else is in the batch.
    assert model.score(x[7:8])[0] == model.score(x)[7]


def test_logistic_regression_standardizes_with_training_scale_and_clips(rows):
    train, _ = rows
    x = matrix(train, SPEC)
    model = fit_logistic(x, labels(train), 0.1, SPEC)
    np.testing.assert_allclose(model.scale, np.sqrt(np.nanmean(x**2, axis=0)))
    # An absurd value counts as CLIP_SD standard deviations, no more.
    row = np.zeros((1, len(FEATURES)))
    row[0, 0] = 1e9
    capped = np.zeros((1, len(FEATURES)))
    capped[0, 0] = CLIP_SD * model.scale[0]
    assert model.score(row)[0] == pytest.approx(model.score(capped)[0])
    restored = logistic_from_dict(model.to_dict(SPEC.names), SPEC.names)
    np.testing.assert_array_equal(restored.score(x), model.score(x))


def test_logistic_regression_finds_the_signal_and_c_is_chosen_on_validation(rows):
    train, valid = rows
    model, record = tune_logistic(train, valid, SPEC)
    coef = dict(zip(FEATURES, model.coef, strict=True))
    assert coef["d_elo"] > 0 > coef["d_age"]  # how the labels were generated
    losses = [row["log_loss"] for row in record["grid"]]
    assert record["log_loss"] == min(losses) < np.log(2)
    assert record["c"] == record["grid"][losses.index(min(losses))]["c"]

    predictions = baseline_predictions(train, valid, SPEC, record["c"])
    assert set(predictions) == {"elo", "logistic"}
    assert log_loss(predictions["logistic"], labels(valid)) == pytest.approx(
        record["log_loss"]
    )


def test_symmetric_features_are_refused_by_the_logistic_baseline(rows):
    train, _ = rows
    spec = FeatureSpec(FEATURES, ("lvl_age",))
    with pytest.raises(ValueError, match="zero coefficient"):
        fit_logistic(matrix(train, spec), labels(train), 0.1, spec)
    # The tuning helper leaves them out by itself.
    model, _ = tune_logistic(train, rows[1], spec)
    assert len(model.coef) == len(FEATURES)


def test_feature_spec_refuses_outcomes_and_duplicates():
    for bad in ("label", "result", "bout_id", "date"):
        with pytest.raises(ValueError, match="not allowed"):
            FeatureSpec(("d_elo", bad))
    with pytest.raises(ValueError, match="twice"):
        FeatureSpec(("d_elo", "d_elo"))
    with pytest.raises(ValueError, match="at least one"):
        FeatureSpec((), ("lvl_age",))
    spec = FeatureSpec(("d_elo", "d_age"), ("lvl_age",))
    assert spec.names == ("d_elo", "d_age", "lvl_age")
    assert spec.without(["d_age", "lvl_age"]) == FeatureSpec(("d_elo",))
    assert FeatureSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ValueError, match="disagree"):
        FeatureSpec.from_dict(
            {**spec.to_dict(), "features": ["d_age", "d_elo", "lvl_age"]}
        )
