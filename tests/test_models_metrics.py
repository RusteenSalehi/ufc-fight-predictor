"""Metrics, checked by hand."""

import math

import numpy as np
import pytest

from ufc.models.metrics import (
    accuracy,
    bootstrap_mean_interval,
    bout_correct,
    bout_log_loss,
    brier,
    confidence_table,
    expected_calibration_error,
    log_loss,
    reliability_table,
    summary,
    wilson_interval,
)


def test_log_loss_brier_and_accuracy_by_hand():
    p = np.array([0.8, 0.6, 0.3, 0.5])
    y = np.array([1.0, 0.0, 0.0, 1.0])
    expected = -(math.log(0.8) + math.log(0.4) + math.log(0.7) + math.log(0.5)) / 4
    assert log_loss(p, y) == pytest.approx(expected)
    assert brier(p, y) == pytest.approx((0.04 + 0.36 + 0.09 + 0.25) / 4)
    # Right, wrong, right, and a coin flip counted as half.
    np.testing.assert_array_equal(bout_correct(p, y), [1.0, 0.0, 1.0, 0.5])
    assert accuracy(p, y) == pytest.approx(2.5 / 4)
    assert log_loss(np.array([0.5, 0.5]), np.array([1.0, 0.0])) == pytest.approx(
        math.log(2)
    )


def test_a_certain_wrong_prediction_is_finite_but_large():
    assert 30 < bout_log_loss(np.array([1.0]), np.array([0.0]))[0] < 40


def test_metrics_do_not_depend_on_which_fighter_is_called_a():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, size=500)
    y = (rng.random(500) < p).astype(float)
    for metric in (log_loss, brier, accuracy, expected_calibration_error):
        assert metric(p, y) == pytest.approx(metric(1 - p, 1 - y), abs=1e-12)
    one_way, other_way = reliability_table(p, y), reliability_table(1 - p, 1 - y)
    assert [r["n"] for r in one_way] == [r["n"] for r in other_way]
    assert [r["favourite_win_rate"] for r in one_way] == pytest.approx(
        [r["favourite_win_rate"] for r in other_way]
    )


def test_reliability_bins_are_on_the_favoured_fighter():
    # Favourite at 52% twice (one win), at 77% four times (three wins).
    p = np.array([0.52, 0.48, 0.77, 0.77, 0.23, 0.23])
    y = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    rows = reliability_table(p, y)
    assert len(rows) == 10
    assert [r["n"] for r in rows] == [2, 0, 0, 0, 0, 4, 0, 0, 0, 0]
    assert rows[0]["mean_confidence"] == pytest.approx(0.52)
    assert rows[0]["favourite_win_rate"] == pytest.approx(0.5)
    assert rows[5]["mean_confidence"] == pytest.approx(0.77)
    assert rows[5]["favourite_win_rate"] == pytest.approx(0.75)
    assert rows[1]["favourite_win_rate"] is None
    # ECE: (2 * |0.5 - 0.52| + 4 * |0.75 - 0.77|) / 6
    assert expected_calibration_error(p, y) == pytest.approx((2 * 0.02 + 4 * 0.02) / 6)


def test_bin_edges_belong_to_the_bin_above_and_one_to_the_last():
    p = np.array([0.5, 0.55, 0.7, 1.0])
    y = np.array([1.0, 1.0, 1.0, 1.0])
    rows = confidence_table(p, y)
    assert [(r["low"], r["high"]) for r in rows] == [
        (0.5, 0.55),
        (0.55, 0.6),
        (0.6, 0.65),
        (0.65, 0.7),
        (0.7, 0.8),
        (0.8, 1.0),
    ]
    assert [r["n"] for r in rows] == [1, 1, 0, 0, 1, 1]
    assert sum(r["n"] for r in reliability_table(p, y)) == 4


def test_perfectly_calibrated_probabilities_have_small_ece():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, size=200_000)
    y = (rng.random(len(p)) < p).astype(float)
    assert expected_calibration_error(p, y) < 0.01
    overconfident = np.clip(0.5 + (p - 0.5) * 1.8, 0.01, 0.99)
    assert expected_calibration_error(overconfident, y) > 0.05


def test_summary_and_empty_inputs():
    p, y = np.array([0.9, 0.2]), np.array([1.0, 0.0])
    assert set(summary(p, y)) == {"n", "log_loss", "brier", "accuracy", "ece"}
    assert summary(p, y)["n"] == 2
    assert math.isnan(log_loss(np.array([]), np.array([])))
    assert math.isnan(expected_calibration_error(np.array([]), np.array([])))


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError, match="labels"):
        log_loss(np.array([0.5]), np.array([np.nan]))
    with pytest.raises(ValueError, match="labels"):
        log_loss(np.array([0.5]), np.array([0.5]))
    with pytest.raises(ValueError, match="probabilities"):
        log_loss(np.array([1.2]), np.array([1.0]))
    with pytest.raises(ValueError):
        log_loss(np.array([0.5, 0.5]), np.array([1.0]))


def test_wilson_interval_and_bootstrap():
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3) and high == pytest.approx(
        0.5962, abs=1e-3
    )
    assert all(math.isnan(v) for v in wilson_interval(0, 0))

    values = np.random.default_rng(2).normal(0.1, 1.0, size=400)
    first = bootstrap_mean_interval(values, seed=0)
    assert first == bootstrap_mean_interval(values, seed=0)  # same seed, same answer
    assert first["low"] < first["mean"] < first["high"]
    assert first["mean"] == pytest.approx(values.mean())
    # About two standard errors either side.
    assert first["high"] - first["low"] == pytest.approx(4 * 1.0 / 20, rel=0.25)
    assert bootstrap_mean_interval(np.array([]))["n"] == 0
