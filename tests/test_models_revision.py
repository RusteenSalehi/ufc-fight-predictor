"""Model revision M4.5: the slice arithmetic (pure functions, invented numbers)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ufc.models.revision import (
    adoption,
    logit_shift,
    older_view,
    paired_slice_change,
    slice_masks,
    slice_metrics,
)


def _frame(d_age, d_elo, label, p) -> pd.DataFrame:
    return pd.DataFrame(
        {"d_age": d_age, "d_elo": d_elo, "label": label, "p": p}, dtype=float
    )


def test_older_view_takes_the_older_fighters_side():
    frame = _frame([7.0, -7.0], [80.0, 80.0], [1.0, 1.0], [0.4, 0.4])
    view = older_view(frame, frame["p"].to_numpy())
    # Bout 0: A is older, rated 80 higher, given 40%, won.
    assert view.loc[0].tolist() == [7.0, 80.0, 0.4, 1.0]
    # Bout 1: B is older and rated 80 lower; B was given 60% and lost.
    assert view.loc[1].tolist() == [7.0, -80.0, pytest.approx(0.6), 0.0]


def test_older_view_is_the_same_whichever_fighter_is_listed_first():
    frame = _frame([8.0], [120.0], [0.0], [0.3])
    swapped = _frame([-8.0], [-120.0], [1.0], [0.7])
    a = older_view(frame, frame["p"].to_numpy())
    b = older_view(swapped, swapped["p"].to_numpy())
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


def test_slice_masks():
    view = pd.DataFrame(
        {
            "age_gap": [6.0, 6.0, 5.9, 10.0, 7.0, np.nan],
            "elo_gap": [50.0, 100.0, 200.0, -60.0, -100.0, 300.0],
        }
    )
    masks = slice_masks(view)
    assert masks["all"].tolist() == [True] * 5 + [False]
    assert masks["age_gap"].tolist() == [True, True, False, True, True, False]
    assert masks["elite_veteran_+50"].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert masks["elite_veteran_+100"].tolist() == [
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert masks["mirror_-50"].tolist() == [False, False, False, True, True, False]
    assert masks["mirror_-100"].tolist() == [False, False, False, False, True, False]


def test_logit_shift_recovers_a_known_offset():
    rng = np.random.default_rng(0)
    z = rng.normal(0.0, 1.0, 200_000)
    y = (rng.random(len(z)) < 1 / (1 + np.exp(-(z + 0.4)))).astype(float)
    p = 1 / (1 + np.exp(-z))
    assert logit_shift(p, y) == pytest.approx(0.4, abs=0.02)


def test_logit_shift_of_a_calibrated_slice_is_zero_and_undefined_without_both_outcomes():
    p = np.array([0.5, 0.5])
    assert logit_shift(p, np.array([1.0, 0.0])) == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(logit_shift(p, np.array([1.0, 1.0])))


def test_slice_metrics_signs():
    p = np.full(100, 0.3)
    y = np.r_[np.ones(60), np.zeros(40)]
    m = slice_metrics(p, y)
    assert m["n"] == 100
    assert m["gap"] == pytest.approx(-0.3)  # underrated by 30 points
    assert m["gap_interval"][1] < 0
    assert m["logit_shift"] > 0
    low, high = m["win_rate_interval"]
    assert low < 0.6 < high
    assert slice_metrics(np.array([]), np.array([])) == {"n": 0}


def test_paired_slice_change_reports_movement_toward_the_older_fighter():
    frame = _frame([8.0, -8.0], [100.0, -100.0], [1.0, 0.0], [0.4, 0.6])
    frame["base"] = frame["p"]
    frame["v"] = [0.5, 0.5]  # moves both bouts 10 points toward the older fighter
    change = paired_slice_change(frame, "v", "base")
    row = change["elite_veteran_+100"]
    assert row["n"] == 2
    assert row["shift_toward_older"] == pytest.approx(0.1)
    assert row["log_loss_delta"] < 0
    assert row["gap_reference"] == pytest.approx(-0.6)
    assert row["gap_variant"] == pytest.approx(-0.5)


def _slice(gap: float, log_loss: float) -> dict:
    return {"gap": gap, "log_loss": log_loss}


def _record(variants: dict) -> tuple[dict, pd.DataFrame]:
    """A run record with the D54 rule's inputs, and per-seed validation
    columns so that the tie-break can be computed."""
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.5).astype(float)
    valid = pd.DataFrame({"label": y})
    record = {
        "variants": {},
        "validation": {"variants": {}, "slices": {}},
        "backtest_2013_2024": {"slices": {}},
    }
    for name, spec in variants.items():
        p = np.clip(0.5 + spec["edge"] * (y - 0.5), 0.01, 0.99)
        valid[f"{name}|0|raw"] = p
        record["variants"][name] = {
            "antisymmetric_added": ["x"] * spec["inputs"],
            "symmetric_added": [],
            "feature_settings": {},
        }
        record["validation"]["variants"][name] = {
            "raw": {"log_loss": spec["val_ll"], "delta_interval": [spec["low"], 0.01]}
        }
        record["validation"]["slices"][name] = {
            "elite_veteran_+50": _slice(0.1, spec["val_slice_ll"])
        }
        record["backtest_2013_2024"]["slices"][name] = {
            "elite_veteran_+50": _slice(spec["gap"], spec["back_slice_ll"])
        }
    return record, valid


BASE = {
    "inputs": 0,
    "edge": 0.10,
    "val_ll": 0.64,
    "low": 0.0,
    "gap": 0.05,
    "back_slice_ll": 0.68,
    "val_slice_ll": 0.63,
}


def test_adoption_applies_each_criterion():
    record, valid = _record(
        {
            "base": BASE,
            # worse beyond noise on validation
            "worse": {
                **BASE,
                "inputs": 1,
                "low": 0.001,
                "gap": 0.01,
                "back_slice_ll": 0.6,
            },
            # better calibrated but higher slice loss on the backtest
            "lossier": {**BASE, "inputs": 1, "gap": 0.01, "back_slice_ll": 0.69},
            # slice loss up on validation
            "val_slice": {
                **BASE,
                "inputs": 1,
                "gap": 0.01,
                "back_slice_ll": 0.6,
                "val_slice_ll": 0.64,
            },
            # over-corrects: |gap| the same size in the other direction
            "flipped": {**BASE, "inputs": 1, "gap": -0.05, "back_slice_ll": 0.6},
            "good": {**BASE, "inputs": 2, "gap": -0.02, "back_slice_ll": 0.6},
        }
    )
    result = adoption(record, valid)
    passes = {n: r["passes"] for n, r in result["variants"].items()}
    assert passes == {
        "worse": False,
        "lossier": False,
        "val_slice": False,
        "flipped": False,
        "good": True,
    }
    assert result["recommended"] == "good"


def test_adoption_tie_break_goes_to_fewer_inputs_inside_noise():
    record, valid = _record(
        {
            "base": BASE,
            "big": {
                **BASE,
                "inputs": 7,
                "val_ll": 0.630,
                "gap": 0.0,
                "back_slice_ll": 0.6,
            },
            "small": {
                **BASE,
                "inputs": 3,
                "val_ll": 0.631,
                "gap": 0.0,
                "back_slice_ll": 0.6,
            },
        }
    )
    # Identical validation probabilities: the paired difference is exactly zero.
    result = adoption(record, valid)
    assert result["passing"] == ["big", "small"]
    assert result["tie_breaks"][0]["inside_noise"]
    assert result["recommended"] == "small"


def test_adoption_keeps_the_lower_loss_when_the_difference_is_clear():
    record, valid = _record(
        {
            "base": BASE,
            "big": {
                **BASE,
                "inputs": 7,
                "edge": 0.6,
                "val_ll": 0.60,
                "gap": 0.0,
                "back_slice_ll": 0.6,
            },
            "small": {
                **BASE,
                "inputs": 3,
                "edge": 0.0,
                "val_ll": 0.69,
                "gap": 0.0,
                "back_slice_ll": 0.6,
            },
        }
    )
    result = adoption(record, valid)
    assert not result["tie_breaks"][0]["inside_noise"]
    assert result["recommended"] == "big"
