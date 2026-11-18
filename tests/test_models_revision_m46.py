"""M4.6 revision helpers: the D63 rule as code, and the scorecard parser."""

from __future__ import annotations

import pytest

from ufc.models.revision_m46 import PERFORMANCE, apply_rule, scorecards


def _records(delta: float, high: float, backtest_delta: float):
    validation = {
        "variants": {
            "performance": {
                "raw": {"delta": delta, "delta_interval": [delta - 0.002, high]}
            }
        }
    }
    backtest = {"variants": {"performance": {"delta": backtest_delta}}}
    return validation, backtest


@pytest.mark.parametrize(
    ("delta", "high", "backtest", "adopt"),
    [
        (-0.0010, 0.0005, -0.0003, True),
        (-0.0010, 0.0005, 0.0001, False),  # backtest does not improve
        (-0.0010, 0.0010, -0.0003, False),  # upper end not below +0.001
        (0.0001, 0.0009, -0.0003, False),  # not negative
        (-0.0030, -0.0005, -0.0001, True),
    ],
)
def test_rule_needs_both_criteria(delta, high, backtest, adopt):
    assert apply_rule(*_records(delta, high, backtest))["adopt"] is adopt


def test_scorecards_read_each_judge_and_ignore_deduction_text():
    details = (
        "Point Deducted: Holding Fence by FabianTony Weeks 27 - 29."
        "Glenn Trowbridge 28 - 28.Jeff Mullen 28 - 28."
    )
    assert scorecards(details) == [(27, 29), (28, 28), (28, 28)]
    assert scorecards("Rear Naked Choke") == []
    assert scorecards(None) == []


def test_the_variant_adds_the_seven_family_16_columns():
    assert len(PERFORMANCE) == 7
    assert all(name.startswith("d_") for name in PERFORMANCE)
