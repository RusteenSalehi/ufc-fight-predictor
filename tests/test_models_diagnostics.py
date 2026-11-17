"""The era-shift diagnostics' helpers, on invented numbers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ufc.models.diagnostics import (
    BEFORE,
    INSIDE,
    attribution,
    debut_dates,
    era_masks,
    one_sided_gaps,
    rule_accuracy,
    rule_direction,
    single_feature_rules,
)


def _frame(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    """(date, d_x, label) rows with no attribute gaps."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime([r[0] for r in rows]),
            "d_x": [r[1] for r in rows],
            "label": [r[2] for r in rows],
            "a_attr_gap": False,
            "b_attr_gap": False,
        }
    )


def test_rule_direction_follows_the_side_that_won() -> None:
    values = np.array([2.0, -1.0, 3.0, np.nan, 0.0])
    assert rule_direction(values, np.array([1.0, 0.0, 1.0, 0.0, 0.0])) == 1
    assert rule_direction(values, np.array([0.0, 1.0, 0.0, 1.0, 1.0])) == -1


def test_rule_accuracy_skips_missing_and_equal_values() -> None:
    values = np.array([2.0, -1.0, 3.0, np.nan, 0.0])
    y = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    result = rule_accuracy(values, y, 1)
    assert result["n"] == 3
    assert result["accuracy"] == 2 / 3
    assert result["coverage"] == 3 / 5
    assert rule_accuracy(values, y, -1)["accuracy"] == 1 / 3
    assert rule_accuracy(np.array([np.nan]), np.array([1.0]), 1)["accuracy"] is None


def test_rule_direction_is_set_on_the_early_period_only() -> None:
    # Larger wins before the cutoff; smaller wins in every bout after it.
    early = [("2012-01-01", 1.0, 1.0), ("2012-02-01", -1.0, 0.0)]
    late = [("2025-01-01", 1.0, 0.0), ("2025-02-01", -2.0, 1.0)]
    rules = single_feature_rules(_frame(early + late), ["d_x"], date(2022, 12, 31))
    assert rules["d_x"]["direction"] == "larger wins"
    assert rules["d_x"]["periods"]["2025"] == {"n": 2, "accuracy": 0.0, "coverage": 1.0}
    assert rules["d_x"]["periods"][INSIDE]["accuracy"] == 0.0


def test_rules_can_leave_out_attribute_gap_bouts() -> None:
    frame = _frame([("2012-01-01", 1.0, 1.0), ("2025-01-01", 1.0, 0.0)])
    frame.loc[1, "a_attr_gap"] = True
    rules = single_feature_rules(frame, ["d_x"], date(2022, 12, 31), fit_only=True)
    assert "2025" not in rules["d_x"]["periods"]


def test_era_masks_split_before_and_inside_the_test_period() -> None:
    masks = era_masks(np.array([2012, 2013, 2024, 2025, 2026]))
    assert masks[BEFORE].tolist() == [False, True, True, False, False]
    assert masks[INSIDE].tolist() == [False, False, False, True, True]
    assert "2012" not in masks
    assert masks["2025"].tolist() == [False, False, False, True, False]


def test_attribution_reports_the_shift_and_the_paired_difference() -> None:
    bouts = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2025-01-01"]),
            "label": [1.0, 0.0, 1.0],
            "a_attr_gap": False,
            "b_attr_gap": False,
            "p_full": [0.6, 0.4, 0.8],
            "p_other": [0.5, 0.5, 0.8],
        }
    )
    result = attribution(bouts, ["full", "other"])
    assert np.isclose(result["full"]["log_loss"][BEFORE], -np.log(0.6))
    assert np.isclose(result["full"]["shift"], -np.log(0.6) + np.log(0.8))
    assert result["full"]["minus_full"][BEFORE]["mean"] == 0.0
    assert np.isclose(
        result["other"]["minus_full"][BEFORE]["mean"], -np.log(0.5) + np.log(0.6)
    )
    assert result["other"]["minus_full"][INSIDE]["mean"] == 0.0


def test_one_sided_gaps_count_wins_of_the_complete_profile() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-02-01", "2026-01-01"]),
            "label": [1.0, 1.0, 0.0],
            "a_attr_gap": [True, False, True],
            "b_attr_gap": [False, True, True],
        }
    )
    result = one_sided_gaps(frame)[INSIDE]
    assert result == {"n": 2, "complete_profile_won": 0.5}


def test_debut_dates_take_the_first_bout() -> None:
    @dataclass
    class Bout:
        date: date
        fighter_a_id: str
        fighter_b_id: str

    @dataclass
    class Data:
        bouts: list[Bout]

    data = Data([Bout(date(2020, 5, 1), "x", "y"), Bout(date(2019, 5, 1), "y", "z")])
    assert debut_dates(data) == {  # type: ignore[arg-type]
        "x": date(2020, 5, 1),
        "y": date(2019, 5, 1),
        "z": date(2019, 5, 1),
    }
