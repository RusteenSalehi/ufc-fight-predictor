"""The three periods, and the lock on the test period."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.model_helpers import synthetic_frame
from ufc.models.splits import (
    SPLIT_DATES,
    SplitDates,
    Splits,
    TestPeriodLocked,
    attribute_gap,
    development_only,
    fit_rows,
    labelled,
    recency_weights,
)


@pytest.fixture(scope="module")
def frame():
    return synthetic_frame()


def test_split_dates_are_the_ones_in_the_requirements():
    assert SPLIT_DATES.train_end == date(2022, 12, 31)
    assert SPLIT_DATES.valid_start == date(2023, 1, 1)
    assert SPLIT_DATES.valid_end == date(2024, 12, 31)
    assert SPLIT_DATES.test_start == date(2025, 1, 1)
    with pytest.raises(ValueError):
        SplitDates(train_end=date(2024, 1, 1), valid_end=date(2023, 1, 1))


def test_periods_partition_the_table_in_date_order(frame):
    splits = Splits(frame)
    train, valid = splits.train(), splits.valid()
    assert train["date"].max() <= pd.Timestamp("2022-12-31") < valid["date"].min()
    assert valid["date"].max() <= pd.Timestamp("2024-12-31")
    assert len(train) and len(valid)
    assert len(splits.development()) == len(train) + len(valid)
    splits.unlock_test("some-version")
    test = splits.test()
    assert test["date"].min() >= pd.Timestamp("2025-01-01")
    assert len(train) + len(valid) + len(test) == len(frame)
    assert not set(train["bout_id"]) & set(valid["bout_id"])
    assert not set(valid["bout_id"]) & set(test["bout_id"])


def test_test_period_is_locked_until_unlocked_for_a_named_candidate(frame):
    splits = Splits(frame)
    assert splits.test_locked
    with pytest.raises(TestPeriodLocked, match="before the final evaluation"):
        splits.test()
    with pytest.raises(TestPeriodLocked):
        splits.between(date(2024, 6, 1), date(2025, 1, 4), "a window")
    with pytest.raises(TestPeriodLocked):
        splits.between(None, None, "everything")
    # The day before the test period is fine.
    assert len(splits.between(date(2024, 6, 1), date(2024, 12, 31), "a window"))
    splits.assert_test_locked("tuning")

    with pytest.raises(ValueError):
        splits.unlock_test("")
    splits.unlock_test("xgboost-abc")
    assert not splits.test_locked and len(splits.test())
    with pytest.raises(TestPeriodLocked, match="selection must finish"):
        splits.assert_test_locked("tuning")


def test_other_period_arrangements_cannot_reach_the_test_period(frame):
    """A walk-forward fold has its own periods, but the lock still starts
    where the project's test period starts."""
    fold = SplitDates(train_end=date(2023, 12, 31), valid_end=date(2025, 12, 31))
    splits = Splits(frame, fold, test_dates=SPLIT_DATES)
    assert splits.locked_from == date(2025, 1, 1)
    assert splits.train()["date"].max() <= pd.Timestamp("2023-12-31")
    with pytest.raises(TestPeriodLocked):
        splits.valid()  # 2024 to 2025 reaches into the test period
    splits.unlock_test("v")
    assert splits.valid()["date"].max() > pd.Timestamp("2025-01-01")


def test_development_only_removes_every_test_period_row(frame):
    kept = development_only(frame)
    assert kept["date"].max() <= pd.Timestamp("2024-12-31")
    assert len(kept) == (frame["date"] <= pd.Timestamp("2024-12-31")).sum()
    splits = Splits(kept)
    splits.unlock_test("v")
    assert splits.test().empty  # nothing there to find, even unlocked


def test_labelled_drops_draws_and_no_contests(frame):
    kept = labelled(frame)
    assert set(frame["result"]) == {"win", "draw", "nc"}
    assert set(kept["result"]) == {"win"} and kept["label"].notna().all()
    assert len(kept) < len(frame)


def test_fit_rows_leave_out_bouts_with_an_attribute_gap(frame):
    kept = fit_rows(frame)
    scored = labelled(frame)
    gap = attribute_gap(scored)
    assert gap.any() and (scored["a_attr_gap"] & ~scored["b_attr_gap"]).any()
    assert len(kept) == len(scored) - gap.sum()
    assert not attribute_gap(kept).any() and kept["label"].notna().all()
    # Bouts where both profiles are equally incomplete stay in.
    assert kept["d_reach_cm"].isna().any()
    # They are left out of fitting only: scoring still sees every bout.
    assert set(scored["bout_id"]) > set(kept["bout_id"])


def test_recency_weights_halve_per_half_life_and_average_one():
    dates = pd.Series(pd.to_datetime(["2022-12-31", "2020-12-31", "2018-12-31"]))
    reference = date(2022, 12, 31)
    np.testing.assert_array_equal(recency_weights(dates, reference, None), np.ones(3))
    weights = recency_weights(dates, reference, 2.0)
    assert weights.mean() == pytest.approx(1.0)
    assert weights[1] / weights[0] == pytest.approx(0.5, rel=1e-3)
    assert weights[2] / weights[0] == pytest.approx(0.25, rel=1e-3)
    slow = recency_weights(dates, reference, 8.0)
    assert slow[2] / slow[0] > weights[2] / weights[0]
    with pytest.raises(ValueError):
        recency_weights(dates, date(2021, 1, 1), 2.0)  # a bout after the reference
    with pytest.raises(ValueError):
        recency_weights(dates, reference, 0.0)
