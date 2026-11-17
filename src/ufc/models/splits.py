"""Train, validation and test periods, with the test period behind a lock.

docs/ML_REQUIREMENTS.md, "Data splits":

- train: bouts through 2022-12-31
- validation: 2023-01-01 through 2024-12-31 (selection, tuning, calibration)
- test: 2025-01-01 onward, evaluated once per finalized candidate

``Splits`` is the only way pipeline code gets rows. It hands out training and
validation rows freely and raises ``TestPeriodLocked`` for anything dated in
the test period until ``unlock_test`` is called, which only the final
evaluation does, after the candidate has been saved. Every selection stage
additionally asserts that the lock is still closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ufc.features.config import TRAIN_END


class TestPeriodLocked(RuntimeError):
    """Test-period rows were requested before the final evaluation."""

    __test__ = False  # not a pytest class, despite the name


@dataclass(frozen=True, slots=True)
class SplitDates:
    """Last day of training, last day of validation. Test is everything after.

    The defaults are the requirements. Other values exist for tests on
    invented data and for the walk-forward backtest, which moves the same
    three periods back one year at a time.
    """

    train_end: date = TRAIN_END
    valid_end: date = date(2024, 12, 31)

    def __post_init__(self) -> None:
        if self.valid_end <= self.train_end:
            raise ValueError("validation must end after training ends")

    @property
    def valid_start(self) -> date:
        return self.train_end + timedelta(days=1)

    @property
    def test_start(self) -> date:
        return self.valid_end + timedelta(days=1)

    def to_dict(self) -> dict[str, str]:
        return {
            "train_end": self.train_end.isoformat(),
            "valid_start": self.valid_start.isoformat(),
            "valid_end": self.valid_end.isoformat(),
            "test_start": self.test_start.isoformat(),
        }


SPLIT_DATES = SplitDates()
assert SPLIT_DATES.valid_start == date(2023, 1, 1)
assert SPLIT_DATES.test_start == date(2025, 1, 1)


def labelled(frame: pd.DataFrame) -> pd.DataFrame:
    """Bouts with a winner. Draws and no contests have no label and are
    excluded from training and from every metric."""
    return frame[frame["label"].notna()]


def _between(frame: pd.DataFrame, start: date | None, end: date | None) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= frame["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= frame["date"] <= pd.Timestamp(end)
    return mask


class Splits:
    """A feature table divided into periods.

    ``test_dates`` fixes where the locked period begins. It is always the
    project's test period, also when a walk-forward fold uses different
    training and validation periods (``dates``), so no arrangement of periods
    can reach a test-period row while the lock is closed.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        dates: SplitDates = SPLIT_DATES,
        test_dates: SplitDates | None = None,
    ) -> None:
        self._frame = frame
        self.dates = dates
        self._locked_from = (test_dates or dates).test_start
        self._unlocked_for: str | None = None

    # --- the lock -----------------------------------------------------------

    @property
    def test_locked(self) -> bool:
        return self._unlocked_for is None

    @property
    def locked_from(self) -> date:
        return self._locked_from

    def unlock_test(self, candidate: str) -> None:
        """Open the test period for the evaluation of one saved candidate."""
        if not candidate:
            raise ValueError("the test period is unlocked for a named candidate")
        self._unlocked_for = candidate

    def assert_test_locked(self, stage: str) -> None:
        """Selection stages call this first: they must never run once the
        test period has been opened."""
        if not self.test_locked:
            raise TestPeriodLocked(
                f"{stage} ran after the test period was unlocked for "
                f"{self._unlocked_for!r}; selection must finish before evaluation"
            )

    def _check(self, rows: pd.DataFrame, what: str) -> pd.DataFrame:
        if self.test_locked and len(rows):
            latest = rows["date"].max()
            if latest >= pd.Timestamp(self._locked_from):
                raise TestPeriodLocked(
                    f"{what} includes a bout on {latest.date()}, inside the test "
                    f"period (from {self._locked_from}), before the final evaluation"
                )
        return rows

    # --- rows ---------------------------------------------------------------

    def between(self, start: date | None, end: date | None, what: str) -> pd.DataFrame:
        """All bouts dated from ``start`` to ``end`` inclusive."""
        return self._check(self._frame[_between(self._frame, start, end)], what)

    def train(self) -> pd.DataFrame:
        return self.between(None, self.dates.train_end, "training rows")

    def valid(self) -> pd.DataFrame:
        return self.between(
            self.dates.valid_start, self.dates.valid_end, "validation rows"
        )

    def test(self) -> pd.DataFrame:
        """Everything after validation. Raises until ``unlock_test``."""
        return self.between(self.dates.test_start, None, "test rows")

    def development(self) -> pd.DataFrame:
        """Training and validation together."""
        return self.between(None, self.dates.valid_end, "development rows")


def development_only(
    frame: pd.DataFrame, dates: SplitDates = SPLIT_DATES
) -> pd.DataFrame:
    """The table with every test-period row removed.

    The training command applies this to the feature table before anything
    else, so the selection code is never handed a test-period row at all; the
    lock in ``Splits`` is the second guard, not the only one.
    """
    kept = frame[frame["date"] <= pd.Timestamp(dates.valid_end)].reset_index(drop=True)
    assert not (kept["date"] >= pd.Timestamp(dates.test_start)).any()
    return kept


def recency_weights(
    dates: pd.Series, reference: date, half_life_years: float | None
) -> np.ndarray:
    """Sample weights that halve every ``half_life_years`` before ``reference``.

    ``None`` means no weighting. Weights are scaled to average 1, so settings
    that count rows or sum weights (minimum leaf size) mean the same thing at
    every half-life.
    """
    if half_life_years is None:
        return np.ones(len(dates))
    if half_life_years <= 0:
        raise ValueError("half-life must be positive")
    age_years = (pd.Timestamp(reference) - dates).dt.days.to_numpy() / 365.25
    if (age_years < 0).any():
        raise ValueError("a training bout is dated after the reference date")
    weights = 0.5 ** (age_years / half_life_years)
    return weights / weights.mean()
