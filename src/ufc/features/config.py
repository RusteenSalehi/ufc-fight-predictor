"""Settings for the feature pipeline. Nothing in this file is fitted to data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# Last day of the training period (docs/ML_REQUIREMENTS.md, "Data splits").
# Everything that is fitted (Elo parameters, league baselines) sees bouts up
# to and including this date and nothing later.
TRAIN_END = date(2022, 12, 31)

# Decayed sums weight a bout by 2 ** ((bout date - DECAY_ORIGIN) / half-life).
# Only ratios of such sums are used, so the origin cancels out; it is a fixed
# constant so that weights never depend on which bouts are in the data.
DECAY_ORIGIN = date(1993, 1, 1).toordinal()

WINDOWS: tuple[str, ...] = ("career", "l3", "l5", "decay")
"""Whole history, the last 3 bouts, the last 5 bouts, time-decayed history."""


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Choices made in advance, not tuned on outcomes.

    ``decay_half_life_days``: a bout this many days before the cutoff counts
    half as much as one on the cutoff in the ``decay`` versions of a stat.

    ``prior_fights``: how many league-average fights are added to an
    opponent's own numbers when estimating what he usually allows. It keeps
    baselines of opponents with one or two other bouts from being extreme.

    ``recent_days``: the window for "fights in the last two years".
    """

    train_end: date = TRAIN_END
    decay_half_life_days: float = 730.0
    prior_fights: float = 2.0
    recent_days: int = 730


_WEIGHT_LIMIT_LBS: dict[str, float] = {
    "Strawweight": 115.0,
    "Flyweight": 125.0,
    "Bantamweight": 135.0,
    "Featherweight": 145.0,
    "Lightweight": 155.0,
    "Welterweight": 170.0,
    "Middleweight": 185.0,
    "Light Heavyweight": 205.0,
    "Heavyweight": 265.0,
}


def weight_limit_lbs(weight_class: str | None) -> float | None:
    """Upper limit of a weight class in pounds, for telling up from down.

    ``None`` for classes without a fixed limit (catch weight, open weight,
    super heavyweight) and for bouts with no recorded class.
    """
    if weight_class is None:
        return None
    return _WEIGHT_LIMIT_LBS.get(weight_class.removeprefix("Women's "))
