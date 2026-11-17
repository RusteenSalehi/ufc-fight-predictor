"""Settings for the feature pipeline. Nothing in this file is fitted to data.

One setting, ``shrink_minutes``, is chosen on validation by the model
pipeline (``python -m ufc.models train``); its default here is "off".
"""

from __future__ import annotations

from collections.abc import Sequence
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
    """Choices made in advance, not tuned on outcomes, except ``shrink_minutes``.

    ``decay_half_life_days``: a bout this many days before the cutoff counts
    half as much as one on the cutoff in the ``decay`` versions of a stat.

    ``prior_fights``: how many league-average fights are added to an
    opponent's own numbers when estimating what he usually allows. It keeps
    baselines of opponents with one or two other bouts from being extreme.

    ``recent_days``: the window for "fights in the last two years".

    ``shrink_minutes``: prior strength, in cage minutes, for per-minute and
    per-15-minute rates. A rate becomes (count + m * class mean per minute) /
    (minutes + m), which pulls a fighter with little cage time toward the
    average of the weight class. 0 leaves rates as plain pooled rates. The
    model pipeline chooses the value on validation (docs/DECISIONS.md, D33).

    ``catch_weight_adjacent``: in "fights at this weight" and the
    weight-change features, count a catch-weight bout as the division of the
    fighter's adjacent bouts (``resolve_catch_weights``) instead of as a
    class of its own. Off by default, which is the M4 behaviour; the M4.5
    revision tests it as a variant (docs/DECISIONS.md, D54, D55).
    """

    train_end: date = TRAIN_END
    decay_half_life_days: float = 730.0
    prior_fights: float = 2.0
    recent_days: int = 730
    shrink_minutes: float = 0.0
    catch_weight_adjacent: bool = False

    def __post_init__(self) -> None:
        if self.shrink_minutes < 0:
            raise ValueError("shrink_minutes cannot be negative")


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


CATCH_WEIGHT = "Catch Weight"


def _known_class(weight_class: str | None) -> bool:
    return weight_class is not None and weight_class != CATCH_WEIGHT


def resolve_catch_weights(
    classes: Sequence[str | None], weight_class: str | None
) -> tuple[list[str | None], str | None]:
    """A fighter's earlier classes and the current bout's class, with catch
    weights replaced by the division of adjacent bouts.

    ``classes`` are the fighter's bouts strictly before the cutoff, oldest
    first; nothing later is passed in, so nothing later can be used. A
    catch-weight bout in that list takes the class of the nearest earlier
    bout with a known division, or failing that the nearest later one in the
    list. A catch-weight bout being predicted takes the class of the
    fighter's nearest earlier bout with a known division. A catch weight with
    no such neighbour stays a catch weight.
    """
    resolved = list(classes)
    for i, value in enumerate(classes):
        if value != CATCH_WEIGHT:
            continue
        earlier = next((c for c in reversed(classes[:i]) if _known_class(c)), None)
        later = next((c for c in classes[i + 1 :] if _known_class(c)), None)
        resolved[i] = earlier or later or value
    current = weight_class
    if weight_class == CATCH_WEIGHT:
        current = next((c for c in reversed(classes) if _known_class(c)), weight_class)
    return resolved, current
