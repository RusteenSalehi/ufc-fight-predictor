"""Median age at bout time per weight class, fitted on the training period.

Used by the age-shape candidates of the M4.5 revision (DECISIONS D55): how
far a fighter is past the typical age of his division. Heavyweights and
flyweights do not age on the same schedule, and a single threshold such as
32 cannot say so.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from ufc.features.config import weight_limit_lbs
from ufc.features.data import BoutFacts, FighterAttrs

MIN_BOUTS = 50
"""Training bouts a weight class needs for a median of its own. Smaller
classes, classes without a fixed limit (catch, open and super heavyweight)
and bouts without a class use the median over every training bout."""

ALL = "*"


@dataclass(frozen=True, slots=True)
class AgeMedians:
    """``by_class`` maps a weight class to its median age in years; ``ALL``
    holds the median over every training bout."""

    by_class: Mapping[str, float]

    def for_class(self, weight_class: str | None) -> float:
        if weight_class is not None and weight_class in self.by_class:
            return self.by_class[weight_class]
        return self.by_class[ALL]

    def to_dict(self) -> dict[str, float]:
        return dict(self.by_class)

    @classmethod
    def from_dict(cls, raw: Mapping[str, float]) -> AgeMedians:
        return cls({str(k): float(v) for k, v in raw.items()})


def fit_age_medians(
    bouts: Sequence[BoutFacts],
    fighters: Mapping[str, FighterAttrs],
    train_end: date,
    min_bouts: int = MIN_BOUTS,
) -> AgeMedians | None:
    """Median age of the fighters in each class's training bouts.

    Each fighter in each bout through ``train_end`` with a known date of
    birth counts once. ``None`` when no training bout has a known age.
    """
    ages: dict[str, list[float]] = defaultdict(list)
    n_bouts: dict[str, int] = defaultdict(int)
    every: list[float] = []
    for bout in bouts:
        if bout.date > train_end:
            continue
        cls = bout.weight_class
        n_bouts[cls or ""] += 1
        for fighter_id in (bout.fighter_a_id, bout.fighter_b_id):
            attrs = fighters.get(fighter_id)
            if attrs is None or attrs.dob is None:
                continue
            age = (bout.date.toordinal() - attrs.dob.toordinal()) / 365.25
            every.append(age)
            if cls is not None:
                ages[cls].append(age)
    if not every:
        return None
    by_class = {ALL: float(np.median(every))}
    for cls, values in ages.items():
        fixed = weight_limit_lbs(cls) is not None
        if fixed and n_bouts[cls] >= min_bouts and values:
            by_class[cls] = float(np.median(values))
    return AgeMedians(by_class)
