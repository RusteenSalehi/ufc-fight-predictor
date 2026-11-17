"""Weight-class average rates: the prior for shrinking per-minute rates.

A fighter with twenty seconds of cage time and one knockdown has a pooled
rate of 45 knockdowns per 15 minutes. Shrinkage replaces a per-minute rate
``count / minutes`` with ``(count + m * mean) / (minutes + m)``: the fighter
is treated as if he had fought ``m`` extra minutes at the average rate of the
weight class. With many real minutes the prior barely matters; with few, the
rate stays near the class average (docs/DECISIONS.md, D33).

The class averages are fitted on the training period only (leakage rule 4).
The fighter's own counts and minutes are as of the cutoff like every other
feature; this module does not read histories at all.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from ufc.features.components import (
    IDX,
    N_COMPONENTS,
    bout_components,
    has_usable_stats,
)
from ufc.features.config import weight_limit_lbs
from ufc.features.data import BoutFacts

RATE_COMPONENTS: tuple[str, ...] = (
    "kd",
    "sig_l",
    "sig_a",
    "td_l",
    "sub_att",
    "opp_kd",
    "opp_sig_l",
    "opp_head_l",
)
"""Counts that are divided by stat time somewhere in the features."""

MIN_CLASS_BOUTS = 50
"""A weight class needs this many training bouts with stats to have its own
average. Smaller classes, and bouts without a fixed weight limit (catch
weight, open weight), use the average over all bouts. Set in advance."""

_T = IDX["t"]
_RATE_IDX = np.array([IDX[name] for name in RATE_COMPONENTS])


def _per_minute(total: np.ndarray) -> np.ndarray:
    """One average minute as a component vector: 60 stat seconds and the
    counts that go with them. All zeros if there is no stat time."""
    out = np.zeros(N_COMPONENTS)
    if total[_T] > 0:
        out[_RATE_IDX] = total[_RATE_IDX] * 60.0 / total[_T]
        out[_T] = 60.0
    return out


@dataclass(frozen=True, slots=True)
class RatePriors:
    """Average counts per minute, overall and for each weight class that has
    enough training bouts. Each vector has ``N_COMPONENTS`` entries; only the
    ``RATE_COMPONENTS`` and the stat-time entry (60 seconds) are non-zero."""

    overall: np.ndarray
    by_class: Mapping[str, np.ndarray] = field(default_factory=dict)
    n_bouts: Mapping[str, int] = field(default_factory=dict)

    def for_class(self, weight_class: str | None) -> np.ndarray:
        if weight_class is None:
            return self.overall
        return self.by_class.get(weight_class, self.overall)

    def to_dict(self) -> dict:
        def rates(vector: np.ndarray) -> dict[str, float]:
            return {name: float(vector[IDX[name]]) for name in RATE_COMPONENTS}

        return {
            "unit": "count per minute of fights with recorded stats",
            "has_overall": bool(self.overall[_T] > 0),
            "overall": rates(self.overall),
            "by_class": {name: rates(v) for name, v in sorted(self.by_class.items())},
            "n_bouts": dict(sorted(self.n_bouts.items())),
        }

    @classmethod
    def from_dict(cls, raw: Mapping) -> RatePriors:
        def vector(rates: Mapping[str, float], present: bool = True) -> np.ndarray:
            out = np.zeros(N_COMPONENTS)
            if present:
                for name in RATE_COMPONENTS:
                    out[IDX[name]] = rates[name]
                out[_T] = 60.0
            return out

        return cls(
            overall=vector(raw["overall"], raw["has_overall"]),
            by_class={name: vector(r) for name, r in raw["by_class"].items()},
            n_bouts=dict(raw["n_bouts"]),
        )


def fit_rate_priors(
    bouts: Iterable[BoutFacts], train_end: date, min_bouts: int = MIN_CLASS_BOUTS
) -> RatePriors:
    """Average per-minute counts from bouts dated on or before ``train_end``.

    Rates are pooled: total count over total stat minutes, both fighters of
    every bout with stats.
    """
    overall = np.zeros(N_COMPONENTS)
    totals: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(N_COMPONENTS))
    counts: dict[str, int] = defaultdict(int)
    for bout in bouts:
        if bout.date > train_end or not has_usable_stats(bout):
            continue
        both = bout_components(bout, bout.fighter_a_id, 0.0) + bout_components(
            bout, bout.fighter_b_id, 0.0
        )
        overall += both
        if bout.weight_class is not None:
            totals[bout.weight_class] += both
            counts[bout.weight_class] += 1
    qualifying = sorted(
        name
        for name, count in counts.items()
        if count >= min_bouts and weight_limit_lbs(name) is not None
    )
    return RatePriors(
        overall=_per_minute(overall),
        by_class={name: _per_minute(totals[name]) for name in qualifying},
        n_bouts={name: counts[name] for name in qualifying},
    )
