"""Opponent-adjusted stats: what a fighter did, minus what his opponents allow.

For fighter F at cutoff D, take each earlier bout of F that has stats. Its
opponent O has a baseline: O's own numbers in bouts strictly before D, with
his bouts against F removed, shrunk toward the league average. The baseline
says what O's other opponents typically managed against him, for example
that they land 40% of their takedown attempts. Multiplied by what F actually
attempted in the bout, it gives an expected count. The adjusted stat is
(actual - expected) / exposure over F's history.

Both sides respect the cutoff: F's bouts and O's bouts are all strictly
before D (docs/DECISIONS.md, D25).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ufc.features.components import (
    IDX,
    N_COMPONENTS,
    bout_components,
    has_usable_stats,
)
from ufc.features.data import BoutFacts

if TYPE_CHECKING:
    from ufc.features.store import FeatureStore, FighterHistory


class Adjusted(NamedTuple):
    """``num / den`` is the fighter's own rate. ``sign`` is -1 where the
    feature is the complement of that rate (a defense), so that positive
    always means "more than expected" of the named stat."""

    name: str
    num: str
    den: str
    scale: float = 1.0
    sign: float = 1.0


ADJUSTED: tuple[Adjusted, ...] = (
    Adjusted("adj_slpm", "sig_l", "t", 60.0),
    Adjusted("adj_sapm", "opp_sig_l", "t", 60.0),
    Adjusted("adj_sig_acc", "sig_l", "sig_a"),
    Adjusted("adj_sig_def", "opp_sig_l", "opp_sig_a", sign=-1.0),
    Adjusted("adj_td_per15", "td_l", "t", 900.0),
    Adjusted("adj_td_acc", "td_l", "td_a"),
    Adjusted("adj_td_def", "opp_td_l", "opp_td_a", sign=-1.0),
    Adjusted("adj_kd_per15", "kd", "t", 900.0),
    Adjusted("adj_ctrl_share", "ctrl", "t_ctrl"),
    Adjusted("adj_sub_att_per15", "sub_att", "t", 900.0),
)

ADJUSTED_WINDOWS: tuple[str, ...] = ("career", "decay")
ADJUSTED_NAMES: tuple[str, ...] = tuple(
    f"{spec.name}_{window}" for window in ADJUSTED_WINDOWS for spec in ADJUSTED
)


def _mirror(component: str) -> str:
    """The same quantity from the other fighter's side of a bout."""
    if component == "t":
        return "t"
    if component == "t_ctrl":
        return "t_opp_ctrl"
    if component == "t_opp_ctrl":
        return "t_ctrl"
    return (
        component.removeprefix("opp_")
        if component.startswith("opp_")
        else f"opp_{component}"
    )


_NUM = np.array([IDX[s.num] for s in ADJUSTED])
_DEN = np.array([IDX[s.den] for s in ADJUSTED])
_MIRROR_NUM = np.array([IDX[_mirror(s.num)] for s in ADJUSTED])
_MIRROR_DEN = np.array([IDX[_mirror(s.den)] for s in ADJUSTED])
_FACTOR = np.array([s.scale * s.sign for s in ADJUSTED])
_T = IDX["t"]


def fit_league_baseline(bouts: Iterable[BoutFacts], train_end: date) -> np.ndarray:
    """Average component vector of a fighter in a bout with stats.

    Fitted on bouts dated on or before ``train_end`` only. Zeros if there are
    none, which leaves opponents without other bouts without a baseline.
    """
    total, count = np.zeros(N_COMPONENTS), 0
    for bout in bouts:
        if bout.date > train_end or not has_usable_stats(bout):
            continue
        for fighter_id in (bout.fighter_a_id, bout.fighter_b_id):
            total += bout_components(bout, fighter_id, opp_elo=0.0)
            count += 1
    league = total / count if count else total
    league[IDX["opp_elo"]] = 0.0  # ratings are not part of any baseline
    return league


def adjusted_stats(
    store: FeatureStore,
    fighter_id: str,
    history: FighterHistory,
    idx: int,
    ordinal: int,
) -> np.ndarray:
    """Values for ``ADJUSTED_NAMES`` from the first ``idx`` bouts of ``history``."""
    n = len(ADJUSTED)
    out = np.full(2 * n, np.nan)
    stat_bouts = [j for j in range(idx) if history.rows[j][_T] > 0]
    if not stat_bouts:
        return out

    own = np.array([history.rows[j] for j in stat_bouts])
    baselines = np.array(
        [
            store.baseline_excluding(history.opp_ids[j], fighter_id, ordinal)
            for j in stat_bouts
        ]
    )
    weights = np.array([history.weights[j] for j in stat_bouts])

    mirror_den = baselines[:, _MIRROR_DEN]
    rate = np.full(mirror_den.shape, np.nan)
    np.divide(baselines[:, _MIRROR_NUM], mirror_den, out=rate, where=mirror_den > 0)
    exposure = own[:, _DEN]
    # No exposure means nothing was expected, whatever the opponent's rate.
    expected = np.where(exposure > 0, rate * exposure, 0.0)
    actual = own[:, _NUM]

    for k, w in enumerate((np.ones(len(stat_bouts)), weights)):
        den = w @ exposure
        values = np.full(n, np.nan)
        np.divide(w @ actual - w @ expected, den, out=values, where=den > 0)
        out[k * n : (k + 1) * n] = values * _FACTOR
    return out
