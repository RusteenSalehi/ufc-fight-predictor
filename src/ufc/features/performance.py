"""Deserved win probability: who was winning a bout on the judges' criteria.

Results are a noisy measure of performance: a fighter can lose every
exchange and land one knockout, and a decision can be close or wrong. This
module turns a bout's in-fight stats into a probability that each fighter
"deserved" to win (model revision M4.6, DECISIONS D63).

The performance model is a logistic regression without intercept on
A-minus-B in-fight differences, fitted on decisions with a winner (label: the
judged winner) from the training period only (leakage rule 4). It is then
applied to every bout with stats, finishes and draws included; for a finish
the stats are those up to the stoppage. Without an intercept and with
antisymmetric inputs, a bout seen from the other side gets exactly the
negated logit, so the two fighters' values are mirror images and do not
depend on which fighter the source lists first.

A bout with control time missing in any round uses a second fit without the
two control inputs. A bout without usable stats has no deserved value
(``None``), never a stand-in of 0.5.

The performance Elo is the D26 tracker with the deserved share as each
fighter's score instead of the result. Its step sizes are chosen on the
training period by the log loss of its pre-fight probabilities against the
actual winners.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import product

import numpy as np
from sklearn.linear_model import LogisticRegression

from ufc.features.components import has_usable_stats
from ufc.features.data import BoutFacts
from ufc.features.elo import (
    EloParams,
    EloTracker,
    RatedBout,
    by_date,
    expected_score,
    rated_bout,
)

ROUND_S = 300.0
"""Rates are per 5 minutes, and a fight counts as at least this long."""

INPUTS: tuple[str, ...] = (
    "sig_landed",  # significant strikes landed, per 5 minutes
    "knockdowns",  # per 5 minutes
    "td_landed",  # takedowns landed, per 5 minutes
    "sub_attempts",  # per 5 minutes
    "ctrl_min",  # control minutes, per 5 minutes
    "rounds_sig",  # (rounds led on significant strikes, minus his) / rounds
    "rounds_ctrl",  # (rounds led on control time, minus his) / rounds
)
CONTROL_INPUTS: tuple[str, ...] = ("ctrl_min", "rounds_ctrl")
NO_CONTROL: tuple[str, ...] = tuple(n for n in INPUTS if n not in CONTROL_INPUTS)
_NO_CONTROL_IDX = np.array([INPUTS.index(n) for n in NO_CONTROL])

C = 1.0
"""L2 strength (scikit-learn's inverse convention); fixed in D63, not tuned."""
MIN_BOUTS = 20
"""Decisions a fit needs; with fewer the model is not fitted (``None``)."""

DECISION_PREFIX = "Decision"


def _dominance(own: Sequence[float], opp: Sequence[float]) -> float:
    """Rounds he led minus rounds the opponent led, over rounds fought."""
    led = sum(1 for o, p in zip(own, opp, strict=True) if o > p)
    trailed = sum(1 for o, p in zip(own, opp, strict=True) if p > o)
    return (led - trailed) / len(own)


def performance_inputs(bout: BoutFacts, fighter_id: str) -> np.ndarray | None:
    """``INPUTS`` for ``fighter_id`` minus his opponent, or ``None`` when the
    bout has no usable stats or no round lines for both fighters.

    The control inputs are NaN when control time is missing in any round.
    Seen from the opponent's side every value is exactly negated.
    """
    if not has_usable_stats(bout):
        return None
    opponent_id = bout.opponent_of(fighter_id)
    own = list(bout.rounds.get(fighter_id, ()))
    opp = list(bout.rounds.get(opponent_id, ()))
    if not own or len(own) != len(opp):
        return None
    per = ROUND_S / max(float(bout.fight_time_s or 0), ROUND_S)

    def total(lines: list, key: str) -> float:
        return float(sum(line[key] for line in lines))

    def diff(key: str) -> float:
        return (total(own, key) - total(opp, key)) * per

    out = np.array(
        [
            diff("sig_str_landed"),
            diff("knockdowns"),
            diff("td_landed"),
            diff("sub_attempts"),
            math.nan,
            _dominance(
                [line["sig_str_landed"] for line in own],
                [line["sig_str_landed"] for line in opp],
            ),
            math.nan,
        ]
    )
    if all(line["ctrl_s"] is not None for line in (*own, *opp)):
        out[INPUTS.index("ctrl_min")] = diff("ctrl_s") / 60.0
        out[INPUTS.index("rounds_ctrl")] = _dominance(
            [line["ctrl_s"] for line in own], [line["ctrl_s"] for line in opp]
        )
    return out


def has_control(x: np.ndarray) -> bool:
    return not np.isnan(x).any()


@dataclass(frozen=True, slots=True)
class LinearFit:
    """``logit = sum(coef_i * x_i / scale_i)`` over ``names``."""

    names: tuple[str, ...]
    scale: tuple[float, ...]
    coef: tuple[float, ...]
    n_bouts: int

    @property
    def weights(self) -> np.ndarray:
        return np.array(self.coef) / np.array(self.scale)

    def to_dict(self) -> dict:
        return {
            "names": list(self.names),
            "scale": list(self.scale),
            "coef": list(self.coef),
            "n_bouts": self.n_bouts,
        }

    @classmethod
    def from_dict(cls, raw: Mapping) -> LinearFit:
        return cls(
            tuple(raw["names"]),
            tuple(float(v) for v in raw["scale"]),
            tuple(float(v) for v in raw["coef"]),
            int(raw["n_bouts"]),
        )


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass(frozen=True, slots=True)
class PerformanceModel:
    """The fitted transform: ``full`` reads every input, ``no_control`` the
    inputs without control time, for bouts that lack it."""

    full: LinearFit
    no_control: LinearFit
    train_end: date

    def logit(self, x: np.ndarray) -> float:
        if has_control(x):
            return float(np.dot(self.full.weights, x))
        return float(np.dot(self.no_control.weights, x[_NO_CONTROL_IDX]))

    def deserved(self, bout: BoutFacts, fighter_id: str) -> float | None:
        """Probability that ``fighter_id`` deserved to win ``bout``."""
        x = performance_inputs(bout, fighter_id)
        return None if x is None else _sigmoid(self.logit(x))

    def deserved_pair(self, bout: BoutFacts) -> tuple[float, float] | None:
        """Both fighters' deserved probabilities, each from his own side."""
        a = self.deserved(bout, bout.fighter_a_id)
        if a is None:
            return None
        b = self.deserved(bout, bout.fighter_b_id)
        assert b is not None
        return a, b

    def to_dict(self) -> dict:
        return {
            "full": self.full.to_dict(),
            "no_control": self.no_control.to_dict(),
            "train_end": self.train_end.isoformat(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping) -> PerformanceModel:
        return cls(
            LinearFit.from_dict(raw["full"]),
            LinearFit.from_dict(raw["no_control"]),
            date.fromisoformat(raw["train_end"]),
        )


def is_judged_win(bout: BoutFacts) -> bool:
    return (
        bout.result == "win"
        and bout.winner_id is not None
        and (bout.method or "").startswith(DECISION_PREFIX)
    )


def training_rows(
    bouts: Iterable[BoutFacts], train_end: date
) -> tuple[np.ndarray, np.ndarray]:
    """Decisions with a winner through ``train_end``, both orientations:
    the inputs from each fighter's side and 1 for the judged winner."""
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for bout in bouts:
        if bout.date > train_end or not is_judged_win(bout):
            continue
        for fighter_id in (bout.fighter_a_id, bout.fighter_b_id):
            x = performance_inputs(bout, fighter_id)
            if x is None:
                break
            xs.append(x)
            ys.append(1.0 if fighter_id == bout.winner_id else 0.0)
    if not xs:
        return np.zeros((0, len(INPUTS))), np.zeros(0)
    return np.array(xs), np.array(ys)


def _fit(x: np.ndarray, y: np.ndarray, names: tuple[str, ...]) -> LinearFit | None:
    n_bouts = len(y) // 2
    if n_bouts < MIN_BOUTS:
        return None
    # Both orientations are present, so every column has mean exactly zero
    # and its standard deviation is the root mean square.
    scale = np.sqrt(np.mean(x**2, axis=0))
    scale = np.where(scale > 0, scale, 1.0)
    model = LogisticRegression(C=C, fit_intercept=False, max_iter=1000)
    model.fit(x / scale, y)
    return LinearFit(
        names,
        tuple(float(v) for v in scale),
        tuple(float(v) for v in model.coef_[0]),
        n_bouts,
    )


def fit_performance_model(
    bouts: Iterable[BoutFacts], train_end: date
) -> PerformanceModel | None:
    """Fit on decisions with a winner dated through ``train_end`` only.

    ``None`` when either fit has fewer than ``MIN_BOUTS`` decisions."""
    x, y = training_rows(bouts, train_end)
    if len(y) == 0:
        return None
    with_control = ~np.isnan(x).any(axis=1)
    # Keep the two orientations of a bout together: both have control or
    # neither does, because missing control is a property of the bout.
    full = _fit(x[with_control], y[with_control], INPUTS)
    no_control = _fit(x[:, _NO_CONTROL_IDX], y, NO_CONTROL)
    if full is None or no_control is None:
        return None
    return PerformanceModel(full, no_control, train_end)


# --- performance Elo -------------------------------------------------------------


def rated_performance(bout: BoutFacts, model: PerformanceModel) -> RatedBout | None:
    """The bout for the performance Elo: each fighter's score is his deserved
    share. ``None`` for a bout the standard Elo does not rate (a no contest)
    or one without a deserved value."""
    if rated_bout(bout) is None:
        return None
    pair = model.deserved_pair(bout)
    if pair is None:
        return None
    return RatedBout(bout.fighter_a_id, bout.fighter_b_id, pair[0], False, pair[1])


PERFORMANCE_ELO_GRID: dict[str, tuple[float, ...]] = {
    "k": (16, 24, 32, 48, 64, 96, 128, 160, 192, 256),
    "provisional_mult": (1.0, 1.5, 2.0, 2.5),
}


@dataclass(frozen=True, slots=True)
class PerformanceEloFit:
    params: EloParams
    log_loss: float
    n_bouts: int
    on_grid_edge: tuple[str, ...]


def _days(
    bouts: Iterable[BoutFacts], model: PerformanceModel, train_end: date
) -> list[tuple[int, list[RatedBout], list[tuple[str, str]]]]:
    """Per training date: the rated bouts, and (winner, loser) of those with
    a winner, which the loss is measured on."""
    out = []
    for day, day_bouts in by_date(b for b in bouts if b.date <= train_end):
        rated, decided = [], []
        for bout in day_bouts:
            r = rated_performance(bout, model)
            if r is None:
                continue
            rated.append(r)
            if bout.result == "win" and bout.winner_id is not None:
                decided.append((bout.winner_id, bout.opponent_of(bout.winner_id)))
        out.append((day.toordinal(), rated, decided))
    return out


def _elo_loss(days, params: EloParams) -> tuple[float, int]:
    tracker = EloTracker(params)
    total, count = 0.0, 0
    for ordinal, rated, decided in days:
        for winner, loser in decided:
            total -= math.log(
                expected_score(
                    tracker.rating(winner), tracker.rating(loser), params.scale
                )
            )
            count += 1
        tracker.commit(ordinal, rated)
    return (total / count if count else math.nan), count


def fit_performance_elo(
    bouts: Iterable[BoutFacts],
    model: PerformanceModel,
    train_end: date,
    grid: Mapping[str, tuple[float, ...]] | None = None,
) -> PerformanceEloFit:
    """Step sizes with the lowest log loss of pre-fight probabilities against
    actual winners, on bouts through ``train_end``. The finish multiplier is
    1: the deserved share already says how a bout went. Ties go to the first
    combination in grid order."""
    grid = dict(grid or PERFORMANCE_ELO_GRID)
    days = _days(list(bouts), model, train_end)
    names = list(grid)
    best: tuple[float, EloParams, int] | None = None
    for values in product(*(grid[name] for name in names)):
        params = EloParams(**dict(zip(names, values, strict=True)))
        loss, count = _elo_loss(days, params)
        if best is None or loss < best[0]:
            best = (loss, params, count)
    if best is None:
        raise ValueError("empty parameter grid")
    loss, params, count = best
    edges = tuple(
        name
        for name in names
        if len(grid[name]) > 1
        and getattr(params, name) in (grid[name][0], grid[name][-1])
    )
    return PerformanceEloFit(params, loss, count, edges)
