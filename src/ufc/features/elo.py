"""Elo ratings, as of any date, with parameters fitted on the training period.

Ratings change once per date: every bout on a date is scored against the
ratings from before that date, and the changes are applied together after it.
A bout therefore never sees a rating that was moved by another bout on the
same card, including the early one-night tournaments where a fighter fought
several times in an evening.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from itertools import groupby, product
from typing import NamedTuple

from ufc.features.components import KO_METHODS, SUB_METHODS
from ufc.features.data import BoutFacts, sort_key


@dataclass(frozen=True, slots=True)
class EloParams:
    """``k`` is the basic step. A finish (KO/TKO or submission) multiplies it
    by ``finish_mult``. A fighter's first ``provisional_bouts`` rated bouts
    multiply his own step by ``provisional_mult``, so newcomers, who all start
    at ``initial``, move to their level faster."""

    k: float = 32.0
    finish_mult: float = 1.0
    provisional_mult: float = 1.0
    provisional_bouts: int = 3
    initial: float = 1500.0
    scale: float = 400.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def expected_score(
    rating: float, opponent_rating: float, scale: float = 400.0
) -> float:
    """Probability that the first fighter wins, from the rating difference."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / scale))


class RatedBout(NamedTuple):
    fighter_a_id: str
    fighter_b_id: str
    score_a: float  # 1 win, 0 loss, 0.5 draw
    is_finish: bool


def rated_bout(bout: BoutFacts) -> RatedBout | None:
    """What Elo needs from a bout. ``None`` for a no contest, which is not rated."""
    if bout.result == "win":
        score = 1.0 if bout.winner_id == bout.fighter_a_id else 0.0
        finish = bout.method in KO_METHODS or bout.method in SUB_METHODS
    elif bout.result == "draw":
        score, finish = 0.5, False
    else:
        return None
    return RatedBout(bout.fighter_a_id, bout.fighter_b_id, score, finish)


class EloTracker:
    """Ratings that advance one date at a time and remember their history."""

    def __init__(self, params: EloParams) -> None:
        self.params = params
        self._rating: dict[str, float] = {}
        self._rated_bouts: dict[str, int] = defaultdict(int)
        self._dates: dict[str, list[int]] = defaultdict(list)
        self._after: dict[str, list[float]] = defaultdict(list)
        self._last_ordinal: int | None = None

    def rating(self, fighter_id: str) -> float:
        """The rating after every date committed so far."""
        return self._rating.get(fighter_id, self.params.initial)

    def rating_before(self, fighter_id: str, ordinal: int) -> float:
        """The rating from bouts on dates strictly before ``ordinal``."""
        dates = self._dates.get(fighter_id)
        if not dates:
            return self.params.initial
        i = bisect_left(dates, ordinal)
        return self._after[fighter_id][i - 1] if i else self.params.initial

    def commit(self, ordinal: int, bouts: Iterable[RatedBout]) -> None:
        """Apply one date's bouts. All of them use the ratings from before it."""
        if self._last_ordinal is not None and ordinal <= self._last_ordinal:
            raise ValueError("dates must be committed in increasing order")
        self._last_ordinal = ordinal
        p = self.params
        delta: dict[str, float] = defaultdict(float)
        seen: dict[str, int] = defaultdict(int)
        for a, b, score_a, is_finish in bouts:
            expected_a = expected_score(self.rating(a), self.rating(b), p.scale)
            step = p.k * (p.finish_mult if is_finish else 1.0)
            for fighter, surprise in (
                (a, score_a - expected_a),
                (b, expected_a - score_a),
            ):
                provisional = self._rated_bouts[fighter] < p.provisional_bouts
                delta[fighter] += (
                    step * (p.provisional_mult if provisional else 1.0) * surprise
                )
                seen[fighter] += 1
        for fighter, change in delta.items():
            rating = self.rating(fighter) + change
            self._rating[fighter] = rating
            self._rated_bouts[fighter] += seen[fighter]
            self._dates[fighter].append(ordinal)
            self._after[fighter].append(rating)

    def commit_date(self, ordinal: int, bouts: Iterable[BoutFacts]) -> None:
        self.commit(ordinal, (r for r in map(rated_bout, bouts) if r is not None))


def by_date(bouts: Iterable[BoutFacts]) -> Iterator[tuple[date, list[BoutFacts]]]:
    """Bouts grouped by date, earliest first."""
    for day, group in groupby(sorted(bouts, key=sort_key), key=lambda b: b.date):
        yield day, list(group)


@dataclass(frozen=True, slots=True)
class EloPrediction:
    """Ratings before a bout and the win probability they imply for fighter A."""

    bout_id: str
    date: date
    rating_a: float
    rating_b: float
    p_a: float
    label: float | None  # 1 if A won, 0 if B won, None for a draw or no contest


def run_elo(bouts: Iterable[BoutFacts], params: EloParams) -> list[EloPrediction]:
    """Replay history and return every bout's pre-fight ratings."""
    tracker = EloTracker(params)
    predictions = []
    for day, day_bouts in by_date(bouts):
        for bout in day_bouts:
            rating_a = tracker.rating(bout.fighter_a_id)
            rating_b = tracker.rating(bout.fighter_b_id)
            label = None
            if bout.result == "win":
                label = 1.0 if bout.winner_id == bout.fighter_a_id else 0.0
            predictions.append(
                EloPrediction(
                    bout.bout_id,
                    day,
                    rating_a,
                    rating_b,
                    expected_score(rating_a, rating_b, params.scale),
                    label,
                )
            )
        tracker.commit_date(day.toordinal(), day_bouts)
    return predictions


def _log_loss(days: Sequence[tuple[int, list[RatedBout]]], params: EloParams) -> float:
    """Mean log loss of pre-fight predictions over bouts that had a winner."""
    tracker = EloTracker(params)
    total, count = 0.0, 0
    for ordinal, rated in days:
        for a, b, score_a, _ in rated:
            if score_a == 0.5:
                continue
            p_a = expected_score(tracker.rating(a), tracker.rating(b), params.scale)
            total -= math.log(p_a if score_a == 1.0 else 1.0 - p_a)
            count += 1
        tracker.commit(ordinal, rated)
    return total / count if count else math.nan


DEFAULT_GRID: dict[str, tuple[float, ...]] = {
    "k": (16, 24, 32, 40, 48, 56, 64, 80, 96, 112),
    "finish_mult": (1.0, 1.15, 1.3, 1.5),
    "provisional_mult": (1.0, 1.5, 2.0, 2.5),
}


@dataclass(frozen=True, slots=True)
class EloFit:
    params: EloParams
    log_loss: float
    n_bouts: int  # bouts with a winner that the loss was averaged over
    last_date: date | None  # latest bout the fit saw
    on_grid_edge: tuple[str, ...]  # parameters whose best value is a grid extreme


def fit_elo_params(
    bouts: Iterable[BoutFacts],
    train_end: date,
    grid: dict[str, tuple[float, ...]] | None = None,
) -> EloFit:
    """Choose the parameters with the lowest log loss on the training period.

    Bouts after ``train_end`` are discarded before anything else happens.
    Each prediction in the loss is made from ratings built on earlier dates
    only, so the loss is an honest one-step-ahead score even though it is
    measured on the data the parameters are chosen from. Ties go to the first
    combination in grid order.
    """
    grid = grid or DEFAULT_GRID
    training = [b for b in bouts if b.date <= train_end]
    days = [
        (day.toordinal(), [r for r in map(rated_bout, day_bouts) if r is not None])
        for day, day_bouts in by_date(training)
    ]
    n_bouts = sum(1 for _, rated in days for r in rated if r.score_a != 0.5)

    names = list(grid)
    best: tuple[float, EloParams] | None = None
    for values in product(*(grid[name] for name in names)):
        params = EloParams(**dict(zip(names, values, strict=True)))
        loss = _log_loss(days, params)
        if best is None or loss < best[0]:
            best = (loss, params)
    if best is None:
        raise ValueError("empty parameter grid")
    loss, params = best
    edges = tuple(
        name
        for name in names
        if len(grid[name]) > 1
        and getattr(params, name) in (grid[name][0], grid[name][-1])
    )
    return EloFit(
        params=params,
        log_loss=loss,
        n_bouts=n_bouts,
        last_date=max((b.date for b in training), default=None),
        on_grid_edge=edges,
    )
