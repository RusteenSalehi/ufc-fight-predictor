"""Per-fighter history that only ever grows forward in time.

This is the one place bouts enter the feature pipeline, and the one place
histories are read. Reads go through ``FighterHistory.index_before``, which
returns how many of a fighter's bouts fall on dates strictly before a cutoff.
A bout on the cutoff date itself, such as another bout on the same card, is
never included.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np

from ufc.features.components import (
    COMPONENTS,
    N_COMPONENTS,
    bout_components,
    outcome_for,
)
from ufc.features.config import DECAY_ORIGIN, FeatureConfig
from ufc.features.data import BoutFacts, FighterAttrs
from ufc.features.elo import EloParams, EloTracker
from ufc.features.shrink import RatePriors


@dataclass(frozen=True, slots=True)
class FittedParams:
    """Everything in the pipeline that is fitted to data.

    Every part is fitted on the training period only (leakage rule 4).
    ``league`` is the average component vector of one fighter in one bout
    with stats; it is the prior for opponent baselines. ``rate_priors`` holds
    the weight-class average rates that per-minute rates are shrunk toward;
    it is only needed when ``FeatureConfig.shrink_minutes`` is positive.
    """

    elo: EloParams
    league: np.ndarray
    rate_priors: RatePriors | None = None

    def to_dict(self) -> dict:
        return {
            "elo": self.elo.to_dict(),
            "league": dict(zip(COMPONENTS, self.league.tolist(), strict=True)),
            "rate_priors": self.rate_priors.to_dict() if self.rate_priors else None,
        }

    @classmethod
    def from_dict(cls, raw: Mapping) -> FittedParams:
        priors = raw.get("rate_priors")
        return cls(
            elo=EloParams(**raw["elo"]),
            league=np.array([raw["league"][name] for name in COMPONENTS], dtype=float),
            rate_priors=RatePriors.from_dict(priors) if priors else None,
        )


class FighterHistory:
    """One fighter's bouts in date order, with running sums.

    ``cum[i]`` is the sum of the first ``i`` component vectors and ``cum_w[i]``
    the same sum with each bout weighted for time decay, so any window over
    the history is one or two lookups.
    """

    __slots__ = (
        "cum",
        "cum_w",
        "dates",
        "opp_ids",
        "outcomes",
        "rows",
        "vs",
        "weight_classes",
        "weights",
    )

    def __init__(self) -> None:
        zeros = np.zeros(N_COMPONENTS)
        self.dates: list[int] = []
        self.rows: list[np.ndarray] = []
        self.weights: list[float] = []
        self.cum: list[np.ndarray] = [zeros]
        self.cum_w: list[np.ndarray] = [zeros]
        self.opp_ids: list[str] = []
        self.outcomes: list[str] = []
        self.weight_classes: list[str | None] = []
        self.vs: dict[str, list[int]] = {}

    def index_before(self, ordinal: int) -> int:
        """Number of bouts on dates strictly before ``ordinal``."""
        return bisect_left(self.dates, ordinal)

    def append(
        self,
        ordinal: int,
        row: np.ndarray,
        weight: float,
        opponent_id: str,
        outcome: str,
        weight_class: str | None,
    ) -> None:
        self.vs.setdefault(opponent_id, []).append(len(self.dates))
        self.dates.append(ordinal)
        self.rows.append(row)
        self.weights.append(weight)
        self.cum.append(self.cum[-1] + row)
        self.cum_w.append(self.cum_w[-1] + weight * row)
        self.opp_ids.append(opponent_id)
        self.outcomes.append(outcome)
        self.weight_classes.append(weight_class)


_NO_HISTORY = FighterHistory()


class FeatureStore:
    """All fighters' histories and Elo ratings up to the last committed date."""

    def __init__(
        self,
        fighters: Mapping[str, FighterAttrs],
        config: FeatureConfig,
        params: FittedParams,
    ) -> None:
        self.fighters = fighters
        self.config = config
        self.params = params
        self.elo = EloTracker(params.elo)
        self._histories: dict[str, FighterHistory] = {}
        self._prior = config.prior_fights * params.league
        if config.shrink_minutes > 0 and params.rate_priors is None:
            raise ValueError("shrink_minutes is set but no rate priors were fitted")
        self.last_date: date | None = None

    def history(self, fighter_id: str) -> FighterHistory:
        return self._histories.get(fighter_id, _NO_HISTORY)

    def attrs(self, fighter_id: str) -> FighterAttrs:
        """Physical attributes; all unknown for a fighter with no profile."""
        return self.fighters.get(fighter_id) or FighterAttrs(fighter_id)

    def decay_weight(self, ordinal: int) -> float:
        return 2.0 ** ((ordinal - DECAY_ORIGIN) / self.config.decay_half_life_days)

    def shrink_prior(self, weight_class: str | None) -> np.ndarray | None:
        """``shrink_minutes`` average minutes of ``weight_class`` as a
        component vector, or ``None`` when shrinkage is off."""
        minutes = self.config.shrink_minutes
        if minutes <= 0 or self.params.rate_priors is None:
            return None
        return minutes * self.params.rate_priors.for_class(weight_class)

    def commit_date(self, day: date, bouts: Iterable[BoutFacts]) -> None:
        """Add every bout of one date. Dates must arrive in increasing order."""
        if self.last_date is not None and day <= self.last_date:
            raise ValueError(f"{day} committed after {self.last_date}")
        bouts = list(bouts)
        ordinal = day.toordinal()
        weight = self.decay_weight(ordinal)
        for bout in bouts:
            if bout.date != day:
                raise ValueError(f"bout {bout.bout_id} is not on {day}")
            for fighter_id in (bout.fighter_a_id, bout.fighter_b_id):
                opponent_id = bout.opponent_of(fighter_id)
                # Elo has not been advanced yet, so this is the opponent's
                # rating from before the date.
                row = bout_components(bout, fighter_id, self.elo.rating(opponent_id))
                self._histories.setdefault(fighter_id, FighterHistory()).append(
                    ordinal,
                    row,
                    weight,
                    opponent_id,
                    outcome_for(bout, fighter_id),
                    bout.weight_class,
                )
        self.elo.commit_date(ordinal, bouts)
        self.last_date = day

    def baseline_excluding(
        self, opponent_id: str, fighter_id: str, ordinal: int
    ) -> np.ndarray:
        """What ``opponent_id`` did against everyone except ``fighter_id``.

        The sum of the opponent's component vectors over bouts strictly before
        ``ordinal``, leaving out his bouts against ``fighter_id``, plus
        ``prior_fights`` league-average bouts.
        """
        history = self.history(opponent_id)
        idx = history.index_before(ordinal)
        total = history.cum[idx].copy()
        for i in history.vs.get(fighter_id, ()):
            if i < idx:
                total -= history.rows[i]
        return total + self._prior
