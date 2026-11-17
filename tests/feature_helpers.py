"""Build small leagues for feature tests.

A ``League`` is assembled bout by bout, loaded into an in-memory SQLite
database with the real loaders, and read back with the real reader, so feature
tests cover the path from tables to features. ``random_league`` generates a
few hundred invented bouts for the leakage and symmetry tests.
"""

from __future__ import annotations

import random
from dataclasses import replace
from datetime import date, timedelta

import numpy as np

from ufc.db import loaders
from ufc.db.connection import connect
from ufc.features.components import N_COMPONENTS
from ufc.features.config import FeatureConfig
from ufc.features.data import BoutFacts, FeatureData, load_feature_data
from ufc.features.elo import EloParams
from ufc.features.store import FittedParams
from ufc.ingest.models import (
    Bout,
    Dataset,
    Event,
    Fighter,
    FightStats,
    RoundStats,
    StatLine,
    sum_stat_lines,
)
from ufc.ingest.parsing import fight_time_s, parse_time_format

THREE_ROUNDS = "3 Rnd (5-5-5)"
FIVE_ROUNDS = "5 Rnd (5-5-5-5-5)"
SEEN_AT = "2020-01-01T00:00:00Z"


def line(
    sig: tuple[int, int] = (0, 0),
    td: tuple[int, int] = (0, 0),
    kd: int = 0,
    sub: int = 0,
    ctrl: int | None = 0,
    head: int | None = None,
    body: int = 0,
    leg: int = 0,
    clinch: int = 0,
    ground: int = 0,
) -> StatLine:
    """A stat line from the few numbers a test cares about.

    Significant strikes default to head strikes at distance.
    """
    landed, attempted = sig
    head_landed = landed - body - leg if head is None else head
    return StatLine(
        knockdowns=kd,
        sig_str_landed=landed,
        sig_str_attempted=attempted,
        total_str_landed=landed,
        total_str_attempted=attempted,
        td_landed=td[0],
        td_attempted=td[1],
        sub_attempts=sub,
        reversals=0,
        ctrl_s=ctrl,
        head_landed=head_landed,
        head_attempted=head_landed,
        body_landed=body,
        body_attempted=body,
        leg_landed=leg,
        leg_attempted=leg,
        distance_landed=landed - clinch - ground,
        distance_attempted=landed - clinch - ground,
        clinch_landed=clinch,
        clinch_attempted=clinch,
        ground_landed=ground,
        ground_attempted=ground,
    )


class League:
    """Events, fighters and bouts, added in any order."""

    def __init__(self) -> None:
        self.dataset = Dataset()
        self._events: dict[str, date] = {}

    def fighter(
        self,
        height_cm: float | None = 180.0,
        reach_cm: float | None = 185.0,
        stance: str | None = "Orthodox",
        dob: date | None = date(1990, 1, 1),
    ) -> str:
        fighter_id = f"f{len(self.dataset.fighters):015x}"
        self.dataset.fighters.append(
            Fighter(
                fighter_id,
                f"Fighter {len(self.dataset.fighters)}",
                None,
                height_cm,
                reach_cm,
                stance,
                dob,
            )
        )
        return fighter_id

    def event(self, day: date) -> str:
        event_id = f"e{len(self.dataset.events):015x}"
        self.dataset.events.append(
            Event(event_id, f"Event {len(self.dataset.events)}", day, None)
        )
        self._events[event_id] = day
        return event_id

    def bout(
        self,
        event_id: str,
        fighter_a: str,
        fighter_b: str,
        *,
        winner: str | None = None,
        result: str | None = None,
        method: str = "Decision - Unanimous",
        rounds_a: list[StatLine] | None = None,
        rounds_b: list[StatLine] | None = None,
        end_round: int | None = None,
        end_time_s: int = 300,
        weight_class: str | None = "Lightweight",
        is_title_bout: bool = False,
        time_format: str = THREE_ROUNDS,
    ) -> str:
        """Add a bout. ``winner`` defaults to fighter A unless ``result`` is
        ``draw`` or ``nc``. Without ``rounds_a`` the bout has no stats."""
        result = result or "win"
        if result == "win":
            winner = winner or fighter_a
        has_stats = rounds_a is not None
        if has_stats:
            assert rounds_b is not None and len(rounds_a) == len(rounds_b)
            end_round = end_round or len(rounds_a)
            assert end_round == len(rounds_a)
        end_round = end_round or 3
        parsed = parse_time_format(time_format)
        bout_id = f"b{len(self.dataset.bouts):015x}"
        self.dataset.bouts.append(
            Bout(
                bout_id=bout_id,
                event_id=event_id,
                fighter_a_id=fighter_a,
                fighter_b_id=fighter_b,
                weight_class=weight_class,
                bout_type_raw=f"{weight_class} Bout",
                is_womens=False,
                is_title_bout=is_title_bout,
                is_interim_title=False,
                is_tournament_final=False,
                scheduled_rounds=parsed.scheduled_rounds,
                time_format=time_format,
                status="completed",
                result=result,
                winner_id=winner if result == "win" else None,
                method=method,
                method_details=None,
                referee=None,
                end_round=end_round,
                end_time_s=end_time_s,
                fight_time_s=fight_time_s(parsed, end_round, end_time_s),
                has_stats=has_stats,
            )
        )
        if has_stats:
            for fighter_id, lines in ((fighter_a, rounds_a), (fighter_b, rounds_b)):
                for number, stats in enumerate(lines, start=1):
                    self.dataset.round_stats.append(
                        RoundStats(bout_id, fighter_id, number, stats)
                    )
                self.dataset.fight_stats.append(
                    FightStats(bout_id, fighter_id, sum_stat_lines(lines))
                )
        return bout_id

    def data(self) -> FeatureData:
        """Load into SQLite and read back the way the pipeline does."""
        conn = connect(":memory:")
        try:
            loaders.load_dataset(conn, self.dataset, seen_at=SEEN_AT)
            return load_feature_data(conn)
        finally:
            conn.close()


def plain_params(k: float = 32.0, league: np.ndarray | None = None) -> FittedParams:
    """Fixed parameters: textbook Elo and, by default, no league prior."""
    return FittedParams(
        elo=EloParams(k=k),
        league=np.zeros(N_COMPONENTS) if league is None else league,
    )


NO_PRIOR = FeatureConfig(prior_fights=0.0)


def without_bouts_after(data: FeatureData, cutoff: date) -> FeatureData:
    return replace(data, bouts=[b for b in data.bouts if b.date <= cutoff])


def swap_corners(bout: BoutFacts) -> BoutFacts:
    return replace(bout, fighter_a_id=bout.fighter_b_id, fighter_b_id=bout.fighter_a_id)


_METHODS = (
    "Decision - Unanimous",
    "Decision - Split",
    "KO/TKO",
    "Submission",
    "TKO - Doctor's Stoppage",
)
_WEIGHT_CLASSES = ("Lightweight", "Welterweight", "Middleweight", "Catch Weight", None)
_STANCES = ("Orthodox", "Southpaw", "Switch", None)


def _random_line(rng: random.Random, with_ctrl: bool) -> StatLine:
    attempted = rng.randint(0, 60)
    landed = rng.randint(0, attempted)
    body = rng.randint(0, landed // 3)
    leg = rng.randint(0, landed // 3)
    clinch = rng.randint(0, landed // 4)
    ground = rng.randint(0, landed // 4)
    td_attempted = rng.choice((0, 0, 1, 2, 5))
    return line(
        sig=(landed, attempted),
        td=(rng.randint(0, td_attempted), td_attempted),
        kd=rng.choice((0, 0, 0, 1)),
        sub=rng.choice((0, 0, 1)),
        ctrl=rng.randint(0, 200) if with_ctrl else None,
        body=body,
        leg=leg,
        clinch=clinch,
        ground=ground,
    )


def random_league(
    seed: int = 7,
    n_fighters: int = 28,
    n_events: int = 60,
    start: date = date(2015, 1, 3),
) -> League:
    """An invented league with the awkward cases real data has.

    It includes draws, no contests, bouts without stats, stat lines without
    control time, fighters missing attributes, title bouts over five rounds,
    two events on one date, and a one-night tournament in which a fighter
    fights twice.
    """
    rng = random.Random(seed)
    league = League()
    fighters = [
        league.fighter(
            height_cm=rng.choice((None, 170.0, 175.0, 180.0, 185.0)),
            reach_cm=rng.choice((None, 175.0, 180.0, 190.0)),
            stance=rng.choice(_STANCES),
            dob=None
            if rng.random() < 0.1
            else date(1985, 1, 1) + timedelta(days=rng.randint(0, 3000)),
        )
        for _ in range(n_fighters)
    ]

    day = start
    for number in range(n_events):
        day += timedelta(days=rng.randint(7, 35))
        # Event 20 shares its date with a second event; event 30 is a tournament.
        event_ids = [league.event(day)]
        if number == 20:
            event_ids.append(league.event(day))
        card = rng.sample(fighters, 8 if number != 20 else 12)
        pairs = list(zip(card[0::2], card[1::2], strict=True))
        if number == 30:
            pairs.append((pairs[0][0], pairs[1][0]))  # two bouts for two fighters

        for i, (a, b) in enumerate(pairs):
            is_title = rng.random() < 0.1
            time_format = FIVE_ROUNDS if is_title else THREE_ROUNDS
            max_rounds = 5 if is_title else 3
            roll = rng.random()
            result = "draw" if roll < 0.04 else "nc" if roll < 0.08 else "win"
            method = rng.choice(_METHODS) if result == "win" else "Overturned"
            went_distance = result != "win" or method.startswith("Decision")
            end_round = max_rounds if went_distance else rng.randint(1, max_rounds)
            end_time_s = 300 if went_distance else rng.randint(5, 299)
            has_stats = rng.random() > 0.08
            with_ctrl = rng.random() > 0.1
            league.bout(
                event_ids[i % len(event_ids)],
                a,
                b,
                winner=rng.choice((a, b)),
                result=result,
                method=method,
                rounds_a=[_random_line(rng, with_ctrl) for _ in range(end_round)]
                if has_stats
                else None,
                rounds_b=[_random_line(rng, with_ctrl) for _ in range(end_round)]
                if has_stats
                else None,
                end_round=end_round,
                end_time_s=end_time_s,
                weight_class=rng.choice(_WEIGHT_CLASSES),
                is_title_bout=is_title,
                time_format=time_format,
            )
    return league
