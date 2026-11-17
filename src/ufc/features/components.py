"""One bout, seen from one fighter, as a vector of additive components.

Every feature that is a rate is computed as (sum of a numerator component) /
(sum of a denominator component) over the bouts in a window. Each numerator
has a denominator that is zero whenever the numerator is unknown: a bout with
no recorded stats adds zero strikes *and* zero stat seconds. Sums therefore
never contain a fake zero, and a window with nothing recorded has a zero
denominator, which becomes NaN.
"""

from __future__ import annotations

import numpy as np

from ufc.features.data import BoutFacts, StatMap
from ufc.ingest.parsing import ParseError, parse_time_format

COMPONENTS: tuple[str, ...] = (
    # Record: every bout, with or without stats.
    "n",  # bouts
    "decided",  # wins + losses + draws (no contests excluded)
    "win",
    "loss",
    "draw",
    "nc",
    "ko_win",
    "sub_win",
    "finish_win",
    "ko_loss",
    "sub_loss",
    "title",  # UFC title bouts
    "title_win",
    "cage_s",  # seconds fought
    "opp_elo",  # opponent's Elo before the bout
    # Stats: bouts with recorded stats only.
    "n_stat",
    "t",  # seconds fought in bouts with stats
    "kd",
    "sig_l",
    "sig_a",
    "td_l",
    "td_a",
    "sub_att",
    "head_l",
    "body_l",
    "leg_l",
    "dist_l",
    "clinch_l",
    "ground_l",
    "opp_kd",
    "opp_sig_l",
    "opp_sig_a",
    "opp_td_l",
    "opp_td_a",
    "opp_sub_att",
    "opp_head_l",
    # Control time is not recorded for some early bouts; it has its own
    # exposure so those bouts drop out of control shares only.
    "ctrl",
    "t_ctrl",
    "opp_ctrl",
    "t_opp_ctrl",
    # Cardio: bouts that reached round 3, from per-round stats.
    "c_n",
    "r1_t",
    "r1_sig_l",
    "r1_sig_a",
    "r1_opp_sig_l",
    "r1_opp_sig_a",
    "late_t",  # seconds fought in rounds 3 and later
    "late_sig_l",
    "late_sig_a",
    "late_opp_sig_l",
    "late_opp_sig_a",
)

IDX: dict[str, int] = {name: i for i, name in enumerate(COMPONENTS)}
N_COMPONENTS = len(COMPONENTS)

KO_METHODS = frozenset({"KO/TKO", "TKO - Doctor's Stoppage"})
SUB_METHODS = frozenset({"Submission"})

CARDIO_FIRST_LATE_ROUND = 3


def outcome_for(bout: BoutFacts, fighter_id: str) -> str:
    """``win``, ``loss``, ``draw`` or ``nc`` from this fighter's side."""
    if bout.result == "win":
        return "win" if bout.winner_id == fighter_id else "loss"
    return bout.result


def has_usable_stats(bout: BoutFacts) -> bool:
    """Stats recorded for both fighters, and a known, positive fight time."""
    return (
        bout.has_stats
        and bool(bout.fight_time_s)
        and bout.fighter_a_id in bout.totals
        and bout.fighter_b_id in bout.totals
    )


def round_seconds(bout: BoutFacts) -> list[int] | None:
    """Seconds fought in each round, or ``None`` if the format cannot say.

    Rounds before the last ran their full length; the last ran until the
    recorded end time.
    """
    if bout.end_round is None or bout.end_time_s is None or not bout.time_format:
        return None
    try:
        time_format = parse_time_format(bout.time_format)
    except ParseError:
        return None
    if time_format.no_time_limit or bout.end_round > len(time_format.round_minutes):
        return None
    full = [minutes * 60 for minutes in time_format.round_minutes[: bout.end_round - 1]]
    return [*full, bout.end_time_s]


def _cardio_rounds(
    bout: BoutFacts, fighter_id: str, opponent_id: str
) -> tuple[list[int], list[StatMap], list[StatMap]] | None:
    """Round lengths and both fighters' round lines, if the bout qualifies.

    A bout qualifies only if it went past round 2 and has a stat line for
    every round fought.
    """
    if not has_usable_stats(bout) or (bout.end_round or 0) < CARDIO_FIRST_LATE_ROUND:
        return None
    seconds = round_seconds(bout)
    own = bout.rounds.get(fighter_id, ())
    opp = bout.rounds.get(opponent_id, ())
    if seconds is None or len(own) != len(seconds) or len(opp) != len(seconds):
        return None
    return seconds, list(own), list(opp)


def bout_components(bout: BoutFacts, fighter_id: str, opp_elo: float) -> np.ndarray:
    """The component vector of ``bout`` for ``fighter_id``.

    ``opp_elo`` is the opponent's rating before the bout's date.
    """
    opponent_id = bout.opponent_of(fighter_id)
    v = np.zeros(N_COMPONENTS)

    def put(name: str, value: float) -> None:
        v[IDX[name]] = value

    outcome = outcome_for(bout, fighter_id)
    put("n", 1)
    put(outcome, 1)
    put("decided", 0 if outcome == "nc" else 1)
    if outcome == "win":
        ko, sub = bout.method in KO_METHODS, bout.method in SUB_METHODS
        put("ko_win", ko)
        put("sub_win", sub)
        put("finish_win", ko or sub)
        put("title_win", bout.is_title_bout)
    elif outcome == "loss":
        put("ko_loss", bout.method in KO_METHODS)
        put("sub_loss", bout.method in SUB_METHODS)
    put("title", bout.is_title_bout)
    put("cage_s", bout.fight_time_s or 0)
    put("opp_elo", opp_elo)

    if has_usable_stats(bout):
        own, opp = bout.totals[fighter_id], bout.totals[opponent_id]
        seconds = float(bout.fight_time_s or 0)
        put("n_stat", 1)
        put("t", seconds)
        put("kd", own["knockdowns"])
        put("sig_l", own["sig_str_landed"])
        put("sig_a", own["sig_str_attempted"])
        put("td_l", own["td_landed"])
        put("td_a", own["td_attempted"])
        put("sub_att", own["sub_attempts"])
        put("head_l", own["head_landed"])
        put("body_l", own["body_landed"])
        put("leg_l", own["leg_landed"])
        put("dist_l", own["distance_landed"])
        put("clinch_l", own["clinch_landed"])
        put("ground_l", own["ground_landed"])
        put("opp_kd", opp["knockdowns"])
        put("opp_sig_l", opp["sig_str_landed"])
        put("opp_sig_a", opp["sig_str_attempted"])
        put("opp_td_l", opp["td_landed"])
        put("opp_td_a", opp["td_attempted"])
        put("opp_sub_att", opp["sub_attempts"])
        put("opp_head_l", opp["head_landed"])
        if own["ctrl_s"] is not None:
            put("ctrl", own["ctrl_s"])
            put("t_ctrl", seconds)
        if opp["ctrl_s"] is not None:
            put("opp_ctrl", opp["ctrl_s"])
            put("t_opp_ctrl", seconds)

    cardio = _cardio_rounds(bout, fighter_id, opponent_id)
    if cardio is not None:
        seconds_by_round, own_rounds, opp_rounds = cardio
        late = slice(CARDIO_FIRST_LATE_ROUND - 1, None)
        put("c_n", 1)
        put("r1_t", seconds_by_round[0])
        put("r1_sig_l", own_rounds[0]["sig_str_landed"])
        put("r1_sig_a", own_rounds[0]["sig_str_attempted"])
        put("r1_opp_sig_l", opp_rounds[0]["sig_str_landed"])
        put("r1_opp_sig_a", opp_rounds[0]["sig_str_attempted"])
        put("late_t", sum(seconds_by_round[late]))
        put("late_sig_l", sum(r["sig_str_landed"] for r in own_rounds[late]))
        put("late_sig_a", sum(r["sig_str_attempted"] for r in own_rounds[late]))
        put("late_opp_sig_l", sum(r["sig_str_landed"] for r in opp_rounds[late]))
        put("late_opp_sig_a", sum(r["sig_str_attempted"] for r in opp_rounds[late]))
    return v
