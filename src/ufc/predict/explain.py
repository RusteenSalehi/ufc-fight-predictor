"""A plain-language "why" for each prediction, built from SHAP factors.

Everything here is deterministic: the contributions come from the model, and
the text comes from fixed templates filled with the fighters' as-of values.
No language model is involved.

1. **Contributions.** TreeSHAP gives each input's contribution to the
   log-odds of one orientation. The model is asked both ways round, so the
   contribution toward fighter A is ``(phi(A, B) - phi(B, A)) / 2``. Swapping
   the fighters swaps the two orientations, so every contribution changes
   sign exactly and the explanation of the swapped bout is the mirror image.
   They describe what the model weighs, before calibration; they are not
   causes, and the wording never says they are.
2. **Reasons.** Inputs that measure the same thing are summed into one
   reason (``group_of``): age with UFC fight count and the age-shape
   columns; Elo with opponent quality; each per-minute or percentage stat
   with its opponent-adjusted version across every window. Otherwise inputs
   are grouped by stat or family.
3. **Text.** The favoured fighter (higher calibrated probability), the three
   reasons the model weighs most toward them, and the one it weighs most
   toward the other fighter. Each reason is written from the side of the
   fighter it points toward, with both fighters' values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ufc.features.matchup import PAIRS, Matchup
from ufc.models.design import matrix, swapped
from ufc.models.recipe import Candidate

METHOD = (
    "TreeSHAP contributions to the model's log-odds before calibration, "
    "averaged over both orientations of the bout (contribution toward A = "
    "(phi(A, B) - phi(B, A)) / 2), summed within each reason. They show what "
    "the model weighs, not what causes a result."
)
N_FOR = 3
WINDOWS = {
    "career": "career",
    "l3": "last 3 fights",
    "l5": "last 5 fights",
    "decay": "recent fights weighted most",
}


@dataclass(frozen=True, slots=True)
class Stat:
    label: str
    kind: str  # rate, pct, pp, ratio, count, years, cm, days, elo, flag


STATS: dict[str, Stat] = {
    # Striking
    "slpm": Stat("significant strikes landed per minute", "rate"),
    "sapm": Stat("significant strikes absorbed per minute", "rate"),
    "sig_acc": Stat("significant-strike accuracy", "pct"),
    "sig_def": Stat("significant-strike defence", "pct"),
    "sig_att_pm": Stat("significant strikes attempted per minute", "rate"),
    "kd_per15": Stat("knockdowns scored per 15 minutes", "rate"),
    "head_share": Stat("share of strikes landed to the head", "pct"),
    "body_share": Stat("share of strikes landed to the body", "pct"),
    "leg_share": Stat("share of strikes landed to the legs", "pct"),
    "distance_share": Stat("share of strikes landed at distance", "pct"),
    "clinch_share": Stat("share of strikes landed in the clinch", "pct"),
    "ground_share": Stat("share of strikes landed on the ground", "pct"),
    # Grappling
    "td_per15": Stat("takedowns per 15 minutes", "rate"),
    "td_acc": Stat("takedown accuracy", "pct"),
    "td_def": Stat("takedown defence", "pct"),
    "sub_att_per15": Stat("submission attempts per 15 minutes", "rate"),
    "ctrl_share": Stat("share of cage time in control", "pct"),
    "ctrl_against_share": Stat("share of cage time controlled by opponents", "pct"),
    # Durability
    "kd_abs_per15": Stat("knockdowns absorbed per 15 minutes", "rate"),
    "head_abs_pm": Stat("head strikes absorbed per minute", "rate"),
    "ko_losses": Stat("KO/TKO losses", "count"),
    "sub_losses": Stat("submission losses", "count"),
    "kd_abs": Stat("knockdowns absorbed", "count"),
    # Record
    "win_rate": Stat("UFC win rate", "pct"),
    "finish_win_share": Stat("share of wins by finish", "pct"),
    "n_fights": Stat("UFC fights", "count"),
    "wins": Stat("UFC wins", "count"),
    "losses": Stat("UFC losses", "count"),
    "ko_wins": Stat("KO/TKO wins", "count"),
    "sub_wins": Stat("submission wins", "count"),
    "title_fights": Stat("title fights", "count"),
    "title_wins": Stat("title-fight wins", "count"),
    "cage_min": Stat("minutes in the UFC cage", "count"),
    "win_streak": Stat("current win streak", "count"),
    "loss_streak": Stat("current losing streak", "count"),
    "debut": Stat("UFC debut", "flag"),
    # Physical and age
    "age": Stat("age", "years"),
    "age_past_32": Stat("years past age 32", "years"),
    "age_past_class_median": Stat("years past the division's median age", "years"),
    "age_rel_class": Stat("age relative to the division's median", "years"),
    "height_cm": Stat("height", "cm"),
    "reach_cm": Stat("reach", "cm"),
    "stance_southpaw": Stat("southpaw stance", "flag"),
    "stance_switch": Stat("switch stance", "flag"),
    "stance_orthodox": Stat("orthodox stance", "flag"),
    # Rating, activity, weight
    "elo": Stat("rating", "elo"),
    "opp_elo": Stat("average opponent rating", "elo"),
    "days_since_last": Stat("days since the last fight", "days"),
    "fights_last_2y": Stat("fights in the last two years", "count"),
    "moved_up": Stat("moving up a weight class", "flag"),
    "moved_down": Stat("moving down a weight class", "flag"),
    "fights_at_weight": Stat("fights at this weight", "count"),
    # Cardio (rounds 3 and later against round 1)
    "cardio_slpm_ratio": Stat(
        "late-round strikes landed per minute vs round 1", "ratio"
    ),
    "cardio_sig_att_ratio": Stat(
        "late-round strikes thrown per minute vs round 1", "ratio"
    ),
    "cardio_sapm_ratio": Stat(
        "late-round strikes absorbed per minute vs round 1", "ratio"
    ),
    "cardio_sig_acc_delta": Stat("late-round change in striking accuracy", "pp"),
    "cardio_sig_def_delta": Stat("late-round change in striking defence", "pp"),
    "cardio_fights": Stat("fights that went past round 2", "count"),
    # Inputs of pair features only
    "sub_loss_rate": Stat("submission losses per fight", "pct"),
}

AGE_EXPERIENCE = {
    "age",
    "age_past_32",
    "age_past_class_median",
    "age_rel_class",
    "n_fights",
    "cage_min",
    "debut",
}
RATING = {"elo", "opp_elo"}
RECORD = {
    "win_rate",
    "wins",
    "losses",
    "win_streak",
    "loss_streak",
    "finish_win_share",
    "ko_wins",
    "sub_wins",
    "title_fights",
    "title_wins",
}
STRIKE_MIX = {
    "head_share",
    "body_share",
    "leg_share",
    "distance_share",
    "clinch_share",
    "ground_share",
}
DURABILITY = {"kd_abs_per15", "head_abs_pm", "ko_losses", "sub_losses", "kd_abs"}
ACTIVITY = {"days_since_last", "fights_last_2y"}
WEIGHT = {"moved_up", "moved_down", "fights_at_weight"}
STANCE = {"stance_southpaw", "stance_switch", "stance_southpaw_vs_orthodox"}
PER_STAT = {
    "slpm",
    "sapm",
    "sig_acc",
    "sig_def",
    "sig_att_pm",
    "kd_per15",
    "td_per15",
    "td_acc",
    "td_def",
    "sub_att_per15",
    "ctrl_share",
    "ctrl_against_share",
}
PAIR_BY_NAME = {p.name: p for p in PAIRS}

TITLES: dict[str, str] = {
    "age_experience": "the age and experience gap",
    "rating": "the rating gap and the opposition faced",
    "record": "the UFC record",
    "height": "the height difference",
    "reach": "the reach difference",
    "stance": "the stance matchup",
    "strike_mix": "where each fighter's strikes land",
    "durability": "damage absorbed so far",
    "activity": "recent activity",
    "weight_class": "weight-class history",
    "cardio": "late-round output",
    "style_matchup": "how the styles match up",
    **{key: STATS[key].label for key in PER_STAT},
}


@dataclass(frozen=True, slots=True)
class Parsed:
    """A model input: the per-fighter value it is built from (``source``),
    the stat (``stem``), its window and whether it is opponent-adjusted."""

    name: str
    source: str
    stem: str
    window: str | None
    adjusted: bool
    level: bool


def parse(name: str) -> Parsed:
    if name.startswith("lvl_"):
        source, level = name[4:], True
    elif name.startswith("d_"):
        source, level = name[2:], False
    else:
        raise ValueError(f"not a model input: {name!r}")
    rest = source
    adjusted = rest.startswith("adj_")
    if adjusted:
        rest = rest[4:]
    window = None
    head, _, tail = rest.rpartition("_")
    if tail in WINDOWS and head:
        rest, window = head, tail
    return Parsed(name, source, rest, window, adjusted, level)


def group_of(name: str) -> str:
    """The reason an input belongs to. Raises for an input it does not know,
    so a new feature cannot reach the site without a reason and a template."""
    p = parse(name)
    stem = p.stem
    if stem in PAIR_BY_NAME:
        return "stance" if stem in STANCE else "style_matchup"
    if stem.startswith("cardio_"):
        if stem not in STATS:
            raise KeyError(f"no reason for model input {name!r}")
        return "cardio"
    for group, members in (
        ("age_experience", AGE_EXPERIENCE),
        ("rating", RATING),
        ("record", RECORD),
        ("strike_mix", STRIKE_MIX),
        ("durability", DURABILITY),
        ("activity", ACTIVITY),
        ("weight_class", WEIGHT),
        ("stance", STANCE),
    ):
        if stem in members:
            return group
    if stem == "height_cm":
        return "height"
    if stem == "reach_cm":
        return "reach"
    if stem in PER_STAT:
        return stem
    raise KeyError(f"no reason for model input {name!r}")


# --- contributions ---------------------------------------------------------------


def contributions(candidate: Candidate, inputs) -> dict[str, float]:
    """Log-odds contribution of each input toward fighter A, for one bout
    (``inputs`` is ``model_inputs`` for it)."""
    spec = candidate.spec
    x = matrix(inputs, spec)
    if len(x) != 1:
        raise ValueError("explain one bout at a time")
    k = len(spec.names)
    ab = candidate.model.contributions(x)[0, :k]
    ba = candidate.model.contributions(swapped(x, spec))[0, :k]
    toward_a = (ab - ba) / 2.0
    return dict(zip(spec.names, (float(v) for v in toward_a), strict=True))


def group_totals(toward_a: Mapping[str, float]) -> dict[str, float]:
    """Contributions summed per reason, in a fixed order of inputs (sorted
    by name), so that the mirrored bout gives exactly the negated totals."""
    totals: dict[str, float] = {}
    for name in sorted(toward_a):
        group = group_of(name)
        totals[group] = totals.get(group, 0.0) + toward_a[name]
    return totals


# --- values and phrases ------------------------------------------------------------


def _finite(v: float | None) -> bool:
    return v is not None and isinstance(v, int | float) and math.isfinite(v)


def _value(kind: str, v: float | None, adjusted: bool = False) -> str:
    if not _finite(v):
        return "no data"
    sign = "+" if adjusted and v > 0 else ""
    if kind == "pct":
        return f"{sign}{100 * v:.0f}%" if not adjusted else f"{sign}{100 * v:.0f} pts"
    if kind == "pp":
        return f"{'+' if v > 0 else ''}{100 * v:.0f} pts"
    if kind in ("rate", "ratio"):
        return f"{sign}{v:.2f}"
    if kind == "years":
        return f"{v:.1f}"
    if kind in ("count", "days", "cm"):
        return f"{v:,.0f}"
    if kind == "elo":
        return f"{v:,.0f}"
    if kind == "flag":
        return "yes" if v else "no"
    raise ValueError(kind)


def _diff(fav: Mapping, other: Mapping, key: str) -> float | None:
    a, b = fav.get(key), other.get(key)
    return a - b if _finite(a) and _finite(b) else None


def _age_experience(f: Mapping, o: Mapping) -> str:
    parts = []
    d = _diff(f, o, "age")
    if d is not None:
        word = "younger" if d < 0 else "older"
        parts.append(
            f"{abs(d):.1f} years {word} ({f['age']:.1f} vs {o['age']:.1f})"
            if abs(d) >= 0.05
            else f"the same age ({f['age']:.1f})"
        )
    else:
        parts.append("age unknown for one fighter")
    n = _diff(f, o, "n_fights")
    if n is not None:
        if n == 0:
            parts.append(f"the same number of UFC fights ({f['n_fights']:.0f})")
        else:
            word = "more" if n > 0 else "fewer"
            count = abs(n)
            plural = "fight" if count == 1 else "fights"
            parts.append(
                f"{count:.0f} {word} UFC {plural} ({f['n_fights']:.0f} vs "
                f"{o['n_fights']:.0f})"
            )
    return " and ".join(parts)


def _rating(f: Mapping, o: Mapping) -> str:
    parts = []
    d = _diff(f, o, "elo")
    if d is not None:
        word = "edge" if d >= 0 else "deficit"
        parts.append(
            f"a {abs(d):,.0f}-point rating {word} ({f['elo']:,.0f} vs {o['elo']:,.0f})"
        )
    q = _diff(f, o, "opp_elo_career")
    if q is not None:
        word = "higher" if q >= 0 else "lower"
        parts.append(
            f"past opponents rated {abs(q):,.0f} points {word} on average "
            f"({f['opp_elo_career']:,.0f} vs {o['opp_elo_career']:,.0f})"
        )
    else:
        parts.append("no past UFC opponents for one fighter")
    return " and ".join(parts)


def _stat_phrase(p: Parsed, f: Mapping, o: Mapping, with_label: bool = True) -> str:
    """``with_label`` is False when the reason's title already names the stat."""
    stat = STATS[p.stem]
    parts = [stat.label] if with_label else []
    if p.adjusted:
        parts.append("relative to what was expected against those opponents")
    if p.window:
        parts.append(WINDOWS[p.window])
    fv, ov = f.get(p.source), o.get(p.source)
    values = (
        f"{_value(stat.kind, fv, p.adjusted)} vs {_value(stat.kind, ov, p.adjusted)}"
    )
    return ", ".join(parts) + ": " + values


def _pair_phrase(p: Parsed, f: Mapping, o: Mapping) -> str:
    pair = PAIR_BY_NAME[p.stem]
    off, dfn = STATS[pair.offense], STATS[pair.defense]
    return (
        f"{off.label} {_value(off.kind, f.get(pair.offense))} against the "
        f"opponent's {dfn.label} {_value(dfn.kind, o.get(pair.defense))}"
    )


def _headline(members: Sequence[tuple[str, float]], f: Mapping, o: Mapping) -> str:
    """The member with the largest contribution in the reason's direction,
    as a phrase with both fighters' values."""
    name = max(members, key=lambda m: (m[1], m[0]))[0]
    p = parse(name)
    if p.stem in PAIR_BY_NAME:
        return _pair_phrase(p, f, o)
    if p.level:
        stat = STATS[p.stem]
        return (
            f"bout average {stat.label}: "
            f"{_value(stat.kind, _mean(f.get(p.source), o.get(p.source)))}"
        )
    return _stat_phrase(p, f, o)


def _mean(a, b):
    return (a + b) / 2 if _finite(a) and _finite(b) else None


def detail(
    group: str, members: Sequence[tuple[str, float]], f: Mapping, o: Mapping
) -> str:
    """The values behind one reason, from the side of fighter ``f``, the one
    the reason points toward. ``members`` are (input, contribution toward
    ``f``)."""
    if group == "age_experience":
        return _age_experience(f, o)
    if group == "rating":
        return _rating(f, o)
    if group == "height":
        return _cm(f, o, "height_cm", "taller", "shorter", "height")
    if group == "reach":
        return _cm(f, o, "reach_cm", "longer", "shorter", "reach")
    if group in PER_STAT:
        name = max(members, key=lambda m: (m[1], m[0]))[0]
        return _stat_phrase(parse(name), f, o, with_label=False)
    return _headline(members, f, o)


def _cm(f: Mapping, o: Mapping, key: str, more: str, less: str, what: str) -> str:
    d = _diff(f, o, key)
    if d is None:
        return f"{what} unknown for one fighter"
    if abs(d) < 0.5:
        return f"the same {what} ({f[key]:.0f} cm)"
    word = more if d > 0 else less
    return f"{abs(d):.0f} cm {word} ({f[key]:.0f} vs {o[key]:.0f} cm)"


# --- the explanation ---------------------------------------------------------------


def explain(
    toward_a: Mapping[str, float],
    matchup: Matchup,
    names: tuple[str, str],
    p_a: float,
    p_b: float,
) -> dict:
    """The "why" for one bout, as JSON: the favoured side, up to three
    reasons toward them, the strongest reason toward the other fighter, and
    a summary sentence. Swapping the fighters (contributions negated, sides
    and names swapped, probabilities swapped) gives the mirror image."""
    totals = group_totals(toward_a)
    by_group: dict[str, list[tuple[str, float]]] = {}
    for name in sorted(toward_a):
        by_group.setdefault(group_of(name), []).append((name, toward_a[name]))

    if p_a == p_b:
        return {
            "favoured": None,
            "for": [],
            "against": None,
            "summary": "The model rates this bout even; no side is favoured.",
            "method": METHOD,
        }
    a_fav = p_a > p_b
    fav_side, other_side = (
        ("fighter_a", "fighter_b") if a_fav else ("fighter_b", "fighter_a")
    )
    fav_name, other_name = names if a_fav else names[::-1]
    fav_vals, other_vals = (matchup.a, matchup.b) if a_fav else (matchup.b, matchup.a)
    sign = 1.0 if a_fav else -1.0

    def reason(group: str, toward_fav: bool) -> dict:
        s = sign if toward_fav else -sign
        members = [(n, s * c) for n, c in by_group[group]]
        f, o = (fav_vals, other_vals) if toward_fav else (other_vals, fav_vals)
        return {
            "reason": group,
            "title": TITLES[group],
            "toward": fav_side if toward_fav else other_side,
            "detail": detail(group, members, f, o),
            "log_odds": s * totals[group],
            "inputs": [n for n, _ in by_group[group]],
        }

    # Strongest first; equal weights fall back to the reason's key, which
    # does not depend on which fighter is listed first.
    ranked = sorted(totals, key=lambda g: (-sign * totals[g], g))
    toward = [g for g in ranked if sign * totals[g] > 0][:N_FOR]
    against = [
        g
        for g in sorted(totals, key=lambda g: (sign * totals[g], g))
        if sign * totals[g] < 0
    ]
    reasons_for = [reason(g, True) for g in toward]
    reason_against = reason(against[0], False) if against else None

    p_fav = max(p_a, p_b)
    text = f"The model favours {fav_name} ({100 * p_fav:.0f}%)."
    if reasons_for:
        text += (
            f" What it weighs most toward {fav_name}: "
            + "; ".join(
                f"({i}) {r['title']}, {r['detail']}"
                for i, r in enumerate(reasons_for, 1)
            )
            + "."
        )
    else:
        # Possible when calibration, not the booster, decides the side (a
        # slope at or below zero, or a margin too small to resolve).
        text += f" No reason the model weighs points toward {fav_name} on its own."
    if reason_against is not None:
        text += (
            f" What it weighs most toward {other_name}: "
            f"{reason_against['title']}, {reason_against['detail']}."
        )
    else:
        text += f" Nothing the model weighs points toward {other_name}."
    return {
        "favoured": fav_side,
        "favoured_name": fav_name,
        "for": reasons_for,
        "against": reason_against,
        "summary": text,
        "method": METHOD,
    }


def explain_bout(
    candidate: Candidate,
    inputs,
    matchup: Matchup,
    names: tuple[str, str],
    p_a: float,
    p_b: float,
) -> dict:
    """Contributions and the explanation, as stored with a prediction."""
    toward_a = contributions(candidate, inputs)
    return {
        "log_odds_toward_a": toward_a,
        "why": explain(toward_a, matchup, names, p_a, p_b),
    }
