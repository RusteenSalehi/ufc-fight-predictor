"""One fighter's as-of feature values at a cutoff date.

``fighter_vector`` is a pure function of the store and the cutoff. It reads a
fighter's history only through ``index_before``, so it sees bouts on dates
strictly before the cutoff and nothing else.

Rates are pooled: total landed over total attempted, total strikes over total
minutes. A rate with a zero denominator is NaN, which is how a debut, a
history without recorded stats, or a fighter who never faced a takedown
attempt shows up. Record counts (bouts, wins) are true zeros for a debutant.

With ``FeatureConfig.shrink_minutes`` set, rates per minute and per 15
minutes are shrunk toward the average of the bout's weight class
(``shrink.py``). A fighter with no recorded stat time still has NaN: the
prior is never used as a stand-in for a history that does not exist.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import NamedTuple

import numpy as np

from ufc.features.adjust import ADJUSTED_NAMES, adjusted_stats
from ufc.features.components import IDX
from ufc.features.config import WINDOWS, weight_limit_lbs
from ufc.features.shrink import RATE_COMPONENTS
from ufc.features.store import FeatureStore, FighterHistory

NAN = float("nan")


class Ratio(NamedTuple):
    """``scale * num / den``, or one minus that when ``complement`` is set."""

    name: str
    family: str
    num: str
    den: str
    scale: float = 1.0
    complement: bool = False


# Computed for every window in WINDOWS.
WINDOWED: tuple[Ratio, ...] = (
    Ratio("win_rate", "record", "win", "decided"),
    Ratio("slpm", "striking", "sig_l", "t", 60.0),
    Ratio("sapm", "striking", "opp_sig_l", "t", 60.0),
    Ratio("sig_acc", "striking", "sig_l", "sig_a"),
    Ratio("sig_def", "striking", "opp_sig_l", "opp_sig_a", complement=True),
    Ratio("sig_att_pm", "striking", "sig_a", "t", 60.0),
    Ratio("kd_per15", "striking", "kd", "t", 900.0),
    Ratio("td_per15", "grappling", "td_l", "t", 900.0),
    Ratio("td_acc", "grappling", "td_l", "td_a"),
    Ratio("td_def", "grappling", "opp_td_l", "opp_td_a", complement=True),
    Ratio("sub_att_per15", "grappling", "sub_att", "t", 900.0),
    Ratio("ctrl_share", "grappling", "ctrl", "t_ctrl"),
    Ratio("ctrl_against_share", "grappling", "opp_ctrl", "t_opp_ctrl"),
    Ratio("kd_abs_per15", "durability", "opp_kd", "t", 900.0),
    Ratio("head_abs_pm", "durability", "opp_head_l", "t", 60.0),
    Ratio("opp_elo", "opponent_quality", "opp_elo", "n"),
)

# Whole history only: where his significant strikes land and from where.
CAREER: tuple[Ratio, ...] = (
    Ratio("finish_win_share", "record", "finish_win", "win"),
    Ratio("head_share", "striking", "head_l", "sig_l"),
    Ratio("body_share", "striking", "body_l", "sig_l"),
    Ratio("leg_share", "striking", "leg_l", "sig_l"),
    Ratio("distance_share", "striking", "dist_l", "sig_l"),
    Ratio("clinch_share", "striking", "clinch_l", "sig_l"),
    Ratio("ground_share", "striking", "ground_l", "sig_l"),
    # Used by a style matchup only, not as a feature of its own.
    Ratio("sub_loss_rate", "aux", "sub_loss", "n"),
)


class Count(NamedTuple):
    name: str
    family: str
    component: str
    scale: float = 1.0


# Whole-history totals. These are records, so a debutant has a true zero.
COUNTS: tuple[Count, ...] = (
    Count("n_fights", "record", "n"),
    Count("wins", "record", "win"),
    Count("losses", "record", "loss"),
    Count("ko_wins", "record", "ko_win"),
    Count("sub_wins", "record", "sub_win"),
    Count("title_fights", "record", "title"),
    Count("title_wins", "record", "title_win"),
    Count("cage_min", "record", "cage_s", 1 / 60),
    Count("ko_losses", "durability", "ko_loss"),
    Count("sub_losses", "durability", "sub_loss"),
)

# Computed one at a time in fighter_vector.
SCALARS: tuple[tuple[str, str], ...] = (
    ("age", "physical"),
    ("height_cm", "physical"),
    ("reach_cm", "physical"),
    ("stance_southpaw", "physical"),
    ("stance_switch", "physical"),
    ("win_streak", "record"),
    ("loss_streak", "record"),
    ("ko_losses_l3", "durability"),
    ("ko_losses_l5", "durability"),
    ("kd_abs", "durability"),
    ("days_since_last", "activity"),
    ("fights_last_2y", "activity"),
    ("moved_up", "weight_change"),
    ("moved_down", "weight_change"),
    ("fights_at_weight", "weight_change"),
    ("elo", "rating"),
    ("debut", "debut"),
    ("cardio_slpm_ratio_career", "cardio"),
    ("cardio_slpm_ratio_decay", "cardio"),
    ("cardio_sig_att_ratio_career", "cardio"),
    ("cardio_sig_att_ratio_decay", "cardio"),
    ("cardio_sapm_ratio_career", "cardio"),
    ("cardio_sapm_ratio_decay", "cardio"),
    ("cardio_sig_acc_delta", "cardio"),
    ("cardio_sig_def_delta", "cardio"),
    ("cardio_fights", "cardio"),
    # Not differenced: inputs to pair features and flags.
    ("stance_orthodox", "aux"),
    ("no_stats", "aux"),
)


class FighterFeature(NamedTuple):
    name: str
    family: str


FIGHTER_FEATURES: tuple[FighterFeature, ...] = (
    *(FighterFeature(f"{r.name}_{w}", r.family) for w in WINDOWS for r in WINDOWED),
    *(FighterFeature(r.name, r.family) for r in CAREER),
    *(FighterFeature(c.name, c.family) for c in COUNTS),
    *(FighterFeature(name, family) for name, family in SCALARS),
    *(FighterFeature(name, "opponent_adjusted") for name in ADJUSTED_NAMES),
)
POS: dict[str, int] = {f.name: i for i, f in enumerate(FIGHTER_FEATURES)}

_N_WINDOWED = len(WINDOWS) * len(WINDOWED)
_SL_WINDOWED = slice(0, _N_WINDOWED)
_SL_CAREER = slice(_N_WINDOWED, _N_WINDOWED + len(CAREER))
_SL_COUNTS = slice(_SL_CAREER.stop, _SL_CAREER.stop + len(COUNTS))
_SL_ADJUSTED = slice(len(FIGHTER_FEATURES) - len(ADJUSTED_NAMES), len(FIGHTER_FEATURES))


def _spec_arrays(specs: tuple[Ratio, ...]) -> tuple[np.ndarray, ...]:
    return (
        np.array([IDX[s.num] for s in specs]),
        np.array([IDX[s.den] for s in specs]),
        np.array([s.scale for s in specs]),
        np.array([s.complement for s in specs]),
    )


_W_NUM, _W_DEN, _W_SCALE, _W_COMPLEMENT = _spec_arrays(WINDOWED)
# Rates over stat time: the ones that shrinkage applies to.
_W_TIME_RATE = np.array([s.den == "t" for s in WINDOWED])
assert {s.num for s in WINDOWED if s.den == "t"} <= set(RATE_COMPONENTS)
_C_NUM, _C_DEN, _C_SCALE, _C_COMPLEMENT = _spec_arrays(CAREER)
_COUNT_IDX = np.array([IDX[c.component] for c in COUNTS])
_COUNT_SCALE = np.array([c.scale for c in COUNTS])


def _ratios(
    sums: np.ndarray,
    num: np.ndarray,
    den: np.ndarray,
    scale: np.ndarray,
    complement: np.ndarray,
) -> np.ndarray:
    """Pooled rates from component sums; NaN where the denominator is zero."""
    denominator = sums[..., den]
    out = np.full(denominator.shape, np.nan)
    np.divide(sums[..., num], denominator, out=out, where=denominator > 0)
    out *= scale
    out[..., complement] = 1.0 - out[..., complement]
    return out


def _streak(outcomes: list[str], idx: int, kind: str) -> float:
    """Consecutive ``kind`` results ending at the latest bout. No contests
    are skipped; anything else ends the streak."""
    count = 0
    for outcome in reversed(outcomes[:idx]):
        if outcome == "nc":
            continue
        if outcome != kind:
            break
        count += 1
    return float(count)


def _rate_change(s: np.ndarray, count: str) -> float:
    """Rounds-3+ rate of ``count`` per second divided by its round-1 rate."""
    early = s[IDX[f"r1_{count}"]] * s[IDX["late_t"]]
    if early <= 0:
        return NAN
    return s[IDX[f"late_{count}"]] * s[IDX["r1_t"]] / early


def _share(s: np.ndarray, num: str, den: str) -> float:
    return s[IDX[num]] / s[IDX[den]] if s[IDX[den]] > 0 else NAN


def _cardio(out: np.ndarray, career: np.ndarray, decay: np.ndarray) -> None:
    """Family 12: output and defense in rounds 3+ against round 1.

    Only bouts that went past round 2 feed these sums; with none, every
    value is NaN.
    """
    for window, s in (("career", career), ("decay", decay)):
        out[POS[f"cardio_slpm_ratio_{window}"]] = _rate_change(s, "sig_l")
        out[POS[f"cardio_sig_att_ratio_{window}"]] = _rate_change(s, "sig_a")
        out[POS[f"cardio_sapm_ratio_{window}"]] = _rate_change(s, "opp_sig_l")
    out[POS["cardio_sig_acc_delta"]] = _share(
        career, "late_sig_l", "late_sig_a"
    ) - _share(career, "r1_sig_l", "r1_sig_a")
    # Defense is one minus the opponent's accuracy, so its change is the
    # opponent's round-1 accuracy minus his late accuracy.
    out[POS["cardio_sig_def_delta"]] = _share(
        career, "r1_opp_sig_l", "r1_opp_sig_a"
    ) - _share(career, "late_opp_sig_l", "late_opp_sig_a")
    out[POS["cardio_fights"]] = career[IDX["c_n"]]


def _weight_change(
    out: np.ndarray, history: FighterHistory, idx: int, weight_class: str | None
) -> None:
    limit = weight_limit_lbs(weight_class)
    previous = weight_limit_lbs(history.weight_classes[idx - 1]) if idx else None
    if limit is not None and previous is not None:
        out[POS["moved_up"]] = float(limit > previous)
        out[POS["moved_down"]] = float(limit < previous)
    if weight_class is not None:
        out[POS["fights_at_weight"]] = float(
            sum(1 for wc in history.weight_classes[:idx] if wc == weight_class)
        )


def fighter_vector(
    store: FeatureStore, fighter_id: str, ordinal: int, weight_class: str | None
) -> np.ndarray:
    """Values for ``FIGHTER_FEATURES`` from bouts strictly before ``ordinal``.

    ``weight_class`` is the class of the bout being predicted; it is only
    used by the weight-change features.
    """
    history = store.history(fighter_id)
    idx = history.index_before(ordinal)
    career = history.cum[idx]
    sums = np.stack(
        (
            career,
            career - history.cum[max(idx - 3, 0)],
            career - history.cum[max(idx - 5, 0)],
            history.cum_w[idx],
        )
    )
    decay = sums[3]

    out = np.full(len(FIGHTER_FEATURES), np.nan)
    windowed = _ratios(sums, _W_NUM, _W_DEN, _W_SCALE, _W_COMPLEMENT)
    prior = store.shrink_prior(weight_class)
    if prior is not None:
        # The prior counts as minutes fought at the cutoff, so in the decayed
        # window it carries the weight of a bout held on the cutoff date.
        strength = np.array([1.0, 1.0, 1.0, store.decay_weight(ordinal)])
        shrunk = _ratios(
            sums + strength[:, None] * prior, _W_NUM, _W_DEN, _W_SCALE, _W_COMPLEMENT
        )
        observed = sums[:, _W_DEN] > 0  # no stat time stays NaN
        windowed = np.where(_W_TIME_RATE & observed, shrunk, windowed)
    out[_SL_WINDOWED] = windowed.ravel()
    out[_SL_CAREER] = _ratios(career, _C_NUM, _C_DEN, _C_SCALE, _C_COMPLEMENT)
    out[_SL_COUNTS] = career[_COUNT_IDX] * _COUNT_SCALE

    attrs = store.attrs(fighter_id)
    if attrs.dob is not None:
        out[POS["age"]] = (ordinal - attrs.dob.toordinal()) / 365.25
    if attrs.height_cm is not None:
        out[POS["height_cm"]] = attrs.height_cm
    if attrs.reach_cm is not None:
        out[POS["reach_cm"]] = attrs.reach_cm
    if attrs.stance is not None:
        out[POS["stance_southpaw"]] = float(attrs.stance == "Southpaw")
        out[POS["stance_switch"]] = float(attrs.stance == "Switch")
        out[POS["stance_orthodox"]] = float(attrs.stance == "Orthodox")

    has_stats = career[IDX["n_stat"]] > 0
    out[POS["win_streak"]] = _streak(history.outcomes, idx, "win")
    out[POS["loss_streak"]] = _streak(history.outcomes, idx, "loss")
    out[POS["ko_losses_l3"]] = sums[1][IDX["ko_loss"]]
    out[POS["ko_losses_l5"]] = sums[2][IDX["ko_loss"]]
    if has_stats:
        out[POS["kd_abs"]] = career[IDX["opp_kd"]]
    if idx:
        out[POS["days_since_last"]] = float(ordinal - history.dates[idx - 1])
    since = ordinal - store.config.recent_days
    out[POS["fights_last_2y"]] = float(idx - bisect_left(history.dates, since, 0, idx))
    _weight_change(out, history, idx, weight_class)
    out[POS["elo"]] = store.elo.rating_before(fighter_id, ordinal)
    out[POS["debut"]] = float(idx == 0)
    out[POS["no_stats"]] = float(not has_stats)
    _cardio(out, career, decay)
    out[_SL_ADJUSTED] = adjusted_stats(store, fighter_id, history, idx, ordinal)
    return out
