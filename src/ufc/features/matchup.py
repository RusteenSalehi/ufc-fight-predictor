"""Features for one bout: fighter A against fighter B at a date.

Every model feature is antisymmetric. It is either ``f(A) - f(B)`` for a
per-fighter value ``f``, or ``m(A, B) - m(B, A)`` for a pair value ``m`` such
as "A's takedown volume against B's takedown defense". Swapping the fighters
therefore negates every feature exactly, and which fighter the source happened
to list first carries no information (docs/ML_REQUIREMENTS.md, "Symmetry").

The same function serves the historical build and a hypothetical upcoming
bout, so training and prediction features cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import NamedTuple

import numpy as np

from ufc.features.fighter import FIGHTER_FEATURES, POS, fighter_vector
from ufc.features.store import FeatureStore

# Family keys in the order of docs/ML_REQUIREMENTS.md, "Feature families".
FAMILIES: tuple[str, ...] = (
    "physical",
    "record",
    "striking",
    "grappling",
    "durability",
    "activity",
    "weight_change",
    "opponent_quality",
    "opponent_adjusted",
    "style_matchup",
    "debut",
    "cardio",
)


class Pair(NamedTuple):
    """``m(A, B) = A[offense] * B[defense]``, with ``1 - B[defense]`` when
    ``complement`` is set (a high defense value then lowers the product)."""

    name: str
    family: str
    offense: str
    defense: str
    complement: bool = False


PAIRS: tuple[Pair, ...] = (
    Pair(
        "stance_southpaw_vs_orthodox", "physical", "stance_southpaw", "stance_orthodox"
    ),
    Pair("matchup_td", "style_matchup", "td_per15_career", "td_def_career", True),
    Pair(
        "matchup_strike", "style_matchup", "sig_att_pm_career", "sig_def_career", True
    ),
    Pair("matchup_kd", "style_matchup", "kd_per15_career", "kd_abs_per15_career"),
    Pair(
        "matchup_ctrl",
        "style_matchup",
        "ctrl_share_career",
        "ctrl_against_share_career",
    ),
    Pair("matchup_sub", "style_matchup", "sub_att_per15_career", "sub_loss_rate"),
)


class Feature(NamedTuple):
    name: str
    family: str


_DIFFED = [f for f in FIGHTER_FEATURES if f.family != "aux"]
_DIFF_IDX = np.array([POS[f.name] for f in _DIFFED])
_OFFENSE_IDX = np.array([POS[p.offense] for p in PAIRS])
_DEFENSE_IDX = np.array([POS[p.defense] for p in PAIRS])
_COMPLEMENT = np.array([p.complement for p in PAIRS])

FEATURES: tuple[Feature, ...] = (
    *(Feature(f"d_{f.name}", f.family) for f in _DIFFED),
    *(Feature(f"d_{p.name}", p.family) for p in PAIRS),
)
"""Model features, in column order. All are A-minus-B and start with ``d_``."""

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
FLAG_NAMES: tuple[str, ...] = ("a_debut", "b_debut", "a_no_stats", "b_no_stats")
"""Per-fighter indicators, not differenced. ``debut``: no earlier UFC bout.
``no_stats``: no earlier bout with recorded stats (true for every debut)."""

assert {f.family for f in FEATURES} == set(FAMILIES)


def _pair_values(offense: np.ndarray, defense: np.ndarray) -> np.ndarray:
    against = defense[_DEFENSE_IDX]
    return offense[_OFFENSE_IDX] * np.where(_COMPLEMENT, 1.0 - against, against)


def matchup_arrays(
    store: FeatureStore,
    fighter_a_id: str,
    fighter_b_id: str,
    ordinal: int,
    weight_class: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Both fighters' vectors and the feature vector, in ``FEATURES`` order."""
    if fighter_a_id == fighter_b_id:
        raise ValueError("a bout needs two different fighters")
    a = fighter_vector(store, fighter_a_id, ordinal, weight_class)
    b = fighter_vector(store, fighter_b_id, ordinal, weight_class)
    features = np.concatenate(
        (a[_DIFF_IDX] - b[_DIFF_IDX], _pair_values(a, b) - _pair_values(b, a))
    )
    return a, b, features


def flags_from(a: np.ndarray, b: np.ndarray) -> tuple[bool, bool, bool, bool]:
    """Values for ``FLAG_NAMES``."""
    debut, no_stats = POS["debut"], POS["no_stats"]
    return (bool(a[debut]), bool(b[debut]), bool(a[no_stats]), bool(b[no_stats]))


@dataclass(frozen=True, slots=True)
class Matchup:
    """Features for one bout, with the per-fighter values they came from.

    ``features`` is what a model consumes. ``a`` and ``b`` hold each fighter's
    own as-of values (for display and explanation).
    """

    fighter_a_id: str
    fighter_b_id: str
    date: date
    weight_class: str | None
    features: dict[str, float]
    flags: dict[str, bool]
    a: dict[str, float]
    b: dict[str, float]


def build_matchup(
    store: FeatureStore,
    fighter_a_id: str,
    fighter_b_id: str,
    on: date,
    weight_class: str | None = None,
) -> Matchup:
    """Features for a bout between two fighters on ``on``, real or hypothetical.

    Only bouts on dates strictly before ``on`` are used, whatever the store
    holds. A fighter ID with no history is treated as a debutant; one with no
    profile has unknown physical attributes. Without ``weight_class`` the
    weight-change features are NaN.
    """
    a, b, features = matchup_arrays(
        store, fighter_a_id, fighter_b_id, on.toordinal(), weight_class
    )
    names = [f.name for f in FIGHTER_FEATURES]
    return Matchup(
        fighter_a_id=fighter_a_id,
        fighter_b_id=fighter_b_id,
        date=on,
        weight_class=weight_class,
        features=dict(zip(FEATURE_NAMES, features.tolist(), strict=True)),
        flags=dict(zip(FLAG_NAMES, flags_from(a, b), strict=True)),
        a=dict(zip(names, a.tolist(), strict=True)),
        b=dict(zip(names, b.tolist(), strict=True)),
    )
