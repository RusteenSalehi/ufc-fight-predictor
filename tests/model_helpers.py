"""Invented feature tables for model tests.

``synthetic_frame`` writes a table in the layout of ``features.parquet`` with
a handful of features and a known signal, directly, without going through the
feature pipeline. Dates run from 2017 into 2026, so the project's real split
dates divide it into training, validation and test periods.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from ufc.models.design import FeatureSpec
from ufc.models.gbm import LIGHTGBM, XGBOOST, GbmConfig
from ufc.models.select import SelectionSettings

FEATURES: tuple[str, ...] = (
    "d_elo",
    "d_age",
    "d_height_cm",
    "d_reach_cm",
    "d_slpm_career",
    "d_td_per15_career",
    "d_win_rate_career",
    "d_n_fights",
)
LEVELS: tuple[str, ...] = ("lvl_age", "lvl_n_fights", "lvl_elo")
FAMILIES: dict[str, list[str]] = {
    "physical": ["d_age", "d_height_cm", "d_reach_cm"],
    "record": ["d_win_rate_career", "d_n_fights"],
    "striking": ["d_slpm_career"],
    "grappling": ["d_td_per15_career"],
    "rating": ["d_elo"],
}
SHRUNK: tuple[str, ...] = ("d_slpm_career", "d_td_per15_career")
SPEC = FeatureSpec(FEATURES)

FAST = {
    XGBOOST: GbmConfig(
        XGBOOST,
        {
            "learning_rate": 0.1,
            "max_depth": 2,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        },
    ),
    LIGHTGBM: GbmConfig(
        LIGHTGBM,
        {
            "learning_rate": 0.1,
            "num_leaves": 4,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        },
    ),
}
"""Small, quick settings for tests."""

SMALL = SelectionSettings(
    shrink_grid=(0.0, 15.0), seeds=(0, 1), n_trials=3, top_k=2, half_lives=(None, 4.0)
)


def synthetic_frame(
    seed: int = 0,
    start: date = date(2017, 1, 7),
    end: date = date(2026, 10, 31),
    bouts_per_event: int = 2,
    shrink: float = 0.0,
) -> pd.DataFrame:
    """One event a week. The winner depends on the Elo and age differences;
    the other features are noise. ``shrink`` scales the two per-minute
    features, standing in for a rebuild with rate shrinkage."""
    rng = np.random.default_rng(seed)
    days = [start + timedelta(days=7 * i) for i in range((end - start).days // 7 + 1)]
    n = len(days) * bouts_per_event
    x = rng.normal(size=(n, len(FEATURES)))
    frame = pd.DataFrame(x, columns=list(FEATURES))
    frame["d_elo"] *= 120.0
    frame["d_age"] *= 4.0
    signal = frame["d_elo"] / 150.0 - frame["d_age"] / 8.0
    won = rng.random(n) < 1.0 / (1.0 + np.exp(-signal))

    # Debutants have no stats; some recent fighters have no height or reach.
    a_debut, b_debut = rng.random(n) < 0.12, rng.random(n) < 0.12
    no_stats = a_debut | b_debut
    for name in ("d_slpm_career", "d_td_per15_career", "d_win_rate_career"):
        frame.loc[no_stats, name] = np.nan
    no_reach = rng.random(n) < 0.08
    frame.loc[no_reach, ["d_height_cm", "d_reach_cm"]] = np.nan
    # In most of those bouts only one profile is incomplete: an attribute gap.
    gap_side = rng.random(n)
    a_gap, b_gap = no_reach & (gap_side < 0.45), no_reach & (gap_side > 0.55)
    frame.loc[a_debut & b_debut, "d_elo"] = 0.0
    for name in SHRUNK:
        frame[name] /= 1.0 + shrink / 15.0
    for name in LEVELS:
        frame[name] = rng.normal(size=n)

    result = np.where(rng.random(n) < 0.03, "draw", "win")
    result = np.where(rng.random(n) < 0.02, "nc", result)
    context = pd.DataFrame(
        {
            "bout_id": [f"b{i:015x}" for i in range(n)],
            "event_id": [f"e{i // bouts_per_event:015x}" for i in range(n)],
            "date": pd.to_datetime(np.repeat(days, bouts_per_event)).astype(
                "datetime64[ms]"
            ),
            "fighter_a_id": [f"a{i:015x}" for i in range(n)],
            "fighter_b_id": [f"c{i:015x}" for i in range(n)],
            "weight_class": rng.choice(["Lightweight", "Welterweight", None], size=n),
            "is_womens": False,
            "is_title_bout": False,
            "scheduled_rounds": 3.0,
            "result": result,
            "label": np.where(result == "win", won.astype(float), np.nan),
            "a_debut": a_debut,
            "b_debut": b_debut,
            "a_no_stats": a_debut,
            "b_no_stats": b_debut,
            "a_attr_gap": a_gap,
            "b_attr_gap": b_gap,
        }
    )
    return pd.concat([context, frame], axis=1)


def mirrored(frame: pd.DataFrame) -> pd.DataFrame:
    """The same bouts with the fighters swapped, written out by hand so that
    tests do not lean on the code they check."""
    out = frame.copy()
    out["fighter_a_id"], out["fighter_b_id"] = (
        frame["fighter_b_id"],
        frame["fighter_a_id"],
    )
    out["a_debut"], out["b_debut"] = frame["b_debut"], frame["a_debut"]
    out["a_no_stats"], out["b_no_stats"] = frame["b_no_stats"], frame["a_no_stats"]
    out["a_attr_gap"], out["b_attr_gap"] = frame["b_attr_gap"], frame["a_attr_gap"]
    out[list(FEATURES)] = -frame[list(FEATURES)]
    out["label"] = 1.0 - frame["label"]
    return out
