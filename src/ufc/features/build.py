"""Build the feature table for every completed bout, in date order.

For each date, features for all of that date's bouts are computed first, from
a store that holds earlier dates only. The date's bouts are added to the
store afterwards. Nothing from a bout's own date, and nothing later, exists
in the store when its features are computed.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ufc.features.adjust import fit_league_baseline
from ufc.features.config import FeatureConfig
from ufc.features.data import BoutFacts, FeatureData
from ufc.features.elo import EloFit, by_date, fit_elo_params
from ufc.features.matchup import (
    FAMILIES,
    FEATURE_NAMES,
    FEATURES,
    FLAG_NAMES,
    LEVEL_NAMES,
    SHRUNK_FEATURE_NAMES,
    flags_from,
    matchup_arrays,
)
from ufc.features.shrink import fit_rate_priors
from ufc.features.store import FeatureStore, FittedParams

logger = logging.getLogger(__name__)

CONTEXT_COLUMNS: tuple[str, ...] = (
    "bout_id",
    "event_id",
    "date",
    "fighter_a_id",
    "fighter_b_id",
    "weight_class",
    "is_womens",
    "is_title_bout",
    "scheduled_rounds",
    "result",
    "label",
)
"""Identity, what is known about the bout beforehand, and the outcome.
``label`` is 1 if fighter A won, 0 if fighter B won, NaN for a draw or no
contest. ``result`` and ``label`` are outcomes and must never be model inputs."""

LEVEL_FAMILY = "level"
"""Label for the swap-invariant candidate columns in summaries. It is not one
of the numbered feature families."""

FEATURES_FILE = "features.parquet"
META_FILE = "features_meta.json"


@dataclass(frozen=True, slots=True)
class FeatureBuild:
    frame: pd.DataFrame
    store: FeatureStore
    params: FittedParams
    elo_fit: EloFit | None  # None when parameters were supplied, not fitted


def fit_params(
    bouts: list[BoutFacts], config: FeatureConfig
) -> tuple[FittedParams, EloFit]:
    """Fit everything that is fitted, on bouts through ``config.train_end``."""
    elo_fit = fit_elo_params(bouts, config.train_end)
    league = fit_league_baseline(bouts, config.train_end)
    rate_priors = fit_rate_priors(bouts, config.train_end)
    return FittedParams(
        elo=elo_fit.params, league=league, rate_priors=rate_priors
    ), elo_fit


def _label(bout: BoutFacts) -> float:
    if bout.result != "win":
        return float("nan")
    return 1.0 if bout.winner_id == bout.fighter_a_id else 0.0


def build_features(
    data: FeatureData,
    config: FeatureConfig | None = None,
    params: FittedParams | None = None,
) -> FeatureBuild:
    """One row per completed bout, plus the store left at the end.

    With ``params`` omitted they are fitted on the training period first.
    The returned store holds every bout and can price hypothetical bouts
    with ``build_matchup``.
    """
    config = config or FeatureConfig()
    bouts = list(data.bouts)
    elo_fit = None
    if params is None:
        params, elo_fit = fit_params(bouts, config)

    store = FeatureStore(data.fighters, config, params)
    context: list[tuple] = []
    flags: list[tuple] = []
    values: list[np.ndarray] = []
    levels: list[np.ndarray] = []
    for day, day_bouts in by_date(bouts):
        ordinal = day.toordinal()
        # Phase 1: features for this date, from earlier dates only.
        for bout in day_bouts:
            a, b, features, level_values = matchup_arrays(
                store, bout.fighter_a_id, bout.fighter_b_id, ordinal, bout.weight_class
            )
            context.append(
                (
                    bout.bout_id,
                    bout.event_id,
                    day,
                    bout.fighter_a_id,
                    bout.fighter_b_id,
                    bout.weight_class,
                    bout.is_womens,
                    bout.is_title_bout,
                    bout.scheduled_rounds,
                    bout.result,
                    _label(bout),
                )
            )
            flags.append(flags_from(a, b))
            values.append(features)
            levels.append(level_values)
        # Phase 2: only now does this date become history.
        store.commit_date(day, day_bouts)

    frame = pd.DataFrame(context, columns=list(CONTEXT_COLUMNS))
    # Milliseconds: the resolution parquet stores, so a file reads back equal.
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ms]")
    frame["scheduled_rounds"] = frame["scheduled_rounds"].astype("float64")
    frame["label"] = frame["label"].astype("float64")
    for column in ("is_womens", "is_title_bout"):
        frame[column] = frame[column].astype(bool)
    flag_frame = pd.DataFrame(flags, columns=list(FLAG_NAMES), dtype=bool)
    feature_frame = pd.DataFrame(
        np.array(values).reshape(len(values), len(FEATURE_NAMES)),
        columns=list(FEATURE_NAMES),
    )
    level_frame = pd.DataFrame(
        np.array(levels).reshape(len(levels), len(LEVEL_NAMES)),
        columns=list(LEVEL_NAMES),
    )
    frame = pd.concat([frame, flag_frame, feature_frame, level_frame], axis=1)
    return FeatureBuild(frame=frame, store=store, params=params, elo_fit=elo_fit)


def mirror(frame: pd.DataFrame) -> pd.DataFrame:
    """The same bouts with fighters A and B swapped.

    Difference features change sign, per-fighter flags and IDs trade places,
    and the label flips. Level columns (means of the two fighters) stay as
    they are. Training uses both orientations of every bout.
    """
    swapped = frame.copy()
    swapped["fighter_a_id"] = frame["fighter_b_id"]
    swapped["fighter_b_id"] = frame["fighter_a_id"]
    for name in FLAG_NAMES:
        other = ("b_" if name.startswith("a_") else "a_") + name[2:]
        swapped[name] = frame[other]
    swapped[list(FEATURE_NAMES)] = -frame[list(FEATURE_NAMES)]
    swapped["label"] = 1.0 - frame["label"]
    return swapped


def feature_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Missing-value counts per feature.

    ``missing_established`` counts only bouts where neither fighter is making
    his debut, where a missing value is not explained by the debut flag.
    """
    established = ~(frame["a_debut"] | frame["b_debut"])
    columns = [(f.name, f.family) for f in FEATURES]
    columns += [(name, LEVEL_FAMILY) for name in LEVEL_NAMES]
    return pd.DataFrame(
        {
            "feature": [name for name, _ in columns],
            "family": [family for _, family in columns],
            "missing": [int(frame[name].isna().sum()) for name, _ in columns],
            "missing_established": [
                int(frame.loc[established, name].isna().sum()) for name, _ in columns
            ],
        }
    )


def build_meta(build: FeatureBuild, source_commit: str | None) -> dict:
    """What a reader of the feature file needs to know about how it was made."""
    frame, config = build.frame, build.store.config
    meta: dict = {
        "source_commit": source_commit,
        "n_bouts": len(frame),
        "n_labelled": int(frame["label"].notna().sum()),
        "n_established": int((~(frame["a_debut"] | frame["b_debut"])).sum()),
        "first_date": frame["date"].min().date().isoformat() if len(frame) else None,
        "last_date": frame["date"].max().date().isoformat() if len(frame) else None,
        "config": {**asdict(config), "train_end": config.train_end.isoformat()},
        "params": build.params.to_dict(),
        "context_columns": list(CONTEXT_COLUMNS),
        "flag_columns": list(FLAG_NAMES),
        "features": [{"name": f.name, "family": f.family} for f in FEATURES],
        "families": {
            family: [f.name for f in FEATURES if f.family == family]
            for family in FAMILIES
        },
        "level_features": list(LEVEL_NAMES),
        "shrunk_features": list(SHRUNK_FEATURE_NAMES),
    }
    if build.elo_fit is not None:
        fit = build.elo_fit
        meta["elo_fit"] = {
            "train_log_loss": fit.log_loss,
            "n_bouts": fit.n_bouts,
            "last_date": fit.last_date.isoformat() if fit.last_date else None,
            "on_grid_edge": list(fit.on_grid_edge),
        }
    return meta


def _replace_atomically(path: Path, write) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write(temporary)
    os.replace(temporary, path)


def write_features(
    build: FeatureBuild, out_dir: Path, source_commit: str | None = None
) -> tuple[Path, Path]:
    """Write the feature table and its metadata. Safe to run again: each file
    is replaced whole, and the same input gives the same content."""
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path, meta_path = out_dir / FEATURES_FILE, out_dir / META_FILE
    _replace_atomically(features_path, lambda p: build.frame.to_parquet(p, index=False))
    meta = json.dumps(build_meta(build, source_commit), indent=2) + "\n"
    _replace_atomically(meta_path, lambda p: p.write_text(meta, encoding="utf-8"))
    logger.info("wrote %d bouts to %s", len(build.frame), features_path)
    return features_path, meta_path
