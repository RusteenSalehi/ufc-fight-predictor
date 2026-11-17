"""Score a set of models on one period: headline metrics and breakdowns.

The same function scores validation during selection and the test period at
the end, so the report code has been run on real data before the test period
is ever opened. It takes probabilities, not models, and has no way to fit
anything.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ufc.models.metrics import (
    bootstrap_mean_interval,
    bout_log_loss,
    confidence_table,
    reliability_table,
    summary,
)
from ufc.models.splits import labelled

MIN_GROUP = 100
"""A weight class is reported on its own only with at least this many scored
bouts in the period ("where sample size allows")."""


def _summaries(
    predictions: Mapping[str, np.ndarray], y: np.ndarray, mask: np.ndarray
) -> dict:
    return {name: summary(p[mask], y[mask]) for name, p in predictions.items()}


def _groups(
    predictions: Mapping[str, np.ndarray],
    y: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> dict:
    return {
        label: _summaries(predictions, y, np.asarray(mask))
        for label, mask in masks.items()
        if np.asarray(mask).any()
    }


def evaluate_period(
    frame: pd.DataFrame, predictions: Mapping[str, np.ndarray], primary: str
) -> dict:
    """Metrics for every model in ``predictions`` over the bouts in ``frame``.

    ``frame`` holds every bout of the period, including draws and no
    contests, which are counted as "no result" and not scored. Each entry of
    ``predictions`` is P(A wins) for the bouts with a winner, in frame order.
    ``primary`` names the model that the paired comparisons are made for.
    """
    scored = labelled(frame)
    y = scored["label"].to_numpy(dtype=np.float64)
    for name, p in predictions.items():
        if len(p) != len(scored):
            raise ValueError(f"{name}: {len(p)} predictions for {len(scored)} bouts")
    everyone = np.ones(len(scored), dtype=bool)
    losses = {name: bout_log_loss(p, y) for name, p in predictions.items()}

    result: dict = {
        "n_bouts": len(frame),
        "n_scored": len(scored),
        "n_no_result": len(frame) - len(scored),
        "no_result": {
            str(kind): int(count)
            for kind, count in frame.loc[frame["label"].isna(), "result"]
            .value_counts()
            .sort_index()
            .items()
        },
        "first_date": frame["date"].min().date().isoformat() if len(frame) else None,
        "last_date": frame["date"].max().date().isoformat() if len(frame) else None,
        "primary": primary,
        "models": _summaries(predictions, y, everyone),
    }
    for name, values in losses.items():
        interval = bootstrap_mean_interval(values)
        result["models"][name]["log_loss_low"] = interval["low"]
        result["models"][name]["log_loss_high"] = interval["high"]

    # Paired: the same bouts, the primary model's loss minus the other's.
    result["primary_minus"] = {
        name: bootstrap_mean_interval(losses[primary] - values)
        for name, values in losses.items()
        if name != primary
    }

    years = scored["date"].dt.year.to_numpy()
    result["by_year"] = _groups(
        predictions, y, {str(year): years == year for year in sorted(set(years))}
    )
    result["by_confidence"] = {
        name: confidence_table(p, y) for name, p in predictions.items()
    }
    result["reliability"] = {
        name: reliability_table(p, y) for name, p in predictions.items()
    }

    debuts = scored["a_debut"].to_numpy().astype(int) + scored["b_debut"].to_numpy()
    result["by_debut"] = _groups(
        predictions,
        y,
        {
            "no debutant": debuts == 0,
            "one debutant": debuts == 1,
            "two debutants": debuts == 2,
            "any debutant": debuts >= 1,
        },
    )

    classes = scored["weight_class"].fillna("(none recorded)").to_numpy()
    counts = pd.Series(classes).value_counts()
    result["by_weight_class"] = _groups(
        predictions,
        y,
        {name: classes == name for name in sorted(counts[counts >= MIN_GROUP].index)},
    )
    result["weight_class_min_bouts"] = MIN_GROUP
    result["weight_classes_too_small"] = {
        str(name): int(count) for name, count in counts[counts < MIN_GROUP].items()
    }

    # Height and reach are often missing for recent debutants
    # (docs/DATA_QUALITY.md), and almost never in the training period.
    incomplete = (scored["d_height_cm"].isna() | scored["d_reach_cm"].isna()).to_numpy()
    result["by_attributes"] = _groups(
        predictions,
        y,
        {
            "height and reach known for both": ~incomplete,
            "height or reach missing": incomplete,
        },
    )
    result["equal_elo"] = int((scored["d_elo"] == 0).sum())
    return result
