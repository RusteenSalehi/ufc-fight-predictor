"""Metrics for win probabilities. Pure functions of arrays.

Every metric here gives the same value whichever fighter is called A,
provided the probabilities are symmetric (P(B) = 1 - P(A)), which the
inference path guarantees. Calibration metrics are computed on the favoured
fighter: "when the model gives its pick 65%, how often does the pick win?"
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

CLIP = 1e-15
ECE_BINS = 10
"""Equal-width bins over the favoured fighter's probability, 0.50 to 1.00."""

CONFIDENCE_EDGES: tuple[float, ...] = (0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0)
"""Buckets for the "by confidence" breakdown; the last is closed on the right."""


def _arrays(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError("probabilities and labels must be 1-D and the same length")
    if np.isnan(y).any() or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels must be 0 or 1; drop bouts without a winner first")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return p, y


def bout_log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Negative log of the probability given to what happened, per bout."""
    p, y = _arrays(p, y)
    p = np.clip(p, CLIP, 1.0 - CLIP)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def bout_brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p, y = _arrays(p, y)
    return (p - y) ** 2


def bout_correct(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1 if the favoured fighter won, 0 if not, 0.5 where the model has no
    favourite (exactly 0.5): a coin flip is right half the time."""
    p, y = _arrays(p, y)
    return np.where(p == 0.5, 0.5, ((p > 0.5) == (y == 1.0)).astype(np.float64))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(bout_log_loss(p, y).mean()) if len(p) else math.nan


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(bout_brier(p, y).mean()) if len(p) else math.nan


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float(bout_correct(p, y).mean()) if len(p) else math.nan


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion; sensible for small bins."""
    if n == 0:
        return (math.nan, math.nan)
    share = successes / n
    centre = share + z * z / (2 * n)
    half = z * math.sqrt(share * (1 - share) / n + z * z / (4 * n * n))
    scale = 1 + z * z / n
    return ((centre - half) / scale, (centre + half) / scale)


def _binned(p: np.ndarray, y: np.ndarray, edges: Sequence[float]) -> list[dict]:
    p, y = _arrays(p, y)
    confidence = np.maximum(p, 1.0 - p)
    correct = bout_correct(p, y)
    losses = bout_log_loss(p, y)
    # A value on an inner edge belongs to the bin above it; 1.0 to the last.
    index = np.clip(
        np.searchsorted(edges, confidence, side="right") - 1, 0, len(edges) - 2
    )
    rows = []
    for i in range(len(edges) - 1):
        members = index == i
        n = int(members.sum())
        wins = float(correct[members].sum())
        low, high = wilson_interval(wins, n)
        rows.append(
            {
                "low": float(edges[i]),
                "high": float(edges[i + 1]),
                "n": n,
                "mean_confidence": float(confidence[members].mean()) if n else None,
                "favourite_win_rate": wins / n if n else None,
                "win_rate_low": low if n else None,
                "win_rate_high": high if n else None,
                "log_loss": float(losses[members].mean()) if n else None,
            }
        )
    return rows


def reliability_table(
    p: np.ndarray, y: np.ndarray, n_bins: int = ECE_BINS
) -> list[dict]:
    """Per bin of the favoured fighter's probability: how many bouts, the mean
    probability given, and how often the favourite actually won."""
    return _binned(p, y, np.linspace(0.5, 1.0, n_bins + 1).tolist())


def confidence_table(p: np.ndarray, y: np.ndarray) -> list[dict]:
    return _binned(p, y, CONFIDENCE_EDGES)


def expected_calibration_error(
    p: np.ndarray, y: np.ndarray, n_bins: int = ECE_BINS
) -> float:
    """Average gap between the favourite's probability and the favourite's
    win rate, weighted by how many bouts fall in each bin."""
    if not len(p):
        return math.nan
    rows = [r for r in reliability_table(p, y, n_bins) if r["n"]]
    total = sum(r["n"] for r in rows)
    return float(
        sum(r["n"] * abs(r["favourite_win_rate"] - r["mean_confidence"]) for r in rows)
        / total
    )


def summary(p: np.ndarray, y: np.ndarray) -> dict:
    """The headline metrics for one set of bouts."""
    return {
        "n": len(p),
        "log_loss": log_loss(p, y),
        "brier": brier(p, y),
        "accuracy": accuracy(p, y),
        "ece": expected_calibration_error(p, y),
    }


def bootstrap_mean_interval(
    values: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> dict:
    """Mean of per-bout values with a percentile bootstrap interval.

    Used on per-bout losses, and on per-bout differences in loss between two
    models scored on the same bouts (a paired comparison: the bouts are
    resampled, and both models keep their loss on each resampled bout).
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return {"mean": math.nan, "low": math.nan, "high": math.nan, "n": 0}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return {
        "mean": float(values.mean()),
        "low": float(low),
        "high": float(high),
        "n": n,
    }
