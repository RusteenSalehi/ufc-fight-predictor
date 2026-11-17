"""Scoring model variants on validation, and the hyperparameter search.

Validation holds about a thousand bouts, so a difference of a few thousandths
in log loss between two variants is mostly noise. Three things keep the
choices honest about that:

- a variant's score is its validation log loss averaged over several seeds,
  so a lucky draw of subsamples does not decide a choice;
- every comparison comes with a paired bootstrap interval over bouts;
- every variant gets the same treatment: fixed settings, number of trees by
  early stopping on validation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed

from ufc.models.gbm import LIBRARIES, LIGHTGBM, XGBOOST, GbmConfig
from ufc.models.metrics import bootstrap_mean_interval, bout_log_loss
from ufc.models.recipe import FitData, fit_early_stopped

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
"""Seeds a variant's validation log loss is averaged over."""

HALF_LIVES: tuple[float | None, ...] = (None, 2.0, 4.0, 8.0)
"""Recency half-lives in years (docs/ML_REQUIREMENTS.md); None is no weighting."""

REFERENCE: dict[str, dict[str, float]] = {
    XGBOOST: {
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    LIGHTGBM: {
        "learning_rate": 0.03,
        "num_leaves": 8,
        "min_child_samples": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
}
"""Ordinary settings fixed before any result was seen. Feature choices are
made with these, before the search, so the search cannot tilt them."""

SEARCH_SPACE: dict[str, dict[str, tuple[float, ...]]] = {
    XGBOOST: {
        "learning_rate": (0.02, 0.03, 0.05),
        "max_depth": (2, 3, 4, 5),
        "min_child_weight": (1, 5, 10, 20, 40),
        "subsample": (0.6, 0.7, 0.8, 0.9, 1.0),
        "colsample_bytree": (0.4, 0.6, 0.8, 1.0),
        "reg_lambda": (1.0, 3.0, 10.0, 30.0),
    },
    LIGHTGBM: {
        "learning_rate": (0.02, 0.03, 0.05),
        "num_leaves": (3, 4, 6, 8, 12, 16),
        "min_child_samples": (20, 40, 80, 160),
        "subsample": (0.6, 0.7, 0.8, 0.9, 1.0),
        "colsample_bytree": (0.4, 0.6, 0.8, 1.0),
        "reg_lambda": (1.0, 3.0, 10.0, 30.0),
    },
}

N_TRIALS = 40
"""Settings tried per library, the reference included. One seed each."""

TOP_K = 5
"""The best of those are scored again on every seed before one is chosen."""

SEARCH_SEED = 0


def sample_params(
    library: str, n_trials: int = N_TRIALS, seed: int = SEARCH_SEED
) -> list[dict[str, float]]:
    """The reference settings followed by distinct random draws from the
    search space. The same seed always gives the same list."""
    space = SEARCH_SPACE[library]
    size = int(np.prod([len(values) for values in space.values()]))
    if n_trials > size:
        raise ValueError("more trials than distinct settings")
    rng = np.random.default_rng(seed + LIBRARIES.index(library))
    chosen = [dict(REFERENCE[library])]
    while len(chosen) < n_trials:
        draw = {
            name: values[int(rng.integers(len(values)))]
            for name, values in space.items()
        }
        if draw not in chosen:
            chosen.append(draw)
    return chosen


@dataclass(frozen=True, slots=True)
class Variant:
    """One thing to score: a name, settings (the seed is supplied by the
    scorer) and the arrays to fit and validate on."""

    name: str
    config: GbmConfig
    data: FitData


@dataclass(frozen=True, slots=True)
class Score:
    """Validation results of one variant over the seeds it was run on."""

    name: str
    log_loss: float  # mean over seeds
    seed_log_losses: tuple[float, ...]
    n_trees: tuple[int, ...]
    bout_losses: np.ndarray  # per validation bout, mean over seeds

    def to_dict(self) -> dict:
        return {
            "log_loss": self.log_loss,
            "seed_log_losses": list(self.seed_log_losses),
            "n_trees": list(self.n_trees),
        }


def _score_one(config: GbmConfig, data: FitData) -> tuple[int, np.ndarray]:
    model, p_valid = fit_early_stopped(config, data)
    return model.n_trees, bout_log_loss(p_valid, data.valid_y)


def score_variants(
    variants: Sequence[Variant], seeds: Sequence[int] = SEEDS, n_jobs: int = 1
) -> dict[str, Score]:
    """Fit every variant once per seed and collect validation losses.

    Fits run in parallel across processes; each fit is single-threaded and
    seeded, so the scores do not depend on ``n_jobs``.
    """
    names = [v.name for v in variants]
    if len(set(names)) != len(names):
        raise ValueError("variant names must be unique")
    tasks = [(v, seed) for v in variants for seed in seeds]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_score_one)(v.config.with_seed(seed), v.data) for v, seed in tasks
    )
    scores: dict[str, Score] = {}
    for variant in variants:
        runs = [r for (v, _), r in zip(tasks, results, strict=True) if v is variant]
        per_seed = np.array([losses for _, losses in runs])
        scores[variant.name] = Score(
            name=variant.name,
            log_loss=float(per_seed.mean()),
            seed_log_losses=tuple(float(v) for v in per_seed.mean(axis=1)),
            n_trees=tuple(int(n) for n, _ in runs),
            bout_losses=per_seed.mean(axis=0),
        )
    return scores


def difference(losses: np.ndarray, reference: np.ndarray, seed: int = 0) -> dict:
    """Mean per-bout loss minus the reference's, with a paired bootstrap
    interval. Negative means better than the reference."""
    interval = bootstrap_mean_interval(losses - reference, seed=seed)
    return {
        "delta": interval["mean"],
        "delta_low": interval["low"],
        "delta_high": interval["high"],
    }


def pooled(scores: Mapping[str, Score], names: Sequence[str]) -> np.ndarray:
    """Per-bout loss averaged over several variants (for example the same
    feature option scored with both libraries)."""
    return np.mean([scores[name].bout_losses for name in names], axis=0)
