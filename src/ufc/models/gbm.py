"""XGBoost and LightGBM behind one small interface.

Both are fitted on rows that already hold both orientations of every bout,
with sample weights. Missing values are passed through as NaN; both libraries
learn which way to send them at each split, so nothing is imputed.

Every fit runs on one thread with a fixed seed. Parallelism, where it is
used, is across fits, so results do not depend on the machine's core count.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import xgboost as xgb

XGBOOST = "xgboost"
LIGHTGBM = "lightgbm"
LIBRARIES: tuple[str, ...] = (XGBOOST, LIGHTGBM)

MAX_TREES = 3000
PATIENCE = 100
"""Early stopping: trees are added until validation log loss has not improved
for ``PATIENCE`` rounds, up to ``MAX_TREES``. The number kept is the best."""

MODEL_FILES = {XGBOOST: "model.json", LIGHTGBM: "model.txt"}
"""Each library's own text format. No pickles: a saved model can be read
without running code from the file."""


@dataclass(frozen=True, slots=True)
class GbmConfig:
    """A library, its tuned settings, and the seed.

    ``params`` uses the names in ``tune.py``'s search spaces. Everything that
    is not tuned is fixed in ``_xgboost_params`` and ``_lightgbm_params``.
    """

    library: str
    params: Mapping[str, float] = field(default_factory=dict)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.library not in LIBRARIES:
            raise ValueError(f"unknown library {self.library!r}")

    def with_seed(self, seed: int) -> GbmConfig:
        return GbmConfig(self.library, self.params, seed)

    def to_dict(self) -> dict:
        return {"library": self.library, "params": dict(self.params), "seed": self.seed}

    @classmethod
    def from_dict(cls, raw: Mapping) -> GbmConfig:
        return cls(raw["library"], dict(raw["params"]), int(raw["seed"]))


def _xgboost_params(config: GbmConfig) -> dict[str, Any]:
    p = config.params
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "nthread": 1,
        "seed": config.seed,
        "eta": p["learning_rate"],
        "max_depth": int(p["max_depth"]),
        "min_child_weight": p["min_child_weight"],
        "subsample": p["subsample"],
        "colsample_bytree": p["colsample_bytree"],
        "lambda": p["reg_lambda"],
    }


def _lightgbm_params(config: GbmConfig) -> dict[str, Any]:
    p = config.params
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_threads": 1,
        "seed": config.seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
        "learning_rate": p["learning_rate"],
        "num_leaves": int(p["num_leaves"]),
        "min_child_samples": int(p["min_child_samples"]),
        "bagging_fraction": p["subsample"],
        "bagging_freq": 1,
        "feature_fraction": p["colsample_bytree"],
        "lambda_l2": p["reg_lambda"],
    }


class GbmModel:
    """A fitted booster holding exactly ``n_trees`` trees."""

    def __init__(self, library: str, booster: Any, n_trees: int) -> None:
        self.library = library
        self.n_trees = n_trees
        self._booster = booster

    def one_sided(self, x: np.ndarray) -> np.ndarray:
        """P(first fighter wins) for each row, as float64."""
        if self.library == XGBOOST:
            out = self._booster.inplace_predict(x)
        else:
            out = self._booster.predict(x, num_threads=1)
        return np.asarray(out, dtype=np.float64)

    def save(self, directory: Path) -> Path:
        path = directory / MODEL_FILES[self.library]
        self._booster.save_model(str(path))
        return path

    @classmethod
    def load(cls, library: str, path: Path) -> GbmModel:
        if library == XGBOOST:
            booster = xgb.Booster(model_file=str(path))
            booster.set_param({"nthread": 1})
            return cls(library, booster, booster.num_boosted_rounds())
        if library == LIGHTGBM:
            booster = lgb.Booster(model_file=str(path))
            return cls(library, booster, booster.num_trees())
        raise ValueError(f"unknown library {library!r}")


def fit_gbm(
    config: GbmConfig,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    n_trees: int | None = None,
    valid: tuple[np.ndarray, np.ndarray] | None = None,
) -> GbmModel:
    """Fit on ``x``, ``y``, ``weights``.

    With ``n_trees`` given, exactly that many trees are grown. Otherwise
    ``valid`` (inputs and labels) is required and the number of trees is
    chosen by early stopping on its log loss.
    """
    if (n_trees is None) == (valid is None):
        raise ValueError("give either n_trees or a validation set, not both")
    if n_trees is not None and n_trees < 1:
        raise ValueError("n_trees must be at least 1")

    if config.library == XGBOOST:
        params = _xgboost_params(config)
        train = xgb.DMatrix(x, label=y, weight=weights, nthread=1)
        if n_trees is not None:
            return GbmModel(XGBOOST, xgb.train(params, train, n_trees), n_trees)
        held_out = xgb.DMatrix(valid[0], label=valid[1], nthread=1)
        booster = xgb.train(
            params,
            train,
            MAX_TREES,
            evals=[(held_out, "valid")],
            early_stopping_rounds=PATIENCE,
            verbose_eval=False,
        )
        best = booster.best_iteration + 1
        return GbmModel(XGBOOST, booster[:best], best)

    params = _lightgbm_params(config)
    train = lgb.Dataset(x, label=y, weight=weights)
    if n_trees is not None:
        return GbmModel(LIGHTGBM, lgb.train(params, train, n_trees), n_trees)
    held_out = lgb.Dataset(valid[0], label=valid[1], reference=train)
    booster = lgb.train(
        params,
        train,
        MAX_TREES,
        valid_sets=[held_out],
        callbacks=[lgb.early_stopping(PATIENCE, verbose=False)],
    )
    best = booster.best_iteration
    trimmed = lgb.Booster(model_str=booster.model_to_string(num_iteration=best))
    return GbmModel(LIGHTGBM, trimmed, best)
