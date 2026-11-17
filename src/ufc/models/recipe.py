"""A recipe is everything that defines the candidate model except the data.

``fit_recipe`` turns a recipe, a training period and a validation period into
a fitted, calibrated candidate. The registered model and every fold of the
walk-forward backtest are made by this one function, so the backtest measures
the procedure that produced the registered model and not a relative of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ufc.models.calibrate import Calibrator, fit_calibrator
from ufc.models.design import (
    FeatureSpec,
    both_orientations,
    labels,
    matrix,
    predict_pair,
    swapped,
)
from ufc.models.gbm import GbmConfig, GbmModel, fit_gbm
from ufc.models.splits import recency_weights
from ufc.models.symmetric import symmetric_probability


@dataclass(frozen=True, slots=True)
class Recipe:
    """Library, settings and seed; recency half-life; inputs; calibration."""

    gbm: GbmConfig
    half_life_years: float | None
    spec: FeatureSpec
    calibration: str

    def to_dict(self) -> dict:
        return {
            "gbm": self.gbm.to_dict(),
            "half_life_years": self.half_life_years,
            "calibration": self.calibration,
            **self.spec.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping) -> Recipe:
        return cls(
            gbm=GbmConfig.from_dict(raw["gbm"]),
            half_life_years=raw["half_life_years"],
            spec=FeatureSpec.from_dict(raw),
            calibration=raw["calibration"],
        )


@dataclass(frozen=True, slots=True)
class FitData:
    """Arrays for one fit: training rows in both orientations with weights,
    and validation rows in each orientation."""

    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    valid_ab: np.ndarray
    valid_ba: np.ndarray
    valid_y: np.ndarray


def fit_data(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    spec: FeatureSpec,
    half_life_years: float | None,
    train_end: date,
) -> FitData:
    """``train`` and ``valid`` must hold bouts with a winner only.

    Recency weights are measured back from ``train_end``, the last day of
    the training period.
    """
    weights = recency_weights(train["date"], train_end, half_life_years)
    x, y, w = both_orientations(matrix(train, spec), labels(train), weights, spec)
    valid_ab = matrix(valid, spec)
    return FitData(x, y, w, valid_ab, swapped(valid_ab, spec), labels(valid))


def fit_early_stopped(config: GbmConfig, data: FitData) -> tuple[GbmModel, np.ndarray]:
    """Fit with the number of trees chosen on validation log loss (both
    orientations of every validation bout). Returns the model and its
    symmetric, uncalibrated P(A wins) on validation."""
    stop_x = np.vstack((data.valid_ab, data.valid_ba))
    stop_y = np.concatenate((data.valid_y, 1.0 - data.valid_y))
    model = fit_gbm(config, data.x, data.y, data.w, valid=(stop_x, stop_y))
    p_valid = symmetric_probability(
        model.one_sided(data.valid_ab), model.one_sided(data.valid_ba)
    )
    return model, p_valid


@dataclass(frozen=True, slots=True)
class Candidate:
    """A fitted model with its calibrator and the inputs it reads."""

    recipe: Recipe
    model: GbmModel
    calibrator: Calibrator

    @property
    def spec(self) -> FeatureSpec:
        return self.recipe.spec

    def predict_pair(
        self, frame: pd.DataFrame, calibrated: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """(P(A wins), P(B wins)): both orientations averaged, calibrated,
        summing to exactly 1."""
        return predict_pair(
            self.model, frame, self.spec, self.calibrator if calibrated else None
        )

    def predict(self, frame: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        return self.predict_pair(frame, calibrated)[0]


def fit_recipe(
    recipe: Recipe, train: pd.DataFrame, valid: pd.DataFrame, train_end: date
) -> tuple[Candidate, np.ndarray]:
    """Fit on ``train``; choose the number of trees and fit the calibrator on
    ``valid``. Returns the candidate and its uncalibrated validation
    probabilities."""
    data = fit_data(train, valid, recipe.spec, recipe.half_life_years, train_end)
    model, p_valid = fit_early_stopped(recipe.gbm, data)
    calibrator = fit_calibrator(recipe.calibration, p_valid, data.valid_y)
    return Candidate(recipe, model, calibrator), p_valid
