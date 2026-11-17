"""Baselines: Elo, and logistic regression on the standardized features.

A candidate model has to beat both on validation log loss to count as better
(docs/ML_REQUIREMENTS.md, "Baselines" and "Candidate models").
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from ufc.features.elo import EloParams
from ufc.models.design import FeatureSpec

ELO_FEATURE = "d_elo"
ELO_SPEC = FeatureSpec((ELO_FEATURE,))
"""The Elo baseline reads one column: the difference of the two as-of
ratings that the feature pipeline computed with training-fit parameters."""


@dataclass(frozen=True, slots=True)
class EloModel:
    """Win probability from the rating difference: 1 / (1 + 10^(-d / scale)).

    No fitting happens here. The ratings come from the feature table, whose
    Elo parameters were fit on the training period only (DECISIONS D32).
    """

    scale: float = EloParams().scale

    def one_sided(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] != 1:
            raise ValueError("the Elo baseline reads the rating difference only")
        if np.isnan(x).any():
            raise ValueError("a rating difference is missing")
        return 1.0 / (1.0 + 10.0 ** (-x[:, 0] / self.scale))


CLIP_SD = 5.0
"""Standardized values are limited to this many training standard deviations
before the regression sees them, so one extreme ratio cannot dominate a
prediction. A constant, not tuned."""

C_GRID: tuple[float, ...] = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
"""Inverse L2 strengths tried on validation."""


@dataclass(frozen=True, slots=True)
class LogisticModel:
    """A linear model on antisymmetric features with no intercept.

    ``scale`` is each feature's training standard deviation about zero. A
    difference feature has mean zero once both orientations are counted, so
    standardizing is division by ``scale``. A missing value becomes 0, "no
    known difference", after scaling. With no intercept the score of (B, A)
    is exactly minus the score of (A, B).
    """

    scale: np.ndarray
    coef: np.ndarray
    c: float

    def score(self, x: np.ndarray) -> np.ndarray:
        z = np.clip(x / self.scale, -CLIP_SD, CLIP_SD)
        z = np.where(np.isnan(z), 0.0, z)
        # Summed row by row, so a bout's score does not depend on which
        # other bouts are in the batch.
        return (z * self.coef).sum(axis=1)

    def one_sided(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.score(x)))

    def to_dict(self, names: tuple[str, ...]) -> dict:
        return {
            "c": self.c,
            "clip_sd": CLIP_SD,
            "coef": dict(zip(names, self.coef.tolist(), strict=True)),
            "scale": dict(zip(names, self.scale.tolist(), strict=True)),
        }


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore"):
        scale = np.sqrt(np.nanmean(x**2, axis=0))
    # A feature that is always zero or always missing carries nothing.
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    z = np.clip(x / scale, -CLIP_SD, CLIP_SD)
    return np.where(np.isnan(z), 0.0, z), scale


def fit_logistic(
    x: np.ndarray, y: np.ndarray, c: float, spec: FeatureSpec
) -> LogisticModel:
    """Fit on orientation (A, B) only.

    With no intercept, the loss of a mirrored row (-x, 1 - y) equals the loss
    of (x, y), so adding the mirrored rows would change nothing.
    """
    if spec.symmetric:
        raise ValueError(
            "symmetric features have a zero coefficient in a model with no "
            "intercept; leave them out of the logistic baseline"
        )
    z, scale = _standardize(x)
    model = LogisticRegression(C=c, fit_intercept=False, max_iter=5000, tol=1e-8)
    model.fit(z, y)
    return LogisticModel(scale=scale, coef=model.coef_[0].astype(np.float64), c=c)


def logistic_from_dict(raw: Mapping, names: tuple[str, ...]) -> LogisticModel:
    return LogisticModel(
        scale=np.array([raw["scale"][n] for n in names]),
        coef=np.array([raw["coef"][n] for n in names]),
        c=raw["c"],
    )
