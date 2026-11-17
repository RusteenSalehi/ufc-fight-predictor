"""Calibration of the selected model on the validation period.

Two methods are compared by cross-validation inside validation
(docs/ML_REQUIREMENTS.md, "Calibration"):

- **Platt scaling**: ``sigmoid(a * logit(p))``. Platt's method fits a slope
  and an intercept on the model's score; here the intercept is zero by
  construction, because a symmetric model must map 0.5 to 0.5. One number.
- **Isotonic regression**: a non-decreasing step function fitted to both
  orientations of every bout, which makes it symmetric about 0.5.

A calibrator is applied to the favoured fighter's probability only
(``symmetric.py``), so a calibrated prediction is exactly as symmetric as an
uncalibrated one. Calibrators are stored as plain numbers in JSON.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ufc.models.metrics import CLIP, bout_log_loss
from ufc.models.symmetric import EPS

PLATT = "platt"
ISOTONIC = "isotonic"
IDENTITY = "none"
METHODS: tuple[str, ...] = (PLATT, ISOTONIC)

CV_FOLDS = 5


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, CLIP, 1.0 - CLIP)
    return np.log(p) - np.log1p(-p)


@dataclass(frozen=True, slots=True)
class Calibrator:
    """``kind`` and its numbers. ``__call__`` maps the favoured fighter's
    probability (0.5 to 1) to a calibrated probability."""

    kind: str
    slope: float | None = None
    x: tuple[float, ...] = ()
    y: tuple[float, ...] = ()

    def __call__(self, favoured: np.ndarray) -> np.ndarray:
        favoured = np.asarray(favoured, dtype=np.float64)
        if self.kind == IDENTITY:
            return favoured
        if self.kind == PLATT:
            return 1.0 / (1.0 + np.exp(-self.slope * _logit(favoured)))
        if self.kind == ISOTONIC:
            return np.interp(favoured, self.x, self.y)
        raise ValueError(f"unknown calibrator {self.kind!r}")

    def apply(self, p_a: np.ndarray) -> np.ndarray:
        """Calibrated P(A wins) from an uncalibrated symmetric P(A wins)."""
        p_a = np.asarray(p_a, dtype=np.float64)
        if self.kind == IDENTITY:
            return p_a
        favoured = np.clip(self(np.maximum(p_a, 1.0 - p_a)), EPS, 1.0 - EPS)
        return np.where(p_a > 0.5, favoured, np.where(p_a < 0.5, 1.0 - favoured, 0.5))

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "slope": self.slope,
            "x": list(self.x),
            "y": list(self.y),
        }

    @classmethod
    def from_dict(cls, raw: Mapping) -> Calibrator:
        return cls(raw["kind"], raw["slope"], tuple(raw["x"]), tuple(raw["y"]))


IDENTITY_CALIBRATOR = Calibrator(IDENTITY)


def _fit_platt(p_a: np.ndarray, y: np.ndarray) -> Calibrator:
    """The slope that minimises log loss of ``sigmoid(a * logit(p))``.

    The loss is convex in ``a``; Newton's method from a = 1 converges in a
    few steps. Mirrored rows add nothing (their loss is identical), so the
    fit uses one orientation.
    """
    z = _logit(p_a)
    slope = 1.0
    for _ in range(100):
        s = 1.0 / (1.0 + np.exp(-slope * z))
        gradient = float(np.sum(z * (s - y)))
        hessian = float(np.sum(z * z * s * (1.0 - s)))
        if hessian <= 0:
            break
        step = gradient / hessian
        slope -= step
        if abs(step) < 1e-12:
            break
    return Calibrator(PLATT, slope=float(slope))


def _fit_isotonic(p_a: np.ndarray, y: np.ndarray) -> Calibrator:
    """Isotonic regression on both orientations: (p, y) and (1 - p, 1 - y)."""
    both_p = np.concatenate((p_a, 1.0 - p_a))
    both_y = np.concatenate((y, 1.0 - y))
    fitted = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    fitted.fit(both_p, both_y)
    return Calibrator(
        ISOTONIC,
        x=tuple(float(v) for v in fitted.X_thresholds_),
        y=tuple(float(v) for v in fitted.y_thresholds_),
    )


def fit_calibrator(kind: str, p_a: np.ndarray, y: np.ndarray) -> Calibrator:
    """Fit ``kind`` on uncalibrated symmetric probabilities and outcomes."""
    p_a = np.asarray(p_a, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if kind == IDENTITY:
        return IDENTITY_CALIBRATOR
    if len(p_a) == 0:
        raise ValueError("no bouts to calibrate on")
    if kind == PLATT:
        return _fit_platt(p_a, y)
    if kind == ISOTONIC:
        return _fit_isotonic(p_a, y)
    raise ValueError(f"unknown calibration method {kind!r}")


def fold_ids(groups: np.ndarray, n_folds: int = CV_FOLDS, seed: int = 0) -> np.ndarray:
    """A fold number per bout, with every bout of a group in the same fold.

    Groups are event dates: bouts on one card are scored together, never
    split between the part a calibrator is fitted on and the part it is
    judged on. Groups are shuffled with a fixed seed and dealt out in turn.
    """
    unique = np.unique(groups)
    order = np.random.default_rng(seed).permutation(len(unique))
    fold_of = {unique[j]: i % n_folds for i, j in enumerate(order)}
    return np.array([fold_of[g] for g in groups])


def cross_validated(
    kind: str, p_a: np.ndarray, y: np.ndarray, folds: np.ndarray
) -> np.ndarray:
    """Calibrated P(A wins) for every bout, each from a calibrator fitted on
    the other folds."""
    out = np.empty(len(p_a))
    for fold in np.unique(folds):
        held_out = folds == fold
        out[held_out] = fit_calibrator(kind, p_a[~held_out], y[~held_out]).apply(
            p_a[held_out]
        )
    return out


def choose_method(
    p_a: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0
) -> dict:
    """Cross-validated log loss of each method, and the one with the lowest.

    Returns the losses (including ``none``, the uncalibrated model, for
    reference), the choice, and the cross-validated predictions of each.
    """
    folds = fold_ids(groups, seed=seed)
    predictions = {
        kind: cross_validated(kind, p_a, y, folds) for kind in (*METHODS, IDENTITY)
    }
    losses = {
        kind: float(bout_log_loss(p, y).mean()) for kind, p in predictions.items()
    }
    chosen = min(METHODS, key=lambda kind: (losses[kind], METHODS.index(kind)))
    return {
        "cv_log_loss": losses,
        "chosen": chosen,
        "n_folds": len(np.unique(folds)),
        "n_bouts": len(p_a),
        "predictions": predictions,
    }
