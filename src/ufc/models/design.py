"""Which columns a model reads, and the two orientations of a bout.

A model's inputs are named by a ``FeatureSpec``. Antisymmetric columns (the
``d_`` features) change sign when the fighters swap; symmetric columns (the
``lvl_`` candidates) do not. That is all a model needs to know to be asked
about a bout both ways round, and nothing else distinguishes the fighters:
there is no corner column to read (docs/ML_REQUIREMENTS.md, "Symmetry").
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from ufc.models.symmetric import Calibration, symmetric_pair

FORBIDDEN_COLUMNS = frozenset({"label", "result", "bout_id", "event_id", "date"})
"""Outcomes and identifiers can never be model inputs."""


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Ordered model inputs: antisymmetric columns first, then symmetric."""

    antisymmetric: tuple[str, ...]
    symmetric: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = self.names
        if len(set(names)) != len(names):
            raise ValueError("a feature is listed twice")
        bad = FORBIDDEN_COLUMNS & set(names)
        if bad:
            raise ValueError(f"not allowed as model inputs: {sorted(bad)}")
        if not self.antisymmetric:
            raise ValueError("a model needs at least one antisymmetric feature")

    @property
    def names(self) -> tuple[str, ...]:
        return (*self.antisymmetric, *self.symmetric)

    def without(self, names: Iterable[str]) -> FeatureSpec:
        drop = set(names)
        return FeatureSpec(
            tuple(n for n in self.antisymmetric if n not in drop),
            tuple(n for n in self.symmetric if n not in drop),
        )

    def with_symmetric(self, names: Iterable[str]) -> FeatureSpec:
        return FeatureSpec(self.antisymmetric, (*self.symmetric, *names))

    def to_dict(self) -> dict:
        return {
            "features": list(self.names),
            "antisymmetric": list(self.antisymmetric),
            "symmetric": list(self.symmetric),
        }

    @classmethod
    def from_dict(cls, raw: Mapping) -> FeatureSpec:
        spec = cls(tuple(raw["antisymmetric"]), tuple(raw["symmetric"]))
        if list(spec.names) != list(raw["features"]):
            raise ValueError("feature list and its two parts disagree")
        return spec


def matrix(frame: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    """The inputs for orientation (A, B), one row per bout, in ``spec`` order."""
    return np.ascontiguousarray(frame[list(spec.names)].to_numpy(dtype=np.float64))


def swapped(x: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    """The inputs for orientation (B, A): antisymmetric columns negated."""
    out = x.copy()
    n = len(spec.antisymmetric)
    out[:, :n] = -out[:, :n]
    return out


def labels(frame: pd.DataFrame) -> np.ndarray:
    y = frame["label"].to_numpy(dtype=np.float64)
    if np.isnan(y).any():
        raise ValueError("a bout without a winner reached a model; drop them first")
    return y


def both_orientations(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, spec: FeatureSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every bout twice: (A, B, label) and (B, A, 1 - label). Training on
    both removes any trace of which fighter the source listed first."""
    return (
        np.vstack((x, swapped(x, spec))),
        np.concatenate((y, 1.0 - y)),
        np.concatenate((weights, weights)),
    )


class OneSided(Protocol):
    """Anything that scores one orientation: P(the first fighter wins)."""

    def one_sided(self, x: np.ndarray) -> np.ndarray: ...


def predict_pair(
    model: OneSided,
    frame: pd.DataFrame,
    spec: FeatureSpec,
    calibrate: Calibration | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(P(A wins), P(B wins)) for every bout in ``frame``.

    The model is asked about both orientations and the two answers are
    combined by ``symmetric_pair``.
    """
    x = matrix(frame, spec)
    return symmetric_pair(
        model.one_sided(x), model.one_sided(swapped(x, spec)), calibrate
    )


def predict(
    model: OneSided,
    frame: pd.DataFrame,
    spec: FeatureSpec,
    calibrate: Calibration | None = None,
) -> np.ndarray:
    """P(A wins) for every bout in ``frame``."""
    return predict_pair(model, frame, spec, calibrate)[0]
