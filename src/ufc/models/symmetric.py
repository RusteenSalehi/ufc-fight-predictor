"""Turn a model's two one-sided outputs into one symmetric probability.

A model is asked twice about every bout: ``u = f(A, B)`` and ``v = f(B, A)``.
The requirement (docs/ML_REQUIREMENTS.md, "Symmetry") is that inference
averages ``u`` and ``1 - v``, that P(A) + P(B) is exactly 1, and that swapping
the fighters swaps the two probabilities exactly.

Averaging alone gives that only up to rounding. This module makes it exact in
floating point by never computing the two sides separately:

1. ``m = u - v``. Swapping the fighters gives ``v - u``, which is exactly
   ``-m``, so ``|m|`` is identical in both orientations.
2. ``q = 0.5 + |m| / 2`` is the average of ``u`` and ``1 - v`` for whichever
   fighter the model favours. It is the same number in both orientations.
3. A calibrator, if any, maps ``q`` to ``c``; still one number per bout.
4. The favoured fighter gets ``c`` and the other gets ``1 - c``. For any
   ``c`` in [0, 1] the floating-point sum ``c + (1 - c)`` is exactly 1.

So the pair of outputs is the same two numbers whichever way the bout is
presented, and they sum to exactly 1.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

EPS = 1e-6
"""Final probabilities are kept inside [EPS, 1 - EPS]. No amount of evidence
in this data justifies certainty, and a probability of exactly 0 or 1 that
turns out wrong has infinite log loss."""

Calibration = Callable[[np.ndarray], np.ndarray]


def symmetric_pair(
    u: np.ndarray, v: np.ndarray, calibrate: Calibration | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(P(A wins), P(B wins)) from ``u = f(A, B)`` and ``v = f(B, A)``.

    ``calibrate`` maps the favoured fighter's averaged probability, a value
    in [0.5, 1], to a calibrated probability.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError("the two orientations must have the same shape")
    if not (np.isfinite(u).all() and np.isfinite(v).all()):
        raise ValueError("model outputs must be finite")
    margin = u - v
    favoured = 0.5 + np.abs(margin) / 2.0
    if calibrate is not None:
        favoured = np.asarray(calibrate(favoured), dtype=np.float64)
    favoured = np.clip(favoured, EPS, 1.0 - EPS)
    other = 1.0 - favoured
    a_favoured, b_favoured = margin > 0, margin < 0
    p_a = np.where(a_favoured, favoured, np.where(b_favoured, other, 0.5))
    p_b = np.where(a_favoured, other, np.where(b_favoured, favoured, 0.5))
    return p_a, p_b


def symmetric_probability(
    u: np.ndarray, v: np.ndarray, calibrate: Calibration | None = None
) -> np.ndarray:
    """P(A wins); see ``symmetric_pair``."""
    return symmetric_pair(u, v, calibrate)[0]


def favoured_probability(p_a: np.ndarray) -> np.ndarray:
    """The probability given to whichever fighter is favoured: max(p, 1 - p)."""
    p_a = np.asarray(p_a, dtype=np.float64)
    return np.maximum(p_a, 1.0 - p_a)
