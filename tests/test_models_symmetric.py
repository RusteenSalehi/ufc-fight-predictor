"""Combining the two orientations: exact in floating point, not approximate."""

import numpy as np
import pytest

from ufc.models.calibrate import Calibrator, fit_calibrator
from ufc.models.symmetric import (
    EPS,
    favoured_probability,
    symmetric_pair,
    symmetric_probability,
)


def outputs(seed: int = 0, n: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    """One-sided model outputs, including awkward values: ties, near-ties,
    extremes, and float32 values as XGBoost returns them."""
    rng = np.random.default_rng(seed)
    u, v = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    u[:100] = v[:100]  # no favourite
    u[100:200] = v[100:200] + 1e-17  # rounds to a tie
    u[200:300] = np.nextafter(v[200:300], 1.0)  # the smallest possible edge
    u[300:320], v[300:320] = 1.0, 0.0
    u[320:340], v[320:340] = 0.0, 1.0
    u[340:400] = rng.uniform(0, 1, 60).astype(np.float32)
    return u, v


CALIBRATORS = {
    "none": None,
    "platt": Calibrator("platt", slope=0.73),
    "steep": Calibrator("platt", slope=3.1),
    "backwards": Calibrator("platt", slope=-0.4),  # maps favourites below 50%
    "isotonic": Calibrator(
        "isotonic", x=(0.0, 0.31, 0.5, 0.69, 1.0), y=(0.02, 0.33, 0.47, 0.71, 0.96)
    ),
}


@pytest.mark.parametrize("kind", CALIBRATORS)
def test_the_two_probabilities_sum_to_exactly_one(kind):
    p_a, p_b = symmetric_pair(*outputs(), CALIBRATORS[kind])
    assert np.all(p_a + p_b == 1.0)  # exact equality, every bout
    assert np.all((p_a >= EPS) & (p_a <= 1 - EPS))


@pytest.mark.parametrize("kind", CALIBRATORS)
def test_swapping_the_fighters_swaps_the_probabilities_exactly(kind):
    u, v = outputs(seed=1)
    p_a, p_b = symmetric_pair(u, v, CALIBRATORS[kind])
    q_a, q_b = symmetric_pair(v, u, CALIBRATORS[kind])
    np.testing.assert_array_equal(p_a, q_b)
    np.testing.assert_array_equal(p_b, q_a)
    np.testing.assert_array_equal(
        symmetric_probability(u, v, CALIBRATORS[kind]),
        symmetric_probability(v, u, CALIBRATORS[kind]) * 0 + q_b,
    )


def test_it_is_the_average_of_one_orientation_and_one_minus_the_other():
    u, v = outputs(seed=2)
    np.testing.assert_allclose(
        symmetric_probability(u, v),
        np.clip((u + (1 - v)) / 2, EPS, 1 - EPS),
        atol=1e-15,
    )
    assert symmetric_probability(np.array([0.7]), np.array([0.2]))[0] == pytest.approx(
        0.75
    )
    # No favourite: exactly one half each.
    p_a, p_b = symmetric_pair(np.array([0.4, 0.4 + 1e-17]), np.array([0.4, 0.4]))
    np.testing.assert_array_equal(p_a, [0.5, 0.5])
    np.testing.assert_array_equal(p_b, [0.5, 0.5])


def test_probabilities_are_never_certain():
    p_a, p_b = symmetric_pair(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    np.testing.assert_array_equal(p_a, [1 - EPS, 1 - (1 - EPS)])
    assert np.all(p_a + p_b == 1.0)


def test_favoured_probability_is_the_same_from_either_side():
    u, v = outputs(seed=3)
    p_a, p_b = symmetric_pair(u, v, CALIBRATORS["platt"])
    np.testing.assert_array_equal(favoured_probability(p_a), favoured_probability(p_b))
    assert np.all(favoured_probability(p_a) >= 0.5)


def test_calibrator_applied_to_a_probability_stays_symmetric():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.02, 0.98, 5000)
    y = (rng.random(5000) < p).astype(float)
    for kind in ("platt", "isotonic"):
        calibrator = fit_calibrator(kind, p, y)
        forward, backward = calibrator.apply(p), calibrator.apply(1.0 - p)
        assert np.all(forward + backward == 1.0)
        # The same numbers as calibrating inside symmetric_pair.
        u, v = p, 1.0 - p  # a model that is already symmetric
        np.testing.assert_allclose(
            symmetric_probability(u, v, calibrator), forward, atol=1e-12
        )


def test_bad_outputs_are_refused():
    with pytest.raises(ValueError, match="finite"):
        symmetric_pair(np.array([np.nan]), np.array([0.5]))
    with pytest.raises(ValueError, match="shape"):
        symmetric_pair(np.array([0.5, 0.5]), np.array([0.5]))
