"""M4.6 performance model and family 16 (DECISIONS D63).

Leakage (perturbation, same card, fitting on training only) and listing-order
symmetry of the family 16 columns are covered by the parametrised fixtures in
test_features_leakage.py and test_features_symmetry.py, which compare whole
feature tables. This file checks the transform and the values.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.feature_helpers import random_league, swap_corners
from ufc.features.build import build_features
from ufc.features.components import outcome_for
from ufc.features.config import FeatureConfig
from ufc.features.data import BoutFacts
from ufc.features.elo import EloParams, EloTracker
from ufc.features.fighter import POS, fighter_vector
from ufc.features.matchup import CANDIDATE_FEATURE_NAMES, FEATURES
from ufc.features.performance import (
    INPUTS,
    NO_CONTROL,
    LinearFit,
    PerformanceModel,
    fit_performance_model,
    performance_inputs,
    rated_performance,
    training_rows,
)
from ufc.features.store import FittedParams

PERFORMANCE = [
    n
    for n in CANDIDATE_FEATURE_NAMES
    if n.startswith(("d_deserved", "d_luck", "d_perf"))
]


def _line(sig=0, kd=0, td=0, sub=0, ctrl: int | None = 0) -> dict:
    return {
        "knockdowns": kd,
        "sig_str_landed": sig,
        "sig_str_attempted": sig * 2,
        "td_landed": td,
        "sub_attempts": sub,
        "ctrl_s": ctrl,
    }


def _bout(
    rounds_a: list[dict],
    rounds_b: list[dict],
    fight_time_s: int,
    *,
    winner: str | None = "A",
    method: str = "Decision - Unanimous",
    day: date = date(2020, 1, 1),
    bout_id: str = "x",
) -> BoutFacts:
    def total(lines):
        return {k: sum(line[k] for line in lines) for k in lines[0] if k != "ctrl_s"}

    return BoutFacts(
        bout_id=bout_id,
        event_id="e",
        date=day,
        fighter_a_id="A",
        fighter_b_id="B",
        weight_class="Lightweight",
        is_womens=False,
        is_title_bout=False,
        scheduled_rounds=3,
        time_format="3 Rnd (5-5-5)",
        result="win" if winner else "draw",
        winner_id=winner,
        method=method,
        end_round=len(rounds_a),
        end_time_s=fight_time_s - 300 * (len(rounds_a) - 1),
        fight_time_s=fight_time_s,
        has_stats=True,
        totals={"A": total(rounds_a), "B": total(rounds_b)},
        rounds={"A": tuple(rounds_a), "B": tuple(rounds_b)},
    )


# --- inputs ----------------------------------------------------------------------


def test_inputs_by_hand_for_a_three_round_decision():
    a = [_line(20, ctrl=60), _line(10, td=2, ctrl=120), _line(30, kd=1, ctrl=0)]
    b = [_line(15, ctrl=0), _line(10, ctrl=30), _line(10, sub=1, ctrl=90)]
    x = performance_inputs(_bout(a, b, 900), "A")
    per = 300 / 900
    expected = {
        "sig_landed": (60 - 35) * per,
        "knockdowns": 1 * per,
        "td_landed": 2 * per,
        "sub_attempts": -1 * per,
        "ctrl_min": (180 - 120) / 60 * per,
        # rounds on strikes: A led 1 and 3, round 2 even
        "rounds_sig": (2 - 0) / 3,
        # rounds on control: A led 1 and 2, B led 3
        "rounds_ctrl": (2 - 1) / 3,
    }
    np.testing.assert_allclose(x, [expected[n] for n in INPUTS])


def test_a_short_finish_counts_as_one_full_round():
    x = performance_inputs(_bout([_line(4, kd=1)], [_line(0)], 30), "A")
    assert x[INPUTS.index("sig_landed")] == 4.0
    assert x[INPUTS.index("knockdowns")] == 1.0


def test_inputs_are_exactly_negated_from_the_other_side():
    a = [_line(13, ctrl=61), _line(7, td=1, ctrl=17)]
    b = [_line(11, ctrl=3), _line(9, kd=1, ctrl=29)]
    bout = _bout(a, b, 487)
    np.testing.assert_array_equal(
        performance_inputs(bout, "A"), -performance_inputs(bout, "B")
    )


def test_missing_control_blanks_only_the_control_inputs():
    bout = _bout([_line(10, ctrl=None)], [_line(5, ctrl=40)], 300)
    x = performance_inputs(bout, "A")
    assert np.isnan(x[INPUTS.index("ctrl_min")])
    assert np.isnan(x[INPUTS.index("rounds_ctrl")])
    assert not np.isnan(x[[INPUTS.index(n) for n in NO_CONTROL]]).any()


def test_a_bout_without_stats_has_no_inputs():
    bout = replace(_bout([_line(1)], [_line(1)], 300), has_stats=False)
    assert performance_inputs(bout, "A") is None


# --- the model --------------------------------------------------------------------


def _model(full_coef, no_control_coef) -> PerformanceModel:
    return PerformanceModel(
        LinearFit(INPUTS, (1.0,) * len(INPUTS), tuple(full_coef), 100),
        LinearFit(NO_CONTROL, (1.0,) * len(NO_CONTROL), tuple(no_control_coef), 100),
        date(2022, 12, 31),
    )


def test_deserved_uses_the_no_control_fit_when_control_is_missing():
    model = _model([0.0] * 7, [1.0, 0, 0, 0, 0])
    with_ctrl = _bout([_line(10, ctrl=0)], [_line(5, ctrl=0)], 300)
    without = _bout([_line(10, ctrl=None)], [_line(5, ctrl=0)], 300)
    assert model.deserved(with_ctrl, "A") == 0.5  # full fit: all zeros
    assert model.deserved(without, "A") == pytest.approx(1 / (1 + math.exp(-5.0)))


def test_deserved_pair_is_a_mirror_image_and_does_not_depend_on_listing():
    model = _model([0.1, 0.8, 0.3, 0.2, 0.1, 0.5, 0.4], [0.1, 0.8, 0.3, 0.2, 0.5])
    bout = _bout(
        [_line(13, ctrl=61), _line(7, td=1, ctrl=17)],
        [_line(11, ctrl=3), _line(9, kd=1, ctrl=29)],
        487,
    )
    a, b = model.deserved_pair(bout)
    assert a + b == pytest.approx(1.0, abs=1e-15)
    assert model.deserved_pair(swap_corners(bout)) == (b, a)


def test_model_round_trips_through_its_dict():
    model = _model([0.1] * 7, [0.2] * 5)
    assert PerformanceModel.from_dict(model.to_dict()) == model


@pytest.fixture(scope="module")
def data():
    return random_league().data()


@pytest.fixture(scope="module")
def config(data):
    return FeatureConfig(train_end=data.bouts[int(len(data.bouts) * 0.6)].date)


@pytest.fixture(scope="module")
def built(data, config):
    return build_features(data, config)


def test_fit_reads_judged_decisions_through_the_cutoff_only(data, config):
    x, y = training_rows(data.bouts, config.train_end)
    decisions = [
        b
        for b in data.bouts
        if b.date <= config.train_end
        and b.result == "win"
        and b.method.startswith("Decision")
        and b.has_stats
    ]
    assert len(y) == 2 * len(decisions) and y.sum() == len(decisions)
    # Both orientations: each column sums to zero.
    np.testing.assert_allclose(np.nansum(x, axis=0), 0.0, atol=1e-9)


def test_fit_learns_that_out_landing_wins_decisions():
    bouts = []
    for i in range(60):
        hi, lo = 20 + i % 7, 5 + i % 3
        winner_first = i % 2 == 0
        a = [_line(hi if winner_first else lo, ctrl=30)] * 3
        b = [_line(lo if winner_first else hi, ctrl=30)] * 3
        bouts.append(
            _bout(a, b, 900, winner="A" if winner_first else "B", bout_id=f"b{i}")
        )
    model = fit_performance_model(bouts, date(2030, 1, 1))
    assert model is not None and model.full.n_bouts == 60
    assert model.full.coef[INPUTS.index("sig_landed")] > 0
    a, b = model.deserved_pair(bouts[0])
    assert a > 0.9 and b < 0.1


def test_too_few_decisions_fit_nothing():
    bouts = [_bout([_line(5)], [_line(1)], 300, bout_id=f"b{i}") for i in range(5)]
    assert fit_performance_model(bouts, date(2030, 1, 1)) is None


# --- family 16 values -----------------------------------------------------------------


_ACTUAL = {"win": 1.0, "loss": 0.0, "draw": 0.5}


def _expected_values(data, store, fighter: str, cutoff: int) -> dict[str, float]:
    """Family 16 recomputed from the bouts and the fitted model, not from
    what the store recorded."""
    model = store.params.performance
    earlier = [
        b
        for b in data.bouts
        if fighter in (b.fighter_a_id, b.fighter_b_id) and b.date.toordinal() < cutoff
    ]
    deserved, luck, weights, strong = [], [], [], []
    for b in earlier:
        d = model.deserved(b, fighter)
        d = math.nan if d is None else d
        actual = _ACTUAL.get(outcome_for(b, fighter), math.nan)
        deserved.append(d)
        luck.append(actual - d)
        weights.append(store.decay_weight(b.date.toordinal()))
        opp = store.elo.rating_before(b.opponent_of(fighter), b.date.toordinal())
        strong.append(max(0.0, opp - store.params.elo.initial))

    def mean(values, w=None):
        pairs = [
            (v, 1.0 if w is None else w[i])
            for i, v in enumerate(values)
            if not math.isnan(v)
        ]
        total = sum(p for _, p in pairs)
        return sum(v * p for v, p in pairs) / total if pairs and total > 0 else math.nan

    return {
        "deserved_career": mean(deserved),
        "deserved_l3": mean(deserved[-3:]),
        "deserved_decay": mean(deserved, weights),
        "luck_career": mean(luck),
        "luck_l3": mean(luck[-3:]),
        "deserved_vs_strong": mean(deserved, strong),
    }


def test_family_values_match_a_recomputation(data, built):
    store = built.store
    checked = 0
    for bout in data.bouts[::5]:
        cutoff = bout.date.toordinal()
        for fighter in (bout.fighter_a_id, bout.fighter_b_id):
            v = fighter_vector(store, fighter, cutoff, bout.weight_class)
            for name, value in _expected_values(data, store, fighter, cutoff).items():
                if math.isnan(value):
                    assert math.isnan(v[POS[name]]), name
                else:
                    assert v[POS[name]] == pytest.approx(value, rel=1e-12, abs=1e-12)
            checked += 1
    assert checked > 50


def test_performance_elo_matches_a_replay_with_deserved_scores(data, built):
    params = built.params
    tracker = EloTracker(params.performance_elo)
    by_day: dict[date, list] = {}
    for b in data.bouts:
        by_day.setdefault(b.date, []).append(b)
    for day in sorted(by_day):
        rated = [rated_performance(b, params.performance) for b in by_day[day]]
        tracker.commit(day.toordinal(), [r for r in rated if r is not None])
    for fighter in {b.fighter_a_id for b in data.bouts}:
        assert tracker.rating(fighter) == built.store.performance_elo.rating(fighter)
    assert params.performance_elo.finish_mult == 1.0


def test_debutant_has_nan_shares_and_the_initial_performance_rating(built):
    v = fighter_vector(built.store, "nobody", date(2030, 1, 1).toordinal(), None)
    for name in ("deserved_career", "deserved_l3", "luck_career", "deserved_vs_strong"):
        assert math.isnan(v[POS[name]])
    assert v[POS["perf_elo"]] == built.params.performance_elo.initial


def test_family_16_is_a_candidate_outside_the_model_features(built):
    assert len(PERFORMANCE) == 7
    assert not set(PERFORMANCE) & {f.name for f in FEATURES}
    assert built.frame[PERFORMANCE].notna().any().all()


def test_parameters_saved_before_m46_give_nan_and_leave_everything_else(data, built):
    raw = built.params.to_dict()
    assert "performance" in raw and "performance_elo" in raw
    restored = FittedParams.from_dict(raw)
    assert restored.performance == built.params.performance
    assert restored.performance_elo == built.params.performance_elo
    old = {k: v for k, v in raw.items() if k not in ("performance", "performance_elo")}
    before = build_features(data, built.store.config, FittedParams.from_dict(old))
    assert before.frame[PERFORMANCE].isna().all().all()
    others = [c for c in built.frame.columns if c not in PERFORMANCE]
    pd.testing.assert_frame_equal(
        before.frame[others], built.frame[others], check_exact=True
    )


def test_standard_elo_is_unchanged_by_the_per_side_score():
    """``score_b`` defaults to ``1 - score_a``, the D26 update."""
    from ufc.features.elo import RatedBout

    one, two = EloTracker(EloParams()), EloTracker(EloParams())
    one.commit(1, [RatedBout("a", "b", 1.0, True)])
    two.commit(1, [RatedBout("a", "b", 1.0, True, 0.0)])
    assert one.rating("a") == two.rating("a") and one.rating("b") == two.rating("b")
