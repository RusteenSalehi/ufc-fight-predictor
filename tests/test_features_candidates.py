"""M4.5 candidate features (DECISIONS D55): strength of schedule, age shape,
catch-weight resolution. Leakage and symmetry for them are covered by the
parametrised fixtures in test_features_leakage.py and
test_features_symmetry.py; this file checks their values."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.feature_helpers import random_league
from ufc.features.age import ALL, AgeMedians, fit_age_medians
from ufc.features.build import build_features
from ufc.features.components import outcome_for
from ufc.features.config import CATCH_WEIGHT, FeatureConfig, resolve_catch_weights
from ufc.features.data import BoutFacts, FighterAttrs
from ufc.features.fighter import POS, fighter_vector
from ufc.features.matchup import FEATURE_NAMES, FEATURES, LEVEL_NAMES

CW = CATCH_WEIGHT


@pytest.fixture(scope="module")
def data():
    return random_league().data()


@pytest.fixture(scope="module")
def built(data):
    return build_features(data, FeatureConfig(shrink_minutes=12.0))


# --- catch weight ----------------------------------------------------------------


def test_catch_weight_takes_the_nearest_earlier_known_division():
    resolved, current = resolve_catch_weights(
        ["Lightweight", CW, "Welterweight", CW], CW
    )
    assert resolved == ["Lightweight", "Lightweight", "Welterweight", "Welterweight"]
    assert current == "Welterweight"


def test_catch_weight_with_no_earlier_division_looks_forward_within_the_history():
    resolved, current = resolve_catch_weights([CW, None, CW, "Featherweight"], None)
    assert resolved == ["Featherweight", None, "Featherweight", "Featherweight"]
    assert current is None


def test_catch_weight_with_no_known_neighbour_stays_a_catch_weight():
    assert resolve_catch_weights([CW, None], CW) == ([CW, None], CW)
    assert resolve_catch_weights([], CW) == ([], CW)
    # A bout of a known class is never changed.
    assert resolve_catch_weights(["Bantamweight"], "Flyweight") == (
        ["Bantamweight"],
        "Flyweight",
    )


def test_catch_weight_setting_changes_only_weight_change_features(data):
    config = FeatureConfig(shrink_minutes=12.0)
    off = build_features(data, config).frame
    on = build_features(
        data, FeatureConfig(shrink_minutes=12.0, catch_weight_adjacent=True)
    ).frame
    weight = ["d_moved_up", "d_moved_down", "d_fights_at_weight"]
    others = [c for c in off.columns if c not in weight]
    pd.testing.assert_frame_equal(on[others], off[others], check_exact=True)
    # The synthetic league has catch-weight bouts, so the setting does bite.
    assert not on[weight].equals(off[weight])


# --- strength of schedule --------------------------------------------------------


def test_schedule_values_match_a_recomputation_from_elo_history(data, built):
    """Best win, worst loss and the counts, recomputed from the Elo tracker's
    own history of ratings rather than from what the store recorded."""
    store = built.store
    bouts_of: dict[str, list[BoutFacts]] = {}
    for bout in data.bouts:
        for fighter in (bout.fighter_a_id, bout.fighter_b_id):
            bouts_of.setdefault(fighter, []).append(bout)
    checked = 0
    for bout in data.bouts:
        cutoff = bout.date.toordinal()
        for fighter in (bout.fighter_a_id, bout.fighter_b_id):
            earlier = [b for b in bouts_of[fighter] if b.date.toordinal() < cutoff]
            wins, losses, wins_up, losses_up = [], [], 0, 0
            for b in earlier:
                day = b.date.toordinal()
                own = store.elo.rating_before(fighter, day)
                opp = store.elo.rating_before(b.opponent_of(fighter), day)
                outcome = outcome_for(b, fighter)
                if outcome == "win":
                    wins.append(opp)
                    wins_up += opp > own
                elif outcome == "loss":
                    losses.append(opp)
                    losses_up += opp > own
            v = fighter_vector(store, fighter, cutoff, bout.weight_class)
            expected_best = max(wins) if wins else math.nan
            expected_worst = min(losses) if losses else math.nan
            np.testing.assert_equal(v[POS["best_win_elo"]], expected_best)
            np.testing.assert_equal(v[POS["worst_loss_elo"]], expected_worst)
            assert v[POS["wins_vs_higher"]] == wins_up
            assert v[POS["losses_vs_higher"]] == losses_up
            checked += 1
    assert checked == 2 * len(data.bouts)


def test_debutant_has_no_best_win_and_zero_counts(built):
    v = fighter_vector(built.store, "not-a-fighter", date(2030, 1, 1).toordinal(), None)
    assert math.isnan(v[POS["best_win_elo"]]) and math.isnan(v[POS["worst_loss_elo"]])
    assert v[POS["wins_vs_higher"]] == 0 and v[POS["losses_vs_higher"]] == 0


# --- age shape -------------------------------------------------------------------


def _bout(i: int, day: date, a: str, b: str, weight_class: str | None) -> BoutFacts:
    return BoutFacts(
        bout_id=f"b{i}",
        event_id=f"e{i}",
        date=day,
        fighter_a_id=a,
        fighter_b_id=b,
        weight_class=weight_class,
        is_womens=False,
        is_title_bout=False,
        scheduled_rounds=3,
        time_format=None,
        result="win",
        winner_id=a,
        method=None,
        end_round=3,
        end_time_s=300,
        fight_time_s=900,
        has_stats=False,
    )


def test_age_medians_per_class_with_a_fallback_and_training_bouts_only():
    fighters = {
        "x": FighterAttrs("x", dob=date(1990, 1, 1)),
        "y": FighterAttrs("y", dob=date(1980, 1, 1)),
        "z": FighterAttrs("z"),  # no date of birth: not counted
    }
    day = date(2020, 1, 1)
    bouts = [_bout(i, day, "x", "y", "Heavyweight") for i in range(3)]
    bouts += [_bout(10, day, "x", "z", CW)]
    bouts += [_bout(20, date(2024, 1, 1), "y", "z", "Heavyweight")]  # after training
    medians = fit_age_medians(bouts, fighters, date(2022, 12, 31), min_bouts=3)
    assert medians is not None
    # Heavyweight: x at 30 three times, y at 40 three times.
    assert medians.for_class("Heavyweight") == pytest.approx(35.0, abs=0.01)
    # Catch weight has no fixed limit, so it falls back to every training bout.
    assert CW not in medians.by_class
    assert medians.for_class(CW) == medians.by_class[ALL]
    assert medians.for_class(None) == medians.by_class[ALL]
    assert AgeMedians.from_dict(medians.to_dict()) == medians


def test_age_shape_values(built):
    frame = built.frame
    store = built.store
    row = frame.dropna(subset=["d_age"]).iloc[50]
    on = row["date"].date().toordinal()
    wc = None if pd.isna(row["weight_class"]) else row["weight_class"]
    a = fighter_vector(store, row["fighter_a_id"], on, wc)
    b = fighter_vector(store, row["fighter_b_id"], on, wc)
    median = store.age_median(wc)
    for v in (a, b):
        assert v[POS["age_past_32"]] == max(0.0, v[POS["age"]] - 32.0)
        assert v[POS["age_rel_class"]] == v[POS["age"]] - median
        assert v[POS["age_past_class_median"]] == max(0.0, v[POS["age_rel_class"]])
    assert (
        row["lvl_age_rel_class"]
        == (a[POS["age_rel_class"]] + b[POS["age_rel_class"]]) / 2
    )
    assert row["d_age_past_32"] == a[POS["age_past_32"]] - b[POS["age_past_32"]]


def test_candidates_leave_the_m4_features_unchanged(data, built):
    """The M4.5 columns are added beside the M4 model features; the M4
    features and level columns are what they were before the candidates
    existed (checked bit for bit on the real data during M4.5, and here by
    building with parameters that lack age medians). The age_shape family
    was adopted later (D58) and is the only model family that reads them."""
    from dataclasses import replace

    no_medians = build_features(
        data, built.store.config, replace(built.params, age_medians=None)
    )
    m4 = [f.name for f in FEATURES if f.family != "age_shape"]
    assert len(m4) == len(FEATURE_NAMES) - 2
    cols = [*m4, *LEVEL_NAMES]
    pd.testing.assert_frame_equal(
        no_medians.frame[cols], built.frame[cols], check_exact=True
    )
    assert no_medians.frame["d_age_past_class_median"].isna().all()
