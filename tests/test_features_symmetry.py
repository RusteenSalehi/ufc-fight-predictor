"""Swapping the two fighters negates every feature exactly.

The source lists the winner first far more often than not, so any trace of
the listing order in the features would leak the outcome.
"""

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from tests.feature_helpers import (
    NO_PRIOR,
    League,
    line,
    plain_params,
    random_league,
    swap_corners,
)
from ufc.features.build import CONTEXT_COLUMNS, build_features, mirror
from ufc.features.config import FeatureConfig
from ufc.features.matchup import (
    CANDIDATE_FEATURE_NAMES,
    CANDIDATE_LEVEL_NAMES,
    CANDIDATE_NAMES,
    FEATURE_NAMES,
    FEATURES,
    FLAG_NAMES,
    LEVEL_NAMES,
    build_matchup,
)

D0 = date(2020, 1, 1)


@pytest.fixture(scope="module")
def data():
    return random_league(seed=11).data()


@pytest.fixture(
    scope="module",
    params=[(0.0, False), (12.0, False), (12.0, True)],
    ids=["plain", "shrunk", "catch_adjacent"],
)
def built(data, request):
    shrink, catch = request.param
    return build_features(
        data, FeatureConfig(shrink_minutes=shrink, catch_weight_adjacent=catch)
    )


def test_swapping_fighters_negates_every_difference_feature_exactly(built):
    for row in built.frame.itertuples(index=False):
        weight_class = None if pd.isna(row.weight_class) else row.weight_class
        on = row.date.date()
        ab = build_matchup(
            built.store, row.fighter_a_id, row.fighter_b_id, on, weight_class
        )
        ba = build_matchup(
            built.store, row.fighter_b_id, row.fighter_a_id, on, weight_class
        )
        forward = np.array([ab.features[n] for n in FEATURE_NAMES])
        backward = np.array([ba.features[n] for n in FEATURE_NAMES])
        # Exact equality; NaN matches NaN.
        np.testing.assert_array_equal(forward, -backward)
        # Level columns are means of the two fighters: exactly unchanged.
        np.testing.assert_array_equal(
            np.array([ab.levels[n] for n in LEVEL_NAMES]),
            np.array([ba.levels[n] for n in LEVEL_NAMES]),
        )
        # M4.5 candidates: differences negate, the level column does not move.
        np.testing.assert_array_equal(
            np.array([ab.candidates[n] for n in CANDIDATE_FEATURE_NAMES]),
            -np.array([ba.candidates[n] for n in CANDIDATE_FEATURE_NAMES]),
        )
        np.testing.assert_array_equal(
            np.array([ab.candidates[n] for n in CANDIDATE_LEVEL_NAMES]),
            np.array([ba.candidates[n] for n in CANDIDATE_LEVEL_NAMES]),
        )
        # Each fighter's own values do not depend on which side he is on.
        assert _same_values(ab.a, ba.b) and _same_values(ab.b, ba.a)
        assert (ab.flags["a_debut"], ab.flags["a_no_stats"]) == (
            ba.flags["b_debut"],
            ba.flags["b_no_stats"],
        )
        assert (ab.flags["b_debut"], ab.flags["b_no_stats"]) == (
            ba.flags["a_debut"],
            ba.flags["a_no_stats"],
        )
        assert (ab.flags["a_attr_gap"], ab.flags["b_attr_gap"]) == (
            ba.flags["b_attr_gap"],
            ba.flags["a_attr_gap"],
        )


def _same_values(left: dict, right: dict) -> bool:
    return np.array_equal(
        np.array(list(left.values())), np.array(list(right.values())), equal_nan=True
    )


def test_listing_order_in_the_source_does_not_matter(data, built):
    """Rebuild with every bout's fighters listed the other way round: the
    result is the mirror image, and nothing else changes."""
    swapped = build_features(
        replace(data, bouts=[swap_corners(b) for b in data.bouts]),
        built.store.config,
        built.params,
    )
    pd.testing.assert_frame_equal(swapped.frame, mirror(built.frame), check_exact=True)


def test_mirror_swaps_flags_and_labels_and_is_its_own_inverse(built):
    frame = built.frame
    mirrored = mirror(frame)
    assert mirrored["a_debut"].equals(frame["b_debut"])
    assert mirrored["b_no_stats"].equals(frame["a_no_stats"])
    assert mirrored["a_attr_gap"].equals(frame["b_attr_gap"])
    assert mirrored["b_attr_gap"].equals(frame["a_attr_gap"])
    labelled = frame["label"].notna()
    assert (mirrored.loc[labelled, "label"] == 1 - frame.loc[labelled, "label"]).all()
    assert mirrored.loc[~labelled, "label"].isna().all()
    pd.testing.assert_frame_equal(
        mirrored[list(LEVEL_NAMES)], frame[list(LEVEL_NAMES)], check_exact=True
    )
    assert mirrored[list(FEATURE_NAMES)].equals(-frame[list(FEATURE_NAMES)])
    assert mirrored[list(CANDIDATE_FEATURE_NAMES)].equals(
        -frame[list(CANDIDATE_FEATURE_NAMES)]
    )
    pd.testing.assert_frame_equal(
        mirrored[list(CANDIDATE_LEVEL_NAMES)],
        frame[list(CANDIDATE_LEVEL_NAMES)],
        check_exact=True,
    )
    pd.testing.assert_frame_equal(mirror(mirrored), frame, check_exact=True)


def test_every_model_feature_is_a_difference_and_none_is_an_outcome():
    assert all(name.startswith("d_") for name in FEATURE_NAMES)
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    assert not {"label", "result"} & (set(FEATURE_NAMES) | set(FLAG_NAMES))
    assert {"label", "result"} <= set(CONTEXT_COLUMNS)
    assert not any("corner" in f.name or f.family == "aux" for f in FEATURES)
    # Swap-invariant columns are kept apart from the antisymmetric features.
    assert all(name.startswith("lvl_") for name in LEVEL_NAMES)
    assert not set(LEVEL_NAMES) & set(FEATURE_NAMES)
    # M4.5 candidates are neither model features nor level features, so no
    # model selected before M4.5 and no run of ``train`` reads them.
    assert all(name.startswith("d_") for name in CANDIDATE_FEATURE_NAMES)
    assert all(name.startswith("lvl_") for name in CANDIDATE_LEVEL_NAMES)
    assert not set(CANDIDATE_NAMES) & (set(FEATURE_NAMES) | set(LEVEL_NAMES))


def test_attribute_gap_flags_the_fighter_whose_profile_lacks_what_the_other_has():
    league = League()
    full = league.fighter()
    no_reach = league.fighter(reach_cm=None)
    no_dob = league.fighter(dob=None)
    bare = league.fighter(height_cm=None, reach_cm=None, stance=None, dob=None)
    also_bare = league.fighter(height_cm=None, reach_cm=None, stance=None, dob=None)
    league.bout(league.event(D0), full, no_reach)
    store = build_features(league.data(), NO_PRIOR, plain_params()).store
    on = D0 + timedelta(days=30)

    def gaps(a, b):
        flags = build_matchup(store, a, b, on).flags
        return flags["a_attr_gap"], flags["b_attr_gap"]

    assert gaps(full, no_reach) == (False, True)
    assert gaps(no_reach, full) == (True, False)
    assert gaps(full, bare) == (False, True)
    # Each lacks something the other has.
    assert gaps(no_reach, no_dob) == (True, True)
    # Both lack everything: nothing tells them apart, so no gap.
    assert gaps(bare, also_bare) == (False, False)
    # A fighter with no profile at all is treated like an empty profile.
    assert gaps(full, "0" * 16) == (False, True)
    assert gaps(bare, "0" * 16) == (False, False)


def test_level_columns_are_the_mean_of_the_two_fighters():
    league = League()
    old = league.fighter(dob=date(1980, 1, 1))
    young = league.fighter(dob=date(1995, 1, 1))
    ageless = league.fighter(dob=None)
    league.bout(league.event(D0), old, young)
    league.bout(league.event(D0 + timedelta(days=30)), old, ageless)
    store = build_features(league.data(), NO_PRIOR, plain_params()).store
    on = D0 + timedelta(days=60)

    matchup = build_matchup(store, old, young, on)
    assert matchup.levels["lvl_age"] == (matchup.a["age"] + matchup.b["age"]) / 2
    assert matchup.levels["lvl_n_fights"] == (2 + 1) / 2
    assert matchup.levels["lvl_elo"] == (matchup.a["elo"] + matchup.b["elo"]) / 2
    assert matchup.levels == build_matchup(store, young, old, on).levels
    # A missing value on either side leaves the mean missing, not halved.
    assert np.isnan(build_matchup(store, old, ageless, on).levels["lvl_age"])


def test_style_matchups_are_offense_against_defense():
    league = League()
    a, b, x, y = (league.fighter() for _ in range(4))

    def bout(offset, first, second, first_line, second_line):
        league.bout(
            league.event(D0 + timedelta(days=offset)),
            first,
            second,
            method="KO/TKO",
            rounds_a=[first_line],
            rounds_b=[second_line],
            end_time_s=300,
        )

    # In 5 minutes: a lands 3 takedowns and stops 1 of 4; b lands 1 and stops 2 of 2.
    bout(
        0,
        a,
        x,
        line(sig=(30, 60), td=(3, 5), kd=1, ctrl=120),
        line(sig=(10, 40), td=(3, 4), ctrl=30),
    )
    bout(
        0,
        b,
        y,
        line(sig=(20, 80), td=(1, 2), ctrl=60),
        line(sig=(15, 30), td=(0, 2), kd=1, ctrl=90),
    )
    build = build_features(league.data(), NO_PRIOR, plain_params())
    features = build_matchup(build.store, a, b, D0 + timedelta(days=30)).features

    a_td, a_td_def, b_td, b_td_def = 3 * 3, 1 - 3 / 4, 1 * 3, 1.0
    assert features["d_matchup_td"] == pytest.approx(
        a_td * (1 - b_td_def) - b_td * (1 - a_td_def)
    )
    a_att, a_def, b_att, b_def = 60 / 5, 1 - 10 / 40, 80 / 5, 1 - 15 / 30
    assert features["d_matchup_strike"] == pytest.approx(
        a_att * (1 - b_def) - b_att * (1 - a_def)
    )
    # Knockdowns per 15 minutes scored, against knockdowns per 15 absorbed.
    assert features["d_matchup_kd"] == pytest.approx(3 * 3 - 0 * 0)
    assert features["d_matchup_ctrl"] == pytest.approx(
        (120 / 300) * (90 / 300) - (60 / 300) * (30 / 300)
    )
    assert features["d_matchup_sub"] == 0.0


def test_stance_matchup_is_signed_by_who_is_the_southpaw():
    league = League()
    southpaw = league.fighter(stance="Southpaw")
    orthodox = league.fighter(stance="Orthodox")
    switch = league.fighter(stance="Switch")
    unknown = league.fighter(stance=None)
    league.bout(league.event(D0), southpaw, orthodox)
    store = build_features(league.data(), NO_PRIOR, plain_params()).store
    on = D0 + timedelta(days=30)

    def stance(a, b):
        features = build_matchup(store, a, b, on).features
        return (
            features["d_stance_southpaw_vs_orthodox"],
            features["d_stance_southpaw"],
            features["d_stance_switch"],
        )

    assert stance(southpaw, orthodox) == (1.0, 1.0, 0.0)
    assert stance(orthodox, southpaw) == (-1.0, -1.0, 0.0)
    assert stance(southpaw, switch) == (0.0, 1.0, -1.0)
    # A fighter with no profile has an unknown stance.
    assert all(np.isnan(v) for v in stance(orthodox, "0" * 16))
    assert all(np.isnan(v) for v in stance(southpaw, unknown))
