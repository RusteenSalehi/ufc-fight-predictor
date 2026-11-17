"""Opponent-adjusted stats: hand-checked, including the cutoff on opponents."""

import math
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from tests.feature_helpers import NO_PRIOR, League, line, plain_params
from ufc.features.adjust import fit_league_baseline
from ufc.features.build import build_features
from ufc.features.components import IDX, N_COMPONENTS
from ufc.features.config import FeatureConfig
from ufc.features.matchup import build_matchup

D0 = date(2020, 1, 1)
NOBODY = "0" * 16


def day(offset: int) -> date:
    return D0 + timedelta(days=offset)


def one_round(league, on, a, b, a_line, b_line, seconds=300):
    return league.bout(
        league.event(on),
        a,
        b,
        method="KO/TKO",
        rounds_a=[a_line],
        rounds_b=[b_line],
        end_time_s=seconds,
    )


def values_for(league, fighter_id, on, config=NO_PRIOR, params=None):
    build = build_features(league.data(), config, params or plain_params())
    return build_matchup(build.store, fighter_id, NOBODY, on).a


def test_takedown_defense_example_from_the_requirements():
    """Opponents land 40% of takedowns on others but 20% on him: strong defense."""
    league = League()
    f, o, x = league.fighter(), league.fighter(), league.fighter()
    one_round(league, day(0), o, x, line(td=(4, 10)), line())
    one_round(league, day(30), f, o, line(), line(td=(2, 10)))
    values = values_for(league, f, day(60))

    assert values["td_def_career"] == pytest.approx(0.8)
    # Expected 4 of 10 landed against him, actually 2 of 10: (4 - 2) / 10.
    assert values["adj_td_def_career"] == pytest.approx(0.2)
    assert values["adj_td_def_decay"] == pytest.approx(0.2)


def test_rates_are_adjusted_for_what_each_opponent_allows():
    league = League()
    f, o1, o2, x, y = (league.fighter() for _ in range(5))
    # o1 absorbs 30 strikes in 300 s from x: 6 per minute against others.
    one_round(league, day(0), x, o1, line(sig=(30, 60)), line(sig=(10, 40)))
    # o2 absorbs 5 strikes in 300 s from y: 1 per minute against others.
    one_round(league, day(0), y, o2, line(sig=(5, 10)), line(sig=(20, 50)))
    # f lands 20 in 300 s on o1 and 20 in 100 s on o2.
    one_round(league, day(30), f, o1, line(sig=(20, 50)), line(sig=(5, 20)))
    one_round(league, day(60), f, o2, line(sig=(20, 25)), line(sig=(2, 10)), 100)
    values = values_for(league, f, day(90))

    minutes = 400 / 60
    expected_landed = 6 * (300 / 60) + 1 * (100 / 60)
    assert values["slpm_career"] == pytest.approx(40 / minutes)
    assert values["adj_slpm_career"] == pytest.approx((40 - expected_landed) / minutes)
    # Accuracy: o1 allows 30 of 60, o2 allows 5 of 10; f attempted 50 and 25.
    assert values["adj_sig_acc_career"] == pytest.approx(
        (40 - (0.5 * 50 + 0.5 * 25)) / 75
    )
    # Absorbed: o1 lands 10 per 300 s on others, o2 lands 20 per 300 s.
    expected_absorbed = 10 / 300 * 300 + 20 / 300 * 100
    assert values["adj_sapm_career"] == pytest.approx((7 - expected_absorbed) / minutes)
    # Defense: o1 lands 10 of 40 on others, o2 lands 20 of 50; they threw 20 and 10 at f.
    assert values["adj_sig_def_career"] == pytest.approx(
        (0.25 * 20 + 0.4 * 10 - 7) / 30
    )


def test_opponent_baseline_respects_the_cutoff_and_excludes_their_own_bouts():
    league = League()
    f, o, x = league.fighter(), league.fighter(), league.fighter()
    one_round(league, day(0), f, o, line(), line(td=(2, 10)))
    # After his bout with f, o lands 4 of 10 on x, then 10 of 10 in a rematch with f.
    one_round(league, day(30), o, x, line(td=(4, 10)), line())
    one_round(league, day(60), f, o, line(), line(td=(10, 10)))
    # Later still, o lands 0 of 30 on x.
    one_round(league, day(90), o, x, line(td=(0, 30)), line())

    # Cutoff before o has fought anyone else: no baseline, so no adjustment.
    assert math.isnan(values_for(league, f, day(30))["adj_td_def_career"])
    # Cutoff after o's bout with x: that bout is his baseline, although it
    # came after the bout being adjusted. It is still before the cutoff.
    assert values_for(league, f, day(31))["adj_td_def_career"] == pytest.approx(0.2)
    # After the rematch: o's two bouts with f are never part of his baseline.
    # Expected 0.4 * 20 attempts, actual 12 landed.
    assert values_for(league, f, day(61))["adj_td_def_career"] == pytest.approx(
        (8 - 12) / 20
    )
    # Once o's 0-for-30 is before the cutoff, his rate against others is 4 of 40.
    assert values_for(league, f, day(91))["adj_td_def_career"] == pytest.approx(
        (0.1 * 20 - 12) / 20
    )


def test_baselines_are_shrunk_toward_the_league_average():
    league = League()
    f, o, x = league.fighter(), league.fighter(), league.fighter()
    one_round(league, day(0), o, x, line(td=(4, 10)), line())
    one_round(league, day(30), f, o, line(), line(td=(2, 10)))
    prior = np.zeros(N_COMPONENTS)
    prior[IDX["td_l"]], prior[IDX["td_a"]] = (
        1.0,
        5.0,
    )  # league: 20% of 5 attempts a bout
    values = values_for(
        league, f, day(60), FeatureConfig(prior_fights=2.0), plain_params(league=prior)
    )
    # o's rate: (4 + 2 * 1) of (10 + 2 * 5) = 0.3, so 3 of f's 10 faced were expected.
    assert values["adj_td_def_career"] == pytest.approx((3 - 2) / 10)

    # An opponent with no other bouts falls back to the league rate alone.
    rookie = League()
    f, o = rookie.fighter(), rookie.fighter()
    one_round(rookie, day(0), f, o, line(), line(td=(1, 10)))
    values = values_for(
        rookie, f, day(30), FeatureConfig(prior_fights=2.0), plain_params(league=prior)
    )
    assert values["adj_td_def_career"] == pytest.approx((0.2 * 10 - 1) / 10)


def test_league_baseline_is_fitted_on_the_training_period_only():
    league = League()
    a, b = league.fighter(), league.fighter()
    one_round(
        league, day(0), a, b, line(sig=(10, 20), ctrl=30), line(sig=(2, 4), ctrl=0)
    )
    league.bout(league.event(day(5)), a, b)  # no stats: not part of the average
    one_round(league, day(10), a, b, line(sig=(90, 90)), line(sig=(90, 90)))
    bouts = league.data().bouts

    baseline = fit_league_baseline(bouts, train_end=day(9))
    # Two fighter rows from the one bout with stats on or before the cutoff.
    assert baseline[IDX["sig_l"]] == pytest.approx((10 + 2) / 2)
    assert baseline[IDX["opp_sig_a"]] == pytest.approx((4 + 20) / 2)
    assert baseline[IDX["t"]] == 300.0
    assert baseline[IDX["ctrl"]] == 15.0
    assert baseline[IDX["n_stat"]] == 1.0

    later_changed = [
        replace(b, totals={}, has_stats=False) if b.date > day(9) else b for b in bouts
    ]
    np.testing.assert_array_equal(fit_league_baseline(later_changed, day(9)), baseline)
    assert not fit_league_baseline(bouts, train_end=day(-1)).any()
