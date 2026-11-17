"""Shrinkage of per-minute rates toward the weight-class average, hand-checked."""

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from tests.feature_helpers import League, line, plain_params, random_league
from ufc.features.build import build_features, fit_params
from ufc.features.components import IDX, N_COMPONENTS
from ufc.features.config import FeatureConfig
from ufc.features.fighter import WINDOWED
from ufc.features.matchup import FEATURE_NAMES, build_matchup
from ufc.features.shrink import RATE_COMPONENTS, RatePriors, fit_rate_priors
from ufc.features.store import FeatureStore, FittedParams

D0 = date(2020, 1, 1)
NOBODY = "0" * 16


def day(offset: int) -> date:
    return D0 + timedelta(days=offset)


def one_round(
    league, on, a, b, a_line, b_line, seconds=300, weight_class="Lightweight"
):
    return league.bout(
        league.event(on),
        a,
        b,
        method="KO/TKO",
        rounds_a=[a_line],
        rounds_b=[b_line],
        end_time_s=seconds,
        weight_class=weight_class,
    )


def priors(**per_minute: float) -> RatePriors:
    """Rate priors with the same averages for every weight class."""
    vector = np.zeros(N_COMPONENTS)
    for name, value in per_minute.items():
        vector[IDX[name]] = value
    vector[IDX["t"]] = 60.0
    return RatePriors(overall=vector)


def values_for(league, fighter_id, on, minutes, rate_priors, weight_class=None):
    config = FeatureConfig(prior_fights=0.0, shrink_minutes=minutes)
    params = replace(plain_params(), rate_priors=rate_priors)
    store = build_features(league.data(), config, params).store
    return build_matchup(store, fighter_id, NOBODY, on, weight_class).a


def test_a_short_history_is_pulled_toward_the_average():
    league = League()
    f, o = league.fighter(), league.fighter()
    # 5 minutes: f lands 30 strikes and scores 1 knockdown, absorbs 10.
    one_round(league, day(0), f, o, line(sig=(30, 60), kd=1), line(sig=(10, 40)))
    average = priors(sig_l=4.0, sig_a=9.0, opp_sig_l=4.0, kd=0.02)

    plain = values_for(league, f, day(30), 0.0, average)
    assert plain["slpm_career"] == pytest.approx(30 / 5)
    assert plain["kd_per15_career"] == pytest.approx(3.0)

    shrunk = values_for(league, f, day(30), 15.0, average)
    # (30 landed + 15 min * 4.0) / (5 min + 15 min)
    assert shrunk["slpm_career"] == pytest.approx((30 + 60) / 20)
    assert shrunk["sig_att_pm_career"] == pytest.approx((60 + 15 * 9) / 20)
    assert shrunk["sapm_career"] == pytest.approx((10 + 60) / 20)
    # (1 knockdown + 15 min * 0.02) / 20 min, per 15 minutes
    assert shrunk["kd_per15_career"] == pytest.approx((1 + 0.3) / 20 * 15)
    # The last-3 and last-5 windows hold the same single bout.
    assert shrunk["slpm_l3"] == shrunk["slpm_l5"] == shrunk["slpm_career"]
    # Ratios that are not per minute are left alone.
    for name in ("sig_acc_career", "sig_def_career", "win_rate_career", "n_fights"):
        assert shrunk[name] == plain[name]


def test_more_cage_time_means_less_shrinkage():
    league = League()
    rookie, veteran, o = league.fighter(), league.fighter(), league.fighter()
    one_round(league, day(0), rookie, o, line(sig=(30, 60)), line())
    for i in range(1, 9):
        one_round(league, day(10 * i), veteran, o, line(sig=(30, 60)), line())
    average = priors(sig_l=2.0)
    on = day(200)
    # Both land 6 a minute; the prior is 2 a minute, worth 15 minutes.
    assert values_for(league, rookie, on, 15.0, average)["slpm_career"] == (
        pytest.approx((30 + 30) / (5 + 15))
    )
    assert values_for(league, veteran, on, 15.0, average)["slpm_career"] == (
        pytest.approx((240 + 30) / (40 + 15))
    )


def test_decayed_rate_treats_the_prior_as_minutes_fought_at_the_cutoff():
    league = League()
    f, o = league.fighter(), league.fighter()
    one_round(league, day(0), f, o, line(sig=(30, 60)), line())
    average = priors(sig_l=4.0)
    # One half-life (730 days) after the bout, its 5 minutes count as 2.5.
    values = values_for(league, f, day(730), 15.0, average)
    assert values["slpm_decay"] == pytest.approx((15 + 60) / (2.5 + 15))
    assert values["slpm_career"] == pytest.approx((30 + 60) / (5 + 15))
    # On the day after the bout almost nothing has decayed.
    fresh = values_for(league, f, day(1), 15.0, average)
    assert fresh["slpm_decay"] == pytest.approx(fresh["slpm_career"], rel=1e-3)


def test_no_recorded_stat_time_stays_missing():
    league = League()
    f, o, debutant = league.fighter(), league.fighter(), league.fighter()
    league.bout(league.event(day(0)), f, o)  # a bout without stats
    average = priors(sig_l=4.0, kd=0.02)
    for fighter in (f, debutant):
        values = values_for(league, fighter, day(30), 15.0, average)
        for window in ("career", "l3", "l5", "decay"):
            assert np.isnan(values[f"slpm_{window}"])
            assert np.isnan(values[f"kd_per15_{window}"])
        assert np.isnan(values["adj_slpm_career"])
    assert values_for(league, f, day(30), 15.0, average)["no_stats"] == 1.0


def test_adjusted_per_minute_stats_shrink_toward_zero():
    league = League()
    f, o, x = league.fighter(), league.fighter(), league.fighter()
    # o absorbs 30 strikes in 5 minutes from x: 6 a minute against others.
    one_round(league, day(0), x, o, line(sig=(30, 60)), line(sig=(10, 40)))
    # f lands 50 on o in 5 minutes: 20 more than expected.
    one_round(league, day(30), f, o, line(sig=(50, 80)), line(sig=(5, 20)))
    average = priors(sig_l=4.0)

    plain = values_for(league, f, day(60), 0.0, average)
    assert plain["adj_slpm_career"] == pytest.approx((50 - 30) / 5)
    shrunk = values_for(league, f, day(60), 15.0, average)
    assert shrunk["adj_slpm_career"] == pytest.approx((50 - 30) / (5 + 15))
    # Accuracy is not per minute: (50 - 0.5 * 80) / 80 either way.
    assert shrunk["adj_sig_acc_career"] == plain["adj_sig_acc_career"]
    assert shrunk["adj_sig_acc_career"] == pytest.approx(10 / 80)
    # Decayed: the 5 minutes have decayed a little, the 15 prior minutes have not.
    weight = 2.0 ** (-30 / 730)
    assert shrunk["adj_slpm_decay"] == pytest.approx(
        weight * (50 - 30) / (weight * 5 + 15)
    )


def test_class_averages_are_pooled_per_class_and_fitted_on_training_only():
    league = League()
    a, b, c, d = (league.fighter() for _ in range(4))
    # Lightweight: 10 minutes for two fighters, 40 strikes landed in total.
    one_round(league, day(0), a, b, line(sig=(30, 60), kd=1), line(sig=(10, 40)))
    # Heavyweight: 4 minutes for two fighters, 4 strikes landed in total.
    one_round(
        league, day(1), c, d, line(sig=(3, 9)), line(sig=(1, 2)), 120, "Heavyweight"
    )
    # Catch weight has no fixed limit, so it never gets an average of its own.
    one_round(
        league, day(2), a, c, line(sig=(8, 9)), line(sig=(8, 9)), 60, "Catch Weight"
    )
    league.bout(league.event(day(3)), a, b)  # no stats: not counted
    one_round(league, day(10), a, b, line(sig=(99, 99)), line(sig=(99, 99)))
    bouts = league.data().bouts

    fitted = fit_rate_priors(bouts, train_end=day(9), min_bouts=1)
    assert set(fitted.by_class) == {"Lightweight", "Heavyweight"}
    assert fitted.n_bouts == {"Lightweight": 1, "Heavyweight": 1}
    lightweight = fitted.for_class("Lightweight")
    assert lightweight[IDX["sig_l"]] == pytest.approx(40 / 10)
    assert lightweight[IDX["opp_sig_l"]] == pytest.approx(40 / 10)
    assert lightweight[IDX["kd"]] == pytest.approx(1 / 10)
    assert lightweight[IDX["t"]] == 60.0
    assert fitted.for_class("Heavyweight")[IDX["sig_l"]] == pytest.approx(4 / 4)
    # Everything on or before the cutoff with stats: 16 minutes, 60 landed.
    overall = (40 + 4 + 16) / (10 + 4 + 2)
    assert fitted.overall[IDX["sig_l"]] == pytest.approx(overall)
    for unknown in ("Catch Weight", "Open Weight", "Flyweight", None):
        np.testing.assert_array_equal(fitted.for_class(unknown), fitted.overall)
    # Only the per-minute counts and the minute itself are filled in.
    filled = {IDX[name] for name in RATE_COMPONENTS} | {IDX["t"]}
    assert set(np.flatnonzero(lightweight)) <= filled

    # A class below the minimum number of bouts uses the overall average.
    strict = fit_rate_priors(bouts, train_end=day(9), min_bouts=2)
    assert not strict.by_class
    np.testing.assert_array_equal(strict.for_class("Lightweight"), strict.overall)

    # Bouts after the training period do not move the averages.
    later_changed = [
        replace(b, totals={}, has_stats=False) if b.date > day(9) else b for b in bouts
    ]
    assert fit_rate_priors(later_changed, day(9), 1).to_dict() == fitted.to_dict()
    # With nothing to fit on, there is no prior and shrinkage does nothing.
    empty = fit_rate_priors(bouts, train_end=day(-1))
    assert not empty.overall.any() and not empty.by_class
    assert RatePriors.from_dict(empty.to_dict()).to_dict() == empty.to_dict()


def test_shrinkage_uses_the_class_of_the_bout_being_predicted():
    league = League()
    f, o = league.fighter(), league.fighter()
    one_round(league, day(0), f, o, line(sig=(30, 60)), line())
    slow, fast = np.zeros(N_COMPONENTS), np.zeros(N_COMPONENTS)
    slow[[IDX["sig_l"], IDX["t"]]] = 2.0, 60.0
    fast[[IDX["sig_l"], IDX["t"]]] = 6.0, 60.0
    by_class = RatePriors(overall=slow, by_class={"Flyweight": fast})

    heavy = values_for(league, f, day(30), 15.0, by_class, "Heavyweight")
    fly = values_for(league, f, day(30), 15.0, by_class, "Flyweight")
    assert heavy["slpm_career"] == pytest.approx((30 + 15 * 2) / 20)
    assert fly["slpm_career"] == pytest.approx((30 + 15 * 6) / 20)


def test_zero_minutes_is_exactly_the_unshrunk_build_and_only_rates_change():
    data = random_league(seed=5).data()
    config = FeatureConfig(train_end=data.bouts[150].date)
    params, _ = fit_params(list(data.bouts), config)
    assert params.rate_priors is not None and params.rate_priors.overall.any()

    plain = build_features(data, config, params).frame
    # The priors are ignored at zero minutes, whatever they hold.
    without = build_features(data, config, replace(params, rate_priors=None)).frame
    pd.testing.assert_frame_equal(plain, without, check_exact=True)

    shrunk = build_features(data, replace(config, shrink_minutes=15.0), params).frame
    per_minute = {r.name for r in WINDOWED if r.den == "t"}
    adjusted = {f"adj_{rate}" for rate in per_minute}
    # Style matchups built on a per-minute rate; control time is a share.
    products = {"d_matchup_td", "d_matchup_strike", "d_matchup_kd", "d_matchup_sub"}
    affected = [
        name
        for name in FEATURE_NAMES
        if name.removeprefix("d_").rsplit("_", 1)[0] in per_minute | adjusted
        or name in products
    ]
    assert len(affected) == 4 * 8 + 2 * 5 + 4
    untouched = [name for name in plain.columns if name not in affected]
    pd.testing.assert_frame_equal(plain[untouched], shrunk[untouched], check_exact=True)
    # Shrinkage never creates or removes a missing value.
    pd.testing.assert_frame_equal(plain[affected].isna(), shrunk[affected].isna())
    changed = (plain[affected] != shrunk[affected]) & plain[affected].notna()
    assert changed.any().all(), "every per-minute feature should move for someone"
    # And the extremes come in.
    assert (
        shrunk["d_kd_per15_career"].abs().max() < plain["d_kd_per15_career"].abs().max()
    )


def test_store_refuses_shrinkage_without_fitted_averages():
    with pytest.raises(ValueError, match="rate priors"):
        FeatureStore({}, FeatureConfig(shrink_minutes=5.0), plain_params())
    with pytest.raises(ValueError, match="negative"):
        FeatureConfig(shrink_minutes=-1.0)
    assert FittedParams.from_dict(plain_params().to_dict()).rate_priors is None
