"""Per-fighter as-of values, each checked against arithmetic written out here."""

import math
from datetime import date, timedelta

import pytest

from tests.feature_helpers import (
    FIVE_ROUNDS,
    NO_PRIOR,
    League,
    line,
    plain_params,
)
from ufc.features.build import build_features
from ufc.features.fighter import FIGHTER_FEATURES
from ufc.features.matchup import build_matchup

D0 = date(2020, 1, 1)
NOBODY = "0" * 16
EMPTY = line()


def day(offset: int) -> date:
    return D0 + timedelta(days=offset)


def snapshot(league: League, fighter_id: str, on: date, weight_class="Lightweight"):
    """The fighter's as-of values on ``on``, against an opponent with no history."""
    build = build_features(league.data(), NO_PRIOR, plain_params())
    matchup = build_matchup(build.store, fighter_id, NOBODY, on, weight_class)
    return matchup.a, matchup


def one_round(league, on, a, b, a_line=EMPTY, b_line=EMPTY, seconds=60, **kwargs):
    """A bout stopped in round 1 after ``seconds``; fighter A wins by default."""
    kwargs.setdefault("method", "KO/TKO")
    return league.bout(
        league.event(on),
        a,
        b,
        rounds_a=[a_line],
        rounds_b=[b_line],
        end_time_s=seconds,
        **kwargs,
    )


def test_debutant_has_missing_stats_true_zero_record_and_flags():
    league = League()
    a, b, c = league.fighter(), league.fighter(), league.fighter()
    one_round(league, day(0), b, c, line(sig=(5, 10)), line(sig=(1, 5)))
    values, matchup = snapshot(league, a, day(30))

    assert matchup.flags["a_debut"] and matchup.flags["a_no_stats"]
    assert values["debut"] == 1.0
    for name in ("n_fights", "wins", "losses", "ko_losses", "cage_min"):
        assert values[name] == 0.0
    assert values["win_streak"] == values["fights_last_2y"] == 0.0
    assert values["elo"] == 1500.0
    # Stats are missing, never zero: every rate and every stat count is NaN.
    for feature in FIGHTER_FEATURES:
        if feature.family in ("striking", "grappling", "opponent_adjusted"):
            assert math.isnan(values[feature.name]), feature.name
    for name in (
        "win_rate_career",
        "kd_abs",
        "kd_abs_per15_career",
        "head_abs_pm_l3",
        "opp_elo_career",
        "days_since_last",
        "finish_win_share",
        "moved_up",
        "cardio_slpm_ratio_career",
        "cardio_sig_def_delta",
    ):
        assert math.isnan(values[name]), name


def test_bout_without_stats_counts_for_the_record_but_not_for_stats():
    league = League()
    a, b = league.fighter(), league.fighter()
    league.bout(
        league.event(day(0)), a, b, method="Submission", end_round=1, end_time_s=95
    )
    values, matchup = snapshot(league, a, day(30))

    assert not matchup.flags["a_debut"]
    assert matchup.flags["a_no_stats"]
    assert (values["n_fights"], values["wins"], values["sub_wins"]) == (1.0, 1.0, 1.0)
    assert values["win_rate_career"] == 1.0
    assert values["cage_min"] == pytest.approx(95 / 60)
    assert values["days_since_last"] == 30.0
    for name in (
        "slpm_career",
        "sig_acc_l3",
        "td_def_decay",
        "kd_abs",
        "ctrl_share_career",
    ):
        assert math.isnan(values[name]), name


def test_rates_are_pooled_over_bouts():
    league = League()
    a, b, c = league.fighter(), league.fighter(), league.fighter()
    # Bout 1: three full rounds, 900 seconds.
    league.bout(
        league.event(day(0)),
        a,
        b,
        rounds_a=[
            line(sig=(10, 20), td=(1, 2), ctrl=60, body=2, leg=3, clinch=1, ground=2),
            line(sig=(5, 10), td=(0, 1), ctrl=30),
            line(sig=(5, 10), sub=1, ctrl=0),
        ],
        rounds_b=[
            line(sig=(4, 10), td=(0, 1), ctrl=10),
            line(sig=(6, 15), td=(1, 3), ctrl=20, kd=1),
            line(sig=(2, 5), ctrl=0),
        ],
    )
    # Bout 2: stopped after 100 seconds.
    one_round(
        league, day(60), a, c, line(sig=(6, 8), kd=1), line(sig=(1, 5), td=(0, 1)), 100
    )
    values, _ = snapshot(league, a, day(90))

    minutes = (900 + 100) / 60
    assert values["slpm_career"] == pytest.approx((20 + 6) / minutes)
    assert values["sapm_career"] == pytest.approx((12 + 1) / minutes)
    assert values["sig_acc_career"] == pytest.approx(26 / 48)
    assert values["sig_def_career"] == pytest.approx(1 - 13 / 35)
    assert values["sig_att_pm_career"] == pytest.approx(48 / minutes)
    assert values["kd_per15_career"] == pytest.approx(1 / minutes * 15)
    assert values["td_per15_career"] == pytest.approx(1 / minutes * 15)
    assert values["td_acc_career"] == pytest.approx(1 / 3)
    assert values["td_def_career"] == pytest.approx(1 - 1 / 5)
    assert values["sub_att_per15_career"] == pytest.approx(1 / minutes * 15)
    assert values["ctrl_share_career"] == pytest.approx(90 / 1000)
    assert values["ctrl_against_share_career"] == pytest.approx(30 / 1000)
    assert values["kd_abs"] == 1.0
    assert values["kd_abs_per15_career"] == pytest.approx(1 / minutes * 15)
    # Head strikes absorbed: every strike of b and c defaults to the head.
    assert values["head_abs_pm_career"] == pytest.approx(13 / minutes)
    # Where a's 26 landed strikes went: 2 body, 3 leg, 1 clinch, 2 ground.
    assert values["head_share"] == pytest.approx(21 / 26)
    assert values["body_share"] == pytest.approx(2 / 26)
    assert values["leg_share"] == pytest.approx(3 / 26)
    assert values["distance_share"] == pytest.approx(23 / 26)
    assert values["clinch_share"] == pytest.approx(1 / 26)
    assert values["ground_share"] == pytest.approx(2 / 26)
    assert values["cage_min"] == pytest.approx(minutes)


def test_last_three_and_last_five_windows():
    league = League()
    a = league.fighter()
    for i, landed in enumerate((1, 2, 3, 4, 5, 6)):
        # One minute each, so strikes landed is strikes per minute.
        one_round(league, day(30 * i), a, league.fighter(), line(sig=(landed, 10)))
    values, _ = snapshot(league, a, day(400))
    assert values["slpm_career"] == pytest.approx(21 / 6)
    assert values["slpm_l3"] == pytest.approx((4 + 5 + 6) / 3)
    assert values["slpm_l5"] == pytest.approx((2 + 3 + 4 + 5 + 6) / 5)
    assert values["sig_acc_l3"] == pytest.approx(15 / 30)

    # As of the fourth bout's date only the first three exist.
    earlier, _ = snapshot(league, a, day(90))
    assert earlier["n_fights"] == 3.0
    assert earlier["slpm_l5"] == earlier["slpm_career"] == pytest.approx(2.0)


def test_decayed_average_halves_the_weight_every_half_life():
    league = League()
    a = league.fighter()
    one_round(league, day(0), a, league.fighter(), line(sig=(10, 10)))
    one_round(league, day(730), a, league.fighter(), line(sig=(20, 20)))
    for cutoff in (day(731), day(2000)):
        values, _ = snapshot(league, a, cutoff)
        # The older bout is one half-life (730 days) older: weights 1 and 2.
        assert values["slpm_decay"] == pytest.approx((1 * 10 + 2 * 20) / 3)
        assert values["slpm_career"] == pytest.approx(15.0)


def test_record_streaks_and_finishes():
    league = League()
    a = league.fighter()

    def opponent():
        return league.fighter()

    one_round(league, day(0), a, opponent(), method="KO/TKO")
    one_round(league, day(30), a, opponent(), method="Submission")
    b = opponent()
    one_round(league, day(60), a, b, winner=b, method="TKO - Doctor's Stoppage")
    one_round(
        league,
        day(90),
        a,
        opponent(),
        method="Decision - Split",
        is_title_bout=True,
        time_format=FIVE_ROUNDS,
    )
    one_round(league, day(120), a, opponent(), result="nc", method="Overturned")
    values, _ = snapshot(league, a, day(150))

    assert values["n_fights"] == 5.0
    assert (values["wins"], values["losses"]) == (3.0, 1.0)
    assert (values["ko_wins"], values["sub_wins"]) == (1.0, 1.0)
    assert values["finish_win_share"] == pytest.approx(2 / 3)
    assert (values["title_fights"], values["title_wins"]) == (1.0, 1.0)
    assert values["ko_losses"] == values["ko_losses_l3"] == 1.0
    assert values["sub_losses"] == 0.0
    # The no contest is left out of the rate and does not break the streak.
    assert values["win_rate_career"] == pytest.approx(3 / 4)
    assert values["win_rate_l3"] == pytest.approx(1 / 2)
    assert (values["win_streak"], values["loss_streak"]) == (1.0, 0.0)

    before_title, _ = snapshot(league, a, day(90))
    assert (before_title["win_streak"], before_title["loss_streak"]) == (0.0, 1.0)
    assert before_title["title_fights"] == 0.0


def test_activity_counts_the_last_two_years():
    league = League()
    a = league.fighter()
    cutoff = day(1000)
    for offset in (1000 - 800, 1000 - 730, 1000 - 10):
        one_round(league, day(offset), a, league.fighter())
    values, _ = snapshot(league, a, cutoff)
    assert values["days_since_last"] == 10.0
    assert values["fights_last_2y"] == 2.0  # 730 days ago is inside the window


def test_weight_change_from_the_previous_bout():
    league = League()
    a = league.fighter()
    one_round(league, day(0), a, league.fighter(), weight_class="Lightweight")
    one_round(league, day(30), a, league.fighter(), weight_class="Lightweight")
    one_round(league, day(60), a, league.fighter(), weight_class="Welterweight")

    up, _ = snapshot(league, a, day(60), "Welterweight")
    assert (up["moved_up"], up["moved_down"], up["fights_at_weight"]) == (1.0, 0.0, 0.0)
    down, _ = snapshot(league, a, day(90), "Lightweight")
    assert (down["moved_up"], down["moved_down"]) == (0.0, 1.0)
    assert down["fights_at_weight"] == 2.0
    same, _ = snapshot(league, a, day(90), "Welterweight")
    assert (same["moved_up"], same["moved_down"], same["fights_at_weight"]) == (
        0.0,
        0.0,
        1.0,
    )

    catch, _ = snapshot(league, a, day(90), "Catch Weight")
    assert math.isnan(catch["moved_up"]) and catch["fights_at_weight"] == 0.0
    unknown, _ = snapshot(league, a, day(90), None)
    assert math.isnan(unknown["moved_down"]) and math.isnan(unknown["fights_at_weight"])


def test_control_share_skips_bouts_where_control_was_not_recorded():
    league = League()
    a = league.fighter()
    one_round(
        league,
        day(0),
        a,
        league.fighter(),
        line(sig=(3, 5), ctrl=None),
        line(ctrl=None),
        200,
    )
    one_round(league, day(30), a, league.fighter(), line(ctrl=100), line(ctrl=20), 300)
    values, matchup = snapshot(league, a, day(60))
    assert values["ctrl_share_career"] == pytest.approx(100 / 300)
    assert values["ctrl_against_share_career"] == pytest.approx(20 / 300)
    assert values["slpm_career"] == pytest.approx(3 / (500 / 60))
    assert not matchup.flags["a_no_stats"]

    only_unrecorded, _ = snapshot(league, a, day(30))
    assert math.isnan(only_unrecorded["ctrl_share_career"])


def test_ratio_without_attempts_is_missing_not_zero():
    league = League()
    a = league.fighter()
    one_round(league, day(0), a, league.fighter(), line(sig=(3, 5)), line(sig=(0, 0)))
    values, _ = snapshot(league, a, day(30))
    assert values["td_per15_career"] == 0.0  # a real zero: 0 takedowns in 1 minute
    assert math.isnan(values["td_acc_career"])  # no attempts made
    assert math.isnan(values["td_def_career"])  # no attempts faced
    assert math.isnan(values["sig_def_career"])


def test_cardio_compares_rounds_three_onward_with_round_one():
    league = League()
    a = league.fighter()
    # Qualifies: three full rounds.
    league.bout(
        league.event(day(0)),
        a,
        league.fighter(),
        rounds_a=[line(sig=(10, 20)), line(sig=(40, 50)), line(sig=(5, 20))],
        rounds_b=[line(sig=(8, 16)), line(sig=(40, 50)), line(sig=(12, 16))],
    )
    # Does not qualify: stopped in round 2.
    league.bout(
        league.event(day(30)),
        a,
        league.fighter(),
        method="KO/TKO",
        rounds_a=[line(sig=(30, 30)), line(sig=(30, 30))],
        rounds_b=[line(sig=(30, 30)), line(sig=(30, 30))],
        end_time_s=120,
    )
    # Qualifies: five-round bout stopped 150 seconds into round 4.
    league.bout(
        league.event(day(60)),
        a,
        league.fighter(),
        method="KO/TKO",
        time_format=FIVE_ROUNDS,
        rounds_a=[
            line(sig=(6, 10)),
            line(sig=(9, 9)),
            line(sig=(3, 10)),
            line(sig=(3, 5)),
        ],
        rounds_b=[
            line(sig=(2, 4)),
            line(sig=(9, 9)),
            line(sig=(4, 8)),
            line(sig=(4, 6)),
        ],
        end_time_s=150,
    )
    values, _ = snapshot(league, a, day(90))

    r1_seconds, late_seconds = 300 + 300, 300 + (300 + 150)
    assert values["cardio_fights"] == 2.0
    # Landed: round 1 = 10 + 6, rounds 3+ = 5 + (3 + 3).
    assert values["cardio_slpm_ratio_career"] == pytest.approx(
        (11 / late_seconds) / (16 / r1_seconds)
    )
    # Attempted: round 1 = 20 + 10, rounds 3+ = 20 + (10 + 5).
    assert values["cardio_sig_att_ratio_career"] == pytest.approx(
        (35 / late_seconds) / (30 / r1_seconds)
    )
    # Absorbed: round 1 = 8 + 2, rounds 3+ = 12 + (4 + 4).
    assert values["cardio_sapm_ratio_career"] == pytest.approx(
        (20 / late_seconds) / (10 / r1_seconds)
    )
    assert values["cardio_sig_acc_delta"] == pytest.approx(11 / 35 - 16 / 30)
    # Opponents' accuracy: round 1 = 10 of 20, rounds 3+ = 20 of 30.
    assert values["cardio_sig_def_delta"] == pytest.approx(
        (1 - 20 / 30) - (1 - 10 / 20)
    )


def test_cardio_is_missing_without_a_bout_past_round_two():
    league = League()
    a = league.fighter()
    league.bout(
        league.event(day(0)),
        a,
        league.fighter(),
        method="KO/TKO",
        rounds_a=[line(sig=(30, 30)), line(sig=(30, 30))],
        rounds_b=[line(sig=(30, 30)), line(sig=(30, 30))],
        end_time_s=299,
    )
    # A decision without stats went three rounds but has no rounds to compare.
    league.bout(league.event(day(30)), a, league.fighter())
    values, _ = snapshot(league, a, day(60))
    assert values["cardio_fights"] == 0.0
    for feature in FIGHTER_FEATURES:
        if feature.family == "cardio" and feature.name != "cardio_fights":
            assert math.isnan(values[feature.name]), feature.name
    assert values["slpm_career"] == pytest.approx(
        60 / (599 / 60)
    )  # ordinary stats still exist


def test_physical_attributes_and_age_at_the_bout_date():
    league = League()
    a = league.fighter(
        height_cm=175.0, reach_cm=183.0, stance="Southpaw", dob=date(1990, 6, 15)
    )
    b = league.fighter(height_cm=None, reach_cm=None, stance=None, dob=None)
    one_round(league, day(0), a, b)
    values, _ = snapshot(league, a, date(2021, 6, 15))
    assert values["age"] == pytest.approx(
        (date(2021, 6, 15) - date(1990, 6, 15)).days / 365.25
    )
    assert (values["height_cm"], values["reach_cm"]) == (175.0, 183.0)
    assert (values["stance_southpaw"], values["stance_switch"]) == (1.0, 0.0)

    build = build_features(league.data(), NO_PRIOR, plain_params())
    unknown = build_matchup(build.store, b, a, date(2021, 6, 15)).a
    for name in ("age", "height_cm", "reach_cm", "stance_southpaw", "stance_switch"):
        assert math.isnan(unknown[name]), name


def test_opponent_quality_uses_ratings_from_before_each_bout():
    league = League()
    a, b, c, d = (league.fighter() for _ in range(4))
    first = league.event(day(0))
    league.bout(first, a, b)  # a 1516, b 1484
    league.bout(first, c, d)  # c 1516, d 1484
    league.bout(league.event(day(30)), a, c)  # a beats c, who was 1516
    values, _ = snapshot(league, a, day(60))

    assert values["opp_elo_career"] == pytest.approx((1500 + 1516) / 2)
    assert values["opp_elo_l3"] == pytest.approx((1500 + 1516) / 2)
    assert values["elo"] == pytest.approx(1516 + 32 * 0.5)
    # On the day of the second bout, only the first is history.
    earlier, _ = snapshot(league, a, day(30))
    assert earlier["opp_elo_career"] == 1500.0
    assert earlier["elo"] == 1516.0
