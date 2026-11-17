"""Elo: hand-checked updates, same-date handling, and training-only fitting."""

import math
from dataclasses import replace
from datetime import date

import pytest

from tests.feature_helpers import League, random_league
from ufc.features.elo import (
    EloParams,
    EloTracker,
    RatedBout,
    expected_score,
    fit_elo_params,
    run_elo,
)

D1, D2, D3 = date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)


def test_expected_score_is_a_probability_split_between_the_two():
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1700, 1500) == pytest.approx(1 / (1 + 10 ** (-200 / 400)))
    assert expected_score(1700, 1500) + expected_score(1500, 1700) == pytest.approx(1)


def test_update_matches_hand_calculation():
    tracker = EloTracker(EloParams(k=32))
    tracker.commit(1, [RatedBout("a", "b", 1.0, False)])
    # Equal ratings: expected 0.5, so the winner gains 32 * 0.5.
    assert tracker.rating("a") == 1516
    assert tracker.rating("b") == 1484

    tracker.commit(2, [RatedBout("b", "a", 1.0, False)])
    expected_b = 1 / (1 + 10 ** ((1516 - 1484) / 400))
    assert tracker.rating("b") == pytest.approx(1484 + 32 * (1 - expected_b))
    assert tracker.rating("a") == pytest.approx(1516 - 32 * (1 - expected_b))


def test_draw_moves_ratings_toward_each_other_and_no_contest_is_not_rated():
    league = League()
    a, b, c = league.fighter(), league.fighter(), league.fighter()
    league.bout(league.event(D1), a, b)  # a 1516, b 1484
    league.bout(league.event(D2), a, b, result="draw")
    league.bout(league.event(D2), a, c, result="nc")
    tracker = EloTracker(EloParams(k=32))
    for day in (D1, D2):
        tracker.commit_date(
            day.toordinal(), [x for x in league.data().bouts if x.date == day]
        )
    expected_a = 1 / (1 + 10 ** ((1484 - 1516) / 400))
    assert tracker.rating(a) == pytest.approx(1516 + 32 * (0.5 - expected_a))
    assert tracker.rating(b) == pytest.approx(1484 - 32 * (0.5 - expected_a))
    assert tracker.rating(c) == 1500


def test_bouts_on_one_date_all_use_the_ratings_from_before_it():
    """A tournament: the winner's second bout does not see his first."""
    tracker = EloTracker(EloParams(k=32))
    tracker.commit(
        1, [RatedBout("a", "b", 1.0, False), RatedBout("a", "c", 1.0, False)]
    )
    assert tracker.rating("a") == 1500 + 16 + 16
    assert tracker.rating("c") == 1484


def test_finish_and_provisional_multipliers():
    params = EloParams(k=20, finish_mult=1.5, provisional_mult=2.0, provisional_bouts=1)
    tracker = EloTracker(params)
    tracker.commit(1, [RatedBout("a", "b", 1.0, True)])
    # Both are in their first bout: 20 * 1.5 * 2.0 * 0.5.
    assert tracker.rating("a") == 1530
    tracker.commit(2, [RatedBout("a", "c", 1.0, False)])
    expected_a = 1 / (1 + 10 ** ((1500 - 1530) / 400))
    # a is past his provisional bout, c is in his.
    assert tracker.rating("a") == pytest.approx(1530 + 20 * (1 - expected_a))
    assert tracker.rating("c") == pytest.approx(1500 - 40 * (1 - expected_a))


def test_rating_before_is_strict_about_the_date():
    tracker = EloTracker(EloParams(k=32))
    tracker.commit(10, [RatedBout("a", "b", 1.0, False)])
    tracker.commit(20, [RatedBout("a", "b", 1.0, False)])
    assert tracker.rating_before("a", 10) == 1500
    assert tracker.rating_before("a", 11) == 1516
    assert tracker.rating_before("a", 20) == 1516
    assert tracker.rating_before("a", 21) == tracker.rating("a")
    assert tracker.rating_before("never fought", 21) == 1500


def test_dates_must_be_committed_in_order():
    tracker = EloTracker(EloParams())
    tracker.commit(5, [])
    with pytest.raises(ValueError):
        tracker.commit(5, [])


def test_run_elo_reports_pre_fight_ratings_and_labels():
    league = League()
    a, b = league.fighter(), league.fighter()
    league.bout(league.event(D1), a, b)
    league.bout(league.event(D2), a, b, winner=b)
    league.bout(league.event(D3), a, b, result="draw")
    first, second, third = run_elo(league.data().bouts, EloParams(k=32))
    assert (first.rating_a, first.rating_b, first.p_a, first.label) == (
        1500,
        1500,
        0.5,
        1.0,
    )
    assert (second.rating_a, second.rating_b, second.label) == (1516, 1484, 0.0)
    assert second.p_a == pytest.approx(1 / (1 + 10 ** (-32 / 400)))
    assert third.label is None


def test_fit_returns_the_grid_point_with_the_lowest_training_loss():
    bouts = random_league().data().bouts
    train_end = bouts[len(bouts) // 2].date
    grid = {"k": (8.0, 32.0, 64.0), "finish_mult": (1.0, 1.5)}
    fit = fit_elo_params(bouts, train_end, grid)

    def loss(params: EloParams) -> float:
        scored = [
            p
            for p in run_elo([b for b in bouts if b.date <= train_end], params)
            if p.label is not None
        ]
        assert len(scored) == fit.n_bouts
        return -sum(
            math.log(p.p_a if p.label == 1.0 else 1 - p.p_a) for p in scored
        ) / len(scored)

    losses = {
        (k, m): loss(EloParams(k=k, finish_mult=m))
        for k in grid["k"]
        for m in grid["finish_mult"]
    }
    best = min(losses, key=losses.get)
    assert (fit.params.k, fit.params.finish_mult) == best
    assert fit.log_loss == pytest.approx(losses[best])
    assert fit.last_date <= train_end


def test_fit_ignores_everything_after_the_training_period():
    bouts = list(random_league().data().bouts)
    train_end = bouts[len(bouts) // 2].date
    fit = fit_elo_params(bouts, train_end)

    truncated = [b for b in bouts if b.date <= train_end]
    flipped = [
        b
        if b.date <= train_end or b.result != "win"
        else replace(b, winner_id=b.opponent_of(b.winner_id))
        for b in bouts
    ]
    assert fit_elo_params(truncated, train_end) == fit
    assert fit_elo_params(flipped, train_end) == fit
