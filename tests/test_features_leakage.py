"""The leakage tests required by docs/ML_REQUIREMENTS.md.

Perturbation: changing or deleting bouts after a date leaves every feature on
or before that date exactly as it was. Same event: a bout's features do not
depend on any other bout held on the same date. Comparisons are exact, not
approximate: a leak of any size fails.
"""

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from tests.feature_helpers import (
    random_league,
    scramble_where,
    without_bouts_after,
)
from ufc.features.build import build_features, fit_params
from ufc.features.config import FeatureConfig
from ufc.features.matchup import FEATURE_NAMES, FLAG_NAMES, build_matchup
from ufc.features.store import FeatureStore

NOBODY = "0" * 16


@pytest.fixture(scope="module")
def data():
    return random_league().data()


@pytest.fixture(scope="module")
def config(data):
    """Training ends about 60% of the way through the league's history."""
    return FeatureConfig(train_end=data.bouts[int(len(data.bouts) * 0.6)].date)


@pytest.fixture(scope="module")
def params(data, config):
    return fit_params(list(data.bouts), config)[0]


@pytest.fixture(scope="module")
def built(data, config, params):
    return build_features(data, config, params)


def rows_through(frame: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    return frame[frame["date"] <= pd.Timestamp(cutoff)].reset_index(drop=True)


def dates_of(data) -> list[date]:
    return sorted({b.date for b in data.bouts})


def busiest_dates(data) -> tuple[date, date]:
    """The date with two events, and the date of the one-night tournament."""
    events_per_date: dict[date, set[str]] = {}
    for bout in data.bouts:
        events_per_date.setdefault(bout.date, set()).add(bout.event_id)
    (two_events,) = [d for d, events in events_per_date.items() if len(events) == 2]
    for day in dates_of(data):
        fighters = [
            f
            for b in data.bouts
            if b.date == day
            for f in (b.fighter_a_id, b.fighter_b_id)
        ]
        if len(fighters) != len(set(fighters)):
            return two_events, day
    raise AssertionError("the generated league has no tournament")


def test_league_has_the_cases_the_tests_rely_on(data, built):
    two_events, tournament = busiest_dates(data)
    assert two_events != tournament
    frame = built.frame
    assert len(frame) == len(data.bouts) > 200
    assert {"win", "draw", "nc"} <= set(frame["result"])
    assert frame["a_debut"].any() and not frame["a_debut"].all()
    assert (frame["a_no_stats"] & ~frame["a_debut"]).any()
    # Enough history that most features are populated for someone.
    assert frame[list(FEATURE_NAMES)].notna().any().all()


@pytest.mark.parametrize("fraction", [0.1, 0.35, 0.6, 0.85])
def test_perturbation_deleting_later_bouts_changes_nothing_earlier(
    data, config, params, built, fraction
):
    cutoff = dates_of(data)[int(len(dates_of(data)) * fraction)]
    truncated = build_features(without_bouts_after(data, cutoff), config, params)
    pd.testing.assert_frame_equal(
        truncated.frame, rows_through(built.frame, cutoff), check_exact=True
    )


@pytest.mark.parametrize("fraction", [0.1, 0.35, 0.6, 0.85])
def test_perturbation_changing_later_bouts_changes_nothing_earlier(
    data, config, params, built, fraction
):
    cutoff = dates_of(data)[int(len(dates_of(data)) * fraction)]
    changed = build_features(
        scramble_where(data, lambda b: b.date > cutoff), config, params
    )
    pd.testing.assert_frame_equal(
        rows_through(changed.frame, cutoff),
        rows_through(built.frame, cutoff),
        check_exact=True,
    )
    # The test has teeth: later rows do change, in features and in flags' inputs.
    later = built.frame["date"] > pd.Timestamp(cutoff)
    assert not changed.frame.loc[later, list(FEATURE_NAMES)].equals(
        built.frame.loc[later, list(FEATURE_NAMES)]
    )


def test_perturbation_has_teeth_changing_an_earlier_bout_is_detected(
    data, config, params, built
):
    cutoff = dates_of(data)[len(dates_of(data)) // 2]
    changed = build_features(
        scramble_where(data, lambda b: b.date < cutoff), config, params
    )
    after = built.frame["date"] >= pd.Timestamp(cutoff)
    differing = (
        changed.frame.loc[after, list(FEATURE_NAMES)]
        .ne(built.frame.loc[after, list(FEATURE_NAMES)])
        .any()
    )
    # Nearly every feature is built from history; physical attributes are not.
    assert differing.mean() > 0.9


def test_perturbation_with_fitting_included(data, config, built, params):
    """End to end: bouts after the training period change neither the fitted
    parameters nor any feature inside the training period."""
    changed_data = scramble_where(data, lambda b: b.date > config.train_end)
    changed = build_features(changed_data, config)  # fits its own parameters
    assert changed.params.elo == params.elo
    np.testing.assert_array_equal(changed.params.league, params.league)
    pd.testing.assert_frame_equal(
        rows_through(changed.frame, config.train_end),
        rows_through(built.frame, config.train_end),
        check_exact=True,
    )

    truncated = build_features(without_bouts_after(data, config.train_end), config)
    assert truncated.params.elo == params.elo
    np.testing.assert_array_equal(truncated.params.league, params.league)


def _same_event_dates(data) -> list[date]:
    two_events, tournament = busiest_dates(data)
    ordinary = dates_of(data)[-3]
    return [two_events, tournament, ordinary]


@pytest.mark.parametrize(
    "which", [0, 1, 2], ids=["two_events", "tournament", "ordinary"]
)
def test_same_event_other_bouts_on_the_card_do_not_matter(
    data, config, params, built, which
):
    day = _same_event_dates(data)[which]
    history = without_bouts_after(data, day)
    card = [b for b in history.bouts if b.date == day]
    assert len(card) >= 4
    for bout in card:
        expected = built.frame[built.frame["bout_id"] == bout.bout_id].reset_index(
            drop=True
        )

        def is_other(b, bout=bout):
            return b.date == day and b.bout_id != bout.bout_id

        changed = build_features(
            scramble_where(history, is_other), config, params
        ).frame
        alone = build_features(
            replace(history, bouts=[b for b in history.bouts if not is_other(b)]),
            config,
            params,
        ).frame
        for frame in (changed, alone):
            actual = frame[frame["bout_id"] == bout.bout_id].reset_index(drop=True)
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_fighter_with_two_bouts_on_one_date_brings_only_earlier_history(data, built):
    _, tournament = busiest_dates(data)
    on_the_day = [b for b in data.bouts if b.date == tournament]
    counts: dict[str, int] = {}
    for bout in on_the_day:
        for fighter in (bout.fighter_a_id, bout.fighter_b_id):
            counts[fighter] = counts.get(fighter, 0) + 1
    twice = [f for f, n in counts.items() if n == 2]
    assert twice
    for fighter in twice:
        before = sum(
            1
            for b in data.bouts
            if b.date < tournament and fighter in (b.fighter_a_id, b.fighter_b_id)
        )
        as_of = build_matchup(built.store, fighter, NOBODY, tournament).a
        assert as_of["n_fights"] == before
        next_day = build_matchup(
            built.store, fighter, NOBODY, tournament + timedelta(days=1)
        ).a
        assert next_day["n_fights"] == before + 2


def test_batch_rows_equal_the_as_of_function_on_the_finished_store(data, built):
    """The store at the end holds every bout, yet asking it about any past
    bout gives exactly the row the chronological build produced."""
    for row in built.frame.itertuples(index=False):
        matchup = build_matchup(
            built.store,
            row.fighter_a_id,
            row.fighter_b_id,
            row.date.date(),
            None if pd.isna(row.weight_class) else row.weight_class,
        )
        np.testing.assert_array_equal(
            np.array([matchup.features[n] for n in FEATURE_NAMES]),
            np.array([getattr(row, n) for n in FEATURE_NAMES]),
        )
        assert tuple(matchup.flags[n] for n in FLAG_NAMES) == tuple(
            getattr(row, n) for n in FLAG_NAMES
        )


def test_hypothetical_bout_after_the_last_event_uses_all_history(data, built):
    last = data.bouts[-1]
    fighter = last.fighter_a_id
    total = sum(1 for b in data.bouts if fighter in (b.fighter_a_id, b.fighter_b_id))
    future = build_matchup(
        built.store,
        fighter,
        last.fighter_b_id,
        last.date + timedelta(days=30),
        "Lightweight",
    )
    assert future.a["n_fights"] == total
    assert future.a["days_since_last"] == 30.0
    assert not future.flags["a_debut"]
    with pytest.raises(ValueError):
        build_matchup(built.store, fighter, fighter, last.date)


def test_store_refuses_dates_out_of_order(data, config, params):
    store = FeatureStore(data.fighters, config, params)
    first, second = dates_of(data)[:2]
    store.commit_date(second, [b for b in data.bouts if b.date == second])
    with pytest.raises(ValueError):
        store.commit_date(first, [b for b in data.bouts if b.date == first])
    with pytest.raises(ValueError):
        store.commit_date(second, [b for b in data.bouts if b.date == second])
    with pytest.raises(ValueError):
        store.commit_date(
            second + timedelta(days=400), [b for b in data.bouts if b.date == first]
        )
