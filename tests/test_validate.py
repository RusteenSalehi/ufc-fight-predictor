"""Dataset-level checks, each triggered by a small hand-built dataset."""

from dataclasses import replace
from datetime import date

from ufc.ingest.models import (
    Bout,
    Dataset,
    Event,
    Fighter,
    FightStats,
    RoundStats,
    StatLine,
)
from ufc.ingest.validate import validate_dataset

AS_OF = date(2026, 11, 15)
EVENT_ID, BOUT_ID, RED, BLUE = "e" * 16, "b" * 16, "1" * 16, "2" * 16

LINE = StatLine(
    knockdowns=0,
    sig_str_landed=10,
    sig_str_attempted=20,
    total_str_landed=15,
    total_str_attempted=25,
    td_landed=1,
    td_attempted=2,
    sub_attempts=0,
    reversals=0,
    ctrl_s=30,
    head_landed=6,
    head_attempted=12,
    body_landed=3,
    body_attempted=5,
    leg_landed=1,
    leg_attempted=3,
    distance_landed=7,
    distance_attempted=15,
    clinch_landed=2,
    clinch_attempted=3,
    ground_landed=1,
    ground_attempted=2,
)
EVENT = Event(EVENT_ID, "Test Night", date(2024, 5, 4), "Somewhere")
FIGHTERS = [
    Fighter(RED, "Red Corner", None, 180.0, 185.0, "Orthodox", date(1994, 1, 1)),
    Fighter(BLUE, "Blue Corner", None, None, None, None, None),
]
BOUT = Bout(
    bout_id=BOUT_ID,
    event_id=EVENT_ID,
    fighter_a_id=RED,
    fighter_b_id=BLUE,
    weight_class="Lightweight",
    bout_type_raw="Lightweight Bout",
    is_womens=False,
    is_title_bout=False,
    is_interim_title=False,
    is_tournament_final=False,
    scheduled_rounds=3,
    time_format="3 Rnd (5-5-5)",
    status="completed",
    result="win",
    winner_id=RED,
    method="KO/TKO",
    method_details=None,
    referee=None,
    end_round=1,
    end_time_s=100,
    fight_time_s=100,
    has_stats=True,
)


def make_dataset(**changes) -> Dataset:
    base = {
        "events": [EVENT],
        "fighters": list(FIGHTERS),
        "bouts": [BOUT],
        "round_stats": [
            RoundStats(BOUT_ID, RED, 1, LINE),
            RoundStats(BOUT_ID, BLUE, 1, LINE),
        ],
        "fight_stats": [
            FightStats(BOUT_ID, RED, LINE),
            FightStats(BOUT_ID, BLUE, LINE),
        ],
        "source_fights": 1,
    }
    return Dataset(**{**base, **changes})


def checks(dataset: Dataset, severity: str) -> list[str]:
    return [i.check for i in validate_dataset(dataset, AS_OF) if i.severity == severity]


def test_clean_dataset_raises_nothing_but_notes():
    issues = validate_dataset(make_dataset(), AS_OF)
    assert {issue.severity for issue in issues} == {"info"}
    assert {issue.check for issue in issues} == {
        "row_counts",
        "date_range",
        "results_breakdown",
    }


def test_row_counts_report_dropped_fights():
    [note] = [
        i
        for i in validate_dataset(make_dataset(source_fights=3), AS_OF)
        if i.check == "row_counts"
    ]
    assert "bouts=1 (of 3 distinct fights in the source; 2 dropped)" in note.message
    assert "bouts_with_stats=1" in note.message


def test_duplicate_ids_are_errors():
    assert checks(make_dataset(bouts=[BOUT, BOUT]), "error") == ["duplicate_id"]
    assert checks(make_dataset(fighters=[*FIGHTERS, FIGHTERS[0]]), "error") == [
        "duplicate_id"
    ]
    assert checks(make_dataset(events=[EVENT, EVENT]), "error") == ["duplicate_id"]


def test_bout_with_missing_fighter_or_event_is_an_error():
    assert "broken_reference" in checks(make_dataset(fighters=FIGHTERS[:1]), "error")
    orphaned = replace(BOUT, event_id="0" * 16)
    assert "broken_reference" in checks(make_dataset(bouts=[orphaned]), "error")


def test_stats_for_a_fighter_outside_the_bout_are_an_error():
    stray = RoundStats(BOUT_ID, "3" * 16, 1, LINE)
    assert checks(make_dataset(round_stats=[stray]), "error") == ["broken_reference"]


def test_no_bouts_is_an_error():
    empty = make_dataset(bouts=[], round_stats=[], fight_stats=[])
    assert "row_counts" in checks(empty, "error")


def test_fighters_without_bouts_are_counted():
    extra = Fighter("9" * 16, "Never Fought", None, None, None, None, None)
    issues = validate_dataset(make_dataset(fighters=[*FIGHTERS, extra]), AS_OF)
    [note] = [i for i in issues if i.check == "fighters_without_bouts"]
    assert note.severity == "info"
    assert note.message.startswith("1 of 3 fighter profiles")


def test_event_without_bouts_is_a_warning():
    extra = Event("d" * 16, "Empty Night", date(2024, 6, 1), None)
    assert checks(make_dataset(events=[EVENT, extra]), "warning") == [
        "events_without_bouts"
    ]


def test_event_dated_after_the_snapshot_is_a_warning():
    future = replace(EVENT, date=date(2027, 1, 1))
    assert "date_sanity" in checks(make_dataset(events=[future]), "warning")


def test_event_dated_before_the_sport_existed_is_a_warning():
    ancient = replace(EVENT, date=date(1985, 1, 1))
    assert "date_sanity" in checks(make_dataset(events=[ancient]), "warning")


def test_implausible_date_of_birth_is_a_warning():
    unborn = replace(FIGHTERS[0], dob=date(2030, 1, 1))
    assert "date_sanity" in checks(
        make_dataset(fighters=[unborn, FIGHTERS[1]]), "warning"
    )


def test_implausible_age_at_a_bout_is_a_warning():
    child = replace(FIGHTERS[0], dob=date(2015, 1, 1))
    issues = validate_dataset(make_dataset(fighters=[child, FIGHTERS[1]]), AS_OF)
    [warning] = [i for i in issues if i.severity == "warning"]
    assert warning.check == "date_sanity"
    assert warning.entity_id == BOUT_ID
    assert "9.3 years old" in warning.message


def test_missing_date_of_birth_is_not_an_age_problem():
    assert checks(make_dataset(), "warning") == []


def test_more_landed_than_attempted_is_a_warning():
    bad = replace(LINE, td_landed=3)
    rows = [RoundStats(BOUT_ID, RED, 1, bad), RoundStats(BOUT_ID, BLUE, 1, LINE)]
    issues = validate_dataset(make_dataset(round_stats=rows), AS_OF)
    [warning] = [i for i in issues if i.severity == "warning"]
    assert warning.check == "stat_sanity"
    assert warning.message.startswith(
        "1 fighter-rounds have more landed than attempted"
    )


def test_strike_breakdowns_that_do_not_add_up_are_warnings():
    bad_target = replace(LINE, head_landed=7)
    bad_position = replace(LINE, ground_attempted=9)
    rows = [
        RoundStats(BOUT_ID, RED, 1, bad_target),
        RoundStats(BOUT_ID, BLUE, 1, bad_position),
    ]
    messages = [
        i.message
        for i in validate_dataset(make_dataset(round_stats=rows), AS_OF)
        if i.check == "stat_sanity"
    ]
    assert len(messages) == 2
    assert any("head + body + leg" in m for m in messages)
    assert any("distance + clinch + ground" in m for m in messages)


def test_more_significant_than_total_strikes_is_a_warning():
    bad = replace(LINE, total_str_landed=5)
    rows = [RoundStats(BOUT_ID, RED, 1, bad), RoundStats(BOUT_ID, BLUE, 1, LINE)]
    messages = [
        i.message
        for i in validate_dataset(make_dataset(round_stats=rows), AS_OF)
        if i.check == "stat_sanity"
    ]
    assert any("more significant strikes than total strikes" in m for m in messages)


def test_win_by_a_no_contest_method_is_a_warning():
    odd = replace(BOUT, method="Overturned")
    assert checks(make_dataset(bouts=[odd]), "warning") == ["result_method_mismatch"]
    no_contest = replace(BOUT, result="nc", winner_id=None, method="Overturned")
    assert checks(make_dataset(bouts=[no_contest]), "warning") == []


def test_bouts_without_a_weight_class_are_counted():
    unclassed = replace(BOUT, weight_class=None)
    issues = validate_dataset(make_dataset(bouts=[unclassed]), AS_OF)
    assert [i.severity for i in issues if i.check == "no_weight_class"] == ["info"]
