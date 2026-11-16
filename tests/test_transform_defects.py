"""How the transform reacts to defects injected into the fixture snapshot's rows."""

from dataclasses import replace

from ufc.ingest.identity import load_overrides
from ufc.ingest.transform import build_dataset

TITLE_FIGHT = "30dbf1eceb2e1f8d"
TITLE_FIGHT_BOUT = "Ciryl Gane vs. Josh Hokit"


def build(raw, **tables):
    return build_dataset(replace(raw, **tables), load_overrides())


def edit(rows, match, **changes):
    """Copy ``rows``, applying ``changes`` to those for which ``match(row)`` is true."""
    edited = [{**row, **changes} if match(row) else dict(row) for row in rows]
    assert edited != rows, "edit matched no rows"
    return edited


def is_title_fight(row):
    return row["URL"].endswith(TITLE_FIGHT)


def is_title_fight_stats(row):
    return row["BOUT"] == TITLE_FIGHT_BOUT


def issues(dataset, check):
    return [issue for issue in dataset.issues if issue.check == check]


def bout(dataset, bout_id=TITLE_FIGHT):
    return next((b for b in dataset.bouts if b.bout_id == bout_id), None)


def test_unknown_outcome_code_drops_the_bout(raw):
    built = build(
        raw, fight_results=edit(raw.fight_results, is_title_fight, OUTCOME="W/W")
    )
    assert bout(built) is None
    [issue] = issues(built, "unparseable_fight")
    assert (issue.severity, issue.entity_id) == ("error", TITLE_FIGHT)


def test_unparseable_end_time_drops_the_bout(raw):
    built = build(
        raw, fight_results=edit(raw.fight_results, is_title_fight, TIME="five")
    )
    assert bout(built) is None
    assert issues(built, "unparseable_fight")


def test_malformed_fight_url_is_reported(raw):
    built = build(
        raw, fight_results=edit(raw.fight_results, is_title_fight, URL="not-a-url")
    )
    assert bout(built) is None
    assert issues(built, "unparseable_fight")
    assert issues(built, "fight_files_disagree")


def test_unknown_method_is_kept_with_a_warning(raw):
    built = build(
        raw, fight_results=edit(raw.fight_results, is_title_fight, METHOD="Forfeit ")
    )
    assert bout(built).method == "Forfeit"
    [issue] = issues(built, "unknown_method")
    assert issue.severity == "warning"


def test_unreadable_time_format_keeps_the_bout_without_a_fight_time(raw):
    rows = edit(raw.fight_results, is_title_fight, **{"TIME FORMAT": "Five rounds"})
    built = build(raw, fight_results=rows)
    kept = bout(built)
    assert kept.time_format == "Five rounds"
    assert kept.scheduled_rounds is None and kept.fight_time_s is None
    assert kept.end_round == 5 and kept.end_time_s == 300
    assert issues(built, "bad_time_format")


def test_ending_in_a_round_the_format_does_not_list_leaves_fight_time_unknown(raw):
    rows = edit(raw.fight_results, is_title_fight, **{"TIME FORMAT": "3 Rnd (5-5-5)"})
    built = build(raw, fight_results=rows)
    assert bout(built).fight_time_s is None
    assert bout(built).scheduled_rounds == 3
    assert issues(built, "bad_time_format")


def test_stats_row_naming_a_fighter_outside_the_bout_drops_that_bouts_stats(raw):
    def gane_round_one(row):
        return (
            is_title_fight_stats(row)
            and row["FIGHTER"] == "Ciryl Gane"
            and row["ROUND"] == "Round 1"
        )

    built = build(
        raw, fight_stats=edit(raw.fight_stats, gane_round_one, FIGHTER="Someone Else")
    )
    assert bout(built) is not None and not bout(built).has_stats
    assert not [row for row in built.round_stats if row.bout_id == TITLE_FIGHT]
    assert not [row for row in built.fight_stats if row.bout_id == TITLE_FIGHT]
    [issue] = issues(built, "malformed_stats")
    assert "Someone Else" in issue.message


def test_round_missing_one_fighters_row_drops_that_bouts_stats(raw):
    rows = [
        row
        for row in raw.fight_stats
        if not (
            is_title_fight_stats(row)
            and row["FIGHTER"] == "Josh Hokit"
            and row["ROUND"] == "Round 3"
        )
    ]
    built = build(raw, fight_stats=rows)
    assert not bout(built).has_stats
    assert "missing one fighter's row" in issues(built, "malformed_stats")[0].message


def test_duplicated_stats_row_drops_that_bouts_stats(raw):
    extra = next(row for row in raw.fight_stats if is_title_fight_stats(row))
    built = build(raw, fight_stats=[*raw.fight_stats, extra])
    assert not bout(built).has_stats
    assert "two rows for one fighter" in issues(built, "malformed_stats")[0].message


def test_unparseable_stat_cell_drops_that_bouts_stats(raw):
    def gane_round_two(row):
        return (
            is_title_fight_stats(row)
            and row["FIGHTER"] == "Ciryl Gane"
            and row["ROUND"] == "Round 2"
        )

    built = build(raw, fight_stats=edit(raw.fight_stats, gane_round_two, TD="lots"))
    assert not bout(built).has_stats
    assert issues(built, "malformed_stats")


def test_skipped_round_drops_that_bouts_stats(raw):
    rows = [
        row
        for row in raw.fight_stats
        if not (is_title_fight_stats(row) and row["ROUND"] == "Round 2")
    ]
    built = build(raw, fight_stats=rows)
    assert not bout(built).has_stats
    assert "not consecutive" in issues(built, "malformed_stats")[0].message


def test_fewer_stat_rounds_than_the_result_says_is_loaded_with_a_warning(raw):
    rows = [
        row
        for row in raw.fight_stats
        if not (is_title_fight_stats(row) and row["ROUND"] == "Round 5")
    ]
    built = build(raw, fight_stats=rows)
    assert bout(built).has_stats
    assert len([row for row in built.round_stats if row.bout_id == TITLE_FIGHT]) == 8
    [issue] = issues(built, "stats_round_mismatch")
    assert "4 rounds" in issue.message and "round 5" in issue.message


def test_other_bouts_are_unaffected_by_one_bouts_bad_stats(raw, dataset):
    extra = next(row for row in raw.fight_stats if is_title_fight_stats(row))
    built = build(raw, fight_stats=[*raw.fight_stats, extra])
    others = [row for row in dataset.round_stats if row.bout_id != TITLE_FIGHT]
    assert [row for row in built.round_stats if row.bout_id != TITLE_FIGHT] == others


def test_duplicate_event_id_keeps_the_first_and_reports(raw):
    built = build(raw, events=[*raw.events, {**raw.events[0], "EVENT": "Copy"}])
    assert len(built.events) == 19
    [issue] = issues(built, "duplicate_id")
    assert issue.entity_type == "event"


def test_unparseable_event_date_drops_the_event_and_orphans_its_fights(raw):
    def is_ufc_334(row):
        return row["EVENT"] == "UFC 334: Gane vs. Hokit"

    built = build(raw, events=edit(raw.events, is_ufc_334, DATE="14/11/2026"))
    assert len(built.events) == 18
    assert issues(built, "unparseable_event")
    assert bout(built) is None
    assert TITLE_FIGHT in {issue.entity_id for issue in issues(built, "orphan_event")}


def test_two_events_sharing_a_name_cannot_place_their_fights(raw):
    twin = {**raw.events[0], "URL": "http://ufcstats.com/event-details/" + "0" * 16}
    built = build(raw, events=[*raw.events, twin])
    assert len(issues(built, "duplicate_event_name")) == 2
    assert bout(built) is None


def test_fight_listed_under_two_known_events_is_dropped(raw):
    other_event = next(
        row["EVENT"] for row in raw.events if "UFC 334" not in row["EVENT"]
    )
    relisted = {
        **next(row for row in raw.fight_results if is_title_fight(row)),
        "EVENT": other_event,
    }
    built = build(raw, fight_results=[*raw.fight_results, relisted])
    assert bout(built) is None
    [issue] = issues(built, "conflicting_event")
    assert issue.entity_id == TITLE_FIGHT


def test_duplicate_fighter_id_is_reported(raw):
    built = build(raw, fighter_tott=[*raw.fighter_tott, raw.fighter_tott[0]])
    assert len(built.fighters) == 55
    [issue] = issues(built, "duplicate_id")
    assert issue.entity_type == "fighter"


def test_unreadable_fighter_attribute_is_left_empty_with_a_warning(raw):
    def is_gane(row):
        return row["FIGHTER"] == "Ciryl Gane"

    built = build(raw, fighter_tott=edit(raw.fighter_tott, is_gane, HEIGHT="193 cm"))
    gane = next(f for f in built.fighters if f.name == "Ciryl Gane")
    assert gane.height_cm is None
    assert gane.reach_cm == 205.74
    [issue] = issues(built, "bad_fighter_attribute")
    assert issue.entity_id == gane.fighter_id


def test_fighter_in_both_a_womens_and_a_mens_class_is_flagged(raw):
    def is_womens_title_fight(row):
        return row["URL"].endswith("32112fd0aaf7fba2")

    rows = edit(
        raw.fight_results, is_womens_title_fight, BOUT="Kayla Harrison vs. Ciryl Gane"
    )
    built = build(raw, fight_results=rows)
    flagged = {issue.entity_id for issue in issues(built, "mixed_division")}
    assert TITLE_FIGHT in flagged
