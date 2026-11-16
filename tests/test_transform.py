"""The transform against the real-data fixture snapshot, one edge case per test."""

from dataclasses import fields
from datetime import date

from ufc.ingest.models import Fighter, sum_stat_lines
from ufc.ingest.transform import build_dataset, names_conflict

# Fighter IDs used below (from the fixture files).
GANE = "787bb1f087ccff8a"
BRUNO_SILVA_MIDDLEWEIGHT = "12ebd7d157e91701"
BRUNO_SILVA_FLYWEIGHT = "294aa73dbf37d281"
JEAN_SILVA_FEATHERWEIGHT = "52ef95b5860fb28c"
JEAN_SILVA_OTHER = "9211aae062b799d6"
ZACH_REESE = "23dec7c47cb418f8"
RAFAEL_CERQUEIRA = "09a4447655fd4b7b"
NICHOLAS_GALANTI = "8988705468d04ef7"
TYRELL_FORTUNE = "bc7f4b2d4c2724a2"


def issues_for(dataset, check):
    return [issue for issue in dataset.issues if issue.check == check]


def stats_for(dataset, bout_id):
    return [row for row in dataset.round_stats if row.bout_id == bout_id]


def totals_for(dataset, bout_id):
    return {
        row.fighter_id: row.stats
        for row in dataset.fight_stats
        if row.bout_id == bout_id
    }


# --- shape -----------------------------------------------------------------


def test_row_counts(dataset):
    assert len(dataset.events) == 19
    assert len(dataset.fighters) == 55
    assert dataset.source_fights == 29
    assert len(dataset.bouts) == 27
    assert len(dataset.fight_stats) == 2 * sum(bout.has_stats for bout in dataset.bouts)


def test_every_bout_is_completed_and_points_at_known_rows(dataset, events, fighters):
    for bout in dataset.bouts:
        assert bout.status == "completed"
        assert bout.event_id in events
        assert bout.fighter_a_id in fighters
        assert bout.fighter_b_id in fighters
        assert bout.fighter_a_id != bout.fighter_b_id


def test_event_fields(events):
    event = events["402ea3b5ee233852"]
    assert event.name == "UFC 334: Gane vs. Hokit"
    assert event.date == date(2026, 11, 14)
    assert event.location == "New York City, New York, USA"
    assert event.status == "completed"


# --- results ---------------------------------------------------------------


def test_title_bout_decision_over_five_rounds(bouts):
    bout = bouts["30dbf1eceb2e1f8d"]
    assert bout.weight_class == "Heavyweight"
    assert bout.bout_type_raw == "UFC Heavyweight Title Bout"
    assert bout.is_title_bout and not bout.is_interim_title and not bout.is_womens
    assert bout.scheduled_rounds == 5
    assert bout.result == "win"
    assert bout.winner_id == bout.fighter_a_id == GANE
    assert bout.method == "Decision - Unanimous"
    assert (bout.end_round, bout.end_time_s, bout.fight_time_s) == (5, 300, 1500)
    assert bout.referee == "Marc Goddard"
    assert bout.has_stats


def test_womens_title_bout_submission(bouts):
    bout = bouts["32112fd0aaf7fba2"]
    assert bout.weight_class == "Women's Bantamweight"
    assert bout.is_womens and bout.is_title_bout
    assert bout.method == "Submission"
    assert bout.method_details == "Rear Naked Choke"
    assert (bout.end_round, bout.end_time_s, bout.fight_time_s) == (4, 89, 989)


def test_interim_title_bout(bouts):
    bout = bouts["5727d5be8c373346"]
    assert bout.is_title_bout and bout.is_interim_title
    assert bout.winner_id == bout.fighter_b_id == GANE


def test_majority_draw_has_no_winner(bouts):
    bout = bouts["79c49d244eab9005"]
    assert bout.result == "draw"
    assert bout.winner_id is None
    assert bout.method == "Decision - Majority"


def test_split_draw_in_a_title_bout(bouts):
    bout = bouts["b395c89e19a3fec4"]
    assert bout.result == "draw" and bout.winner_id is None
    assert bout.method == "Decision - Split"
    assert bout.is_title_bout and bout.is_womens


def test_split_decision_won_by_second_named_fighter(bouts, fighters):
    bout = bouts["697efaf0d162e877"]
    assert bout.method == "Decision - Split"
    assert bout.result == "win"
    assert bout.winner_id == bout.fighter_b_id
    assert fighters[bout.winner_id].name == "Jack Della Maddalena"


def test_no_contest_overturned(bouts):
    bout = bouts["5d3e046b6bcd49f5"]
    assert bout.result == "nc" and bout.winner_id is None
    assert bout.method == "Overturned"


def test_no_contest_could_not_continue(bouts):
    bout = bouts["d822ebc7dbc6c259"]
    assert bout.result == "nc" and bout.winner_id is None
    assert bout.method == "Could Not Continue"
    assert bout.fight_time_s == 51


def test_disqualification_is_a_win_for_the_other_fighter(bouts):
    bout = bouts["3a70ed81e5611d24"]
    assert bout.method == "DQ"
    assert bout.result == "win" and bout.winner_id == bout.fighter_b_id
    assert bout.method_details == "Illegal Knee to downed opponent"
    assert bout.time_format == "3 Rnd + OT (5-5-5-5)"
    assert bout.scheduled_rounds == 3


def test_doctor_stoppage(bouts):
    assert bouts["54de03e04f571735"].method == "TKO - Doctor's Stoppage"


def test_tournament_final_is_not_a_title_bout(bouts):
    bout = bouts["3dd2fde7bff79c51"]
    assert bout.is_tournament_final and not bout.is_title_bout
    assert bout.weight_class == "Welterweight"


def test_catch_weight_bout(bouts):
    bout = bouts["2a4b33c38e7b16b0"]
    assert bout.weight_class == "Catch Weight"
    assert not bout.is_womens


def test_womens_catch_weight_bout_is_inferred_from_the_fighters(dataset, bouts):
    bout = bouts["44d58943ac8e47e1"]
    assert bout.bout_type_raw == "Catch Weight Bout"
    assert bout.is_womens
    assert len(issues_for(dataset, "womens_bout_inferred")) == 1
    assert not issues_for(dataset, "mixed_division")


# --- early era -------------------------------------------------------------


def test_early_bout_without_stats(dataset, bouts):
    bout = bouts["a1db4c917777aa79"]
    assert bout.weight_class == "Open Weight"
    assert bout.time_format == "No Time Limit"
    assert bout.scheduled_rounds is None
    assert bout.fight_time_s == 14
    assert not bout.has_stats
    assert stats_for(dataset, bout.bout_id) == []
    assert totals_for(dataset, bout.bout_id) == {}


def test_early_bout_with_stats_has_no_control_time(dataset, bouts):
    bout = bouts["4acab67848e78327"]
    assert bout.has_stats
    rounds = stats_for(dataset, bout.bout_id)
    assert len(rounds) == 2
    assert all(row.stats.ctrl_s is None for row in rounds)
    totals = totals_for(dataset, bout.bout_id)
    assert all(line.ctrl_s is None for line in totals.values())
    assert totals[bout.fighter_a_id].sig_str_landed == 1
    assert totals[bout.fighter_b_id].sig_str_attempted == 4


def test_early_tournament_final_has_no_weight_class(bouts):
    bout = bouts["00835554f95fa911"]
    assert bout.weight_class is None
    assert bout.is_tournament_final and not bout.is_title_bout


def test_superfight_draw_decided_in_overtime(bouts):
    bout = bouts["3932f8e9a74f3d11"]
    assert bout.is_title_bout and bout.weight_class is None
    assert bout.result == "draw" and bout.method == "Other"
    assert bout.time_format == "1 Rnd + OT (30-3)"
    assert bout.scheduled_rounds == 1
    assert (bout.end_round, bout.end_time_s, bout.fight_time_s) == (
        2,
        180,
        30 * 60 + 180,
    )


# --- stats -----------------------------------------------------------------


def test_round_stats_for_a_five_round_fight(dataset):
    rounds = [
        row for row in stats_for(dataset, "30dbf1eceb2e1f8d") if row.fighter_id == GANE
    ]
    assert [row.round for row in rounds] == [1, 2, 3, 4, 5]
    assert [row.stats.sig_str_landed for row in rounds] == [41, 56, 30, 40, 70]
    first = rounds[0].stats
    assert (first.sig_str_landed, first.sig_str_attempted) == (41, 65)
    assert (first.total_str_landed, first.total_str_attempted) == (41, 66)
    assert (first.head_landed, first.head_attempted) == (34, 56)
    assert (first.body_landed, first.leg_landed) == (7, 0)
    assert (first.distance_landed, first.clinch_landed, first.ground_landed) == (
        39,
        2,
        0,
    )
    assert first.ctrl_s == 3
    assert first.knockdowns == 0


def test_fight_totals_are_the_sum_of_the_rounds(dataset):
    totals = totals_for(dataset, "30dbf1eceb2e1f8d")[GANE]
    assert totals.sig_str_landed == 41 + 56 + 30 + 40 + 70
    assert totals.sig_str_attempted == 343
    assert totals.ctrl_s == 4
    for fight in dataset.fight_stats:
        rounds = [
            row.stats
            for row in dataset.round_stats
            if (row.bout_id, row.fighter_id) == (fight.bout_id, fight.fighter_id)
        ]
        assert fight.stats == sum_stat_lines(rounds)


def test_stats_cover_exactly_the_rounds_fought(dataset, bouts):
    for bout in dataset.bouts:
        rounds = stats_for(dataset, bout.bout_id)
        if not bout.has_stats:
            assert rounds == []
            continue
        assert {row.fighter_id for row in rounds} == {
            bout.fighter_a_id,
            bout.fighter_b_id,
        }
        assert sorted({row.round for row in rounds}) == list(
            range(1, bout.end_round + 1)
        )
        assert len(rounds) == 2 * bout.end_round
    assert not issues_for(dataset, "stats_round_mismatch")
    assert not issues_for(dataset, "malformed_stats")


def test_same_pairing_twice_on_one_card_loads_both_bouts_without_stats(dataset, bouts):
    win, no_contest = bouts["ec1bda9a4c2aab42"], bouts["2750ac5854e8b28b"]
    assert (win.result, no_contest.result) == ("win", "nc")
    assert win.event_id == no_contest.event_id
    assert not win.has_stats and not no_contest.has_stats
    flagged = {issue.entity_id for issue in issues_for(dataset, "ambiguous_stats")}
    assert flagged == {win.bout_id, no_contest.bout_id}


# --- fighters --------------------------------------------------------------


def test_fighter_record_holds_physical_attributes_only():
    assert [f.name for f in fields(Fighter)] == [
        "fighter_id",
        "name",
        "nickname",
        "height_cm",
        "reach_cm",
        "stance",
        "dob",
    ]


def test_fighter_with_all_attributes(fighters):
    gane = fighters[GANE]
    assert gane.name == "Ciryl Gane"
    assert gane.nickname == "Bon Gamin"
    assert gane.height_cm == 193.04
    assert gane.reach_cm == 205.74
    assert gane.stance == "Orthodox"
    assert gane.dob == date(1990, 4, 12)


def test_fighter_missing_reach(fighters):
    sakuraba = fighters["9b5b5a75523728f3"]
    assert sakuraba.height_cm == 182.88
    assert sakuraba.reach_cm is None
    assert sakuraba.dob == date(1969, 7, 14)


def test_fighter_missing_date_of_birth(fighters):
    rosier = fighters["598a58db87b890ee"]
    assert rosier.dob is None
    assert rosier.height_cm == 193.04
    assert rosier.stance == "Orthodox"


def test_fighter_missing_everything_but_date_of_birth(fighters):
    aaron = fighters["93fe7332d16c6ad9"]
    assert (aaron.height_cm, aaron.reach_cm, aaron.stance) == (None, None, None)
    assert aaron.dob == date(1978, 7, 13)


def test_fighter_name_whitespace_is_trimmed(fighters):
    assert fighters["2b33e57c4d0a4502"].name == "Joel Thomas"


def test_fighter_present_in_only_one_fighter_file(fighters):
    tran = fighters["5f440fdb94f626c5"]
    assert tran.name == "Kim Tran"
    assert tran.nickname is None
    assert tran.dob == date(1996, 11, 22)


def test_attributes_recorded_under_another_persons_name_are_not_used(dataset, fighters):
    fortune = fighters[TYRELL_FORTUNE]
    assert fortune.name == "Tyrell Fortune"
    assert (fortune.height_cm, fortune.reach_cm, fortune.stance, fortune.dob) == (
        None,
    ) * 4
    flagged = {
        issue.entity_id for issue in issues_for(dataset, "profile_name_conflict")
    }
    assert TYRELL_FORTUNE in flagged
    assert len(flagged) == 3


def test_names_conflict():
    assert names_conflict("Tyrell Fortune", "Regina Malpica")
    assert not names_conflict("Zach Reese", "Zachary Reese")
    assert not names_conflict("Colley Bradford", "ColleyBradford")
    assert not names_conflict("Regezhen", "Gezhen Re")
    assert not names_conflict("Waldo Cortes Acosta", "Waldo Cortes-Acosta")


# --- joining fights to events ----------------------------------------------


def test_fight_listed_under_a_renamed_event_is_loaded_once(dataset, bouts, events):
    assert sum(bout.bout_id == "b395c89e19a3fec4" for bout in dataset.bouts) == 1
    event = events[bouts["b395c89e19a3fec4"].event_id]
    assert event.name == "Noche UFC: Grasso vs. Shevchenko 2"
    [note] = issues_for(dataset, "duplicate_fight_rows")
    assert note.message.startswith("6 fights")


def test_fight_whose_event_is_missing_is_dropped_and_reported(dataset, bouts):
    assert "abf5f4d6f777cf7b" not in bouts
    [issue] = issues_for(dataset, "orphan_event")
    assert issue.severity == "error"
    assert issue.entity_id == "abf5f4d6f777cf7b"
    assert "UFC - Road to UFC 4.6" in issue.message


# --- resolving fighter names to IDs ----------------------------------------


def test_name_listed_differently_in_the_two_fighter_files_resolves(bouts, fighters):
    bout = bouts["d822ebc7dbc6c259"]
    assert bout.fighter_a_id == ZACH_REESE
    assert fighters[ZACH_REESE].name == "Zach Reese"


def test_shared_name_is_settled_by_weight_class(bouts):
    assert bouts["6ecfa439ef900bcc"].fighter_b_id == BRUNO_SILVA_MIDDLEWEIGHT
    assert bouts["20d74ed23d3e9b3a"].fighter_a_id == BRUNO_SILVA_FLYWEIGHT
    assert bouts["de1a3734be60e6a1"].fighter_b_id == JEAN_SILVA_FEATHERWEIGHT


def test_shared_name_resolved_by_override_when_weight_class_cannot(bouts):
    assert bouts["57ff0eb2351979c4"].fighter_b_id == BRUNO_SILVA_FLYWEIGHT
    assert bouts["54de03e04f571735"].fighter_b_id == JEAN_SILVA_FEATHERWEIGHT
    assert bouts["54de03e04f571735"].winner_id == JEAN_SILVA_FEATHERWEIGHT


def test_misspelled_and_shortened_names_resolve_by_override(bouts):
    assert bouts["781a2ea50f5ae68f"].fighter_b_id == RAFAEL_CERQUEIRA
    assert bouts["b6ab5f1be3d3eb9e"].fighter_b_id == NICHOLAS_GALANTI


def test_non_trivial_resolutions_are_recorded(dataset):
    notes = {
        issue.entity_id: issue.message
        for issue in issues_for(dataset, "fighter_resolution")
    }
    assert "by name+weight" in notes["6ecfa439ef900bcc"]
    assert "by override" in notes["57ff0eb2351979c4"]
    assert len(notes) == 7


def test_shared_name_nothing_can_settle_drops_the_bout(dataset, bouts):
    assert "64829a32c78ab656" not in bouts
    [issue] = issues_for(dataset, "unresolved_fighter")
    assert issue.severity == "error"
    assert issue.entity_id == "64829a32c78ab656"
    assert "Jose Montanha" in issue.message


def test_without_overrides_unsettled_names_are_dropped_not_guessed(raw):
    built = build_dataset(raw, overrides=[])
    loaded = {bout.bout_id for bout in built.bouts}
    dropped = {
        issue.entity_id for issue in built.issues if issue.check == "unresolved_fighter"
    }
    assert dropped == {
        "57ff0eb2351979c4",  # Bruno Silva at bantamweight
        "54de03e04f571735",  # Jean Silva at lightweight
        "781a2ea50f5ae68f",  # Rafael Cerquiera (misspelled)
        "b6ab5f1be3d3eb9e",  # Nick Galanti (shortened)
        "64829a32c78ab656",  # Jose Montanha (two heavyweight profiles)
    }
    assert not dropped & loaded
    assert JEAN_SILVA_OTHER not in {
        fighter_id
        for bout in built.bouts
        for fighter_id in (bout.fighter_a_id, bout.fighter_b_id)
    }


def test_every_shipped_override_is_used(dataset):
    assert not issues_for(dataset, "identity_override")


# --- overall ---------------------------------------------------------------


def test_only_the_two_known_drops_are_errors(dataset):
    errors = {
        (issue.check, issue.entity_id)
        for issue in dataset.issues
        if issue.severity == "error"
    }
    assert errors == {
        ("orphan_event", "abf5f4d6f777cf7b"),
        ("unresolved_fighter", "64829a32c78ab656"),
    }


def test_real_stats_pass_the_arithmetic_checks(dataset):
    assert not issues_for(dataset, "stat_sanity")
    assert not issues_for(dataset, "date_sanity")
    assert not issues_for(dataset, "result_method_mismatch")


def test_results_breakdown_is_reported(dataset):
    messages = [issue.message for issue in issues_for(dataset, "results_breakdown")]
    assert messages[0] == "by result: 'win'=20 'nc'=4 'draw'=3"
    assert "'DQ'=1" in messages[1]
