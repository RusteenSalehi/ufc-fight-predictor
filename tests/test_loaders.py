"""Database loaders: idempotency, upsert behaviour, constraints, atomicity."""

import sqlite3
from dataclasses import replace

import pytest

from tests.conftest import SNAPSHOT_AT
from ufc.db import loaders
from ufc.db.connection import connect
from ufc.ingest.models import STAT_COLUMNS, Dataset

LATER = "2026-11-22T18:00:00Z"
TITLE_FIGHT = "b000000000000001"
VALCOURT = "f100000000000001"


@pytest.fixture
def conn():
    connection = connect(":memory:")
    yield connection
    connection.close()


def load(conn, dataset, seen_at=SNAPSHOT_AT):
    return loaders.load_dataset(
        conn, dataset, seen_at=seen_at, repo="owner/repo", commit="abc123"
    )


def dump(conn) -> dict[str, list[tuple]]:
    """Every row of every table, in primary-key order."""
    return {
        table: [
            tuple(row)
            for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1, 2, 3")
        ]
        for table in loaders.TABLES
    }


def columns(conn, table) -> list[str]:
    return [row["name"] for row in conn.execute(f"PRAGMA table_info({table})")]


def test_loads_every_table(conn, dataset):
    load(conn, dataset)
    counts = loaders.table_counts(conn)
    assert counts["events"] == 19
    assert counts["fighters"] == 55
    assert counts["bouts"] == 27
    assert counts["bout_round_stats"] == len(dataset.round_stats) == 104
    assert counts["bout_fighter_stats"] == len(dataset.fight_stats) == 48
    assert counts["ingest_issues"] == len(dataset.issues)


def test_loading_the_same_data_twice_gives_identical_tables(conn, dataset):
    load(conn, dataset)
    first = dump(conn)
    load(conn, dataset)
    assert dump(conn) == first
    assert all(
        first[table] for table in ("events", "fighters", "bouts", "bout_round_stats")
    )


def test_loading_three_times_into_a_file_database_is_stable(tmp_path, dataset):
    path = tmp_path / "nested" / "ufc.db"
    snapshots = []
    for _ in range(3):
        connection = connect(path)
        load(connection, dataset)
        snapshots.append(dump(connection))
        connection.close()
    assert snapshots[0] == snapshots[1] == snapshots[2]


def test_rows_round_trip(conn, dataset):
    load(conn, dataset)
    bout = conn.execute(
        "SELECT * FROM bouts WHERE bout_id = ?", (TITLE_FIGHT,)
    ).fetchone()
    assert bout["weight_class"] == "Heavyweight"
    assert (bout["is_title_bout"], bout["is_womens"], bout["has_stats"]) == (1, 0, 1)
    assert (bout["result"], bout["winner_id"]) == ("win", VALCOURT)
    assert (bout["end_round"], bout["end_time_s"], bout["fight_time_s"]) == (
        5,
        300,
        1500,
    )
    assert bout["status"] == "completed"

    event = conn.execute(
        "SELECT * FROM events WHERE event_id = ?", (bout["event_id"],)
    ).fetchone()
    assert (event["name"], event["date"]) == (
        "UFC 900: Valcourt vs. Hollin",
        "2026-11-14",
    )

    fighter = conn.execute(
        "SELECT * FROM fighters WHERE fighter_id = ?", (VALCOURT,)
    ).fetchone()
    assert (fighter["height_cm"], fighter["reach_cm"], fighter["dob"]) == (
        193.04,
        205.74,
        "1991-02-03",
    )

    totals = conn.execute(
        "SELECT * FROM bout_fighter_stats WHERE bout_id = ? AND fighter_id = ?",
        (TITLE_FIGHT, VALCOURT),
    ).fetchone()
    assert (
        totals["sig_str_landed"],
        totals["sig_str_attempted"],
        totals["ctrl_s"],
    ) == (170, 291, 4)


def test_stored_totals_equal_the_sum_of_stored_rounds(conn, dataset):
    load(conn, dataset)
    sums = ", ".join(f"sum({column})" for column in STAT_COLUMNS if column != "ctrl_s")
    plain = ", ".join(column for column in STAT_COLUMNS if column != "ctrl_s")
    from_rounds = conn.execute(
        f"SELECT bout_id, fighter_id, {sums} FROM bout_round_stats GROUP BY 1, 2 ORDER BY 1, 2"
    ).fetchall()
    stored = conn.execute(
        f"SELECT bout_id, fighter_id, {plain} FROM bout_fighter_stats ORDER BY 1, 2"
    ).fetchall()
    assert [tuple(row) for row in from_rounds] == [tuple(row) for row in stored]


def test_missing_values_are_null_not_zero(conn, dataset):
    load(conn, dataset)
    no_ctrl = conn.execute(
        "SELECT ctrl_s FROM bout_fighter_stats WHERE bout_id = 'b00000000000001d'"
    ).fetchall()
    assert [row[0] for row in no_ctrl] == [None, None]
    no_stats = conn.execute(
        "SELECT count(*) FROM bout_fighter_stats WHERE bout_id = 'b00000000000001b'"
    ).fetchone()[0]
    assert no_stats == 0
    reach = conn.execute(
        "SELECT reach_cm, dob FROM fighters WHERE fighter_id = 'f100000000000025'"
    ).fetchone()
    assert tuple(reach) == (None, None)


def test_fighters_table_has_no_career_aggregate_columns(conn):
    assert columns(conn, "fighters") == [
        "fighter_id",
        "name",
        "nickname",
        "height_cm",
        "reach_cm",
        "stance",
        "dob",
    ]


def test_stat_tables_match_the_stat_record(conn):
    assert columns(conn, "bout_round_stats") == [
        "bout_id",
        "fighter_id",
        "round",
        *STAT_COLUMNS,
    ]
    assert columns(conn, "bout_fighter_stats") == [
        "bout_id",
        "fighter_id",
        *STAT_COLUMNS,
    ]


def test_reload_keeps_first_seen_and_advances_last_seen(conn, dataset):
    load(conn, dataset, SNAPSHOT_AT)
    load(conn, dataset, LATER)
    for table in ("events", "bouts"):
        seen = conn.execute(
            f"SELECT DISTINCT first_seen_at, last_seen_at FROM {table}"
        ).fetchall()
        assert [tuple(row) for row in seen] == [(SNAPSHOT_AT, LATER)]


def test_loading_an_older_snapshot_afterwards_does_not_move_last_seen_back(
    conn, dataset
):
    load(conn, dataset, LATER)
    load(conn, dataset, SNAPSHOT_AT)
    seen = conn.execute(
        "SELECT DISTINCT first_seen_at, last_seen_at FROM bouts"
    ).fetchall()
    assert [tuple(row) for row in seen] == [(SNAPSHOT_AT, LATER)]


def test_changed_values_are_updated_in_place(conn, dataset):
    load(conn, dataset)
    fighters = [
        replace(f, reach_cm=200.0) if f.fighter_id == VALCOURT else f
        for f in dataset.fighters
    ]
    bouts = [
        replace(b, referee="Someone Else") if b.bout_id == TITLE_FIGHT else b
        for b in dataset.bouts
    ]
    load(conn, replace_dataset(dataset, fighters=fighters, bouts=bouts))
    assert (
        conn.execute(
            "SELECT reach_cm FROM fighters WHERE fighter_id = ?", (VALCOURT,)
        ).fetchone()[0]
        == 200.0
    )
    assert (
        conn.execute(
            "SELECT referee FROM bouts WHERE bout_id = ?", (TITLE_FIGHT,)
        ).fetchone()[0]
        == "Someone Else"
    )
    assert loaders.table_counts(conn)["fighters"] == 55
    assert loaders.table_counts(conn)["bouts"] == 27


def test_stats_are_replaced_when_a_bout_loses_them(conn, dataset):
    load(conn, dataset)
    bouts = [
        replace(b, has_stats=False) if b.bout_id == TITLE_FIGHT else b
        for b in dataset.bouts
    ]
    without = replace_dataset(
        dataset,
        bouts=bouts,
        round_stats=[r for r in dataset.round_stats if r.bout_id != TITLE_FIGHT],
        fight_stats=[s for s in dataset.fight_stats if s.bout_id != TITLE_FIGHT],
    )
    load(conn, without)
    for table in ("bout_round_stats", "bout_fighter_stats"):
        remaining = conn.execute(
            f"SELECT count(*) FROM {table} WHERE bout_id = ?", (TITLE_FIGHT,)
        )
        assert remaining.fetchone()[0] == 0
    assert loaders.table_counts(conn)["bout_round_stats"] == 104 - 10


def test_issues_and_source_files_are_replaced_not_appended(conn, dataset, raw):
    for _ in range(2):
        loaders.load_dataset(
            conn,
            dataset,
            seen_at=SNAPSHOT_AT,
            files=raw.files,
            repo="owner/repo",
            commit="abc123",
        )
    counts = loaders.table_counts(conn)
    assert counts["ingest_issues"] == len(dataset.issues)
    assert counts["source_files"] == 6
    row = conn.execute(
        "SELECT * FROM source_files WHERE file_name = 'ufc_fight_stats.csv'"
    ).fetchone()
    assert (row["row_count"], row["upstream_repo"], row["upstream_commit"]) == (
        144,
        "owner/repo",
        "abc123",
    )
    assert row["snapshot_at"] == SNAPSHOT_AT
    severities = conn.execute(
        "SELECT severity, count(*) FROM ingest_issues GROUP BY 1"
    ).fetchall()
    assert dict(map(tuple, severities))["error"] == 2


def test_a_failed_load_writes_nothing(conn, dataset):
    load(conn, dataset)
    before = dump(conn)
    broken_bout = replace(dataset.bouts[0], bout_id="f" * 16, fighter_a_id="0" * 16)
    broken = replace_dataset(dataset, bouts=[*dataset.bouts, broken_bout], events=[])
    with pytest.raises(sqlite3.IntegrityError):
        load(conn, broken, LATER)
    assert dump(conn) == before


def test_schema_rejects_inconsistent_results(conn, dataset):
    load(conn, dataset)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE bouts SET winner_id = NULL WHERE bout_id = ?", (TITLE_FIGHT,)
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE bouts SET result = 'loss' WHERE bout_id = ?", (TITLE_FIGHT,)
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE bouts SET winner_id = ? WHERE bout_id = ?",
            ("f100000000000023", TITLE_FIGHT),
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM fighters WHERE fighter_id = ?", (VALCOURT,))


def test_bouts_missing_from_a_later_snapshot_are_reported_and_kept(conn, dataset):
    load(conn, dataset)
    smaller = replace_dataset(
        dataset, bouts=[b for b in dataset.bouts if b.bout_id != TITLE_FIGHT]
    )
    smaller.round_stats = [r for r in dataset.round_stats if r.bout_id != TITLE_FIGHT]
    smaller.fight_stats = [s for s in dataset.fight_stats if s.bout_id != TITLE_FIGHT]
    assert load(conn, smaller) == [TITLE_FIGHT]
    assert loaders.table_counts(conn)["bouts"] == 27


def test_schema_can_be_applied_to_an_existing_database(tmp_path, dataset):
    path = tmp_path / "ufc.db"
    first = connect(path)
    load(first, dataset)
    expected = dump(first)
    first.close()
    second = connect(path)
    assert dump(second) == expected
    second.close()


def replace_dataset(dataset: Dataset, **changes) -> Dataset:
    base = {
        "events": dataset.events,
        "fighters": dataset.fighters,
        "bouts": dataset.bouts,
        "round_stats": dataset.round_stats,
        "fight_stats": dataset.fight_stats,
        "issues": dataset.issues,
        "source_fights": dataset.source_fights,
    }
    return Dataset(**{**base, **changes})
