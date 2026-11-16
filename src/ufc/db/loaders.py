"""Idempotent loaders: typed records in, SQLite rows out.

Everything is keyed on UFCStats IDs and written with upserts, so loading the
same snapshot twice leaves every table byte-for-byte the same. No wall-clock
time is written: ``seen_at`` is supplied by the caller and is the snapshot's
own timestamp.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import astuple

from ufc.ingest.models import (
    STAT_COLUMNS,
    Bout,
    Dataset,
    Event,
    Fighter,
    FightStats,
    Issue,
    RoundStats,
    SourceFile,
)

logger = logging.getLogger(__name__)

TABLES = (
    "events",
    "fighters",
    "bouts",
    "bout_round_stats",
    "bout_fighter_stats",
    "source_files",
    "ingest_issues",
)

_BOUT_COLUMNS = (
    "bout_id",
    "event_id",
    "fighter_a_id",
    "fighter_b_id",
    "weight_class",
    "bout_type_raw",
    "is_womens",
    "is_title_bout",
    "is_interim_title",
    "is_tournament_final",
    "scheduled_rounds",
    "time_format",
    "status",
    "result",
    "winner_id",
    "method",
    "method_details",
    "referee",
    "end_round",
    "end_time_s",
    "fight_time_s",
    "has_stats",
)


def _placeholders(count: int) -> str:
    return ", ".join("?" * count)


def upsert_events(
    conn: sqlite3.Connection, events: Iterable[Event], seen_at: str
) -> None:
    conn.executemany(
        """
        INSERT INTO events (event_id, name, date, location, status, first_seen_at, last_seen_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (event_id) DO UPDATE SET
            name = excluded.name,
            date = excluded.date,
            location = excluded.location,
            status = excluded.status,
            first_seen_at = min(first_seen_at, excluded.first_seen_at),
            last_seen_at = max(last_seen_at, excluded.last_seen_at)
        """,
        [
            (
                e.event_id,
                e.name,
                e.date.isoformat(),
                e.location,
                e.status,
                seen_at,
                seen_at,
            )
            for e in events
        ],
    )


def upsert_fighters(conn: sqlite3.Connection, fighters: Iterable[Fighter]) -> None:
    conn.executemany(
        """
        INSERT INTO fighters (fighter_id, name, nickname, height_cm, reach_cm, stance, dob)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (fighter_id) DO UPDATE SET
            name = excluded.name,
            nickname = excluded.nickname,
            height_cm = excluded.height_cm,
            reach_cm = excluded.reach_cm,
            stance = excluded.stance,
            dob = excluded.dob
        """,
        [
            (
                f.fighter_id,
                f.name,
                f.nickname,
                f.height_cm,
                f.reach_cm,
                f.stance,
                f.dob.isoformat() if f.dob else None,
            )
            for f in fighters
        ],
    )


def upsert_bouts(conn: sqlite3.Connection, bouts: Iterable[Bout], seen_at: str) -> None:
    updates = ", ".join(f"{c} = excluded.{c}" for c in _BOUT_COLUMNS if c != "bout_id")
    conn.executemany(
        f"""
        INSERT INTO bouts ({", ".join(_BOUT_COLUMNS)}, first_seen_at, last_seen_at)
        VALUES ({_placeholders(len(_BOUT_COLUMNS) + 2)})
        ON CONFLICT (bout_id) DO UPDATE SET
            {updates},
            first_seen_at = min(first_seen_at, excluded.first_seen_at),
            last_seen_at = max(last_seen_at, excluded.last_seen_at)
        """,
        [(*(getattr(b, c) for c in _BOUT_COLUMNS), seen_at, seen_at) for b in bouts],
    )


def replace_bout_stats(
    conn: sqlite3.Connection,
    bouts: Iterable[Bout],
    round_stats: Iterable[RoundStats],
    fight_stats: Iterable[FightStats],
) -> None:
    """Replace the stats of every given bout, including bouts that now have none."""
    bout_ids = [(b.bout_id,) for b in bouts]
    conn.executemany("DELETE FROM bout_round_stats WHERE bout_id = ?", bout_ids)
    conn.executemany("DELETE FROM bout_fighter_stats WHERE bout_id = ?", bout_ids)
    stat_columns = ", ".join(STAT_COLUMNS)
    conn.executemany(
        f"""
        INSERT INTO bout_round_stats (bout_id, fighter_id, round, {stat_columns})
        VALUES ({_placeholders(3 + len(STAT_COLUMNS))})
        """,
        [(r.bout_id, r.fighter_id, r.round, *astuple(r.stats)) for r in round_stats],
    )
    conn.executemany(
        f"""
        INSERT INTO bout_fighter_stats (bout_id, fighter_id, {stat_columns})
        VALUES ({_placeholders(2 + len(STAT_COLUMNS))})
        """,
        [(s.bout_id, s.fighter_id, *astuple(s.stats)) for s in fight_stats],
    )


def replace_source_files(
    conn: sqlite3.Connection,
    files: Iterable[SourceFile],
    repo: str,
    commit: str,
    snapshot_at: str,
) -> None:
    conn.execute("DELETE FROM source_files")
    conn.executemany(
        "INSERT INTO source_files VALUES (?, ?, ?, ?, ?, ?)",
        [
            (f.file_name, f.sha256, f.row_count, repo, commit, snapshot_at)
            for f in files
        ],
    )


def replace_issues(conn: sqlite3.Connection, issues: Sequence[Issue]) -> None:
    conn.execute("DELETE FROM ingest_issues")
    conn.executemany(
        "INSERT INTO ingest_issues VALUES (?, ?, ?, ?, ?, ?)",
        [
            (seq, i.severity, i.check, i.entity_type, i.entity_id, i.message)
            for seq, i in enumerate(issues, 1)
        ],
    )


def find_stale_bouts(conn: sqlite3.Connection, bouts: Iterable[Bout]) -> list[str]:
    """IDs of completed bouts in the database that the dataset no longer contains."""
    current = {b.bout_id for b in bouts}
    rows = conn.execute(
        "SELECT bout_id FROM bouts WHERE status = 'completed' ORDER BY bout_id"
    )
    return [row[0] for row in rows if row[0] not in current]


def load_dataset(
    conn: sqlite3.Connection,
    dataset: Dataset,
    *,
    seen_at: str,
    files: Iterable[SourceFile] = (),
    repo: str = "",
    commit: str = "",
) -> list[str]:
    """Write a whole dataset in one transaction; all of it lands or none does.

    Returns the IDs of completed bouts already in the database that this
    dataset does not contain. They are left in place and only reported.
    """
    with conn:
        upsert_events(conn, dataset.events, seen_at)
        upsert_fighters(conn, dataset.fighters)
        upsert_bouts(conn, dataset.bouts, seen_at)
        replace_bout_stats(
            conn, dataset.bouts, dataset.round_stats, dataset.fight_stats
        )
        replace_source_files(conn, files, repo, commit, seen_at)
        replace_issues(conn, dataset.issues)
        stale = find_stale_bouts(conn, dataset.bouts)
    if stale:
        logger.warning(
            "%d completed bouts in the database are not in this snapshot and were left as is",
            len(stale),
        )
    return stale


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in TABLES
    }
