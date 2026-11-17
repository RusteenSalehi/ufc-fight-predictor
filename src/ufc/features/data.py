"""Read completed bouts and fighter attributes from SQLite into plain records.

The feature code works on these records only, so it can be tested without a
database and never sees a table it should not (there is no career-aggregate
table to see; see docs/ML_REQUIREMENTS.md, leakage rule 2).
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

StatMap = Mapping[str, int | None]
"""One stat line: column name to value, as stored in the stats tables."""

_STAT_KEYS = frozenset({"bout_id", "fighter_id", "round"})


@dataclass(frozen=True, slots=True)
class FighterAttrs:
    """Static physical attributes (leakage rule 3: assumed not to change)."""

    fighter_id: str
    height_cm: float | None = None
    reach_cm: float | None = None
    stance: str | None = None
    dob: date | None = None


@dataclass(frozen=True, slots=True)
class BoutFacts:
    """Everything known about one completed bout.

    ``totals`` maps each fighter to his per-fight stat line and ``rounds`` to
    his per-round lines in round order. Both are empty when ``has_stats`` is
    false. ``fighter_a_id`` / ``fighter_b_id`` keep the source's order, which
    is never used as a feature.
    """

    bout_id: str
    event_id: str
    date: date
    fighter_a_id: str
    fighter_b_id: str
    weight_class: str | None
    is_womens: bool
    is_title_bout: bool
    scheduled_rounds: int | None
    time_format: str | None
    result: str
    winner_id: str | None
    method: str | None
    end_round: int | None
    end_time_s: int | None
    fight_time_s: int | None
    has_stats: bool
    totals: Mapping[str, StatMap] = field(default_factory=dict)
    rounds: Mapping[str, Sequence[StatMap]] = field(default_factory=dict)

    def opponent_of(self, fighter_id: str) -> str:
        if fighter_id == self.fighter_a_id:
            return self.fighter_b_id
        if fighter_id == self.fighter_b_id:
            return self.fighter_a_id
        raise ValueError(f"{fighter_id} is not in bout {self.bout_id}")


@dataclass(frozen=True, slots=True)
class FeatureData:
    """Completed bouts in chronological order, and every fighter profile."""

    bouts: Sequence[BoutFacts]
    fighters: Mapping[str, FighterAttrs]
    source_commit: str | None = None


def sort_key(bout: BoutFacts) -> tuple[date, str, str]:
    """Processing order. Only the date matters for correctness; the rest
    makes the order, and so the output, deterministic."""
    return (bout.date, bout.event_id, bout.bout_id)


def _stat_map(row: sqlite3.Row) -> dict[str, int | None]:
    # Iterating a sqlite3.Row yields values, so .keys() is needed here.
    return {key: row[key] for key in row.keys() if key not in _STAT_KEYS}  # noqa: SIM118


def load_feature_data(conn: sqlite3.Connection) -> FeatureData:
    """Load every completed bout with its stats, and all fighter profiles."""
    previous_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        totals: dict[str, dict[str, StatMap]] = defaultdict(dict)
        for row in conn.execute("SELECT * FROM bout_fighter_stats"):
            totals[row["bout_id"]][row["fighter_id"]] = _stat_map(row)

        rounds: dict[str, dict[str, list[StatMap]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in conn.execute(
            "SELECT * FROM bout_round_stats ORDER BY bout_id, fighter_id, round"
        ):
            rounds[row["bout_id"]][row["fighter_id"]].append(_stat_map(row))

        bouts = [
            BoutFacts(
                bout_id=row["bout_id"],
                event_id=row["event_id"],
                date=date.fromisoformat(row["date"]),
                fighter_a_id=row["fighter_a_id"],
                fighter_b_id=row["fighter_b_id"],
                weight_class=row["weight_class"],
                is_womens=bool(row["is_womens"]),
                is_title_bout=bool(row["is_title_bout"]),
                scheduled_rounds=row["scheduled_rounds"],
                time_format=row["time_format"],
                result=row["result"],
                winner_id=row["winner_id"],
                method=row["method"],
                end_round=row["end_round"],
                end_time_s=row["end_time_s"],
                fight_time_s=row["fight_time_s"],
                has_stats=bool(row["has_stats"]),
                totals=dict(totals.get(row["bout_id"], {})),
                rounds={
                    fighter_id: tuple(lines)
                    for fighter_id, lines in rounds.get(row["bout_id"], {}).items()
                },
            )
            for row in conn.execute(
                """
                SELECT b.*, e.date AS date
                FROM bouts b JOIN events e USING (event_id)
                WHERE b.status = 'completed'
                ORDER BY e.date, b.event_id, b.bout_id
                """
            )
        ]

        fighters = {
            row["fighter_id"]: FighterAttrs(
                fighter_id=row["fighter_id"],
                height_cm=row["height_cm"],
                reach_cm=row["reach_cm"],
                stance=row["stance"],
                dob=date.fromisoformat(row["dob"]) if row["dob"] else None,
            )
            for row in conn.execute("SELECT * FROM fighters")
        }

        commit_row = conn.execute(
            "SELECT upstream_commit FROM source_files ORDER BY file_name LIMIT 1"
        ).fetchone()
    finally:
        conn.row_factory = previous_factory

    return FeatureData(
        bouts=bouts,
        fighters=fighters,
        source_commit=commit_row[0] if commit_row and commit_row[0] else None,
    )
