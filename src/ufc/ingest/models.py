"""Typed records produced by ingestion. These mirror the database tables."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from datetime import date


@dataclass(frozen=True, slots=True)
class Event:
    event_id: str
    name: str
    date: date
    location: str | None
    status: str = "completed"


@dataclass(frozen=True, slots=True)
class Fighter:
    """A fighter profile. Physical attributes only; no career aggregates."""

    fighter_id: str
    name: str
    nickname: str | None
    height_cm: float | None
    reach_cm: float | None
    stance: str | None
    dob: date | None


@dataclass(frozen=True, slots=True)
class Bout:
    bout_id: str
    event_id: str
    fighter_a_id: str
    fighter_b_id: str
    weight_class: str | None
    bout_type_raw: str
    is_womens: bool
    is_title_bout: bool
    is_interim_title: bool
    is_tournament_final: bool
    scheduled_rounds: int | None
    time_format: str | None
    status: str
    result: str | None
    winner_id: str | None
    method: str | None
    method_details: str | None
    referee: str | None
    end_round: int | None
    end_time_s: int | None
    fight_time_s: int | None
    has_stats: bool


@dataclass(frozen=True, slots=True)
class StatLine:
    """One fighter's counting stats over one round or one whole fight.

    ``ctrl_s`` is ``None`` where control time was not recorded (early events).
    """

    knockdowns: int
    sig_str_landed: int
    sig_str_attempted: int
    total_str_landed: int
    total_str_attempted: int
    td_landed: int
    td_attempted: int
    sub_attempts: int
    reversals: int
    ctrl_s: int | None
    head_landed: int
    head_attempted: int
    body_landed: int
    body_attempted: int
    leg_landed: int
    leg_attempted: int
    distance_landed: int
    distance_attempted: int
    clinch_landed: int
    clinch_attempted: int
    ground_landed: int
    ground_attempted: int


STAT_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(StatLine))


def sum_stat_lines(lines: Iterable[StatLine]) -> StatLine:
    """Add per-round lines into a fight total.

    Control time is ``None`` if it is missing for any round, so a partial sum
    is never presented as a fight total.
    """
    lines = list(lines)
    if not lines:
        raise ValueError("cannot total zero stat lines")
    totals: dict[str, int | None] = {}
    for column in STAT_COLUMNS:
        values = [getattr(line, column) for line in lines]
        totals[column] = None if any(v is None for v in values) else sum(values)
    return StatLine(**totals)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class RoundStats:
    bout_id: str
    fighter_id: str
    round: int
    stats: StatLine


@dataclass(frozen=True, slots=True)
class FightStats:
    """Per-fight totals, derived by summing a fighter's rounds."""

    bout_id: str
    fighter_id: str
    stats: StatLine


@dataclass(frozen=True, slots=True)
class Issue:
    """One line of the validation report.

    Severity ``error`` means data was dropped, ``warning`` means it was loaded
    but looks wrong or was altered, ``info`` is a summary or an audit note.
    """

    severity: str
    check: str
    message: str
    entity_type: str | None = None
    entity_id: str | None = None


@dataclass(frozen=True, slots=True)
class SourceFile:
    file_name: str
    sha256: str
    row_count: int


@dataclass(slots=True)
class Dataset:
    """Everything one snapshot of the published dataset yields."""

    events: list[Event] = field(default_factory=list)
    fighters: list[Fighter] = field(default_factory=list)
    bouts: list[Bout] = field(default_factory=list)
    round_stats: list[RoundStats] = field(default_factory=list)
    fight_stats: list[FightStats] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    source_fights: int = 0

    def count_issues(self, severity: str) -> int:
        return sum(1 for issue in self.issues if issue.severity == severity)
