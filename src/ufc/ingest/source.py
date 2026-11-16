"""Read a snapshot of the published dataset from disk, as raw strings.

Nothing is interpreted here. Files are checked only for being present,
non-empty and having exactly the columns the upstream project documents, so a
changed upstream format stops the load instead of producing wrong tables.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ufc.ingest.models import SourceFile

UPSTREAM_REPO = "Greco1899/scrape_ufc_stats"
SOURCE_INFO_FILE = "SOURCE.json"

EVENTS_FILE = "ufc_event_details.csv"
FIGHT_DETAILS_FILE = "ufc_fight_details.csv"
FIGHT_RESULTS_FILE = "ufc_fight_results.csv"
FIGHT_STATS_FILE = "ufc_fight_stats.csv"
FIGHTER_DETAILS_FILE = "ufc_fighter_details.csv"
FIGHTER_TOTT_FILE = "ufc_fighter_tott.csv"

EXPECTED_COLUMNS: dict[str, tuple[str, ...]] = {
    EVENTS_FILE: ("EVENT", "URL", "DATE", "LOCATION"),
    FIGHT_DETAILS_FILE: ("EVENT", "BOUT", "URL"),
    FIGHT_RESULTS_FILE: (
        "EVENT",
        "BOUT",
        "OUTCOME",
        "WEIGHTCLASS",
        "METHOD",
        "ROUND",
        "TIME",
        "TIME FORMAT",
        "REFEREE",
        "DETAILS",
        "URL",
    ),
    FIGHT_STATS_FILE: (
        "EVENT",
        "BOUT",
        "ROUND",
        "FIGHTER",
        "KD",
        "SIG.STR.",
        "SIG.STR. %",
        "TOTAL STR.",
        "TD",
        "TD %",
        "SUB.ATT",
        "REV.",
        "CTRL",
        "HEAD",
        "BODY",
        "LEG",
        "DISTANCE",
        "CLINCH",
        "GROUND",
    ),
    FIGHTER_DETAILS_FILE: ("FIRST", "LAST", "NICKNAME", "URL"),
    FIGHTER_TOTT_FILE: ("FIGHTER", "HEIGHT", "WEIGHT", "REACH", "STANCE", "DOB", "URL"),
}

Row = dict[str, str]


class SourceError(Exception):
    """The snapshot on disk is missing, empty, or not in the expected format."""


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Provenance of a snapshot, written next to the CSVs by the fetch command."""

    repo: str
    commit: str
    commit_date: str
    fetched_at: str | None = None


@dataclass(frozen=True, slots=True)
class RawTables:
    events: list[Row]
    fight_details: list[Row]
    fight_results: list[Row]
    fight_stats: list[Row]
    fighter_details: list[Row]
    fighter_tott: list[Row]
    files: tuple[SourceFile, ...]


def check_header(file_name: str, header: list[str] | None) -> None:
    """Raise unless ``header`` is exactly the documented column list."""
    expected = EXPECTED_COLUMNS[file_name]
    if header is None or tuple(header) != expected:
        raise SourceError(
            f"{file_name}: expected columns {list(expected)}, found {header}"
        )


def read_table(path: Path) -> list[Row]:
    """Read one CSV as a list of ``{column: cell}`` dicts of strings."""
    if not path.is_file():
        raise SourceError(f"missing dataset file: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        check_header(path.name, reader.fieldnames)
        rows: list[Row] = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise SourceError(
                    f"{path.name}: line {reader.line_num} has the wrong cell count"
                )
            rows.append(row)
    if not rows:
        raise SourceError(f"{path.name}: no data rows")
    return rows


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_tables(directory: Path) -> RawTables:
    """Read all six CSVs from ``directory``."""
    tables: dict[str, list[Row]] = {}
    files: list[SourceFile] = []
    for file_name in EXPECTED_COLUMNS:
        path = directory / file_name
        tables[file_name] = read_table(path)
        files.append(SourceFile(file_name, sha256_of(path), len(tables[file_name])))
    return RawTables(
        events=tables[EVENTS_FILE],
        fight_details=tables[FIGHT_DETAILS_FILE],
        fight_results=tables[FIGHT_RESULTS_FILE],
        fight_stats=tables[FIGHT_STATS_FILE],
        fighter_details=tables[FIGHTER_DETAILS_FILE],
        fighter_tott=tables[FIGHTER_TOTT_FILE],
        files=tuple(files),
    )


def read_source_info(directory: Path) -> SourceInfo | None:
    """Read ``SOURCE.json`` if the snapshot has one."""
    path = directory / SOURCE_INFO_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return SourceInfo(
            repo=data["repo"],
            commit=data["commit"],
            commit_date=data["commit_date"],
            fetched_at=data.get("fetched_at"),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SourceError(f"{path}: not a valid snapshot description") from exc
