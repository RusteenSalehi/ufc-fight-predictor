"""The data quality report, run on the synthetic fixture loaded into SQLite."""

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

from tests.conftest import SNAPSHOT_AT
from ufc.db import loaders
from ufc.db.connection import connect

SCRIPT = Path(__file__).parents[1] / "scripts" / "data_quality.py"
TODAY = date(2026, 11, 16)


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("data_quality", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules["data_quality"] = loaded
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture(scope="module")
def conn(dataset):
    connection = connect(":memory:")
    loaders.load_dataset(
        connection, dataset, seen_at=SNAPSHOT_AT, repo="owner/repo", commit="abc123"
    )
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def report(module, conn):
    return module.build_report(conn, TODAY, "2024-01-01", "2025-11-22")


def test_freshness_compares_the_latest_event_with_today(report):
    assert "- Most recent event in the database: 2026-11-14" in report
    assert "- Days from most recent event to today: 2" in report


def test_attribute_counts_match_the_loaded_fighters(report, dataset):
    fighters = dataset.fighters
    n = len(fighters)
    cells = []
    for attribute in ("height_cm", "reach_cm", "stance", "dob"):
        missing = sum(getattr(f, attribute) is None for f in fighters)
        cells += [str(missing), f"{100 * missing / n:.1f}%"]
    assert f"| all profiles | {n} | " + " | ".join(cells) + " |" in report


def test_active_groups_partition_by_debut_date(report, conn):
    rows = {
        line.split("|")[1].strip(): int(line.split("|")[2].replace(",", ""))
        for line in report.splitlines()
        if line.startswith("| active")
    }
    since = rows["active: a bout since 2024-01-01"]
    before = rows["active since 2024-01-01, debut before 2025-11-22"]
    after = rows["active since 2024-01-01, debut on or after 2025-11-22"]
    assert since == before + after
    # In the fixture, the 2026 cards hold fighters who debuted after 2025-11-22.
    assert after > 0


def test_per_year_table_counts_events_and_bouts(report):
    # Nine 2026 events; 15 fights on them, one dropped because its fighter is ambiguous.
    assert "| 2026 | 9 | 14 |" in report


def test_issues_are_grouped_by_check(report):
    assert "## ingest_issues, grouped by check" in report
    assert "| orphan_event | error | 1 |" in report
    assert "| unresolved_fighter | error | 1 |" in report


def test_duplicate_checks_find_the_repeated_pairing(report):
    assert "| fighter pairs meeting twice at one event | 1 |" in report


def test_writing_replaces_only_the_generated_block(module, tmp_path):
    path = tmp_path / "report.md"
    path.write_text(
        f"intro\n\n{module.BEGIN}\nold\n{module.END}\n\nhand-written review\n",
        encoding="utf-8",
    )
    module.write_block(path, "new report\n")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("intro\n")
    assert "new report" in text and "old" not in text
    assert text.rstrip().endswith("hand-written review")
    module.write_block(path, "new report\n")
    assert path.read_text(encoding="utf-8") == text
