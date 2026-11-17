"""End to end: snapshot directory -> validation gate -> database, and the CLI around it."""

import shutil
import sqlite3

import pytest

from tests.conftest import OVERRIDES_FILE, SNAPSHOT_AT
from ufc.db import loaders
from ufc.ingest import pipeline, source
from ufc.ingest.__main__ import format_report, main

# The fixture snapshot deliberately contains 2 undroppable fights out of 29 (6.9%),
# so loads of it need a limit above the 1% default.
LOOSE = 10.0


def dump(path) -> dict[str, list[tuple]]:
    conn = sqlite3.connect(path)
    try:
        return {
            table: conn.execute(f"SELECT * FROM {table} ORDER BY 1, 2, 3").fetchall()
            for table in loaders.TABLES
        }
    finally:
        conn.close()


def test_run_load_writes_the_database(fixture_dir, tmp_path):
    db_path = tmp_path / "ufc.db"
    result = pipeline.run_load(
        fixture_dir, db_path, max_dropped_pct=LOOSE, overrides_path=OVERRIDES_FILE
    )
    assert result.table_counts == {
        "events": 19,
        "fighters": 55,
        "bouts": 27,
        "bout_round_stats": 104,
        "bout_fighter_stats": 48,
        "source_files": 6,
        "ingest_issues": len(result.prepared.dataset.issues),
    }
    assert result.stale_bouts == []
    assert result.prepared.snapshot_at == SNAPSHOT_AT


def test_running_the_whole_load_twice_gives_identical_tables(fixture_dir, tmp_path):
    db_path = tmp_path / "ufc.db"
    pipeline.run_load(
        fixture_dir, db_path, max_dropped_pct=LOOSE, overrides_path=OVERRIDES_FILE
    )
    first = dump(db_path)
    pipeline.run_load(
        fixture_dir, db_path, max_dropped_pct=LOOSE, overrides_path=OVERRIDES_FILE
    )
    assert dump(db_path) == first


def test_snapshot_provenance_is_recorded(fixture_dir, tmp_path):
    db_path = tmp_path / "ufc.db"
    pipeline.run_load(
        fixture_dir, db_path, max_dropped_pct=LOOSE, overrides_path=OVERRIDES_FILE
    )
    rows = dump(db_path)["source_files"]
    assert {row[0] for row in rows} == set(source.EXPECTED_COLUMNS)
    assert {row[3:] for row in rows} == {
        (
            "Greco1899/scrape_ufc_stats",
            "0123456789abcdef0123456789abcdef01234567",
            SNAPSHOT_AT,
        )
    }


def test_dry_run_writes_nothing(fixture_dir, tmp_path):
    db_path = tmp_path / "ufc.db"
    result = pipeline.run_load(
        fixture_dir,
        db_path,
        dry_run=True,
        max_dropped_pct=LOOSE,
        overrides_path=OVERRIDES_FILE,
    )
    assert result.table_counts is None
    assert len(result.prepared.dataset.bouts) == 27
    assert not db_path.exists()


def test_gate_refuses_to_load_when_too_many_fights_are_dropped(fixture_dir, tmp_path):
    db_path = tmp_path / "ufc.db"
    with pytest.raises(pipeline.ValidationFailed, match="2 of 29 fights"):
        pipeline.run_load(fixture_dir, db_path, overrides_path=OVERRIDES_FILE)
    assert not db_path.exists()


def test_failed_gate_leaves_the_last_good_load_untouched(fixture_dir, tmp_path):
    db_path = tmp_path / "ufc.db"
    pipeline.run_load(
        fixture_dir, db_path, max_dropped_pct=LOOSE, overrides_path=OVERRIDES_FILE
    )
    good = dump(db_path)

    broken = tmp_path / "broken"
    shutil.copytree(fixture_dir, broken)
    results = broken / source.FIGHT_RESULTS_FILE
    results.write_text(results.read_text("utf-8").replace(",W/L,", ",WIN,"), "utf-8")
    with pytest.raises(pipeline.ValidationFailed):
        pipeline.run_load(
            broken, db_path, max_dropped_pct=LOOSE, overrides_path=OVERRIDES_FILE
        )
    assert dump(db_path) == good


def test_gate_failures_lists_structural_errors(dataset):
    from dataclasses import replace

    from ufc.ingest.validate import validate_dataset

    doubled = replace(dataset, bouts=[*dataset.bouts, dataset.bouts[0]], issues=[])
    doubled.issues = validate_dataset(doubled, pipeline.datetime(2026, 11, 15).date())
    reasons = pipeline.gate_failures(doubled, max_dropped_pct=100)
    assert any(reason.startswith("duplicate_id") for reason in reasons)
    assert pipeline.gate_failures(dataset, max_dropped_pct=LOOSE) == []


def test_snapshot_without_provenance_needs_an_explicit_as_of(fixture_dir, tmp_path):
    bare = tmp_path / "bare"
    shutil.copytree(fixture_dir, bare)
    (bare / source.SOURCE_INFO_FILE).unlink()
    with pytest.raises(source.SourceError, match="--as-of"):
        pipeline.prepare(bare, overrides_path=OVERRIDES_FILE)
    prepared = pipeline.prepare(
        bare, overrides_path=OVERRIDES_FILE, as_of="2026-11-15T13:03:06-05:00"
    )
    assert prepared.snapshot_at == SNAPSHOT_AT
    assert prepared.commit == "unknown"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-11-15T18:03:06Z", "2026-11-15T18:03:06Z"),
        ("2026-11-15T18:03:06+00:00", "2026-11-15T18:03:06Z"),
        ("2026-11-15T19:03:06+01:00", "2026-11-15T18:03:06Z"),
        ("2026-11-15 18:03:06", "2026-11-15T18:03:06Z"),
        ("2026-11-15", "2026-11-15T00:00:00Z"),
    ],
)
def test_normalise_timestamp(text, expected):
    assert pipeline.normalise_timestamp(text) == expected


def test_overrides_file_can_be_replaced(fixture_dir, tmp_path):
    empty = tmp_path / "overrides.csv"
    empty.write_text("name,fight_id,fighter_id,note\n", "utf-8")
    prepared = pipeline.prepare(fixture_dir, overrides_path=empty)
    assert len(prepared.dataset.bouts) == 23


def test_cli_load_prints_report_and_row_counts(fixture_dir, tmp_path, capsys):
    db_path = tmp_path / "ufc.db"
    code = main(
        [
            "load",
            "--source",
            str(fixture_dir),
            "--db",
            str(db_path),
            "--overrides",
            str(OVERRIDES_FILE),
            "--max-dropped-pct",
            "10",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Greco1899/scrape_ufc_stats@0123456789abcdef0123456789abcdef01234567" in out
    assert "Validation: 2 errors, 6 warnings, 17 notes" in out
    assert "ERROR   orphan_event [bout b000000000000013]" in out
    assert "7 fighter names were resolved by override or weight class" in out
    assert "bouts                     27" in out
    assert db_path.exists()


def test_cli_dry_run(fixture_dir, tmp_path, capsys):
    db_path = tmp_path / "ufc.db"
    code = main(
        [
            "load",
            "--source",
            str(fixture_dir),
            "--db",
            str(db_path),
            "--overrides",
            str(OVERRIDES_FILE),
            "--max-dropped-pct",
            "10",
            "--dry-run",
        ]
    )
    assert code == 0
    assert "Dry run: nothing written." in capsys.readouterr().out
    assert not db_path.exists()


def test_cli_reports_a_failed_gate_with_exit_code_2(fixture_dir, tmp_path, capsys):
    db_path = tmp_path / "ufc.db"
    code = main(
        [
            "load",
            "--source",
            str(fixture_dir),
            "--db",
            str(db_path),
            "--overrides",
            str(OVERRIDES_FILE),
        ]
    )
    assert code == 2
    assert "database not changed" in capsys.readouterr().err
    assert not db_path.exists()


def test_cli_reports_an_unreadable_snapshot_with_exit_code_1(tmp_path, capsys):
    code = main(["load", "--source", str(tmp_path), "--db", str(tmp_path / "ufc.db")])
    assert code == 1
    assert "Cannot read snapshot" in capsys.readouterr().err


def test_report_lists_audit_notes_only_when_verbose(dataset):
    brief = format_report(dataset.issues)
    full = format_report(dataset.issues, verbose=True)
    assert "resolved to" not in brief
    assert full.count("resolved to") == 7
    assert brief.splitlines()[1].lstrip().startswith("ERROR")
