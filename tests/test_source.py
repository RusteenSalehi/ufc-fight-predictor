"""Reading snapshot files: format checks and provenance."""

import json
import shutil

import pytest

from ufc.ingest import source


def copy_snapshot(fixture_dir, tmp_path):
    target = tmp_path / "snapshot"
    shutil.copytree(fixture_dir, target)
    return target


def test_load_raw_tables_reads_every_file_as_strings(raw):
    assert len(raw.events) == 19
    assert len(raw.fight_results) == 35
    assert len(raw.fight_stats) == 144
    assert raw.events[0]["EVENT"] == "UFC 900: Valcourt vs. Hollin"
    assert all(
        isinstance(value, str) for row in raw.fight_stats for value in row.values()
    )


def test_load_raw_tables_records_hash_and_row_count_per_file(raw, fixture_dir):
    by_name = {f.file_name: f for f in raw.files}
    assert set(by_name) == set(source.EXPECTED_COLUMNS)
    events = by_name[source.EVENTS_FILE]
    assert events.row_count == 19
    assert events.sha256 == source.sha256_of(fixture_dir / source.EVENTS_FILE)
    assert len(events.sha256) == 64


def test_missing_file_is_an_error(fixture_dir, tmp_path):
    snapshot = copy_snapshot(fixture_dir, tmp_path)
    (snapshot / source.FIGHT_STATS_FILE).unlink()
    with pytest.raises(source.SourceError, match="missing dataset file"):
        source.load_raw_tables(snapshot)


def test_changed_columns_are_an_error(fixture_dir, tmp_path):
    snapshot = copy_snapshot(fixture_dir, tmp_path)
    path = snapshot / source.EVENTS_FILE
    path.write_text(
        path.read_text("utf-8").replace("EVENT,URL", "NAME,URL", 1), "utf-8"
    )
    with pytest.raises(source.SourceError, match="expected columns"):
        source.load_raw_tables(snapshot)


def test_file_with_no_rows_is_an_error(fixture_dir, tmp_path):
    snapshot = copy_snapshot(fixture_dir, tmp_path)
    (snapshot / source.EVENTS_FILE).write_text("EVENT,URL,DATE,LOCATION\n", "utf-8")
    with pytest.raises(source.SourceError, match="no data rows"):
        source.load_raw_tables(snapshot)


@pytest.mark.parametrize("line", ["only,three,cells", "one,two,three,four,five"])
def test_row_with_wrong_cell_count_is_an_error(fixture_dir, tmp_path, line):
    snapshot = copy_snapshot(fixture_dir, tmp_path)
    with (snapshot / source.EVENTS_FILE).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    with pytest.raises(source.SourceError, match="wrong cell count"):
        source.load_raw_tables(snapshot)


def test_read_source_info(fixture_dir):
    info = source.read_source_info(fixture_dir)
    assert info.repo == "Greco1899/scrape_ufc_stats"
    assert len(info.commit) == 40
    assert info.commit_date == "2026-11-15T18:03:06Z"


def test_read_source_info_absent(tmp_path):
    assert source.read_source_info(tmp_path) is None


def test_read_source_info_malformed(tmp_path):
    (tmp_path / source.SOURCE_INFO_FILE).write_text(json.dumps({"repo": "x"}), "utf-8")
    with pytest.raises(source.SourceError):
        source.read_source_info(tmp_path)
