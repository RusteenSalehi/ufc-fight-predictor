"""Shared fixtures: the synthetic snapshot under tests/fixtures/greco.

It is written by scripts/make_synthetic_fixtures.py and contains no real rows."""

from datetime import date
from pathlib import Path

import pytest

from ufc.ingest.identity import load_overrides
from ufc.ingest.models import Dataset
from ufc.ingest.source import RawTables, load_raw_tables
from ufc.ingest.transform import build_dataset
from ufc.ingest.validate import validate_dataset

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "greco"
OVERRIDES_FILE = FIXTURE_DIR / "identity_overrides.csv"
SNAPSHOT_DATE = date(2026, 11, 15)
SNAPSHOT_AT = "2026-11-15T18:03:06Z"


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def raw() -> RawTables:
    return load_raw_tables(FIXTURE_DIR)


@pytest.fixture(scope="session")
def dataset(raw: RawTables) -> Dataset:
    """The fixture snapshot built with the fixture overrides and validated."""
    built = build_dataset(raw, load_overrides(OVERRIDES_FILE))
    built.issues.extend(validate_dataset(built, SNAPSHOT_DATE))
    return built


@pytest.fixture(scope="session")
def bouts(dataset: Dataset) -> dict:
    return {bout.bout_id: bout for bout in dataset.bouts}


@pytest.fixture(scope="session")
def fighters(dataset: Dataset) -> dict:
    return {fighter.fighter_id: fighter for fighter in dataset.fighters}


@pytest.fixture(scope="session")
def events(dataset: Dataset) -> dict:
    return {event.event_id: event for event in dataset.events}
