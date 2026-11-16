"""Read a snapshot, build and validate the tables, and load them.

The load is gated: if validation finds structural errors, or more bouts were
dropped than the allowed share, nothing is written and the database keeps its
last good contents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ufc.db import loaders
from ufc.db.connection import connect
from ufc.ingest import source
from ufc.ingest.identity import load_overrides
from ufc.ingest.models import Dataset, Issue, SourceFile
from ufc.ingest.transform import build_dataset
from ufc.ingest.validate import validate_dataset

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_DIR = Path("data/external/greco")
DEFAULT_DB_PATH = Path("data/ufc.db")
DEFAULT_MAX_DROPPED_PCT = 1.0


class ValidationFailed(Exception):
    """The snapshot did not pass the gate; nothing was written."""


@dataclass(slots=True)
class Prepared:
    dataset: Dataset
    files: tuple[SourceFile, ...]
    repo: str
    commit: str
    snapshot_at: str


@dataclass(slots=True)
class LoadResult:
    prepared: Prepared
    table_counts: dict[str, int] | None = None
    stale_bouts: list[str] = field(default_factory=list)


def normalise_timestamp(text: str) -> str:
    """Any ISO-8601 instant -> ``YYYY-MM-DDTHH:MM:SSZ`` in UTC, so text order is time order."""
    moment = datetime.fromisoformat(text.strip())
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def prepare(
    source_dir: Path, *, overrides_path: Path | None = None, as_of: str | None = None
) -> Prepared:
    """Read, build and validate a snapshot without touching the database.

    ``as_of`` is the instant the snapshot represents. It defaults to the
    upstream commit date recorded in ``SOURCE.json``.
    """
    raw = source.load_raw_tables(source_dir)
    info = source.read_source_info(source_dir)
    if as_of is None and info is None:
        raise source.SourceError(
            f"{source_dir} has no {source.SOURCE_INFO_FILE}; fetch the snapshot with "
            "'python -m ufc.ingest fetch' or pass --as-of"
        )
    snapshot_at = normalise_timestamp(as_of if as_of is not None else info.commit_date)

    dataset = build_dataset(raw, load_overrides(overrides_path))
    snapshot_date = datetime.fromisoformat(snapshot_at).date()
    dataset.issues.extend(validate_dataset(dataset, snapshot_date))
    return Prepared(
        dataset=dataset,
        files=raw.files,
        repo=info.repo if info else source.UPSTREAM_REPO,
        commit=info.commit if info else "unknown",
        snapshot_at=snapshot_at,
    )


def gate_failures(dataset: Dataset, max_dropped_pct: float) -> list[str]:
    """Reasons this dataset must not be loaded; empty if it may be."""
    reasons: list[str] = []
    structural = [
        i for i in dataset.issues if i.severity == "error" and _is_structural(i)
    ]
    reasons += [f"{issue.check}: {issue.message}" for issue in structural]
    if dataset.source_fights:
        dropped = dataset.source_fights - len(dataset.bouts)
        dropped_pct = 100 * dropped / dataset.source_fights
        if dropped_pct > max_dropped_pct:
            reasons.append(
                f"{dropped} of {dataset.source_fights} fights ({dropped_pct:.2f}%) were dropped, "
                f"above the {max_dropped_pct:.2f}% limit"
            )
    return reasons


def _is_structural(issue: Issue) -> bool:
    """Errors that mean the tables themselves would be inconsistent, not just smaller."""
    return issue.check in {"duplicate_id", "broken_reference", "row_counts"} and (
        issue.entity_type is None or issue.check == "broken_reference"
    )


def run_load(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    overrides_path: Path | None = None,
    as_of: str | None = None,
    dry_run: bool = False,
    max_dropped_pct: float = DEFAULT_MAX_DROPPED_PCT,
) -> LoadResult:
    """Validate a snapshot and, unless ``dry_run``, load it into the database."""
    prepared = prepare(source_dir, overrides_path=overrides_path, as_of=as_of)
    failures = gate_failures(prepared.dataset, max_dropped_pct)
    if failures:
        raise ValidationFailed("; ".join(failures))
    result = LoadResult(prepared)
    if dry_run:
        return result

    conn = connect(db_path)
    try:
        result.stale_bouts = loaders.load_dataset(
            conn,
            prepared.dataset,
            seen_at=prepared.snapshot_at,
            files=prepared.files,
            repo=prepared.repo,
            commit=prepared.commit,
        )
        result.table_counts = loaders.table_counts(conn)
    finally:
        conn.close()
    logger.info("Loaded snapshot %s into %s", prepared.commit, db_path)
    return result
