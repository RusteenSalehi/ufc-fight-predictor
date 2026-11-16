"""Command line for dataset ingestion.

uv run python -m ufc.ingest fetch            download the latest snapshot
uv run python -m ufc.ingest load             validate it and load data/ufc.db
uv run python -m ufc.ingest load --dry-run   validate only
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from ufc.ingest import pipeline
from ufc.ingest.fetch import FetchError, fetch_snapshot
from ufc.ingest.models import Issue
from ufc.ingest.source import SourceError

_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}
_AUDIT_CHECKS = frozenset({"fighter_resolution"})


def format_report(issues: list[Issue], verbose: bool = False) -> str:
    """Render the validation report, most severe first.

    Per-bout audit notes (which rule resolved a shared name) are counted, and
    listed only with ``verbose``.
    """
    counts = Counter(issue.severity for issue in issues)
    summary = (
        f"Validation: {counts['error']} errors, {counts['warning']} warnings, "
        f"{counts['info']} notes"
    )
    lines = [summary]
    audit = [i for i in issues if i.check in _AUDIT_CHECKS]
    shown = issues if verbose else [i for i in issues if i.check not in _AUDIT_CHECKS]
    for issue in sorted(shown, key=lambda i: _SEVERITY_ORDER[i.severity]):
        target = f" [{issue.entity_type} {issue.entity_id}]" if issue.entity_id else ""
        lines.append(
            f"  {issue.severity.upper():7} {issue.check}{target}: {issue.message}"
        )
    if audit and not verbose:
        lines.append(
            f"  INFO    {len(audit)} fighter names were resolved by override or weight class "
            "(use --verbose to list them)"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ufc.ingest", description=__doc__.split("\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser(
        "fetch", help="download a snapshot of the published dataset"
    )
    fetch.add_argument(
        "--ref", default="main", help="branch or commit SHA (default: main)"
    )
    fetch.add_argument("--dest", type=Path, default=pipeline.DEFAULT_SOURCE_DIR)

    load = commands.add_parser(
        "load", help="validate a snapshot and load it into SQLite"
    )
    load.add_argument("--source", type=Path, default=pipeline.DEFAULT_SOURCE_DIR)
    load.add_argument("--db", type=Path, default=pipeline.DEFAULT_DB_PATH)
    load.add_argument(
        "--overrides", type=Path, help="identity overrides CSV (default: built in)"
    )
    load.add_argument(
        "--as-of", help="snapshot instant, ISO-8601 (default: upstream commit date)"
    )
    load.add_argument(
        "--dry-run", action="store_true", help="validate only; write nothing"
    )
    load.add_argument(
        "--max-dropped-pct",
        type=float,
        default=pipeline.DEFAULT_MAX_DROPPED_PCT,
        help="refuse to load if more than this share of fights is dropped (default: %(default)s)",
    )
    load.add_argument("--verbose", action="store_true", help="list every audit note")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)

    if args.command == "fetch":
        try:
            info = fetch_snapshot(args.dest, args.ref)
        except FetchError as exc:
            print(f"Fetch failed: {exc}", file=sys.stderr)
            return 1
        print(f"Snapshot {info.commit} ({info.commit_date}) saved to {args.dest}")
        return 0

    try:
        result = pipeline.run_load(
            args.source,
            args.db,
            overrides_path=args.overrides,
            as_of=args.as_of,
            dry_run=args.dry_run,
            max_dropped_pct=args.max_dropped_pct,
        )
    except SourceError as exc:
        print(f"Cannot read snapshot: {exc}", file=sys.stderr)
        return 1
    except pipeline.ValidationFailed as exc:
        print(f"Validation failed; database not changed: {exc}", file=sys.stderr)
        return 2

    prepared = result.prepared
    print(f"Snapshot: {prepared.repo}@{prepared.commit} as of {prepared.snapshot_at}")
    print(format_report(prepared.dataset.issues, args.verbose))
    if result.table_counts is None:
        print("Dry run: nothing written.")
        return 0
    print(f"Loaded into {args.db}:")
    for table, count in result.table_counts.items():
        print(f"  {table:20} {count:>7}")
    if result.stale_bouts:
        print(
            f"Note: {len(result.stale_bouts)} completed bouts already in the database are not in "
            "this snapshot and were left unchanged."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
