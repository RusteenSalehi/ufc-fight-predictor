"""Command line for upcoming cards and predictions.

uv run python -m ufc.predict refresh   read cards from Wikipedia, predict with the
                                       approved model, write data/publish/predictions.json
uv run python -m ufc.predict show      print stored predictions (read-only)
uv run python -m ufc.predict matchup   predict a hypothetical bout (read-only, not stored)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from ufc.db.connection import connect
from ufc.ingest import source
from ufc.models import registry
from ufc.predict import publish, show
from ufc.predict.refresh import RefreshResult, refresh
from ufc.predict.resolve import load_wikipedia_overrides, resolver_from_snapshot
from ufc.predict.wikipedia import DEFAULT_CACHE, Wikipedia, http_transport


def _pct(p: float) -> str:
    return f"{100 * p:.1f}%"


def print_summary(result: RefreshResult, publish_dir: Path) -> None:
    print(f"Refresh at {result.run_at}: {result.requests} Wikipedia request(s)")
    if result.stopped:
        print(f"STOPPED, nothing changed: {result.stopped}")
        return
    added = sum(len(c.added) for c in result.changes)
    cancelled = sum(len(c.cancelled) for c in result.changes)
    restored = sum(len(c.restored) for c in result.changes)
    print(
        f"Cards: {len(result.changes)} read, {added} bouts added, {cancelled} cancelled, "
        f"{restored} restored"
    )
    for page, error in result.event_errors.items():
        print(f"  NOT APPLIED {page}: {error}")
    s = result.staleness
    if s is not None:
        print(f"Dataset: last event {s.dataset_last_event}; staleness {s.status}")
        for m in s.missing:
            print(f"  missing: {m['name']} ({m['date']}), read: {m['read']}")
    if result.backfill is not None:
        print(
            f"Attribute backfill: {result.backfill.fields_written} field(s) from "
            f"{result.backfill.pages_requested} article(s)"
        )
    if result.model_error:
        print(f"Predictions: none ({result.model_error})")
    elif result.predictions is not None:
        p = result.predictions
        print(
            f"Predictions with {result.model_version}: {len(p.made)} new, "
            f"{len(p.unchanged)} unchanged, {len(p.superseded)} superseded"
        )
    if result.published:
        print(f"Published: {result.published}")
    else:
        print("Not published; the last good file is unchanged:")
        for problem in result.publish_problems[:10]:
            print(f"  - {problem}")

    path = publish_dir / publish.FILE_NAME
    if not path.exists():
        return
    record = json.loads(path.read_text(encoding="utf-8"))
    if not record["events"]:
        return
    event = record["events"][0]
    print()
    print(
        f"Next card: {event['name']}, {event['date']} (Wikipedia rev {event['wikipedia_revision']})"
    )
    for bout in event["bouts"]:
        a, b = bout["fighter_a"], bout["fighter_b"]
        line = (
            f"  {a['name']} {_pct(bout['p_fighter_a'])}  vs.  "
            f"{_pct(bout['p_fighter_b'])} {b['name']}  [{bout['weight_class']}]"
        )
        print(line)
        for reason in bout["limited_data_reasons"]:
            print(f"      limited data: {reason}")
    for bout in event["unresolved_bouts"]:
        print(
            f"  UNRESOLVED {bout['fighter_a']['name']} vs. {bout['fighter_b']['name']}: "
            f"{bout['reason']}"
        )
    later = record["events"][1:]
    if later:
        counts = ", ".join(f"{e['name']} ({len(e['bouts'])})" for e in later)
        print(f"Later cards with predictions: {counts}")


def run_refresh(args: argparse.Namespace) -> int:
    if not args.db.exists():
        print(f"No database at {args.db}; run the ingest load first.", file=sys.stderr)
        return 1
    if not (args.snapshot / source.FIGHTER_TOTT_FILE).exists():
        print(
            f"No snapshot in {args.snapshot}; run the ingest fetch first.",
            file=sys.stderr,
        )
        return 1
    resolver = resolver_from_snapshot(args.snapshot, load_wikipedia_overrides())
    client = Wikipedia(http_transport(), args.cache)
    moment = datetime.now(UTC)
    conn = connect(args.db)
    try:
        result = refresh(
            conn,
            client,
            resolver,
            models_root=args.models,
            publish_dir=args.publish,
            today=moment.date(),
            now=moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
    finally:
        conn.close()
    print_summary(result, args.publish)
    return 0 if result.ok else 1


def _today() -> date:
    return date.today()


def _width(args: argparse.Namespace) -> int:
    return args.width or shutil.get_terminal_size((100, 24)).columns


def run_show(args: argparse.Namespace) -> int:
    try:
        conn = show.connect_readonly(args.db)
        try:
            lines = show.show(conn, _today(), args.event, args.all, _width(args))
        finally:
            conn.close()
    except show.ShowError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("\n".join(lines).rstrip())
    return 0


def run_matchup(args: argparse.Namespace) -> int:
    try:
        on = date.fromisoformat(args.date) if args.date else _today()
    except ValueError:
        print(f"--date must be YYYY-MM-DD, got {args.date!r}", file=sys.stderr)
        return 1
    if not (args.snapshot / source.FIGHTER_TOTT_FILE).exists():
        print(
            f"No snapshot in {args.snapshot}; run the ingest fetch first.",
            file=sys.stderr,
        )
        return 1
    try:
        resolver = resolver_from_snapshot(args.snapshot, load_wikipedia_overrides())
        conn = show.connect_readonly(args.db)
        try:
            result = show.predict_matchup(
                conn,
                resolver,
                args.models,
                args.fighter_a,
                args.fighter_b,
                on,
                args.weight_class,
            )
        finally:
            conn.close()
    except show.ShowError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("\n".join(show.format_hypothetical(result, _width(args))))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        prog="python -m ufc.predict", description=__doc__.split("\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    cmd = commands.add_parser(
        "refresh", help="read cards, predict, write the JSON dump"
    )
    cmd.add_argument("--db", type=Path, default=Path("data/ufc.db"))
    cmd.add_argument("--snapshot", type=Path, default=Path("data/external/greco"))
    cmd.add_argument("--models", type=Path, default=registry.DEFAULT_ROOT)
    cmd.add_argument("--publish", type=Path, default=publish.DEFAULT_DIR)
    cmd.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    shown = commands.add_parser("show", help="print stored predictions (read-only)")
    shown.add_argument("--db", type=Path, default=Path("data/ufc.db"))
    shown.add_argument(
        "--width", type=int, default=None, help="default: terminal width"
    )
    which = shown.add_mutually_exclusive_group()
    which.add_argument("--event", help="event name (exact, else a substring)")
    which.add_argument("--all", action="store_true", help="every stored upcoming card")
    hyp = commands.add_parser(
        "matchup", help="predict a hypothetical bout; nothing is stored"
    )
    hyp.add_argument("fighter_a")
    hyp.add_argument("fighter_b")
    hyp.add_argument("--date", help="YYYY-MM-DD, default today")
    hyp.add_argument(
        "--weight-class", help="default: the division they share most recently"
    )
    hyp.add_argument("--db", type=Path, default=Path("data/ufc.db"))
    hyp.add_argument("--snapshot", type=Path, default=Path("data/external/greco"))
    hyp.add_argument("--models", type=Path, default=registry.DEFAULT_ROOT)
    hyp.add_argument("--width", type=int, default=None, help="default: terminal width")
    args = parser.parse_args(argv)
    if args.command == "show":
        return run_show(args)
    if args.command == "matchup":
        return run_matchup(args)
    return run_refresh(args)


if __name__ == "__main__":
    sys.exit(main())
