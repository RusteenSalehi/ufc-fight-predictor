"""uv run python -m ufc.odds snapshot   one MMA h2h call, raw response stored in data/odds/raw/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ufc.odds import capture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ufc.odds")
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot", help="one call to the MMA h2h market")
    snap.add_argument(
        "--regions", default="us,eu", help="comma list; each region costs 1 credit"
    )
    snap.add_argument("--out", type=Path, default=capture.DEFAULT_DIR)
    args = parser.parse_args(argv)

    try:
        key = capture.load_key()
        snapshot = capture.take_snapshot(key, out_dir=args.out, regions=args.regions)
    except capture.OddsError as exc:
        print(f"Not captured: {exc}", file=sys.stderr)
        return 1
    s = capture.summarize(snapshot.body)
    c = snapshot.meta["credit_headers"]
    print(f"Saved {snapshot.raw_path} (+ {snapshot.meta_path.name})")
    print(
        f"Credits: used {c['x-requests-used']}, remaining {c['x-requests-remaining']}, last call cost {c['x-requests-last']}"
    )
    print(f"Events: {s['events']}; with a moneyline: {s['with_moneyline']}")
    print(
        "Bookmakers (events carried): "
        + (", ".join(f"{k} {v}" for k, v in s["bookmakers"].items()) or "none")
    )
    print(
        "Start times (events): "
        + ", ".join(f"{k} x{v}" for k, v in s["commence_times"].items())
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
