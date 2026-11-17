"""Command line for the feature pipeline.

uv run python -m ufc.features build      fit on the training period, build, write
uv run python -m ufc.features summary    list features with missing-value counts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from ufc.db.connection import connect
from ufc.features.build import (
    FEATURES_FILE,
    LEVEL_FAMILY,
    META_FILE,
    build_features,
    feature_summary,
    write_features,
)
from ufc.features.config import FeatureConfig
from ufc.features.data import load_feature_data
from ufc.features.matchup import FAMILIES, FEATURES

DEFAULT_DB_PATH = Path("data/ufc.db")
DEFAULT_OUT_DIR = Path("data/features")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ufc.features", description=__doc__.split("\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build data/features/features.parquet")
    build.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    build.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    build.add_argument(
        "--shrink-minutes",
        type=float,
        default=0.0,
        help="prior strength for per-minute rates, in cage minutes (0 = off); "
        "`python -m ufc.models train` chooses it on validation and writes the "
        "feature file with the chosen value",
    )
    summary = commands.add_parser(
        "summary", help="list every feature with its missing-value counts"
    )
    summary.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def run_build(db_path: Path, out_dir: Path, shrink_minutes: float = 0.0) -> int:
    if not db_path.exists():
        print(f"No database at {db_path}; run the ingest load first.", file=sys.stderr)
        return 1
    conn = connect(db_path)
    try:
        data = load_feature_data(conn)
    finally:
        conn.close()
    if not data.bouts:
        print("The database has no completed bouts.", file=sys.stderr)
        return 1
    build = build_features(data, FeatureConfig(shrink_minutes=shrink_minutes))
    features_path, meta_path = write_features(build, out_dir, data.source_commit)
    frame = build.frame
    print(f"Snapshot commit: {data.source_commit}")
    print(
        f"Bouts: {len(frame)} ({frame['date'].min().date()} to {frame['date'].max().date()}), "
        f"{int(frame['label'].notna().sum())} with a winner"
    )
    print(f"Features: {len(FEATURES)} in {features_path}")
    print(f"Rate shrinkage: {build.store.config.shrink_minutes:g} cage minutes")
    print(
        f"Elo parameters (fit through {build.store.config.train_end}): {build.params.elo}"
    )
    if build.elo_fit is not None:
        fit = build.elo_fit
        print(
            f"Elo training log loss: {fit.log_loss:.4f} over {fit.n_bouts} bouts"
            + (
                f"; on grid edge: {', '.join(fit.on_grid_edge)}"
                if fit.on_grid_edge
                else ""
            )
        )
    print(f"Metadata: {meta_path}")
    return 0


def run_summary(out_dir: Path) -> int:
    features_path, meta_path = out_dir / FEATURES_FILE, out_dir / META_FILE
    if not features_path.exists():
        print(f"No feature file at {features_path}; run build first.", file=sys.stderr)
        return 1
    frame = pd.read_parquet(features_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    summary = feature_summary(frame)
    established = int((~(frame["a_debut"] | frame["b_debut"])).sum())
    print(f"Snapshot commit: {meta['source_commit']}")
    print(
        f"{len(frame)} bouts, {established} with no debutant, {len(summary)} features. "
        "Columns: missing in all bouts / missing where neither fighter is a debutant."
    )
    for family in (*FAMILIES, LEVEL_FAMILY):
        rows = summary[summary["family"] == family]
        print(f"\n{family} ({len(rows)})")
        for row in rows.itertuples():
            print(f"  {row.feature:38} {row.missing:>6} {row.missing_established:>6}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)
    if args.command == "build":
        return run_build(args.db, args.out, args.shrink_minutes)
    return run_summary(args.out)


if __name__ == "__main__":
    sys.exit(main())
