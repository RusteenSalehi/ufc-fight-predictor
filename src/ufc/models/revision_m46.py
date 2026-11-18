"""Model revision M4.6: expected-outcome ("deserved win") features.

    uv run python -m ufc.models.revision_m46 transform   # sanity checks, examples
    uv run python -m ufc.models.revision_m46 run         # D63 rule: validation, backtest
    uv run python -m ufc.models.revision_m46 ablation    # per-family ablation with family 16

Analysis only: nothing here saves a model, writes the registry or changes
which model is live (DECISIONS D63). The base is the recipe of the M4.5
candidate (``--eval data/reports/m45``), the recipe of the approved refit.

Evidence is kept apart as in M4.5 (``revision.py``): validation (2023 to
2024, pipeline fitted through 2022, five seeds) and the walk-forward
backtest 2013 to 2024 decide; the 2025 and 2026 folds read the test period,
are reported as a secondary, already-seen check and logged in
``data/models/test_diagnostics.jsonl``. ``transform`` also reads test-period
bouts (for example bouts) and is logged the same way; it scores no model.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ufc.db.connection import connect
from ufc.features.build import build_features
from ufc.features.components import KO_METHODS, SUB_METHODS
from ufc.features.data import BoutFacts, FeatureData, load_feature_data
from ufc.features.matchup import CANDIDATE_FEATURES, FAMILY_MEMBERS
from ufc.features.performance import INPUTS, is_judged_win, performance_inputs
from ufc.models import registry
from ufc.models.metrics import CLIP, bootstrap_mean_interval, bout_log_loss
from ufc.models.revision import (
    BACKTEST,
    N_BOOT,
    SECONDARY,
    VALIDATION,
    Variant,
    _feature_config,
    _log_test_look,
    _write_json,
    reproduces_backtest,
    run_backtest,
    run_validation,
    seed_columns,
    seed_mean_loss,
)
from ufc.models.select import ablate
from ufc.models.splits import SPLIT_DATES, Splits, labelled
from ufc.models.tune import SEEDS
from ufc.models.walkforward import FIRST_YEAR

logger = logging.getLogger(__name__)

DEFAULT_DB = Path("data/ufc.db")
DEFAULT_EVAL = Path("data/reports/m45")
DEFAULT_OUT = Path("data/reports/m46")

PERFORMANCE: tuple[str, ...] = tuple(
    f.name for f in CANDIDATE_FEATURES if f.family == "performance"
)
VARIANTS: tuple[Variant, ...] = (Variant("base"), Variant("performance", PERFORMANCE))
"""Registered in DECISIONS D63 before any variant was run."""

UPPER_BOUND = 0.001
"""D63 criterion 1: the interval's upper end must be below this."""


# --- the rule ------------------------------------------------------------------


def apply_rule(validation: Mapping, backtest: Mapping) -> dict:
    """D63: adopt only if (1) the paired validation difference (uncalibrated,
    seed mean) is negative and its 95% interval's upper end is below +0.001,
    and (2) pooled backtest log loss 2013 to 2024 is lower than base's."""
    v = validation["variants"]["performance"]["raw"]
    b = backtest["variants"]["performance"]
    criteria = {
        "1_validation_negative_and_upper_below_0.001": bool(
            v["delta"] < 0 and v["delta_interval"][1] < UPPER_BOUND
        ),
        "2_backtest_improves": bool(b["delta"] < 0),
    }
    return {
        "rule": "DECISIONS D63",
        "criteria": criteria,
        "validation_delta": v["delta"],
        "validation_delta_interval": v["delta_interval"],
        "backtest_delta": b["delta"],
        "adopt": all(criteria.values()),
    }


def validation_summary(frame: pd.DataFrame, names: Sequence[str]) -> dict:
    """Seed-mean log loss per variant, raw (the rule) and calibrated, paired
    against base with intervals; per-seed paired differences."""
    y = frame["label"].to_numpy(dtype=np.float64)
    out: dict = {"n": len(frame), "variants": {}}
    for name in names:
        row: dict = {}
        for kind in ("raw", "cal"):
            losses = seed_mean_loss(frame, name, kind)
            base = seed_mean_loss(frame, "base", kind)
            delta = bootstrap_mean_interval(losses - base, n_boot=N_BOOT)
            seed_deltas = [
                float(
                    (
                        bout_log_loss(frame[c].to_numpy(), y)
                        - bout_log_loss(frame[c.replace(name, "base", 1)].to_numpy(), y)
                    ).mean()
                )
                for c in seed_columns(frame, name, kind)
            ]
            row[kind] = {
                "log_loss": float(losses.mean()),
                "seed_log_losses": [
                    float(bout_log_loss(frame[c].to_numpy(), y).mean())
                    for c in seed_columns(frame, name, kind)
                ],
                "seed_deltas": seed_deltas,
                "delta": delta["mean"],
                "delta_interval": [delta["low"], delta["high"]],
            }
        out["variants"][name] = row
    out["note"] = (
        "raw = uncalibrated (the rule), cal = Platt cross-validated inside "
        "validation; log_loss = mean over seeds of per-bout loss"
    )
    return out


def backtest_summary(bouts: pd.DataFrame, names: Sequence[str]) -> dict:
    y = bouts["label"].to_numpy(dtype=np.float64)
    base = bout_log_loss(bouts["base"].to_numpy(), y)
    out: dict = {
        "years": sorted(int(v) for v in bouts["year"].unique()),
        "n": len(bouts),
        "variants": {},
        "elo_log_loss": float(bout_log_loss(bouts["p_elo"].to_numpy(), y).mean()),
    }
    for name in names:
        losses = bout_log_loss(bouts[name].to_numpy(), y)
        delta = bootstrap_mean_interval(losses - base, n_boot=N_BOOT)
        out["variants"][name] = {
            "log_loss": float(losses.mean()),
            "delta": delta["mean"],
            "delta_interval": [delta["low"], delta["high"]],
        }
    return out


def _context(args: argparse.Namespace) -> tuple[str, object, object, FeatureData]:
    version = json.loads((args.eval / "selection.json").read_text())["version"]
    candidate, _ = registry.load_candidate(args.models, version)
    if registry.is_refit(registry.read_manifest(args.models, version)):
        raise ValueError("the revision starts from the tested candidate, not a refit")
    config = _feature_config(args.models, version)
    conn = connect(args.db)
    try:
        data = load_feature_data(conn)
    finally:
        conn.close()
    return version, candidate, config, data


def run(args: argparse.Namespace) -> int:
    version, candidate, config, data = _context(args)
    names = [v.name for v in VARIANTS]
    record: dict = {
        "candidate": version,
        "variants": {
            v.name: {"antisymmetric_added": list(v.antisymmetric)} for v in VARIANTS
        },
        "seeds": list(args.seeds),
    }
    logger.info("validation: %d variants x %d seeds", len(VARIANTS), len(args.seeds))
    valid = run_validation(
        data, candidate.recipe, config, list(VARIANTS), args.seeds, args.jobs
    )
    registered = candidate.predict(
        labelled(Splits(build_features(data, config).frame).valid()),
        calibrated=False,
    )
    record["validation_reproduces_candidate"] = (
        bool(np.array_equal(valid[f"base|{args.seeds[0]}|raw"].to_numpy(), registered))
        if args.seeds[0] == SEEDS[0]
        else None
    )
    record[VALIDATION] = validation_summary(valid, names)

    years = list(range(FIRST_YEAR, SPLIT_DATES.test_start.year))
    reason = None
    if args.secondary:
        last = max(b.date for b in data.bouts).year
        years += list(range(SPLIT_DATES.test_start.year, last + 1))
        reason = (
            f"model revision M4.6 of {version} (analysis only; 2025-2026 reported "
            "as a secondary, already-seen check, not used for any decision)"
        )
        _log_test_look(args.models, reason)
    logger.info("backtest: %d years x %d variants", len(years), len(VARIANTS))
    bouts = run_backtest(
        data, candidate.recipe, config, list(VARIANTS), years, reason, args.jobs
    )
    saved = json.loads((args.eval / "walk_forward.json").read_text())
    record["backtest_reproduces_saved"] = reproduces_backtest(bouts, saved)
    before = bouts[bouts["year"] < SPLIT_DATES.test_start.year]
    record[BACKTEST] = backtest_summary(before, names)
    record["backtest_by_year"] = {
        str(year): {
            "n": len(rows),
            **{
                name: float(
                    bout_log_loss(
                        rows[name].to_numpy(), rows["label"].to_numpy(dtype=np.float64)
                    ).mean()
                )
                for name in names
            },
        }
        for year, rows in bouts.groupby("year")
    }
    if args.secondary:
        after = bouts[bouts["year"] >= SPLIT_DATES.test_start.year]
        record[SECONDARY] = backtest_summary(after, names)
    record["rule"] = apply_rule(record[VALIDATION], record[BACKTEST])

    args.out.mkdir(parents=True, exist_ok=True)
    _write_json(args.out / "revision.json", record)
    valid.to_parquet(args.out / "validation.parquet", index=False)
    bouts.to_parquet(args.out / "backtest.parquet", index=False)
    print(f"Wrote {args.out / 'revision.json'}")
    print(
        f"validation reproduces candidate: {record['validation_reproduces_candidate']}"
    )
    print(f"backtest reproduces saved: {record['backtest_reproduces_saved']}")
    for name in names:
        v = record[VALIDATION]["variants"][name]["raw"]
        b = record[BACKTEST]["variants"][name]
        print(
            f"  {name:12} validation {v['log_loss']:.4f} delta {v['delta']:+.4f} "
            f"({v['delta_interval'][0]:+.4f} to {v['delta_interval'][1]:+.4f}) | "
            f"backtest {b['log_loss']:.4f} delta {b['delta']:+.4f}"
        )
    print(f"D63 rule: {record['rule']['criteria']} -> adopt={record['rule']['adopt']}")
    return 0


def ablation(args: argparse.Namespace) -> int:
    """The per-family ablation with family 16 in the model (seeds 0 to 4),
    on the candidate's validation pipeline. Reported, not part of the rule."""
    version, candidate, config, data = _context(args)
    splits = Splits(build_features(data, config).frame)
    splits.assert_test_locked("revision M4.6: ablation")
    recipe = VARIANTS[1].recipe(candidate.recipe)
    families = {**FAMILY_MEMBERS, "performance": PERFORMANCE}
    result = ablate(splits, recipe, families, args.seeds, args.jobs)
    splits.assert_test_locked("revision M4.6: ablation")
    out = {"candidate": version, "with": "performance", "ablation": result}
    _write_json(args.out / "ablation.json", out)
    print(f"Wrote {args.out / 'ablation.json'}; full {result['full_log_loss']:.4f}")
    for row in result["families"]:
        print(
            f"  {row['family']:18} n={row['n_features']:3d} "
            f"delta {row['delta']:+.4f} ({row['delta_low']:+.4f} to {row['delta_high']:+.4f})"
        )
    return 0


# --- the transform: sanity checks and examples ------------------------------------


_CARD = re.compile(r"(\d{1,2}) - (\d{1,2})\.")


def scorecards(details: str | None) -> list[tuple[int, int]]:
    """Each judge's bout total as written (for a win, the loser's first)."""
    return [(int(a), int(b)) for a, b in _CARD.findall(details or "")]


def _bout_text(conn: sqlite3.Connection) -> dict[str, tuple[str, str | None]]:
    return {
        row[0]: (row[1], row[2])
        for row in conn.execute("SELECT bout_id, method, method_details FROM bouts")
    }


def _names(conn: sqlite3.Connection) -> dict[str, str]:
    return dict(conn.execute("SELECT fighter_id, name FROM fighters"))


def _period(day: date) -> str:
    if day <= SPLIT_DATES.train_end:
        return "training (in sample for the transform)"
    if day <= SPLIT_DATES.valid_end:
        return "validation"
    return "test period (already seen)"


def _stat_line(bout: BoutFacts, fighter_id: str) -> dict:
    t = bout.totals[fighter_id]
    return {
        "sig_landed": t["sig_str_landed"],
        "knockdowns": t["knockdowns"],
        "td_landed": t["td_landed"],
        "sub_attempts": t["sub_attempts"],
        "ctrl_s": t["ctrl_s"],
    }


def _describe(bout, model, names, text, deserved_of: str) -> dict:
    a, b = model.deserved_pair(bout)
    winner = bout.winner_id
    return {
        "date": bout.date.isoformat(),
        "period": _period(bout.date),
        "bout_id": bout.bout_id,
        "fighters": [names.get(bout.fighter_a_id), names.get(bout.fighter_b_id)],
        "winner": names.get(winner) if winner else None,
        "result": bout.result,
        "method": bout.method,
        "end": f"R{bout.end_round} {bout.end_time_s // 60}:{bout.end_time_s % 60:02d}"
        if bout.end_round and bout.end_time_s is not None
        else None,
        "deserved": {
            names.get(bout.fighter_a_id): a,
            names.get(bout.fighter_b_id): b,
        },
        "deserved_of_" + deserved_of: a if winner == bout.fighter_a_id else b,
        "stats": {
            names.get(f): _stat_line(bout, f)
            for f in (bout.fighter_a_id, bout.fighter_b_id)
        },
        "details": text.get(bout.bout_id, (None, None))[1],
    }


def _accuracy(p_winner: np.ndarray) -> dict:
    n = len(p_winner)
    if n == 0:
        return {"n": 0}
    losses = -np.log(np.clip(p_winner, CLIP, 1 - CLIP))
    return {
        "n": n,
        "accuracy": float((p_winner > 0.5).mean()),
        "log_loss": float(losses.mean()),
        "mean_deserved_of_winner": float(p_winner.mean()),
    }


def transform(args: argparse.Namespace) -> int:
    """The performance model fitted through 2022 (the validation pipeline's):
    its coefficients, accuracy on decisions by period and decision type,
    agreement with judges' scorecards, finishes, point deductions, and the
    example bouts. Logged as a look at the test period because examples and
    finish summaries include 2025-2026 bouts; no model is scored."""
    version, _, config, data = _context(args)
    _log_test_look(
        args.models,
        "model revision M4.6: performance transform sanity checks and example "
        "bouts (analysis only; no model scored, nothing used for a decision)",
    )
    build = build_features(data, config)
    model = build.params.performance
    if model is None:
        raise ValueError("no performance model was fitted")
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        text, names = _bout_text(conn), _names(conn)
    finally:
        conn.close()

    rows = []
    for bout in data.bouts:
        pair = model.deserved_pair(bout)
        if pair is None:
            continue
        x = performance_inputs(bout, bout.fighter_a_id)
        method = bout.method or ""
        rows.append(
            {
                "bout": bout,
                "date": bout.date,
                "period": "train"
                if bout.date <= SPLIT_DATES.train_end
                else "validation"
                if bout.date <= SPLIT_DATES.valid_end
                else "test",
                "result": bout.result,
                "method": method,
                "judged": is_judged_win(bout),
                "p_winner": (
                    pair[0] if bout.winner_id == bout.fighter_a_id else pair[1]
                )
                if bout.winner_id
                else math.nan,
                "has_control": not np.isnan(x).any(),
                "cards": scorecards(text.get(bout.bout_id, (None, None))[1]),
                "deducted": "Deduct" in (text.get(bout.bout_id, (None, ""))[1] or ""),
            }
        )
    frame = pd.DataFrame(rows)

    out: dict = {
        "candidate_pipeline": version,
        "fitted_through": model.train_end.isoformat(),
        "model": model.to_dict(),
        "inputs": list(INPUTS),
        "performance_elo": build.params.performance_elo.to_dict()
        if build.params.performance_elo
        else None,
        "performance_elo_fit": None,
        "elo": build.params.elo.to_dict(),
        "elo_train_log_loss": build.elo_fit.log_loss if build.elo_fit else None,
        "n_bouts_scored": len(frame),
    }
    if build.performance_elo_fit is not None:
        fit = build.performance_elo_fit
        out["performance_elo_fit"] = {
            "train_log_loss": fit.log_loss,
            "n_bouts": fit.n_bouts,
            "on_grid_edge": list(fit.on_grid_edge),
        }

    # Decisions: accuracy on the judged winner, by period and decision type.
    decisions = frame[frame["judged"]]
    out["decisions"] = {}
    for period in ("train", "validation", "test"):
        part = decisions[decisions["period"] == period]
        out["decisions"][period] = {
            "all": _accuracy(part["p_winner"].to_numpy()),
            **{
                kind: _accuracy(
                    part[part["method"].str.contains(kind)]["p_winner"].to_numpy()
                )
                for kind in ("Unanimous", "Split", "Majority")
            },
            "without_control": _accuracy(
                part[~part["has_control"]]["p_winner"].to_numpy()
            ),
        }

    # Agreement with the judges' margin, validation decisions with three cards.
    v = decisions[(decisions["period"] == "validation")].copy()
    v = v[v["cards"].map(len) == 3]
    v["margin"] = v["cards"].map(
        lambda cards: float(np.mean([b - a for a, b in cards]))
    )
    v["judges_for_winner"] = v["cards"].map(lambda cards: sum(b > a for a, b in cards))
    buckets = [(-np.inf, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, np.inf)]
    out["validation_by_judges_margin"] = [
        {
            "mean_margin_points": [lo, hi],
            **_accuracy(
                v[(v["margin"] > lo) & (v["margin"] <= hi)]["p_winner"].to_numpy()
            ),
        }
        for lo, hi in buckets
    ]
    out["validation_margin_correlation"] = (
        float(np.corrcoef(v["margin"], v["p_winner"])[0, 1]) if len(v) > 2 else None
    )

    # Finishes: is the finisher also ahead on the judges' criteria?
    out["finishes"] = {}
    for period in ("train", "validation", "test"):
        part = frame[(frame["period"] == period) & (frame["result"] == "win")]
        out["finishes"][period] = {
            "ko_tko": _accuracy(
                part[part["method"].isin(KO_METHODS)]["p_winner"].to_numpy()
            ),
            "submission": _accuracy(
                part[part["method"].isin(SUB_METHODS)]["p_winner"].to_numpy()
            ),
        }

    # Draws and point deductions.
    draws = frame[frame["result"] == "draw"]
    out["draws"] = {
        "n": len(draws),
        "n_with_deduction": int(draws["deducted"].sum()),
        "with_deduction": [
            _describe(r["bout"], model, names, text, "winner")
            for _, r in draws[draws["deducted"]].iterrows()
        ],
    }
    out["n_decisions_with_deduction"] = int(
        frame[frame["method"].str.startswith("Decision")]["deducted"].sum()
    )

    # Examples.
    wanted = [("Lerone Murphy", "Aaron Pico"), ("Carlos Prates", "Geoff Neal")]
    out["requested_examples"] = []
    for pair in wanted:
        found = [
            r["bout"]
            for _, r in frame.iterrows()
            if {names.get(r["bout"].fighter_a_id), names.get(r["bout"].fighter_b_id)}
            == set(pair)
        ]
        out["requested_examples"].append(
            {
                "pair": list(pair),
                "in_data": bool(found),
                "bouts": [_describe(b, model, names, text, "winner") for b in found],
            }
        )
    judged = frame[frame["judged"]].nsmallest(args.n_examples, "p_winner")
    out["lowest_deserved_judged_wins"] = [
        {
            **_describe(r["bout"], model, names, text, "winner"),
            "scorecards_loser_first": r["cards"],
        }
        for _, r in judged.iterrows()
    ]
    kos = frame[(frame["result"] == "win") & (frame["method"] == "KO/TKO")]
    out["lowest_deserved_ko_wins"] = [
        _describe(r["bout"], model, names, text, "winner")
        for _, r in kos.nsmallest(args.n_examples, "p_winner").iterrows()
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    _write_json(args.out / "transform.json", out)
    print(f"Wrote {args.out / 'transform.json'}")
    for period, row in out["decisions"].items():
        a = row["all"]
        print(
            f"  decisions {period:10} n={a['n']:5d} acc {a['accuracy']:.3f} ll {a['log_loss']:.3f}"
        )
    return 0


# --- command ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ufc.models.revision_m46", description=__doc__.split("\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--db", type=Path, default=DEFAULT_DB)
        p.add_argument("--models", type=Path, default=registry.DEFAULT_ROOT)
        p.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
        p.add_argument("--out", type=Path, default=DEFAULT_OUT)
        p.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
        p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))

    r = sub.add_parser("run", help="validation and backtest under the D63 rule")
    common(r)
    r.add_argument(
        "--no-secondary",
        dest="secondary",
        action="store_false",
        help="leave out the 2025 and 2026 folds (the test period)",
    )
    r.set_defaults(func=run)
    a = sub.add_parser("ablation", help="per-family ablation with family 16")
    common(a)
    a.set_defaults(func=ablation)
    t = sub.add_parser("transform", help="performance model checks and examples")
    common(t)
    t.add_argument("--n-examples", type=int, default=10)
    t.set_defaults(func=transform)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
