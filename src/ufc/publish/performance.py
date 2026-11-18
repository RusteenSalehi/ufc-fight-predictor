"""``performance.json``: backtest numbers from the model reports, the
era-shift caveat, and the live record, kept apart.

Every backtest number is copied from JSON that ``python -m ufc.models
evaluate`` wrote (``test_metrics.json``, ``walk_forward.json``); nothing here
computes a backtest metric. The reports are those of the candidate whose
recipe the approved model uses: the approved version itself if it is a
candidate, or its lineage parent if it is a production refit, which has no
held-out test of its own (D47).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_REPORTS = Path("data/reports")
METRIC_KEYS = (
    "n",
    "log_loss",
    "brier",
    "accuracy",
    "ece",
    "log_loss_low",
    "log_loss_high",
)
BASELINES = ("model", "elo", "logistic")

SEPARATION = (
    "Backtest numbers were produced after the fact on bouts the model did not "
    "train on. Live numbers come only from predictions published before each "
    "event. The two are reported separately and never pooled."
)


def find_reports(root: Path, version: str) -> Path | None:
    """The reports directory whose test and walk-forward records are for
    ``version``; ``None`` if there is none."""
    if not root.is_dir():
        return None
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            test = json.loads((directory / "test_metrics.json").read_text())
            walk = json.loads((directory / "walk_forward.json").read_text())
        except (OSError, ValueError):
            continue
        if test.get("version") == version and walk.get("version") == version:
            return directory
    return None


def _metrics(raw: Mapping) -> dict:
    return {k: raw[k] for k in METRIC_KEYS if k in raw}


def _models(raw: Mapping) -> dict:
    return {name: _metrics(raw[name]) for name in BASELINES if name in raw}


def backtest(reports: Path, candidate: str, production: str) -> dict:
    test = json.loads((reports / "test_metrics.json").read_text())
    walk = json.loads((reports / "walk_forward.json").read_text())
    seen = test.get("already_seen") or {}
    pooled_test = walk.get("pooled_test_period")
    note = (
        f"{production} is a production refit of {candidate} on every completed "
        "bout (DECISIONS D47). A refit has no held-out test: the numbers here "
        "measure the candidate's recipe, and the refit's only out-of-sample "
        "evidence is the live record."
        if production != candidate
        else f"{production} is the candidate these numbers were measured on."
    )
    return {
        "label": "Backtest: scored after the fact, on bouts the model was not fitted on",
        "reports": str(reports),
        "commands": [
            "uv run python -m ufc.models evaluate"
            + (f" --reports {reports}" if reports.name != "eval" else ""),
        ],
        "candidate_version": candidate,
        "production_version": production,
        "production_note": note,
        "test_period": {
            "candidate_version": test["version"],
            "first_date": test["first_date"],
            "last_date": test["last_date"],
            "n_scored": test["n_scored"],
            "n_no_result": test["n_no_result"],
            "already_seen": bool(seen.get("already_seen", False)),
            "already_seen_note": seen.get("note"),
            "metrics": _models(test["models"]),
            "calibration": test["by_confidence"]["model"],
            "calibration_buckets": "the favoured fighter's probability",
        },
        "walk_forward": {
            "schedule": walk["schedule"],
            "folds": [
                {
                    "year": fold["year"],
                    "in_test_period": fold["in_test_period"],
                    "n_scored": fold["n_scored"],
                    "metrics": _models(fold["metrics"]),
                }
                for fold in walk["folds"]
            ],
            "pooled_before_test_period": {
                "years": walk["pooled_before_test_period"]["years"],
                "metrics": _models(walk["pooled_before_test_period"]["metrics"]),
            },
            "pooled_test_period": {
                "years": pooled_test["years"],
                "metrics": _models(pooled_test["metrics"]),
            }
            if pooled_test
            else None,
        },
    }


def era_shift(walk: Mapping) -> dict | None:
    """The caveat that the test period scored better than any backtest year,
    with the numbers from the walk-forward record; ``None`` if the record
    has no test-period folds."""
    pooled_test = walk.get("pooled_test_period")
    if not pooled_test:
        return None
    before = walk["pooled_before_test_period"]
    years = before["years"]
    earlier = [
        f["metrics"]["model"]["log_loss"]
        for f in walk["folds"]
        if not f["in_test_period"]
    ]
    test_loss = pooled_test["metrics"]["model"]["log_loss"]
    before_loss = before["metrics"]["model"]["log_loss"]
    better = test_loss < min(earlier)
    text = (
        f"The test period ({pooled_test['years'][0]} onward) scored "
        + (
            "better than every earlier backtest year"
            if better
            else "differently from the earlier backtest years"
        )
        + f": log loss {test_loss:.4f}, against {before_loss:.4f} pooled over "
        f"{years[0]} to {years[1]} (single years {min(earlier):.4f} to "
        f"{max(earlier):.4f}). The era-shift diagnostic found no leak but did not "
        "establish the cause (DECISIONS D45, docs/MODEL_REPORT.md). Expect live "
        "results somewhere between the backtest and the test figure; the live "
        "record is the real measurement."
    )
    return {
        "text": text,
        "backtest_years": tuple(years),
        "backtest_log_loss": before_loss,
        "backtest_year_range": (min(earlier), max(earlier)),
        "test_period_log_loss": test_loss,
    }


def performance(
    *,
    generated_at: str,
    model: Mapping,
    reports_root: Path,
    live: Mapping,
) -> dict:
    """The whole artifact. ``model`` is the approved model's record (version,
    status, parent)."""
    production = model["version"]
    candidate = model.get("parent") or production
    reports = find_reports(reports_root, candidate)
    record: dict = {
        "generated_at": generated_at,
        "model": dict(model),
        "separation": SEPARATION,
        "backtest": None,
        "backtest_unavailable": None,
        "era_shift_caveat": None,
        "live": dict(live),
    }
    if reports is None:
        record["backtest_unavailable"] = (
            f"no evaluation record for {candidate} under {reports_root}"
        )
        logger.warning("performance: %s", record["backtest_unavailable"])
        return record
    record["backtest"] = backtest(reports, candidate, production)
    record["era_shift_caveat"] = era_shift(
        json.loads((reports / "walk_forward.json").read_text())
    )
    return record
