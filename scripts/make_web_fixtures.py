"""Write the frontend's synthetic test fixtures under web/src/test/fixtures/.

Usage:
    uv run python scripts/make_web_fixtures.py [FIXTURES_DIR]

Every fixture is built through the pydantic artifact models
(``ufc.publish.schemas``) and written as the gate writes artifacts, so a
fixture is always a valid artifact; ``tests/test_publish_jsonschema.py``
checks the committed files again. Nothing is copied from the dataset or a
published artifact: fighters, events and every number are invented, and
event dates are in 2099 so the "next event" never passes. The output is
deterministic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel

from ufc.publish.schemas import (
    BoutArtifact,
    Performance,
    Upcoming,
    fair_moneyline,
    winner_of,
)

OUT = Path("web/src/test/fixtures")
GENERATED = "2099-01-02T12:00:00Z"
PREDICTED = "2099-01-02T11:30:00Z"
MODEL = "xgboost-0000aaaa-1111bbbb"
CANDIDATE = "xgboost-2222cccc-3333dddd"
ATTRIBUTION = "Synthetic test data. No real fighters, events or results."
METHOD = "Synthetic method note: contributions to the model's log-odds."

EVENT_1 = {
    "event_key": "wp:1001",
    "name": "Test Fight Night: Stone vs. Vale",
    "date": "2099-01-10",
    "wikipedia_page": "Test Fight Night: Stone vs. Vale",
    "wikipedia_revision": 1,
}
EVENT_2 = {
    "event_key": "wp:1002",
    "name": "Test 900: Okafor vs. Lind",
    "date": "2099-02-07",
    "wikipedia_page": "Test 900: Okafor vs. Lind",
    "wikipedia_revision": 2,
}


def brief(name: str, ref: str | None, provisional: bool = False) -> dict:
    return {
        "name": name,
        "ref": ref,
        "provisional": provisional,
        "wikipedia_page": name,
    }


def upcoming_bout(
    bout_id: str,
    order: int,
    weight_class: str,
    a: dict,
    b: dict,
    p_a: float,
    reasons: list[str] | None = None,
    segment: str = "Main card",
    title: bool = False,
) -> dict:
    reasons = reasons or []
    p_b = 1.0 - p_a
    return {
        "bout_id": bout_id,
        "order": order,
        "segment": segment,
        "weight_class": weight_class,
        "title_bout": title,
        "fighter_a": a,
        "fighter_b": b,
        "p_fighter_a": p_a,
        "p_fighter_b": p_b,
        "predicted_winner": winner_of(p_a, p_b),
        "model_version": MODEL,
        "predicted_at": PREDICTED,
        "limited_data": bool(reasons),
        "limited_data_reasons": reasons,
        "stale_stats": any(r.startswith("stale stats") for r in reasons),
        "detail": f"bouts/{bout_id}.json",
    }


def upcoming() -> Upcoming:
    stone = brief("Avery Stone", "aaaaaaaaaaaaaaa1")
    vale = brief("Bruno Vale", "aaaaaaaaaaaaaaa2")
    kade = brief("Kade Moreno", "aaaaaaaaaaaaaaa3")
    new = brief("Tomas Newman", "wpname:Tomas Newman", provisional=True)
    reyes = brief("Iris Reyes", "aaaaaaaaaaaaaaa4")
    holt = brief("June Holt", "aaaaaaaaaaaaaaa5")
    okafor = brief("Dele Okafor", "aaaaaaaaaaaaaaa6")
    lind = brief("Eric Lind", "aaaaaaaaaaaaaaa7")
    return Upcoming(
        generated_at=GENERATED,
        model={"version": MODEL, "status": "synthetic", "parent": CANDIDATE},
        dataset={"source_commit": "0" * 40, "last_event": "2099-01-01"},
        staleness={
            "status": "current",
            "dataset_last_event": "2099-01-01",
            "missing_events": [],
        },
        events=[
            {
                **EVENT_1,
                "card_checked_at": GENERATED,
                "bouts": [
                    upcoming_bout(
                        "wb0000000000000001",
                        0,
                        "Middleweight",
                        stone,
                        vale,
                        0.3,
                        title=True,
                    ),
                    upcoming_bout(
                        "wb0000000000000002",
                        1,
                        "Lightweight",
                        kade,
                        new,
                        0.62,
                        reasons=[
                            "debut: Tomas Newman has no earlier UFC bout in the dataset",
                            "attribute gap: no height, reach for Tomas Newman",
                        ],
                    ),
                    upcoming_bout(
                        "wb0000000000000003",
                        2,
                        "Women's Strawweight",
                        reyes,
                        holt,
                        0.5,
                        reasons=[
                            "stale stats: June Holt fought at an event the dataset lacks"
                        ],
                        segment="Preliminary card",
                    ),
                ],
                "unresolved_bouts": [
                    {
                        "bout_id": "wb0000000000000004",
                        "order": 3,
                        "segment": "Preliminary card",
                        "weight_class": "Flyweight",
                        "title_bout": False,
                        "fighter_a": brief("Sam Park", None),
                        "fighter_b": brief("Sam Park Jr.", None),
                        "status": "unresolved",
                        "reason": "ambiguous name: Sam Park",
                    }
                ],
            },
            {
                **EVENT_2,
                "card_checked_at": GENERATED,
                "bouts": [
                    upcoming_bout(
                        "wb0000000000000005", 0, "Heavyweight", okafor, lind, 0.71
                    )
                ],
                "unresolved_bouts": [],
            },
        ],
        attribution=ATTRIBUTION,
    )


def reason(key: str, title: str, toward: str, detail: str, weight: float) -> dict:
    return {
        "reason": key,
        "component": None,
        "title": title,
        "toward": toward,
        "detail": detail,
        "log_odds": weight,
        "inputs": [f"d_{key}"],
    }


def past(date: str, opponent: str, result: str, method: str, rnd: int, deserved):
    return {
        "date": date,
        "event": f"Test event {date}",
        "opponent": opponent,
        "opponent_id": "bbbbbbbbbbbbbbb" + str(len(opponent) % 10),
        "result": result,
        "method": method,
        "end_round": rnd,
        "end_time": "5:00" if method.startswith("Decision") else "2:10",
        "weight_class": "Middleweight",
        "title_bout": False,
        "deserved_win_probability": deserved,
    }


def bout() -> BoutArtifact:
    p_a = 0.3
    b_toward = [
        reason(
            "age", "the age gap", "fighter_b", "4.0 years younger (28.0 vs 32.0)", 0.4
        ),
        reason(
            "sapm", "strikes absorbed per minute", "fighter_b", "2.10 vs 3.40", 0.25
        ),
        reason("record", "the UFC record", "fighter_b", "win rate 80% vs 60%", 0.1),
    ]
    a_toward = [
        reason(
            "rating",
            "the opposition faced",
            "fighter_a",
            "past opponents rated 90 higher",
            0.2,
        ),
        reason("reach", "reach", "fighter_a", "5 cm longer (193 vs 188)", 0.05),
    ]
    return BoutArtifact(
        generated_at=GENERATED,
        bout_id="wb0000000000000001",
        event=EVENT_1,
        order=0,
        segment="Main card",
        weight_class="Middleweight",
        title_bout=True,
        fighter_a={**brief("Avery Stone", "aaaaaaaaaaaaaaa1"), "resolution": "name"},
        fighter_b={**brief("Bruno Vale", "aaaaaaaaaaaaaaa2"), "resolution": "name"},
        prediction={
            "prediction_id": 1,
            "model_version": MODEL,
            "predicted_at": PREDICTED,
            "p_fighter_a": p_a,
            "p_fighter_b": 1 - p_a,
            "predicted_winner": winner_of(p_a, 1 - p_a),
            "fair_moneyline": {
                "fighter_a": fair_moneyline(p_a),
                "fighter_b": fair_moneyline(1 - p_a),
            },
            "fair_moneyline_note": "Synthetic note: a conversion, not a price.",
        },
        limited_data={"limited": False, "reasons": [], "stale_stats": False},
        why={
            "favoured": "fighter_b",
            "favoured_name": "Bruno Vale",
            "for": b_toward,
            "against": a_toward[0],
            "summary": "The model favours Bruno Vale (70%). Synthetic summary.",
            "method": METHOD,
        },
        top_factors={"fighter_a": a_toward, "fighter_b": b_toward, "method": METHOD},
        stats={
            "note": "Synthetic stats note.",
            "shrink_minutes": 15.0,
            "stance": {"fighter_a": "Orthodox", "fighter_b": "Southpaw"},
            "attribute_sources": {
                "fighter_a": {
                    "dob": "upstream",
                    "height_cm": "upstream",
                    "reach_cm": "upstream",
                },
                "fighter_b": {
                    "dob": "upstream",
                    "height_cm": "wikipedia",
                    "reach_cm": None,
                },
            },
            "rows": [
                {
                    "key": "age",
                    "group": "physical",
                    "label": "age",
                    "kind": "years",
                    "fighter_a": 32.0,
                    "fighter_b": 28.0,
                    "shrunk": False,
                },
                {
                    "key": "height_cm",
                    "group": "physical",
                    "label": "height",
                    "kind": "cm",
                    "fighter_a": 185.0,
                    "fighter_b": 183.0,
                    "shrunk": False,
                },
                {
                    "key": "reach_cm",
                    "group": "physical",
                    "label": "reach",
                    "kind": "cm",
                    "fighter_a": 193.0,
                    "fighter_b": None,
                    "shrunk": False,
                },
                {
                    "key": "wins",
                    "group": "record",
                    "label": "UFC wins",
                    "kind": "count",
                    "fighter_a": 6.0,
                    "fighter_b": 8.0,
                    "shrunk": False,
                },
                {
                    "key": "elo",
                    "group": "rating",
                    "label": "rating",
                    "kind": "elo",
                    "fighter_a": 1620.4,
                    "fighter_b": 1588.9,
                    "shrunk": False,
                },
                {
                    "key": "slpm_career",
                    "group": "striking",
                    "label": "strikes landed per minute",
                    "kind": "rate",
                    "fighter_a": 4.123,
                    "fighter_b": 3.456,
                    "shrunk": True,
                },
                {
                    "key": "td_def_career",
                    "group": "grappling",
                    "label": "takedown defence",
                    "kind": "pct",
                    "fighter_a": 0.7,
                    "fighter_b": 0.55,
                    "shrunk": False,
                },
            ],
        },
        recent_bouts={
            "note": "Synthetic note: a description of past fights, not a model input.",
            "performance_model": {"train_end": "2098-12-31", "n_decisions": 10},
            "fighter_a": [
                past("2098-10-01", "Cole Ward", "win", "Decision - Unanimous", 3, 0.81),
                past("2098-05-01", "Dan Price", "loss", "KO/TKO", 1, 0.12),
            ],
            "fighter_b": [
                past("2098-11-01", "Eli Frost", "win", "Submission", 2, None)
            ],
        },
        market={},
        attribution=ATTRIBUTION,
    )


def metrics(n: int, ll: float, br: float, acc: float, interval: bool = False) -> dict:
    out = {"n": n, "log_loss": ll, "brier": br, "accuracy": acc, "ece": 0.04}
    if interval:
        out |= {"log_loss_low": ll - 0.02, "log_loss_high": ll + 0.02}
    return out


def calibration(ns: list[int]) -> list[dict]:
    edges = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
    rows = []
    for (low, high), n in zip(edges, ns, strict=True):
        mid = (low + high) / 2
        rows.append(
            {
                "low": low,
                "high": high,
                "n": n,
                "mean_confidence": mid if n else None,
                "favourite_win_rate": mid + 0.01 if n else None,
                "win_rate_low": mid - 0.05 if n else None,
                "win_rate_high": mid + 0.06 if n else None,
                "log_loss": 0.6 if n else None,
            }
        )
    return rows


STATUSES = [
    "scored",
    "no_result",
    "pending",
    "cancelled",
    "no_live_prediction",
    "not_found",
    "unlinked",
]


def performance(live_bouts: list[dict]) -> Performance:
    folds = [
        {
            "year": y,
            "in_test_period": y >= 2097,
            "n_scored": 100,
            "metrics": {
                "model": metrics(100, 0.66 - 0.005 * i, 0.23, 0.6),
                "logistic": metrics(100, 0.665 - 0.004 * i, 0.235, 0.59),
                "elo": metrics(100, 0.68, 0.24, 0.56),
            },
        }
        for i, y in enumerate(range(2093, 2099))
    ]
    counts = {s: sum(1 for b in live_bouts if b["status"] == s) for s in STATUSES}
    scored = [b for b in live_bouts if b["status"] == "scored"]
    n = len(scored)
    return Performance(
        generated_at=GENERATED,
        model={"version": MODEL, "status": "synthetic", "parent": CANDIDATE},
        separation="Synthetic separation note: backtest and live are never pooled.",
        backtest={
            "label": "Backtest: synthetic",
            "reports": "data/reports/test",
            "commands": ["uv run python -m ufc.models evaluate"],
            "candidate_version": CANDIDATE,
            "production_version": MODEL,
            "production_note": "Synthetic production note.",
            "test_period": {
                "candidate_version": CANDIDATE,
                "first_date": "2097-01-05",
                "last_date": "2098-12-20",
                "n_scored": 200,
                "n_no_result": 3,
                "already_seen": True,
                "already_seen_note": "synthetic already-seen note",
                "metrics": {
                    "model": metrics(200, 0.63, 0.22, 0.64, interval=True),
                    "logistic": metrics(200, 0.635, 0.222, 0.65, interval=True),
                    "elo": metrics(200, 0.68, 0.24, 0.56, interval=True),
                },
                "calibration": calibration([60, 60, 50, 30]),
                "calibration_buckets": "the favoured fighter's probability",
            },
            "walk_forward": {
                "schedule": "synthetic schedule",
                "folds": folds,
                "pooled_before_test_period": {
                    "years": (2093, 2096),
                    "metrics": {"model": metrics(400, 0.655, 0.23, 0.6)},
                },
                "pooled_test_period": {
                    "years": (2097, 2098),
                    "metrics": {"model": metrics(200, 0.635, 0.22, 0.64)},
                },
            },
        },
        backtest_unavailable=None,
        era_shift_caveat={
            "text": "Synthetic era-shift caveat.",
            "backtest_years": (2093, 2096),
            "backtest_log_loss": 0.655,
            "backtest_year_range": (0.64, 0.66),
            "test_period_log_loss": 0.635,
        },
        live={
            "label": "Live: synthetic",
            "rules": ["Synthetic rule one.", "Synthetic rule two."],
            "counts": counts,
            "metrics": metrics(n, 0.6, 0.21, 1.0) if n else None,
            "metrics_without_limited_data": metrics(n, 0.6, 0.21, 1.0) if n else None,
            "by_model_version": {MODEL: metrics(n, 0.6, 0.21, 1.0)} if n else {},
            "calibration": calibration([0, n, 0, 0]),
            "calibration_buckets": "the favoured fighter's probability",
            "bouts": live_bouts,
        },
    )


def live_bout(bout_id: str, status: str, **kw) -> dict:
    base = {
        "bout_id": bout_id,
        "event_key": "wp:900",
        "event_name": "Test Fight Night: Past",
        "event_date": "2098-12-01",
        "fighter_a": "Lena Moss",
        "fighter_b": "Rita Gale",
        "status": status,
        "detail": None,
        "prediction_id": None,
        "model_version": None,
        "predicted_at": None,
        "p_fighter_a": None,
        "limited_data": None,
        "result_bout_id": None,
        "outcome": None,
        "log_loss": None,
        "brier": None,
        "correct": None,
    }
    return base | kw


def write(model: BaseModel, path: Path) -> None:
    text = json.dumps(
        model.model_dump(mode="json", by_alias=True), indent=2, ensure_ascii=False
    )
    path.write_text(text + "\n", encoding="utf-8")


def main(out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write(upcoming(), out / "upcoming.json")
    write(bout(), out / "bout.json")
    write(performance([]), out / "performance-empty.json")
    scored = live_bout(
        "wb00000000000000a1",
        "scored",
        prediction_id=7,
        model_version=MODEL,
        predicted_at="2098-11-28T09:00:00Z",
        p_fighter_a=0.6,
        limited_data=False,
        result_bout_id="cccccccccccccccc",
        outcome="fighter_a",
        log_loss=0.5108,
        brier=0.16,
        correct=1.0,
    )
    pending = live_bout(
        "wb00000000000000a2",
        "pending",
        fighter_a="Nora Pike",
        fighter_b="Vera Lam",
        prediction_id=8,
        model_version=MODEL,
        predicted_at="2098-11-28T09:00:00Z",
        p_fighter_a=0.45,
        limited_data=True,
    )
    write(performance([scored, pending]), out / "performance-live.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
