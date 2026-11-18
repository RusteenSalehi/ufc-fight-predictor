"""The current predictions as one record, and the card checks of the
validation gate.

``snapshot`` reads every scheduled event and its bouts with their current
predictions; ``validate`` lists the reasons it must not be published
(docs/ARCHITECTURE.md, "Validation gate"). The site's artifacts are built
from this record and written by ``ufc.publish.gate`` (M6); ``show`` prints
it. The "why" is rendered from the contributions and values stored with
each prediction, so the current wording applies to every row (D65).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from ufc.predict.explain import explain_values

DEFAULT_DIR = Path("data/publish")
ATTRIBUTION = (
    "Fight cards and some fighter attributes: English Wikipedia, CC BY-SA 4.0 "
    "(https://creativecommons.org/licenses/by-sa/4.0/). Fight statistics: "
    "Greco1899/scrape_ufc_stats, derived from ufcstats.com."
)


def _fighter(row: sqlite3.Row, side: str) -> dict:
    ref = row[f"fighter_{side}_ref"]
    return {
        "name": row[f"name_{side}"],
        "wikipedia_page": row[f"page_{side}"],
        "ref": ref,
        "provisional": bool(ref and ref.startswith("wp")),
        "resolution": row[f"method_{side}"],
    }


def stored_why(row: sqlite3.Row, p_a: float, p_b: float) -> dict | None:
    """The "why", rendered with the current templates from the contributions
    and as-of values stored with the prediction; None for a row made before
    explanations were stored."""
    if not row["shap_json"]:
        return None
    toward_a = json.loads(row["shap_json"])["log_odds_toward_a"]
    features = json.loads(row["features_json"])
    return explain_values(
        toward_a,
        features["fighter_a"],
        features["fighter_b"],
        (row["name_a"], row["name_b"]),
        p_a,
        p_b,
    )


def snapshot(
    conn: sqlite3.Connection,
    *,
    today: date,
    generated_at: str,
    model: dict | None,
    staleness: dict,
    dataset: dict,
) -> dict:
    """Every scheduled event after ``today`` that has bouts, with each bout's
    current prediction; unresolved bouts are listed apart, unpredicted."""
    events = []
    for event in conn.execute(
        """
        SELECT * FROM upcoming_events
        WHERE status = 'scheduled' AND date > ? ORDER BY date, event_key
        """,
        (today.isoformat(),),
    ).fetchall():
        bouts, unresolved = [], []
        for row in conn.execute(
            """
            SELECT b.*, p.prediction_id, p.model_version, p.created_at,
                   p.p_fighter_a, p.p_fighter_b, p.limited_data,
                   p.limited_reasons_json, p.is_live, p.features_json,
                   p.shap_json
            FROM upcoming_bouts b
            LEFT JOIN predictions p
              ON p.bout_id = b.bout_id AND p.superseded_at IS NULL
            WHERE b.event_key = ? AND b.status = 'scheduled'
            ORDER BY b.card_order
            """,
            (event["event_key"],),
        ).fetchall():
            base = {
                "bout_id": row["bout_id"],
                "order": row["card_order"],
                "segment": row["segment"],
                "weight_class": row["weight_class"] or row["weight_class_raw"],
                "title_bout": bool(row["is_title_bout"]),
                "fighter_a": _fighter(row, "a"),
                "fighter_b": _fighter(row, "b"),
                "first_seen_at": row["first_seen_at"],
            }
            if row["unresolved_reason"] is not None:
                unresolved.append({**base, "reason": row["unresolved_reason"]})
                continue
            if row["prediction_id"] is None:
                unresolved.append({**base, "reason": "not predicted"})
                continue
            p_a, p_b = row["p_fighter_a"], row["p_fighter_b"]
            reasons = json.loads(row["limited_reasons_json"])
            bouts.append(
                {
                    **base,
                    "prediction_id": row["prediction_id"],
                    "model_version": row["model_version"],
                    "predicted_at": row["created_at"],
                    "is_live": bool(row["is_live"]),
                    "p_fighter_a": p_a,
                    "p_fighter_b": p_b,
                    "predicted_winner": "fighter_a"
                    if p_a > p_b
                    else "fighter_b"
                    if p_b > p_a
                    else None,
                    "limited_data": bool(row["limited_data"]),
                    "limited_data_reasons": reasons,
                    "stale_stats": any(r.startswith("stale stats") for r in reasons),
                    "model_input_names": sorted(
                        json.loads(row["features_json"])["model_inputs"]
                    ),
                    # Rows made before explanations existed have none.
                    "why": stored_why(row, p_a, p_b),
                }
            )
        if not bouts and not unresolved:
            continue
        events.append(
            {
                "event_key": event["event_key"],
                "name": event["name"],
                "date": event["date"],
                "wikipedia_page": event["page_title"],
                "wikipedia_revision": event["revid"],
                "card_checked_at": event["card_applied_at"],
                "bouts": bouts,
                "unresolved_bouts": unresolved,
            }
        )
    return {
        "generated_at": generated_at,
        "model": model,
        "dataset": dataset,
        "staleness": staleness,
        "events": events,
        "attribution": ATTRIBUTION,
    }


def validate(
    record: dict, today: date, model_inputs: Sequence[str] | None
) -> list[str]:
    """Reasons the snapshot must not be published; empty if it may be."""
    problems: list[str] = []
    if record["model"] is None:
        problems.append("no approved model, so nothing was predicted")
    predicted = [b for e in record["events"] for b in e["bouts"]]
    if not predicted:
        problems.append("no predicted bouts")
    expected = sorted(model_inputs or [])
    for event in record["events"]:
        if date.fromisoformat(event["date"]) <= today:
            problems.append(f"{event['name']} is not in the future")
        for bout in event["bouts"]:
            label = f"{event['name']}: {bout['fighter_a']['name']} vs. {bout['fighter_b']['name']}"
            p_a, p_b = bout["p_fighter_a"], bout["p_fighter_b"]
            if (
                not (0.0 <= p_a <= 1.0 and 0.0 <= p_b <= 1.0)
                or abs(p_a + p_b - 1.0) > 1e-12
            ):
                problems.append(f"{label}: probabilities {p_a}, {p_b}")
            if not (bout["fighter_a"]["ref"] and bout["fighter_b"]["ref"]):
                problems.append(f"{label}: a fighter is not resolved")
            if bout["model_input_names"] != expected:
                problems.append(
                    f"{label}: inputs do not match the model's feature list"
                )
            if record["model"] and bout["model_version"] != record["model"]["version"]:
                problems.append(f"{label}: predicted by {bout['model_version']}")
    return problems
