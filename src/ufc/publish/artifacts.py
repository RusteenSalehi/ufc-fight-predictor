"""Builders for the per-bout artifacts, ``upcoming.json`` and the archive
snapshots, from the current-predictions record (``ufc.predict.publish.
snapshot``) and what is stored with each prediction.

Everything a bout page shows comes from the prediction row: its calibrated
probabilities, the contributions and as-of values stored with it, and its
limited-data reasons. The "why" and the factors are rendered from the stored
contributions with the current templates (``snapshot`` renders the "why";
D65), so a wording fix reaches every published bout without touching an
immutable prediction. The only
thing computed from the dataset at publish time is the recent-bouts context
(``history.py``), which no prediction reads.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from ufc.predict.explain import METHOD, STATS, factors
from ufc.predict.publish import ATTRIBUTION
from ufc.publish.history import NOTE as HISTORY_NOTE
from ufc.publish.history import History
from ufc.publish.schemas import fair_moneyline

N_FACTORS = 5

DISPLAY: tuple[tuple[str, str], ...] = (
    ("age", "physical"),
    ("height_cm", "physical"),
    ("reach_cm", "physical"),
    ("n_fights", "record"),
    ("wins", "record"),
    ("losses", "record"),
    ("win_streak", "record"),
    ("loss_streak", "record"),
    ("ko_wins", "record"),
    ("sub_wins", "record"),
    ("ko_losses", "record"),
    ("title_fights", "record"),
    ("elo", "rating"),
    ("opp_elo_career", "rating"),
    ("days_since_last", "activity"),
    ("fights_last_2y", "activity"),
    ("slpm_career", "striking"),
    ("sapm_career", "striking"),
    ("sig_acc_career", "striking"),
    ("sig_def_career", "striking"),
    ("kd_per15_career", "striking"),
    ("td_per15_career", "grappling"),
    ("td_acc_career", "grappling"),
    ("td_def_career", "grappling"),
    ("sub_att_per15_career", "grappling"),
    ("ctrl_share_career", "grappling"),
)
"""The as-of values shown for each fighter, in display order."""

STATS_NOTE = (
    "As of the event date, from UFC bouts in the dataset only (no other "
    "promotions). Per-minute rates are shrunk toward the weight-class average "
    "by the stated number of cage minutes (DECISIONS D33), as the model reads "
    "them. Height, reach and date of birth may come from Wikipedia where the "
    "dataset has none (attribute_sources)."
)
MONEYLINE_NOTE = (
    "American odds with no bookmaker margin, from the calibrated probability. "
    "A conversion of the model's number, not a price and not betting advice."
)
STANCES = (
    ("stance_orthodox", "Orthodox"),
    ("stance_southpaw", "Southpaw"),
    ("stance_switch", "Switch"),
)


@dataclass(frozen=True, slots=True)
class Stored:
    """What is stored with one prediction, parsed."""

    inputs_sha256: str
    values: dict[str, dict]  # fighter_a / fighter_b as-of values
    attribute_sources: dict[str, dict[str, str]]
    toward_a: dict[str, float] | None  # None for rows made before D59


def read_stored(
    conn: sqlite3.Connection, prediction_ids: Sequence[int]
) -> dict[int, Stored]:
    out = {}
    for i in range(0, len(prediction_ids), 500):
        chunk = list(prediction_ids[i : i + 500])
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(
            f"""SELECT prediction_id, inputs_sha256, features_json, shap_json
            FROM predictions WHERE prediction_id IN ({marks})""",
            chunk,
        ):
            features = json.loads(row["features_json"])
            shap = json.loads(row["shap_json"]) if row["shap_json"] else None
            out[row["prediction_id"]] = Stored(
                row["inputs_sha256"],
                {
                    "fighter_a": features["fighter_a"],
                    "fighter_b": features["fighter_b"],
                },
                features["attribute_sources"],
                shap["log_odds_toward_a"] if shap else None,
            )
    return out


def _label(key: str) -> str:
    if key.endswith("_career"):
        return f"{STATS[key[: -len('_career')]].label}, career"
    return STATS[key].label


def _kind(key: str) -> str:
    return STATS[key.removesuffix("_career")].kind


def _stance(values: Mapping) -> str | None:
    for key, name in STANCES:
        if values.get(key):
            return name
    return None


def stats_block(stored: Stored, shrink_minutes: float | None, shrunk: set[str]) -> dict:
    a, b = stored.values["fighter_a"], stored.values["fighter_b"]
    return {
        "note": STATS_NOTE,
        "shrink_minutes": shrink_minutes,
        "stance": {"fighter_a": _stance(a), "fighter_b": _stance(b)},
        "attribute_sources": stored.attribute_sources,
        "rows": [
            {
                "key": key,
                "group": group,
                "label": _label(key),
                "kind": _kind(key),
                "fighter_a": a.get(key),
                "fighter_b": b.get(key),
                "shrunk": key in shrunk,
            }
            for key, group in DISPLAY
        ],
    }


def _brief(fighter: Mapping) -> dict:
    return {
        "name": fighter["name"],
        "ref": fighter["ref"],
        "provisional": fighter["provisional"],
        "wikipedia_page": fighter["wikipedia_page"],
    }


def _event_info(event: Mapping) -> dict:
    return {
        "event_key": event["event_key"],
        "name": event["name"],
        "date": event["date"],
        "wikipedia_page": event["wikipedia_page"],
        "wikipedia_revision": event["wikipedia_revision"],
    }


def dataset_id(ref: str | None, links: Mapping[str, str | None]) -> str | None:
    if ref is None:
        return None
    return links.get(ref) if ref.startswith("wp") else ref


@dataclass(frozen=True, slots=True)
class ModelDetails:
    """What the bout pages need about the model that made a prediction."""

    shrink_minutes: float | None
    shrunk: set[str]


def bout_artifact(
    event: Mapping,
    bout: Mapping,
    stored: Stored,
    history: History,
    links: Mapping[str, str | None],
    model: ModelDetails,
    generated_at: str,
) -> dict:
    p_a, p_b = bout["p_fighter_a"], bout["p_fighter_b"]
    top = (
        {
            **factors(
                stored.toward_a,
                stored.values["fighter_a"],
                stored.values["fighter_b"],
                N_FACTORS,
            ),
            "method": METHOD,
        }
        if stored.toward_a is not None
        else None
    )
    on = date.fromisoformat(event["date"])
    return {
        "generated_at": generated_at,
        "bout_id": bout["bout_id"],
        "event": _event_info(event),
        "order": bout["order"],
        "segment": bout["segment"],
        "weight_class": bout["weight_class"],
        "title_bout": bout["title_bout"],
        "fighter_a": {
            **_brief(bout["fighter_a"]),
            "resolution": bout["fighter_a"]["resolution"],
        },
        "fighter_b": {
            **_brief(bout["fighter_b"]),
            "resolution": bout["fighter_b"]["resolution"],
        },
        "prediction": {
            "prediction_id": bout["prediction_id"],
            "model_version": bout["model_version"],
            "predicted_at": bout["predicted_at"],
            "p_fighter_a": p_a,
            "p_fighter_b": p_b,
            "predicted_winner": bout["predicted_winner"],
            "fair_moneyline": {
                "fighter_a": fair_moneyline(p_a),
                "fighter_b": fair_moneyline(p_b),
            },
            "fair_moneyline_note": MONEYLINE_NOTE,
        },
        "limited_data": {
            "limited": bout["limited_data"],
            "reasons": bout["limited_data_reasons"],
            "stale_stats": bout["stale_stats"],
        },
        "why": bout["why"],
        "top_factors": top,
        "stats": stats_block(stored, model.shrink_minutes, model.shrunk),
        "recent_bouts": {
            "note": HISTORY_NOTE,
            "performance_model": history.performance_model_info(),
            "fighter_a": history.recent(
                dataset_id(bout["fighter_a"]["ref"], links), on
            ),
            "fighter_b": history.recent(
                dataset_id(bout["fighter_b"]["ref"], links), on
            ),
        },
        "market": {},
        "attribution": ATTRIBUTION,
    }


def upcoming_artifact(record: Mapping) -> dict:
    """``upcoming.json`` from the current-predictions record."""
    events = []
    for event in record["events"]:
        bouts = [
            {
                "bout_id": b["bout_id"],
                "order": b["order"],
                "segment": b["segment"],
                "weight_class": b["weight_class"],
                "title_bout": b["title_bout"],
                "fighter_a": _brief(b["fighter_a"]),
                "fighter_b": _brief(b["fighter_b"]),
                "p_fighter_a": b["p_fighter_a"],
                "p_fighter_b": b["p_fighter_b"],
                "predicted_winner": b["predicted_winner"],
                "model_version": b["model_version"],
                "predicted_at": b["predicted_at"],
                "limited_data": b["limited_data"],
                "limited_data_reasons": b["limited_data_reasons"],
                "stale_stats": b["stale_stats"],
                "detail": f"bouts/{b['bout_id']}.json",
            }
            for b in event["bouts"]
        ]
        unresolved = [
            {
                "bout_id": b["bout_id"],
                "order": b["order"],
                "segment": b["segment"],
                "weight_class": b["weight_class"],
                "title_bout": b["title_bout"],
                "fighter_a": _brief(b["fighter_a"]),
                "fighter_b": _brief(b["fighter_b"]),
                "status": "not_predicted"
                if b["reason"] == "not predicted"
                else "unresolved",
                "reason": b["reason"],
            }
            for b in event["unresolved_bouts"]
        ]
        events.append(
            {
                **_event_info(event),
                "card_checked_at": event["card_checked_at"],
                "bouts": bouts,
                "unresolved_bouts": unresolved,
            }
        )
    return {
        "generated_at": record["generated_at"],
        "model": record["model"],
        "dataset": record["dataset"],
        "staleness": record["staleness"],
        "events": events,
        "attribution": record["attribution"],
    }


def content_sha256(content: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def archive_snapshots(
    upcoming: Mapping, bouts: Mapping[str, Mapping], stored: Mapping[int, Stored]
) -> list[dict]:
    """One snapshot per event with predicted bouts: what the card showed.
    ``bouts`` are the bout artifacts by ID (for the rendered "why")."""
    out = []
    for event in upcoming["events"]:
        if not event["bouts"]:
            continue
        archived = []
        for b in event["bouts"]:
            artifact = bouts[b["bout_id"]]
            why = artifact["why"]
            archived.append(
                {
                    "bout_id": b["bout_id"],
                    "order": b["order"],
                    "weight_class": b["weight_class"],
                    "fighter_a": b["fighter_a"],
                    "fighter_b": b["fighter_b"],
                    "prediction_id": artifact["prediction"]["prediction_id"],
                    "model_version": b["model_version"],
                    "predicted_at": b["predicted_at"],
                    "inputs_sha256": stored[
                        artifact["prediction"]["prediction_id"]
                    ].inputs_sha256,
                    "p_fighter_a": b["p_fighter_a"],
                    "p_fighter_b": b["p_fighter_b"],
                    "predicted_winner": b["predicted_winner"],
                    "limited_data_reasons": b["limited_data_reasons"],
                    "why_summary": why["summary"] if why else None,
                }
            )
        info = {
            k: event[k]
            for k in (
                "event_key",
                "name",
                "date",
                "wikipedia_page",
                "wikipedia_revision",
            )
        }
        content = {
            "event": info,
            "model_version": upcoming["model"]["version"],
            "bouts": archived,
            "unresolved_bouts": event["unresolved_bouts"],
        }
        # A Wikipedia edit elsewhere on the page changes the revision but not
        # what was predicted, so the revision is kept but not hashed.
        hashed = {
            **content,
            "event": {k: v for k, v in info.items() if k != "wikipedia_revision"},
        }
        out.append(
            {
                "published_at": upcoming["generated_at"],
                **content,
                "content_sha256": content_sha256(hashed),
            }
        )
    return out
