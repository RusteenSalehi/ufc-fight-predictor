"""Predictions for upcoming bouts with the approved model (D53).

The model is the version ``data/models/approved.json`` names (D4); nothing
here chooses one. Inputs are built the way the model's training inputs were:
the feature settings and fitted parameters stored beside it
(``feature_pipeline.json``) and ``build_matchup`` at the event date, which
reads only bouts strictly before that date (D23). Physical attributes are the
dataset's, with Wikipedia values filling gaps (D51).

A prediction is a row that is never edited except to mark it superseded. A
bout gets a new row when its inputs change (model version, either fighter,
the feature values, or its limited-data reasons); otherwise the current row
stands, so refreshing twice does not churn the record.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np
import pandas as pd

from ufc.features.build import build_features
from ufc.features.config import FeatureConfig
from ufc.features.data import FeatureData, FighterAttrs
from ufc.features.matchup import Matchup, build_matchup
from ufc.features.store import FeatureStore, FittedParams
from ufc.ingest.parsing import name_key
from ufc.models.recipe import Candidate
from ufc.predict.backfill import FIELDS, attribute_sources
from ufc.predict.cards_db import supersede
from ufc.predict.staleness import Staleness

ATTRIBUTE_LABELS = {
    "height_cm": "height",
    "reach_cm": "reach",
    "dob": "date of birth",
    "stance": "stance",
}


def feature_config(pipeline: Mapping) -> FeatureConfig:
    raw = dict(pipeline["config"])
    raw["train_end"] = date.fromisoformat(raw["train_end"])
    return FeatureConfig(**raw)


def feature_store(
    data: FeatureData, pipeline: Mapping, fighters: Mapping[str, FighterAttrs]
) -> FeatureStore:
    """Every completed bout, with the model's own settings and fitted
    parameters; attributes from ``fighters``."""
    params = FittedParams.from_dict(pipeline["params"])
    build = build_features(
        replace(data, fighters=dict(fighters)), feature_config(pipeline), params
    )
    return build.store


def model_inputs(matchup: Matchup, names: tuple[str, ...]) -> pd.DataFrame:
    values = {**matchup.features, **matchup.levels, **matchup.candidates}
    missing = [n for n in names if n not in values]
    if missing:
        raise ValueError(f"the matchup lacks model inputs: {missing}")
    return pd.DataFrame([[values[n] for n in names]], columns=list(names))


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.floating | np.integer | np.bool_):
        return _clean(value.item())
    return value


def _clean_map(values: Mapping) -> dict:
    return {k: _clean(v) for k, v in values.items()}


def limited_reasons(
    matchup: Matchup,
    names: tuple[str, str],
    attrs: tuple[FighterAttrs | None, FighterAttrs | None],
    stale: list[str],
) -> list[str]:
    reasons: list[str] = []
    for side, name in zip("ab", names, strict=True):
        if matchup.flags[f"{side}_debut"]:
            reasons.append(f"debut: {name} has no earlier UFC bout in the dataset")
    for side, name, own in zip("ab", names, attrs, strict=True):
        if matchup.flags[f"{side}_attr_gap"]:
            lacking = [
                ATTRIBUTE_LABELS[f]
                for f in (*FIELDS, "stance")
                if own is None or getattr(own, f) is None
            ]
            reasons.append(f"attribute gap: no {', '.join(lacking)} for {name}")
    return reasons + stale


@dataclass(slots=True)
class PredictionSummary:
    made: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    superseded: list[str] = field(default_factory=list)


def predict_upcoming(
    conn: sqlite3.Connection,
    candidate: Candidate,
    version: str,
    store: FeatureStore,
    upstream: Mapping[str, FighterAttrs],
    backfilled: Mapping[str, Mapping[str, str | None]],
    staleness: Staleness,
    today: date,
    now: str,
) -> PredictionSummary:
    """Predict every resolved, scheduled bout of every event after ``today``."""
    names = candidate.spec.names
    summary = PredictionSummary()
    bouts = conn.execute(
        """
        SELECT b.*, e.date AS event_date
        FROM upcoming_bouts b JOIN upcoming_events e USING (event_key)
        WHERE b.status = 'scheduled' AND b.unresolved_reason IS NULL
          AND e.status = 'scheduled' AND e.date > ?
        ORDER BY e.date, b.event_key, b.card_order
        """,
        (today.isoformat(),),
    ).fetchall()
    for bout in bouts:
        on = date.fromisoformat(bout["event_date"])
        a_ref, b_ref = bout["fighter_a_ref"], bout["fighter_b_ref"]
        matchup = build_matchup(store, a_ref, b_ref, on, bout["weight_class"])
        p_a, p_b = candidate.predict_pair(model_inputs(matchup, names))
        p_a, p_b = float(p_a[0]), float(p_b[0])
        fighter_names = (bout["name_a"], bout["name_b"])
        stale = staleness.bout_reasons(
            [
                (bout["name_a"], a_ref, name_key(bout["name_a"])),
                (bout["name_b"], b_ref, name_key(bout["name_b"])),
            ]
        )
        reasons = limited_reasons(
            matchup,
            fighter_names,
            (store.fighters.get(a_ref), store.fighters.get(b_ref)),
            stale,
        )
        inputs = {
            "model_version": version,
            "fighter_a_ref": a_ref,
            "fighter_b_ref": b_ref,
            "event_date": on.isoformat(),
            "weight_class": bout["weight_class"],
            "inputs": [_clean(float(v)) for v in model_inputs(matchup, names).iloc[0]],
            "limited_reasons": reasons,
        }
        digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        current = conn.execute(
            "SELECT inputs_sha256 FROM predictions WHERE bout_id = ? AND superseded_at IS NULL",
            (bout["bout_id"],),
        ).fetchone()
        if current is not None and current[0] == digest:
            summary.unchanged.append(bout["bout_id"])
            continue
        if current is not None:
            supersede(conn, bout["bout_id"], now, "inputs changed")
            summary.superseded.append(bout["bout_id"])
        features = {
            "model_inputs": dict(zip(names, inputs["inputs"], strict=True)),
            "flags": _clean_map(matchup.flags),
            "fighter_a": _clean_map(matchup.a),
            "fighter_b": _clean_map(matchup.b),
            "attribute_sources": {
                "fighter_a": attribute_sources(a_ref, upstream, backfilled),
                "fighter_b": attribute_sources(b_ref, upstream, backfilled),
            },
            "weight_class": bout["weight_class"],
            "event_date": on.isoformat(),
        }
        conn.execute(
            """
            INSERT INTO predictions
                (bout_id, model_version, created_at, event_date, fighter_a_ref,
                 fighter_b_ref, p_fighter_a, p_fighter_b, inputs_sha256,
                 features_json, shap_json, limited_data, limited_reasons_json, is_live)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
            """,
            (
                bout["bout_id"],
                version,
                now,
                on.isoformat(),
                a_ref,
                b_ref,
                p_a,
                p_b,
                digest,
                json.dumps(features, sort_keys=True),
                int(bool(reasons)),
                json.dumps(reasons),
                int(now[:10] < on.isoformat()),
            ),
        )
        summary.made.append(bout["bout_id"])
    return summary
