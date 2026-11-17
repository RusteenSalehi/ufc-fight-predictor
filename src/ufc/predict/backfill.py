"""Fill missing height, reach and date of birth from Wikipedia (D46, D51).

The dataset captures a profile when a fighter first appears and refreshes it
rarely, so recent debutants often have no height or reach. For a fighter on
an upcoming card, a field the dataset lacks is taken from the infobox of the
article the card links to, if it has one. Rules:

- only fields that are empty upstream are ever looked up;
- the ``fighters`` table is never written; backfilled values live in
  ``attribute_backfill``, one row per fighter and field, with their source;
- when attributes are merged for prediction, an upstream value always wins,
  including one that appears upstream after a value was backfilled.

These attributes feed predictions only. Training uses upstream profiles
alone, so nothing here can change what a model was fitted on.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date

from ufc.features.data import FighterAttrs
from ufc.predict.cards_db import log
from ufc.predict.infobox import parse_infobox
from ufc.predict.wikipedia import PROFILE_MAX_AGE_S, Wikipedia, WikipediaError

logger = logging.getLogger(__name__)

FIELDS: tuple[str, ...] = ("height_cm", "reach_cm", "dob")
UPSTREAM = "upstream"
WIKIPEDIA = "wikipedia"


def missing_fields(attrs: FighterAttrs | None) -> tuple[str, ...]:
    if attrs is None:
        return FIELDS
    return tuple(f for f in FIELDS if getattr(attrs, f) is None)


@dataclass(frozen=True, slots=True)
class BackfillResult:
    pages_requested: int
    fields_written: int
    failed: bool = False


def backfill(
    conn: sqlite3.Connection,
    client: Wikipedia,
    wanted: Mapping[str, str],
    upstream: Mapping[str, FighterAttrs],
    now: str,
) -> BackfillResult:
    """``wanted`` maps fighter refs to the article their card links to.
    Looks up only fields the dataset lacks; writes only those."""
    todo = {
        ref: (page, missing_fields(upstream.get(ref)))
        for ref, page in wanted.items()
        if page and missing_fields(upstream.get(ref))
    }
    if not todo:
        return BackfillResult(0, 0)
    titles = sorted({page for page, _ in todo.values()})
    try:
        pages = client.pages(titles, PROFILE_MAX_AGE_S)
    except WikipediaError as exc:
        log(conn, now, "warning", f"attribute backfill skipped: {exc}")
        return BackfillResult(len(titles), 0, failed=True)
    written = 0
    for ref, (title, fields) in sorted(todo.items()):
        page = pages.get(title)
        if page is None or not page.exists:
            continue
        found = parse_infobox(page.wikitext or "")
        for name in fields:
            value = getattr(found, name)
            if value is None:
                continue
            text = value.isoformat() if isinstance(value, date) else repr(float(value))
            conn.execute(
                """
                INSERT INTO attribute_backfill
                    (fighter_ref, field, value, source, page_title, revid, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (fighter_ref, field) DO UPDATE SET
                    value = excluded.value, page_title = excluded.page_title,
                    revid = excluded.revid, fetched_at = excluded.fetched_at
                """,
                (ref, name, text, WIKIPEDIA, page.title, page.revid, now),
            )
            written += 1
    return BackfillResult(len(titles), written)


def _value(field: str, text: str):
    return date.fromisoformat(text) if field == "dob" else float(text)


def merged_attributes(
    conn: sqlite3.Connection, upstream: Mapping[str, FighterAttrs]
) -> tuple[dict[str, FighterAttrs], dict[str, dict[str, str | None]]]:
    """Upstream attributes with backfilled values in the gaps only, and the
    source of every field of every fighter that has a backfilled row."""
    merged = dict(upstream)
    sources: dict[str, dict[str, str | None]] = {}
    rows = conn.execute(
        "SELECT fighter_ref, field, value FROM attribute_backfill ORDER BY fighter_ref"
    ).fetchall()
    for row in rows:
        ref, field_name = row["fighter_ref"], row["field"]
        base = merged.get(ref) or FighterAttrs(ref)
        if getattr(upstream.get(ref), field_name, None) is not None:
            continue  # the dataset has it: never overwritten
        merged[ref] = replace(base, **{field_name: _value(field_name, row["value"])})
        sources.setdefault(ref, {})[field_name] = WIKIPEDIA
    return merged, sources


def attribute_sources(
    ref: str,
    upstream: Mapping[str, FighterAttrs],
    backfilled: Mapping[str, Mapping[str, str | None]],
) -> dict[str, str | None]:
    """Per field: 'upstream', 'wikipedia', or None if unknown."""
    base = upstream.get(ref)
    out: dict[str, str | None] = {}
    for name in (*FIELDS, "stance"):
        if base is not None and getattr(base, name) is not None:
            out[name] = UPSTREAM
        else:
            out[name] = backfilled.get(ref, {}).get(name)
    return out
