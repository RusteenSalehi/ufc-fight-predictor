"""One refresh: cards from Wikipedia, resolution, backfill, staleness,
predictions, and the JSON dump.

Order and failure handling (D50):

1. Read "List of UFC events". If it cannot be fetched or parsed, stop:
   nothing in the database changes and nothing is published.
2. Read every scheduled event page in one request. Each event's card is
   applied, or, if its page is missing, unreadable or implausible, left
   exactly as it was, with the failure logged.
3. Check staleness against completed events after the dataset's last one.
4. Fill missing height, reach and date of birth from linked articles.
5. Predict every resolved bout with the approved model; if no model is
   approved, skip predictions.
6. Validate and write ``data/publish/predictions.json``, or keep the last
   good file.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from ufc.features.data import load_feature_data
from ufc.ingest.identity import FighterResolver
from ufc.models import registry
from ufc.predict import publish
from ufc.predict.backfill import BackfillResult, backfill, merged_attributes
from ufc.predict.cards import (
    LIST_PAGE,
    CardParseError,
    EventList,
    parse_card,
    parse_event_list,
)
from ufc.predict.cards_db import (
    CardChange,
    apply_card,
    bout_id,
    log,
    retire_events,
    upsert_event,
)
from ufc.predict.predict import PredictionSummary, feature_store, predict_upcoming
from ufc.predict.resolve import resolve_bout
from ufc.predict.staleness import Staleness, check_staleness, dataset_last_event
from ufc.predict.wikipedia import CARD_MAX_AGE_S, Wikipedia, WikipediaError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RefreshResult:
    run_at: str
    ok: bool = False
    stopped: str | None = None  # why the refresh stopped before changing anything
    changes: list[CardChange] = field(default_factory=list)
    event_errors: dict[str, str] = field(default_factory=dict)
    staleness: Staleness | None = None
    backfill: BackfillResult | None = None
    predictions: PredictionSummary | None = None
    model_version: str | None = None
    model_error: str | None = None
    published: Path | None = None
    publish_problems: list[str] = field(default_factory=list)
    requests: int = 0


def _event_list(client: Wikipedia) -> EventList:
    page = client.pages([LIST_PAGE], CARD_MAX_AGE_S)[LIST_PAGE]
    if not page.exists:
        raise CardParseError(f"{LIST_PAGE!r} does not exist")
    return parse_event_list(page.wikitext or "")


def refresh(
    conn: sqlite3.Connection,
    client: Wikipedia,
    resolver: FighterResolver,
    *,
    models_root: Path,
    publish_dir: Path,
    today: date,
    now: str,
) -> RefreshResult:
    result = RefreshResult(run_at=now)

    # 1. The schedule. Nothing changes if it cannot be read.
    try:
        events = _event_list(client)
    except (WikipediaError, CardParseError) as exc:
        result.stopped = f"could not read {LIST_PAGE!r}: {exc}"
        with conn:
            log(
                conn,
                now,
                "error",
                f"refresh stopped, nothing changed: {result.stopped}",
            )
        result.requests = client.requests_made
        return result

    # 2. Event pages, in one request, and their cards.
    try:
        pages = client.pages([e.page for e in events.scheduled], CARD_MAX_AGE_S)
    except WikipediaError as exc:
        result.stopped = f"could not read the event pages: {exc}"
        with conn:
            log(
                conn,
                now,
                "error",
                f"refresh stopped, nothing changed: {result.stopped}",
            )
        result.requests = client.requests_made
        return result

    listed_keys: set[str] = set()
    for listed in events.scheduled:
        page = pages.get(listed.page)
        with conn:
            if page is None or not page.exists or page.pageid is None:
                result.event_errors[listed.page] = "event page not found"
                # Still listed: keep the stored event as it is.
                listed_keys.update(
                    r["event_key"]
                    for r in conn.execute(
                        "SELECT event_key FROM upcoming_events WHERE page_title = ?",
                        (listed.page,),
                    )
                )
                log(
                    conn,
                    now,
                    "error",
                    f"{listed.page!r}: event page not found; card unchanged",
                )
                continue
            key = upsert_event(conn, listed, page, now, today)
            listed_keys.add(key)
            if listed.date <= today:
                continue  # under way or over: its card is no longer changed
            try:
                card = parse_card(page.wikitext or "")
            except CardParseError as exc:
                result.event_errors[listed.page] = str(exc)
                log(conn, now, "error", f"card not read ({exc}); card unchanged", key)
                continue
            bouts = [
                (
                    b,
                    resolve_bout(
                        resolver, b, bout_id(page.pageid, b.fighter_a, b.fighter_b)
                    ),
                )
                for b in card.bouts
                if not b.completed
            ]
            change = apply_card(conn, key, page.pageid, bouts, now)
            result.changes.append(change)
            if not change.applied:
                result.event_errors[listed.page] = change.reason or "card not applied"
    with conn:
        retire_events(conn, listed_keys, today, now)

    # 3. Staleness.
    result.staleness = check_staleness(
        client, resolver, events, dataset_last_event(conn), today
    )
    if result.staleness.status != "current":
        with conn:
            log(
                conn,
                now,
                "warning",
                f"dataset is {result.staleness.status}: {result.staleness.missing}",
            )

    # 4. Attribute backfill for fighters on upcoming cards.
    data = load_feature_data(conn)
    wanted: dict[str, str] = {}
    for row in conn.execute(
        """
        SELECT b.fighter_a_ref, b.page_a, b.fighter_b_ref, b.page_b
        FROM upcoming_bouts b JOIN upcoming_events e USING (event_key)
        WHERE b.status = 'scheduled' AND b.unresolved_reason IS NULL
          AND e.status = 'scheduled' AND e.date > ?
        """,
        (today.isoformat(),),
    ):
        for side in "ab":
            if row[f"page_{side}"]:
                wanted.setdefault(row[f"fighter_{side}_ref"], row[f"page_{side}"])
    with conn:
        result.backfill = backfill(conn, client, wanted, data.fighters, now)
    merged, sources = merged_attributes(conn, data.fighters)

    # 5. Predictions with the approved model.
    model_record = None
    spec_names = None
    try:
        candidate, manifest = registry.load_approved(models_root)
    except registry.RegistryError as exc:
        result.model_error = str(exc)
        with conn:
            log(conn, now, "warning", f"no predictions: {exc}")
    else:
        version = manifest["version"]
        result.model_version = version
        pipeline = registry.read_json(models_root, version, "feature_pipeline.json")
        store = feature_store(data, pipeline, merged)
        with conn:
            result.predictions = predict_upcoming(
                conn,
                candidate,
                version,
                store,
                data.fighters,
                sources,
                result.staleness,
                today,
                now,
            )
        spec_names = list(candidate.spec.names)
        model_record = {
            "version": version,
            "status": (manifest.get("lineage") or {}).get("status", "candidate"),
            "parent": (manifest.get("lineage") or {}).get("parent"),
        }

    # 6. Validate and publish.
    record = publish.snapshot(
        conn,
        today=today,
        generated_at=now,
        model=model_record,
        staleness=result.staleness.to_dict(),
        dataset={
            "source_commit": data.source_commit,
            "last_event": result.staleness.to_dict()["dataset_last_event"],
        },
    )
    result.publish_problems = publish.validate(record, today, spec_names)
    with conn:
        if result.publish_problems:
            log(
                conn,
                now,
                "error",
                "not published: " + "; ".join(result.publish_problems[:5]),
            )
        else:
            result.published = publish.write(record, publish_dir)
    result.ok = not result.event_errors and not result.publish_problems
    result.requests = client.requests_made
    return result
