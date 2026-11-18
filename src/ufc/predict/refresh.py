"""One refresh: cards from Wikipedia, resolution, backfill, staleness,
predictions, and the JSON dump.

Order and failure handling (D50):

1. Read "List of UFC events". If it cannot be fetched or parsed, stop:
   nothing in the database changes and nothing is published.
2. Read every scheduled event page in one request, parse each card and
   resolve its fighters. Check provisional fighters' articles (one request);
   if that fails, stop before any card is applied. Then each event's card is
   applied, or, if its page is missing, unreadable or implausible, left
   exactly as it was, with the failure logged.
3. Check staleness against completed events after the dataset's last one.
4. Fill missing height, reach and date of birth from linked articles.
5. Predict every resolved bout with the approved model; if no model is
   approved, skip predictions.
6. Link provisional fighters of past cards who now have a dataset profile.
7. Build the site's artifacts (bout pages, ``upcoming.json``,
   ``performance.json`` with the live record), validate every one, and write
   them with archive snapshots, or keep the last good artifacts
   (``ufc.publish.gate``).
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
    link_past_provisionals,
    log,
    retire_events,
    upsert_event,
)
from ufc.predict.predict import PredictionSummary, feature_store, predict_upcoming
from ufc.predict.resolve import check_provisional, cited_profiles, resolve_bout
from ufc.predict.staleness import Staleness, check_staleness, dataset_last_event
from ufc.predict.wikipedia import (
    CARD_MAX_AGE_S,
    PROFILE_MAX_AGE_S,
    Wikipedia,
    WikipediaError,
)
from ufc.publish import gate
from ufc.publish.history import History, read_names
from ufc.publish.performance import DEFAULT_REPORTS

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
    archived: list[Path] = field(default_factory=list)
    linked: list[tuple[str, str]] = field(default_factory=list)
    live_counts: dict[str, int] = field(default_factory=dict)
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
    reports_root: Path = DEFAULT_REPORTS,
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

    # Parse every card first; nothing is written until the checks are done.
    listed_keys: set[str] = set()
    parsed: list[tuple[str, int, list]] = []
    for listed in events.scheduled:
        page = pages.get(listed.page)
        with conn:
            if page is None or not page.exists or page.pageid is None:
                result.event_errors[listed.page] = "event page not found"
                log(
                    conn,
                    now,
                    "error",
                    f"{listed.page!r}: event page not found; card unchanged",
                )
                # Still listed: keep the stored event as it is.
                listed_keys.update(
                    r["event_key"]
                    for r in conn.execute(
                        "SELECT event_key FROM upcoming_events WHERE page_title = ?",
                        (listed.page,),
                    )
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
        parsed.append((key, page.pageid, bouts))

    # A provisional fighter whose article cites a profile in the dataset is
    # not a debutant (resolve.check_provisional). One request for all of them.
    articles = sorted(
        {
            fighter.page
            for _, _, bouts in parsed
            for b, r in bouts
            for fighter, side in ((b.fighter_a, r.a), (b.fighter_b, r.b))
            if side.provisional and fighter.page
        }
    )
    try:
        fetched = client.pages(articles, PROFILE_MAX_AGE_S) if articles else {}
    except WikipediaError as exc:
        result.stopped = f"could not check provisional fighters' articles: {exc}"
        with conn:
            log(
                conn,
                now,
                "error",
                f"refresh stopped, no card applied: {result.stopped}",
            )
        result.requests = client.requests_made
        return result
    cited = {t: cited_profiles(p.wikitext or "") for t, p in fetched.items()}
    known = {r[0] for r in conn.execute("SELECT fighter_id FROM fighters")}

    for key, pageid, bouts in parsed:
        checked = [(b, check_provisional(r, b, cited, known)) for b, r in bouts]
        with conn:
            change = apply_card(conn, key, pageid, checked, now)
        result.changes.append(change)
        if not change.applied:
            page_title = conn.execute(
                "SELECT page_title FROM upcoming_events WHERE event_key = ?", (key,)
            ).fetchone()[0]
            result.event_errors[page_title] = change.reason or "card not applied"
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

    # 6. Link provisional fighters of past cards who now have a profile, so
    # that live scoring can match their bouts (the card rules, D49).
    with conn:
        result.linked = link_past_provisionals(conn, resolver, today, now)

    # 7. Build, validate and publish the site's artifacts, or keep the last
    # good ones (ufc.publish.gate).
    events_by_id, fighter_names = read_names(conn)
    staleness = result.staleness.to_dict()
    published = gate.publish(
        conn,
        publish_dir,
        today=today,
        now=now,
        model=model_record,
        model_inputs=spec_names,
        staleness=staleness,
        dataset={
            "source_commit": data.source_commit,
            "last_event": staleness["dataset_last_event"],
        },
        models_root=models_root,
        reports_root=reports_root,
        history=History.build(data.bouts, events_by_id, fighter_names),
    )
    result.publish_problems = published.problems
    result.archived = published.archived
    result.live_counts = published.live_counts
    with conn:
        if published.problems:
            log(
                conn,
                now,
                "error",
                "not published: " + "; ".join(published.problems[:5]),
            )
        else:
            result.published = publish_dir / gate.PUBLIC / gate.UPCOMING_FILE
            log(
                conn,
                now,
                "info",
                f"published {len(published.written)} file(s), "
                f"archived {len(published.archived)} snapshot(s)",
            )
    result.ok = not result.event_errors and not result.publish_problems
    result.requests = client.requests_made
    return result
