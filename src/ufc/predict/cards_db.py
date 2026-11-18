"""Upcoming events and bouts in SQLite, and the rules for changing them.

A refresh never trusts a card it cannot read or that looks broken
(D16, D50). For each event:

- the event page could not be fetched or parsed: nothing about the event's
  bouts changes; the failure is logged;
- the page parsed but the card is implausible (no bouts where there were
  some, or most bouts gone at once): nothing changes; the failure is logged;
- otherwise the card is applied in one transaction: new pairings are added,
  pairings still listed are refreshed (``last_seen_at``), and pairings no
  longer listed are cancelled, with their current predictions superseded.

A bout's ID comes from the event's page ID and the two fighters' normalised
names, sorted, so it is stable across refreshes and page renames, does not
depend on corner, and a replacement opponent is a new bout.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from ufc.ingest.identity import FighterResolver
from ufc.predict.cards import CardBout, CardFighter, ListedEvent
from ufc.predict.resolve import BoutResolution, provisional_id, resolve_fighter
from ufc.predict.wikipedia import Page

logger = logging.getLogger(__name__)

MIN_BOUTS_FOR_SHARE_CHECK = 4
MAX_MISSING_SHARE = 0.5
"""With at least four bouts on record, losing more than half of them in one
refresh is treated as a broken page, not a real change."""


def event_key(pageid: int) -> str:
    return f"wp:{pageid}"


def bout_id(pageid: int, a: CardFighter, b: CardFighter) -> str:
    first, second = sorted((a.key, b.key))
    digest = hashlib.sha256(f"{pageid}|{first}|{second}".encode()).hexdigest()
    return f"wb{digest[:16]}"


def log(
    conn: sqlite3.Connection,
    run_at: str,
    level: str,
    message: str,
    event: str | None = None,
) -> None:
    getattr(logger, "warning" if level == "warning" else level)(
        "%s%s", f"[{event}] " if event else "", message
    )
    conn.execute(
        "INSERT INTO refresh_log (run_at, level, event_key, message) VALUES (?, ?, ?, ?)",
        (run_at, level, event, message),
    )


def implausible(previous: set[str], current: set[str]) -> str | None:
    """Why a parsed card must not replace the stored one, or None."""
    if not current:
        return (
            f"the card has no bouts, where {len(previous)} were listed"
            if previous
            else None
        )
    missing = previous - current
    if len(previous) >= MIN_BOUTS_FOR_SHARE_CHECK and len(
        missing
    ) > MAX_MISSING_SHARE * len(previous):
        return f"{len(missing)} of {len(previous)} listed bouts are missing at once"
    return None


def upsert_event(
    conn: sqlite3.Connection, listed: ListedEvent, page: Page, now: str, today: date
) -> str:
    """Record an event from the scheduled list and its page."""
    key = event_key(page.pageid)  # type: ignore[arg-type]
    status = "scheduled" if listed.date > today else "past"
    conn.execute(
        """
        INSERT INTO upcoming_events
            (event_key, pageid, page_title, name, date, revid, status,
             first_seen_at, last_seen_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (event_key) DO UPDATE SET
            page_title = excluded.page_title, name = excluded.name,
            date = excluded.date, revid = excluded.revid, status = excluded.status,
            last_seen_at = excluded.last_seen_at
        """,
        (
            key,
            page.pageid,
            page.title,
            listed.name,
            listed.date.isoformat(),
            page.revid,
            status,
            now,
            now,
        ),
    )
    return key


def retire_events(
    conn: sqlite3.Connection, listed_keys: set[str], today: date, now: str
) -> None:
    """Events whose date has passed become 'past'. A scheduled event that is
    no longer in the list becomes 'unlisted': its bouts are left as they are
    and it is not published, because a missing row is not evidence that the
    card was cancelled."""
    conn.execute(
        "UPDATE upcoming_events SET status = 'past' WHERE status != 'past' AND date <= ?",
        (today.isoformat(),),
    )
    for row in conn.execute(
        "SELECT event_key, name FROM upcoming_events WHERE status = 'scheduled'"
    ).fetchall():
        if row["event_key"] not in listed_keys:
            conn.execute(
                "UPDATE upcoming_events SET status = 'unlisted' WHERE event_key = ?",
                (row["event_key"],),
            )
            log(
                conn,
                now,
                "warning",
                f"{row['name']} is no longer in the scheduled list",
                row["event_key"],
            )


@dataclass(slots=True)
class CardChange:
    event: str
    applied: bool
    reason: str | None = None
    added: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    restored: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)


def supersede(conn: sqlite3.Connection, bout: str, now: str, reason: str) -> int:
    """Mark the bout's current prediction superseded. Returns rows changed."""
    return conn.execute(
        """
        UPDATE predictions SET superseded_at = ?, superseded_reason = ?
        WHERE bout_id = ? AND superseded_at IS NULL
        """,
        (now, reason, bout),
    ).rowcount


def _upsert_provisional(
    conn: sqlite3.Connection, fighter: CardFighter, now: str
) -> None:
    conn.execute(
        """
        INSERT INTO provisional_fighters
            (provisional_id, page_title, name, first_seen_at, last_seen_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (provisional_id) DO UPDATE SET
            name = excluded.name, last_seen_at = excluded.last_seen_at
        """,
        (provisional_id(fighter), fighter.page, fighter.name, now, now),
    )


def _link_provisional(
    conn: sqlite3.Connection, fighter: CardFighter, fighter_id: str, now: str
) -> None:
    """A fighter once provisional now matches a UFCStats profile."""
    conn.execute(
        """
        UPDATE provisional_fighters SET fighter_id = ?, linked_at = ?
        WHERE provisional_id = ? AND fighter_id IS NULL
        """,
        (fighter_id, now, provisional_id(fighter)),
    )


def link_past_provisionals(
    conn: sqlite3.Connection, resolver: FighterResolver, today: date, now: str
) -> list[tuple[str, str]]:
    """Link provisional fighters on cards dated today or earlier whose name
    now resolves to a dataset profile (the debutant's first bout has reached
    the dataset). Card resolution only links fighters on cards it applies,
    which are future ones, so without this a debutant's bout could never be
    matched to its result. Same rules as the card (D49): a name that is
    ambiguous or still unknown stays unlinked. Returns (provisional ID,
    fighter ID) pairs linked."""
    rows = conn.execute(
        """
        SELECT DISTINCT p.provisional_id, p.name, p.page_title, b.weight_class
        FROM provisional_fighters p
        JOIN upcoming_bouts b
          ON p.provisional_id IN (b.fighter_a_ref, b.fighter_b_ref)
        JOIN upcoming_events e USING (event_key)
        WHERE p.fighter_id IS NULL AND e.date <= ?
        ORDER BY p.provisional_id
        """,
        (today.isoformat(),),
    ).fetchall()
    linked: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if row["provisional_id"] in seen:
            continue
        seen.add(row["provisional_id"])
        fighter = CardFighter(row["name"], row["page_title"])
        r = resolve_fighter(resolver, fighter, "", row["weight_class"])
        if r.ref is None or r.provisional:
            continue
        _link_provisional(conn, fighter, r.ref, now)
        log(conn, now, "info", f"linked provisional {row['provisional_id']} to {r.ref}")
        linked.append((row["provisional_id"], r.ref))
    return linked


def apply_card(
    conn: sqlite3.Connection,
    event: str,
    pageid: int,
    bouts: Sequence[tuple[CardBout, BoutResolution]],
    now: str,
) -> CardChange:
    """Apply one parsed card to the event's stored bouts, or nothing at all."""
    ids: dict[str, tuple[CardBout, BoutResolution]] = {}
    for bout, resolution in bouts:
        key = bout_id(pageid, bout.fighter_a, bout.fighter_b)
        if key in ids:
            log(
                conn,
                now,
                "warning",
                f"pairing listed twice: {bout.fighter_a.name} vs. {bout.fighter_b.name}",
                event,
            )
            continue
        ids[key] = (bout, resolution)
    previous = {
        row["bout_id"]
        for row in conn.execute(
            "SELECT bout_id FROM upcoming_bouts WHERE event_key = ? AND status = 'scheduled'",
            (event,),
        )
    }
    reason = implausible(previous, set(ids))
    if reason is not None:
        log(conn, now, "error", f"card not applied: {reason}", event)
        return CardChange(event, applied=False, reason=reason)

    known = {
        row["bout_id"]: row["status"]
        for row in conn.execute(
            "SELECT bout_id, status FROM upcoming_bouts WHERE event_key = ?", (event,)
        )
    }
    change = CardChange(event, applied=True)
    for key, (bout, resolution) in ids.items():
        for fighter, r in (
            (bout.fighter_a, resolution.a),
            (bout.fighter_b, resolution.b),
        ):
            if r.provisional:
                _upsert_provisional(conn, fighter, now)
            elif r.ref is not None:
                _link_provisional(conn, fighter, r.ref, now)
        resolved = resolution.resolved
        conn.execute(
            """
            INSERT INTO upcoming_bouts
                (bout_id, event_key, card_order, segment, weight_class_raw,
                 weight_class, is_title_bout, name_a, page_a, name_b, page_b,
                 fighter_a_ref, fighter_b_ref, method_a, method_b,
                 unresolved_reason, status, first_seen_at, last_seen_at, cancelled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'scheduled', ?, ?, NULL)
            ON CONFLICT (bout_id) DO UPDATE SET
                card_order = excluded.card_order, segment = excluded.segment,
                weight_class_raw = excluded.weight_class_raw,
                weight_class = excluded.weight_class,
                is_title_bout = excluded.is_title_bout,
                name_a = excluded.name_a, page_a = excluded.page_a,
                name_b = excluded.name_b, page_b = excluded.page_b,
                fighter_a_ref = excluded.fighter_a_ref,
                fighter_b_ref = excluded.fighter_b_ref,
                method_a = excluded.method_a, method_b = excluded.method_b,
                unresolved_reason = excluded.unresolved_reason,
                status = 'scheduled', cancelled_at = NULL,
                last_seen_at = excluded.last_seen_at
            """,
            (
                key,
                event,
                bout.order,
                bout.segment,
                bout.weight_class_raw,
                bout.weight_class,
                int(bout.is_title_bout),
                bout.fighter_a.name,
                bout.fighter_a.page,
                bout.fighter_b.name,
                bout.fighter_b.page,
                resolution.a.ref if resolved else None,
                resolution.b.ref if resolved else None,
                resolution.a.method,
                resolution.b.method,
                resolution.unresolved_reason,
                now,
                now,
            ),
        )
        if key not in known:
            change.added.append(key)
        elif known[key] == "cancelled":
            change.restored.append(key)
        else:
            change.kept.append(key)
        if not resolved:
            change.unresolved.append(key)
            if supersede(conn, key, now, "bout unresolved"):
                log(
                    conn,
                    now,
                    "warning",
                    f"{key} became unresolved; prediction superseded",
                    event,
                )
    for key in sorted(previous - set(ids)):
        conn.execute(
            """
            UPDATE upcoming_bouts SET status = 'cancelled', cancelled_at = ?
            WHERE bout_id = ?
            """,
            (now, key),
        )
        supersede(conn, key, now, "bout cancelled")
        change.cancelled.append(key)
    conn.execute(
        "UPDATE upcoming_events SET card_applied_at = ? WHERE event_key = ?",
        (now, event),
    )
    if change.added or change.cancelled or change.restored:
        log(
            conn,
            now,
            "info",
            f"card applied: {len(change.added)} added, {len(change.cancelled)} "
            f"cancelled, {len(change.restored)} restored, {len(change.kept)} unchanged",
            event,
        )
    return change
