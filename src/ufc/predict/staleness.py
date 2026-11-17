"""Is the dataset behind the sport? (D52)

The published dataset is refreshed daily upstream but can lag a few days
behind the last event, or stop. Features for an upcoming bout are only as
recent as the dataset, so a fighter who fought last weekend on an event the
dataset does not have yet is predicted from stats that miss his latest fight.

- Globally: completed UFC events on Wikipedia dated after the dataset's
  latest event are "missing". Status is ``current`` with none, ``behind``
  with some, ``unknown`` if a missing event's page could not be read.
- Per bout: a fighter who appears on a missing event's card makes the bout
  "stale stats"; if a missing event could not be read, every bout is flagged,
  because it cannot be ruled out.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date

from ufc.ingest.identity import FighterResolver
from ufc.predict.cards import CardParseError, EventList, ListedEvent, parse_card
from ufc.predict.resolve import resolve_fighter
from ufc.predict.wikipedia import CARD_MAX_AGE_S, Wikipedia, WikipediaError

CURRENT, BEHIND, UNKNOWN = "current", "behind", "unknown"


def dataset_last_event(conn: sqlite3.Connection) -> date | None:
    row = conn.execute(
        "SELECT max(date) FROM events WHERE status = 'completed'"
    ).fetchone()
    return date.fromisoformat(row[0]) if row and row[0] else None


def missing_events(
    events: EventList, last: date | None, today: date
) -> list[ListedEvent]:
    """Completed events after the dataset's last one: everything in the past
    table after it, and scheduled-table events dated before today that have
    not been moved to the past table yet."""
    if last is None:
        return []
    found = {e.page: e for e in events.past if e.date > last}
    for e in events.scheduled:
        if last < e.date < today:
            found.setdefault(e.page, e)
    return sorted(found.values(), key=lambda e: (e.date, e.page))


@dataclass(slots=True)
class Staleness:
    dataset_last_event: date | None
    status: str = CURRENT
    missing: list[dict] = field(default_factory=list)
    # fighter ref or name key -> the missing event he fought on
    fought: dict[str, str] = field(default_factory=dict)

    def bout_reasons(self, sides: list[tuple[str, str | None, str]]) -> list[str]:
        """``sides``: (displayed name, fighter ref, name key) per fighter."""
        if self.status == UNKNOWN:
            unread = ", ".join(m["name"] for m in self.missing if not m["read"])
            return [f"stale stats: could not check recent results ({unread})"]
        reasons = []
        for name, ref, key in sides:
            event = self.fought.get(ref or "") or self.fought.get(f"name:{key}")
            if event:
                reasons.append(
                    f"stale stats: {name} fought at {event}, which the dataset does not have yet"
                )
        return reasons

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "dataset_last_event": (
                self.dataset_last_event.isoformat() if self.dataset_last_event else None
            ),
            "missing_events": self.missing,
        }


def check_staleness(
    client: Wikipedia,
    resolver: FighterResolver,
    events: EventList,
    last: date | None,
    today: date,
) -> Staleness:
    result = Staleness(last)
    missing = missing_events(events, last, today)
    if not missing:
        return result
    result.status = BEHIND
    try:
        pages = client.pages([e.page for e in missing], CARD_MAX_AGE_S)
    except WikipediaError:
        pages = {}
    for event in missing:
        label = f"{event.name} ({event.date.isoformat()})"
        page = pages.get(event.page)
        entry = {"name": event.name, "page": event.page, "date": event.date.isoformat()}
        try:
            if page is None or not page.exists:
                raise CardParseError("page could not be fetched")
            card = parse_card(page.wikitext or "")
        except CardParseError:
            result.status = UNKNOWN
            result.missing.append({**entry, "read": False, "bouts": None})
            continue
        result.missing.append({**entry, "read": True, "bouts": len(card.bouts)})
        for bout in card.bouts:
            for fighter in (bout.fighter_a, bout.fighter_b):
                if fighter.is_placeholder:
                    continue
                result.fought.setdefault(f"name:{fighter.key}", label)
                r = resolve_fighter(resolver, fighter, "", bout.weight_class)
                if r.ref:
                    result.fought.setdefault(r.ref, label)
    return result
