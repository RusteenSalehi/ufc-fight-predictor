"""Parse UFC cards from Wikipedia wikitext.

- ``parse_event_list``: the "Scheduled events" and "Past events" tables of
  "List of UFC events": each row's event page and date.
- ``parse_card``: the ``{{MMAevent bout|class|fighter|vs.|fighter|...}}``
  templates of one event page, in card order.

Both raise ``CardParseError`` when the page does not have the expected
shape. A card that parses but looks wrong (no bouts, most bouts gone) is the
caller's decision, not the parser's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from ufc.ingest.parsing import name_key
from ufc.predict.wikitext import (
    Template,
    first_link,
    plain,
    remove_templates,
    section,
    strip_noise,
    table_rows,
    templates,
)

LIST_PAGE = "List of UFC events"
SCHEDULED_HEADING = "Scheduled events"
PAST_HEADING = "Past events"

_MONTHS = {
    m: i
    for i, names in enumerate(
        (
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ),
        start=1,
    )
    for m in names
}

WEIGHT_CLASSES = (
    "Strawweight",
    "Flyweight",
    "Bantamweight",
    "Featherweight",
    "Lightweight",
    "Welterweight",
    "Middleweight",
    "Light Heavyweight",
    "Heavyweight",
)
"""The dataset's labels; women's classes are these with "Women's " in front."""

PLACEHOLDERS = frozenset({"", "tba", "tbd", "tbc", "unknown", "opponent tba"})


class CardParseError(ValueError):
    """A page did not have the structure the parser expects."""


@dataclass(frozen=True, slots=True)
class ListedEvent:
    """One row of an events table."""

    page: str  # the linked page title
    name: str  # as displayed
    date: date


@dataclass(frozen=True, slots=True)
class EventList:
    scheduled: tuple[ListedEvent, ...]
    past: tuple[ListedEvent, ...]


@dataclass(frozen=True, slots=True)
class CardFighter:
    name: str  # displayed name, markup removed
    page: str | None  # linked article, if any

    @property
    def key(self) -> str:
        return name_key(self.name)

    @property
    def is_placeholder(self) -> bool:
        return self.key in PLACEHOLDERS


@dataclass(frozen=True, slots=True)
class CardBout:
    order: int  # position on the page, 0 = first listed
    segment: str | None  # "Main card", "Preliminary card", ...
    weight_class_raw: str
    weight_class: str | None  # the dataset's label, or None if not recognised
    fighter_a: CardFighter
    fighter_b: CardFighter
    connector: str  # "vs." (not yet fought, draw or no contest) or "def."
    result: str  # method text; empty for a bout not yet fought
    is_title_bout: bool

    @property
    def completed(self) -> bool:
        return self.connector == "def." or bool(self.result)


@dataclass(frozen=True, slots=True)
class Card:
    bouts: tuple[CardBout, ...]
    announced: tuple[str, ...]  # "Announced bouts" list items, not predicted


def parse_dts(template: Template) -> date:
    """``{{dts|2027|Feb|20}}`` (or numeric month) -> date."""
    values = [v for v in template.positional if v]
    if len(values) < 3:
        raise CardParseError(f"incomplete date template: {template}")
    year, month, day = values[:3]
    month_number = (
        int(month) if month.isdigit() else _MONTHS.get(month.casefold().rstrip("."))
    )
    try:
        return date(int(year), month_number, int(day))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise CardParseError(f"bad date template: {template}") from exc


def _rows(body: str) -> list[ListedEvent]:
    events: list[ListedEvent] = []
    for row in table_rows(strip_noise(body)):
        link = first_link(row)
        dts = templates(row, "dts")
        if link is None or not dts:
            raise CardParseError(f"event row without a link or a date: {row[:120]!r}")
        events.append(ListedEvent(page=link[0], name=link[1], date=parse_dts(dts[0])))
    return events


def parse_event_list(wikitext: str) -> EventList:
    """Scheduled and past events from "List of UFC events"."""
    scheduled = section(wikitext, SCHEDULED_HEADING)
    past = section(wikitext, PAST_HEADING)
    if scheduled is None or past is None:
        raise CardParseError("events list without its scheduled or past section")
    upcoming, done = _rows(scheduled), _rows(past)
    if not done:
        raise CardParseError("past events table is empty")
    return EventList(scheduled=tuple(upcoming), past=tuple(done))


def normalise_weight_class(raw: str) -> str | None:
    """Wikipedia's class name -> the dataset's label ("Catchweight (160.2
    lb)" -> "Catch Weight"). Anything unrecognised -> None."""
    text = plain(raw)
    text = re.sub(r"\(.*?\)", "", text).strip().casefold()
    womens = text.startswith(("women's", "women’s", "womens"))
    text = re.sub(r"^women[’']?s\s+", "", text)
    text = text.replace("-", " ").removesuffix(" bout").strip()
    for label in WEIGHT_CLASSES:
        if text == label.casefold():
            return f"Women's {label}" if womens else label
    if text.replace(" ", "") == "catchweight":
        return "Catch Weight"
    if text.replace(" ", "") == "openweight":
        return "Open Weight"
    return None


_CHAMPION_MARK = re.compile(r"\((?:c|ic|uc)\)\s*$", re.IGNORECASE)


def parse_fighter(cell: str) -> CardFighter:
    cell = strip_noise(cell)
    link = first_link(remove_templates(cell))
    name = _CHAMPION_MARK.sub("", plain(cell)).strip()
    if link is not None:
        return CardFighter(name=name, page=link[0])
    return CardFighter(name=name, page=None)


def parse_card(wikitext: str) -> Card:
    """Bouts of one event page, in the order the page lists them."""
    text = strip_noise(wikitext)
    bouts: list[CardBout] = []
    # Card segments and bouts in page order.
    marks = sorted(
        [
            (m.start(), "card", m)
            for m in re.finditer(r"\{\{\s*MMAevent card", text, re.IGNORECASE)
        ]
        + [
            (m.start(), "bout", m)
            for m in re.finditer(r"\{\{\s*MMAevent bout", text, re.IGNORECASE)
        ]
    )
    segments = templates(text, "MMAevent card")
    all_bouts = templates(text, "MMAevent bout")
    if len(all_bouts) != sum(1 for _, kind, _ in marks if kind == "bout"):
        raise CardParseError("an MMAevent bout template could not be read")
    segment, s_i, b_i = None, 0, 0
    for _, kind, _ in marks:
        if kind == "card":
            if s_i < len(segments):
                values = segments[s_i].positional
                segment = plain(values[0]) if values else None
                if segment:
                    segment = re.sub(r"\s*\(.*\)$", "", segment) or None
            s_i += 1
            continue
        template = all_bouts[b_i]
        b_i += 1
        values = list(template.positional) + [""] * 8
        weight_raw, a_cell, connector, b_cell, method = values[:5]
        notes = values[7]
        connector = plain(connector).casefold()
        if connector not in ("vs.", "vs", "def.", "def"):
            raise CardParseError(
                f"bout {b_i} has {connector!r} where 'vs.' or 'def.' belongs"
            )
        bouts.append(
            CardBout(
                order=len(bouts),
                segment=segment,
                weight_class_raw=plain(weight_raw),
                weight_class=normalise_weight_class(weight_raw),
                fighter_a=parse_fighter(a_cell),
                fighter_b=parse_fighter(b_cell),
                connector="def." if connector.startswith("def") else "vs.",
                result=plain(method),
                is_title_bout=bool(
                    re.search(r"\bchampionship\b", plain(notes), re.IGNORECASE)
                    and re.search(r"\bfor the\b", plain(notes), re.IGNORECASE)
                ),
            )
        )
    announced_text = section(text, "Announced bouts") or ""
    announced = tuple(
        plain(line[1:])
        for line in announced_text.splitlines()
        if line.startswith("*") and plain(line[1:])
    )
    return Card(bouts=tuple(bouts), announced=announced)
