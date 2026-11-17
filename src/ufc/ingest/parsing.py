"""Pure field parsers for the published UFCStats-derived CSV dataset.

Every function takes the raw string from a CSV cell and returns a typed value.
Missing values (the dataset writes ``--``, ``---`` or an empty cell) become
``None``; anything else that does not match the expected shape raises
``ParseError`` so that bad data is reported instead of silently coerced.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime

MISSING_MARKERS = frozenset({"", "--", "---"})

_ID_URL = re.compile(
    r"https?://(?:www\.)?ufcstats\.com/(event|fight|fighter)-details/([0-9a-f]{16})/?"
)
_HEIGHT = re.compile(r"(\d)'\s*(\d{1,2})\"")
_REACH = re.compile(r"(\d{2,3}(?:\.\d+)?)\"")
_WEIGHT = re.compile(r"(\d{2,3}) lbs\.")
_LANDED_ATTEMPTED = re.compile(r"(\d+) of (\d+)")
_CLOCK = re.compile(r"(\d{1,2}):([0-5]\d)")
_ROUND_LABEL = re.compile(r"Round (\d+)")
_TIME_FORMAT = re.compile(r"(\d+) Rnd(?: \+ (\d*)OT)? \((\d+(?:-\d+)*)\)")

_CM_PER_INCH = 2.54

# Longest names first so that "Light Heavyweight" is not read as "Heavyweight".
_WEIGHT_CLASSES = (
    "Super Heavyweight",
    "Light Heavyweight",
    "Heavyweight",
    "Middleweight",
    "Welterweight",
    "Lightweight",
    "Featherweight",
    "Bantamweight",
    "Flyweight",
    "Strawweight",
    "Catch Weight",
    "Open Weight",
)
_UNGENDERED_CLASSES = frozenset({"Catch Weight", "Open Weight"})

_OUTCOMES: dict[str, tuple[str, str | None]] = {
    "W/L": ("win", "a"),
    "L/W": ("win", "b"),
    "D/D": ("draw", None),
    "NC/NC": ("nc", None),
}


class ParseError(ValueError):
    """A cell did not match the format the dataset is expected to use."""


def clean(text: str) -> str:
    """Collapse internal whitespace and strip the ends."""
    return " ".join(text.split())


def is_missing(text: str) -> bool:
    return clean(text) in MISSING_MARKERS


def name_key(name: str) -> str:
    """Normalise a fighter name for matching: ASCII, lower case, no punctuation.

    ``"Jiří Procházka"`` and ``"Jiri Prochazka"`` share a key, as do
    ``"Waldo Cortes-Acosta"`` and ``"Waldo Cortes Acosta"``.
    """
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return clean(re.sub(r"[^a-z0-9 ]+", " ", ascii_name.casefold()))


def id_from_url(url: str, kind: str) -> str:
    """Extract the 16-character hex ID from a UFCStats URL of the given kind.

    ``kind`` is ``"event"``, ``"fight"`` or ``"fighter"``. The URL is only
    parsed, never requested.
    """
    match = _ID_URL.fullmatch(url.strip())
    if match is None or match.group(1) != kind:
        raise ParseError(f"not a UFCStats {kind} URL: {url!r}")
    return match.group(2)


def parse_event_date(text: str) -> date:
    """``"November 14, 2026"`` -> ``date(2026, 11, 14)``."""
    try:
        return datetime.strptime(clean(text), "%B %d, %Y").date()
    except ValueError as exc:
        raise ParseError(f"bad event date: {text!r}") from exc


def parse_dob(text: str) -> date | None:
    """``"Jul 13, 1978"`` -> ``date(1978, 7, 13)``; ``"--"`` -> ``None``."""
    if is_missing(text):
        return None
    try:
        return datetime.strptime(clean(text), "%b %d, %Y").date()
    except ValueError as exc:
        raise ParseError(f"bad date of birth: {text!r}") from exc


def parse_height_cm(text: str) -> float | None:
    """``5' 11"`` -> ``180.34``; ``"--"`` -> ``None``."""
    if is_missing(text):
        return None
    match = _HEIGHT.fullmatch(clean(text))
    if match is None or int(match.group(2)) > 11:
        raise ParseError(f"bad height: {text!r}")
    inches = int(match.group(1)) * 12 + int(match.group(2))
    return round(inches * _CM_PER_INCH, 2)


def parse_reach_cm(text: str) -> float | None:
    """``72"`` -> ``182.88``; ``"--"`` -> ``None``."""
    if is_missing(text):
        return None
    match = _REACH.fullmatch(clean(text))
    if match is None:
        raise ParseError(f"bad reach: {text!r}")
    return round(float(match.group(1)) * _CM_PER_INCH, 2)


def parse_weight_lbs(text: str) -> int | None:
    """``"155 lbs."`` -> ``155``; ``"--"`` -> ``None``."""
    if is_missing(text):
        return None
    match = _WEIGHT.fullmatch(clean(text))
    if match is None:
        raise ParseError(f"bad weight: {text!r}")
    return int(match.group(1))


def parse_stance(text: str) -> str | None:
    return None if is_missing(text) else clean(text)


def parse_landed_attempted(text: str) -> tuple[int, int]:
    """``"41 of 65"`` -> ``(41, 65)``."""
    match = _LANDED_ATTEMPTED.fullmatch(clean(text))
    if match is None:
        raise ParseError(f"bad 'landed of attempted' value: {text!r}")
    return int(match.group(1)), int(match.group(2))


def parse_count(text: str) -> int:
    """``"2"`` or ``"2.0"`` -> ``2``. Older rows in the dataset are float-formatted."""
    try:
        value = float(clean(text))
    except ValueError as exc:
        raise ParseError(f"bad count: {text!r}") from exc
    if value < 0 or value != int(value):
        raise ParseError(f"bad count: {text!r}")
    return int(value)


def parse_clock(text: str) -> int:
    """``"4:21"`` -> ``261`` seconds."""
    match = _CLOCK.fullmatch(clean(text))
    if match is None:
        raise ParseError(f"bad clock time: {text!r}")
    return int(match.group(1)) * 60 + int(match.group(2))


def parse_control_time(text: str) -> int | None:
    """Control time in seconds; ``"--"`` (not recorded in early events) -> ``None``."""
    return None if is_missing(text) else parse_clock(text)


def parse_round_label(text: str) -> int:
    """``"Round 3"`` -> ``3``."""
    match = _ROUND_LABEL.fullmatch(clean(text))
    if match is None or int(match.group(1)) < 1:
        raise ParseError(f"bad round label: {text!r}")
    return int(match.group(1))


def parse_round_number(text: str) -> int:
    """The round a fight ended in: ``"3"`` -> ``3``."""
    value = clean(text)
    if not value.isdigit() or int(value) < 1:
        raise ParseError(f"bad round number: {text!r}")
    return int(value)


def parse_outcome(text: str) -> tuple[str, str | None]:
    """Map the dataset's outcome code to ``(result, winner_slot)``.

    The code is written from the point of view of the bout string
    ``"A vs. B"``: ``"W/L"`` means A won, ``"L/W"`` means B won.
    ``winner_slot`` is ``"a"``, ``"b"`` or ``None`` (draw or no contest).
    """
    try:
        return _OUTCOMES[clean(text)]
    except KeyError as exc:
        raise ParseError(f"unknown outcome code: {text!r}") from exc


def split_bout(text: str) -> tuple[str, str]:
    """``"Orin Valcourt vs. Dax Hollin"`` -> ``("Orin Valcourt", "Dax Hollin")``."""
    parts = [clean(part) for part in clean(text).split(" vs. ")]
    if len(parts) != 2 or not all(parts):
        raise ParseError(f"cannot split bout into two fighters: {text!r}")
    return parts[0], parts[1]


@dataclass(frozen=True, slots=True)
class BoutType:
    """What the dataset's WEIGHTCLASS cell says about a bout."""

    weight_class: str | None
    is_womens: bool
    is_title_bout: bool
    is_interim_title: bool
    is_tournament_final: bool


def parse_bout_type(text: str) -> BoutType:
    """Read weight class and title flags from e.g. ``"UFC Women's Flyweight Title Bout"``.

    ``is_title_bout`` means a UFC championship bout, including interim titles
    and the 1990s "Superfight Championship". Tournament finals (early UFC
    tournaments, The Ultimate Fighter, Road to UFC) are flagged separately and
    are not title bouts. ``weight_class`` is ``None`` when the text names no
    weight class, which is the case for early tournament finals and superfights.
    """
    lowered = clean(text).casefold().replace("catchweight", "catch weight")
    lowered = lowered.replace("openweight", "open weight")
    weight_class = next((c for c in _WEIGHT_CLASSES if c.casefold() in lowered), None)
    is_womens = "women" in lowered
    if weight_class and is_womens and weight_class not in _UNGENDERED_CLASSES:
        weight_class = f"Women's {weight_class}"
    is_tournament_final = "tournament" in lowered
    has_title = "title" in lowered or "championship" in lowered
    is_title_bout = has_title and not is_tournament_final
    return BoutType(
        weight_class=weight_class,
        is_womens=is_womens,
        is_title_bout=is_title_bout,
        is_interim_title=is_title_bout and "interim" in lowered,
        is_tournament_final=is_tournament_final,
    )


@dataclass(frozen=True, slots=True)
class TimeFormat:
    """A bout's scheduled format, e.g. three five-minute rounds.

    ``round_minutes`` lists every period that could be fought, including
    overtime periods; ``scheduled_rounds`` counts regulation rounds only.
    Both are empty/``None`` for the "No Time Limit" bouts of 1993 to 1995.
    """

    scheduled_rounds: int | None
    round_minutes: tuple[int, ...]
    no_time_limit: bool


def parse_time_format(text: str) -> TimeFormat:
    """``"3 Rnd (5-5-5)"`` or ``"1 Rnd + OT (12-3)"`` or ``"No Time Limit"``."""
    value = clean(text)
    if value == "No Time Limit":
        return TimeFormat(scheduled_rounds=None, round_minutes=(), no_time_limit=True)
    match = _TIME_FORMAT.fullmatch(value)
    if match is None:
        raise ParseError(f"bad time format: {text!r}")
    rounds = int(match.group(1))
    minutes = tuple(int(m) for m in match.group(3).split("-"))
    if len(minutes) < rounds:
        raise ParseError(f"time format lists fewer periods than rounds: {text!r}")
    return TimeFormat(
        scheduled_rounds=rounds, round_minutes=minutes, no_time_limit=False
    )


def fight_time_s(
    time_format: TimeFormat, end_round: int, end_time_s: int
) -> int | None:
    """Total seconds fought: full earlier rounds plus the time into the last one.

    Returns ``None`` when it cannot be known from the format, which only
    happens if a no-time-limit bout is recorded as ending after round 1 or a
    bout ends in a round its format does not list.
    """
    if time_format.no_time_limit:
        return end_time_s if end_round == 1 else None
    if end_round > len(time_format.round_minutes):
        return None
    return sum(time_format.round_minutes[: end_round - 1]) * 60 + end_time_s
