"""Height, reach and date of birth from a fighter's Wikipedia infobox.

Used only to fill attributes that the published dataset lacks (D46, D48).
Only a few unambiguous shapes are read; anything else is "not found", never a
guess. Values outside plausible ranges are rejected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from ufc.predict.wikitext import Template, plain, strip_noise, templates

INFOBOXES = ("Infobox martial artist", "Infobox MMA fighter", "Infobox boxer")
IN_TO_CM = 2.54
HEIGHT_RANGE_CM = (140.0, 220.0)
REACH_RANGE_CM = (140.0, 230.0)
BIRTH_YEAR_RANGE = (1950, 2010)


@dataclass(frozen=True, slots=True)
class InfoboxAttributes:
    height_cm: float | None = None
    reach_cm: float | None = None
    dob: date | None = None


def _infobox(wikitext: str) -> Template | None:
    text = strip_noise(wikitext)
    for name in INFOBOXES:
        found = templates(text, name)
        if found:
            return found[0]
    return None


def _in_range(value: float, bounds: tuple[float, float]) -> float | None:
    return round(value, 2) if bounds[0] <= value <= bounds[1] else None


def parse_height(raw: str) -> float | None:
    """``6 ft 4 in``, ``5 ft 11 in (180 cm)``, ``{{height|ft=5|in=8}}``,
    ``{{convert|6|ft|4|in|cm}}`` -> centimetres."""
    for t in templates(raw, "height"):
        if "ft" in t.named:
            try:
                inches = float(t.named["ft"]) * 12 + float(t.named.get("in", "0") or 0)
            except ValueError:
                return None
            return _in_range(inches * IN_TO_CM, HEIGHT_RANGE_CM)
        if "m" in t.named or "cm" in t.named:
            try:
                cm = (
                    float(t.named["cm"])
                    if "cm" in t.named
                    else float(t.named["m"]) * 100
                )
            except ValueError:
                return None
            return _in_range(cm, HEIGHT_RANGE_CM)
    for t in templates(raw, "convert"):
        v = t.positional
        if len(v) >= 4 and v[1] == "ft" and v[3] == "in":
            try:
                return _in_range(
                    (float(v[0]) * 12 + float(v[2])) * IN_TO_CM, HEIGHT_RANGE_CM
                )
            except ValueError:
                return None
    match = re.fullmatch(r"(\d)\s*ft\s*(\d{1,2}(?:\.\d+)?)\s*in\b.*", plain(raw))
    if match:
        inches = int(match.group(1)) * 12 + float(match.group(2))
        return _in_range(inches * IN_TO_CM, HEIGHT_RANGE_CM)
    return None


def parse_reach(raw: str) -> float | None:
    """``79 in``, ``72 in (183 cm)``, ``{{convert|72|in|cm}}`` -> centimetres."""
    for t in templates(raw, "convert"):
        v = t.positional
        if len(v) >= 2 and v[1] == "in":
            try:
                return _in_range(float(v[0]) * IN_TO_CM, REACH_RANGE_CM)
            except ValueError:
                return None
    match = re.fullmatch(r"(\d{2}(?:\.\d+)?)\s*(?:in|inches|″|\")(?:\W.*)?", plain(raw))
    if match:
        return _in_range(float(match.group(1)) * IN_TO_CM, REACH_RANGE_CM)
    return None


def parse_birth_date(raw: str) -> date | None:
    """``{{Birth date and age|1997|11|07|df=y}}`` and its variants -> date."""
    for name in ("birth date and age", "birth date", "bda", "birth-date and age"):
        for t in templates(raw, name):
            numbers = [v for v in t.positional if v.isdigit()]
            if len(numbers) < 3:
                return None
            year, month, day = (int(n) for n in numbers[:3])
            if not BIRTH_YEAR_RANGE[0] <= year <= BIRTH_YEAR_RANGE[1]:
                return None
            try:
                return date(year, month, day)
            except ValueError:
                return None
    return None


def parse_infobox(wikitext: str) -> InfoboxAttributes:
    box = _infobox(wikitext)
    if box is None:
        return InfoboxAttributes()
    get = box.named.get
    return InfoboxAttributes(
        height_cm=parse_height(get("height", "")),
        reach_cm=parse_reach(get("reach", "")),
        dob=parse_birth_date(get("birth_date", "")),
    )
