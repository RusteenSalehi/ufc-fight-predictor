"""Resolve the fighter names on a bout to UFCStats fighter IDs.

The published dataset identifies events, fights and fighter profiles by
UFCStats URL, but a fight row names its two fighters as text only. This
module maps those names to fighter IDs with rules that never guess:

1. an explicit override from ``identity_overrides.csv``;
2. a name that belongs to exactly one fighter profile;
3. a name shared by several profiles where exactly one profile's listed weight
   fits the bout's weight class;
4. otherwise the name is unresolved and the caller drops the bout and reports it.

There is deliberately no fuzzy matching. A near-miss match can silently attach
a fight to the wrong person, which is worse than a reported gap.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from ufc.ingest.parsing import clean, name_key

OVERRIDE_COLUMNS = ("name", "fight_id", "fighter_id", "note")

# Upper limit of each class in pounds. Fighter profiles list the limit of the
# class the fighter usually competes in, except heavyweights, whose listed
# weight is anywhere in the class's range.
WEIGHT_CLASS_LIMITS_LBS: dict[str, int] = {
    "Strawweight": 115,
    "Flyweight": 125,
    "Bantamweight": 135,
    "Featherweight": 145,
    "Lightweight": 155,
    "Welterweight": 170,
    "Middleweight": 185,
    "Light Heavyweight": 205,
    "Heavyweight": 265,
}

METHOD_OVERRIDE = "override"
METHOD_NAME = "name"
METHOD_WEIGHT = "name+weight"
METHOD_UNKNOWN_NAME = "unknown-name"
METHOD_AMBIGUOUS = "ambiguous-name"


@dataclass(frozen=True, slots=True)
class Profile:
    """What identity resolution needs to know about one fighter profile."""

    fighter_id: str
    names: tuple[str, ...]
    weight_lbs: int | None


@dataclass(frozen=True, slots=True)
class Override:
    """A hand-checked mapping from a name on a bout to a fighter ID.

    With ``fight_id`` set it applies to that fight only; without, to every
    fight that uses the name.
    """

    name: str
    fight_id: str | None
    fighter_id: str
    note: str


@dataclass(frozen=True, slots=True)
class Resolution:
    fighter_id: str | None
    method: str
    candidates: tuple[str, ...] = ()


def weight_fits(weight_lbs: int | None, weight_class: str | None) -> bool:
    """Whether a profile's listed weight is consistent with a bout's weight class.

    Returns ``False`` whenever there is nothing to compare: unknown weight, no
    weight class, or a class without a limit (catch weight, open weight).
    """
    if weight_lbs is None or weight_class is None:
        return False
    division = weight_class.removeprefix("Women's ")
    limit = WEIGHT_CLASS_LIMITS_LBS.get(division)
    if limit is None:
        return False
    if division == "Heavyweight":
        return WEIGHT_CLASS_LIMITS_LBS["Light Heavyweight"] < weight_lbs <= limit
    return weight_lbs == limit


def load_overrides(path: Path | None = None) -> list[Override]:
    """Read identity overrides; by default the file shipped with the package."""
    if path is None:
        text = (
            resources.files("ufc.ingest")
            .joinpath("identity_overrides.csv")
            .read_text("utf-8")
        )
    else:
        text = path.read_text(encoding="utf-8")
    reader = csv.DictReader(text.splitlines())
    if tuple(reader.fieldnames or ()) != OVERRIDE_COLUMNS:
        raise ValueError(
            f"identity overrides must have columns {list(OVERRIDE_COLUMNS)}"
        )
    return [
        Override(
            name=clean(row["name"]),
            fight_id=clean(row["fight_id"]) or None,
            fighter_id=clean(row["fighter_id"]),
            note=clean(row["note"]),
        )
        for row in reader
    ]


class FighterResolver:
    """Maps (name, fight, weight class) to a fighter ID using the rules above."""

    def __init__(
        self, profiles: Iterable[Profile], overrides: Iterable[Override] = ()
    ) -> None:
        self._weights: dict[str, int | None] = {}
        self._ids_by_key: dict[str, set[str]] = {}
        for profile in profiles:
            self._weights[profile.fighter_id] = profile.weight_lbs
            for name in profile.names:
                self._ids_by_key.setdefault(name_key(name), set()).add(
                    profile.fighter_id
                )

        self._overrides: dict[tuple[str, str | None], Override] = {}
        self.invalid_overrides: list[Override] = []
        for override in overrides:
            if override.fighter_id in self._weights:
                self._overrides[(name_key(override.name), override.fight_id)] = override
            else:
                self.invalid_overrides.append(override)
        self._used: set[tuple[str, str | None]] = set()

    def resolve(self, name: str, fight_id: str, weight_class: str | None) -> Resolution:
        key = name_key(name)
        for override_key in ((key, fight_id), (key, None)):
            override = self._overrides.get(override_key)
            if override is not None:
                self._used.add(override_key)
                return Resolution(override.fighter_id, METHOD_OVERRIDE)

        candidates = tuple(sorted(self._ids_by_key.get(key, ())))
        if not candidates:
            return Resolution(None, METHOD_UNKNOWN_NAME)
        if len(candidates) == 1:
            return Resolution(candidates[0], METHOD_NAME, candidates)
        fitting = [c for c in candidates if weight_fits(self._weights[c], weight_class)]
        if len(fitting) == 1:
            return Resolution(fitting[0], METHOD_WEIGHT, candidates)
        return Resolution(None, METHOD_AMBIGUOUS, candidates)

    def unused_overrides(self) -> list[Override]:
        """Valid overrides that no bout needed. Usually a sign they are stale."""
        return [o for key, o in self._overrides.items() if key not in self._used]
