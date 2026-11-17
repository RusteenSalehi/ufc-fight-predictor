"""Card fighters to fighter references, with the ingest's rules (D17, D49).

A card names a fighter by displayed name and, usually, a linked article. Both
are matched exactly (accents, case and punctuation ignored, as in the
ingest) against the same profiles and overrides the load uses, plus
``wikipedia_overrides.csv`` for spellings that only Wikipedia uses:

1. an override, a name of exactly one profile, or a shared name settled by
   weight class: the fighter's UFCStats ID;
2. a name that matches no profile at all: a provisional fighter, keyed by
   the article title (or the name, if there is no article). On a card this
   is almost always a debutant;
3. anything else (a shared name the weight class does not settle, the two
   names pointing at different profiles, a placeholder such as "TBD", or both
   corners resolving to one fighter): unresolved. The bout is stored and
   reported, not predicted.

There is no fuzzy matching. A spelling that differs from the dataset's in
more than accents and punctuation makes a provisional fighter, which the
refresh report lists so that an override can be added.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from ufc.ingest import source
from ufc.ingest.identity import (
    METHOD_AMBIGUOUS,
    METHOD_UNKNOWN_NAME,
    FighterResolver,
    Override,
    load_overrides,
)
from ufc.ingest.parsing import name_key
from ufc.ingest.transform import build_profiles
from ufc.predict.cards import CardBout, CardFighter

METHOD_PROVISIONAL = "provisional"
METHOD_PLACEHOLDER = "placeholder"
METHOD_CONFLICT = "names-disagree"

_DISAMBIGUATION = re.compile(r"\s*\([^()]*\)\s*$")


def load_wikipedia_overrides(path: Path | None = None) -> list[Override]:
    """Overrides for names as Wikipedia writes them; same columns as the
    ingest's file, with ``fight_id`` holding an upcoming bout ID if set."""
    if path is not None:
        return load_overrides(path)
    text = resources.files("ufc.predict").joinpath("wikipedia_overrides.csv")
    return load_overrides(Path(str(text)))


def resolver_from_snapshot(
    snapshot_dir: Path, extra_overrides: Iterable[Override] = ()
) -> FighterResolver:
    """The ingest's resolver, built from the snapshot's two profile files."""
    listing = source.read_table(snapshot_dir / source.FIGHTER_DETAILS_FILE)
    tape = source.read_table(snapshot_dir / source.FIGHTER_TOTT_FILE)
    return FighterResolver(
        build_profiles(listing, tape), [*load_overrides(), *extra_overrides]
    )


def provisional_id(fighter: CardFighter) -> str:
    if fighter.page:
        return f"wp:{fighter.page}"
    return f"wpname:{fighter.key}"


@dataclass(frozen=True, slots=True)
class FighterResolution:
    ref: str | None  # UFCStats fighter ID or provisional ID; None if unresolved
    method: str
    provisional: bool = False
    detail: str = ""


def _names(fighter: CardFighter) -> list[str]:
    names = [fighter.name]
    if fighter.page:
        title = _DISAMBIGUATION.sub("", fighter.page)
        if name_key(title) != fighter.key:
            names.append(title)
    return names


def resolve_fighter(
    resolver: FighterResolver,
    fighter: CardFighter,
    bout_id: str,
    weight_class: str | None,
) -> FighterResolution:
    if fighter.is_placeholder:
        return FighterResolution(
            None, METHOD_PLACEHOLDER, detail=fighter.name or "(empty)"
        )
    results = [(n, resolver.resolve(n, bout_id, weight_class)) for n in _names(fighter)]
    found = {r.fighter_id for _, r in results if r.fighter_id}
    ambiguous = [(n, r) for n, r in results if r.method == METHOD_AMBIGUOUS]
    if len(found) > 1:
        return FighterResolution(
            None, METHOD_CONFLICT, detail=f"names point to {sorted(found)}"
        )
    if len(found) == 1:
        (fighter_id,) = found
        if any(fighter_id not in r.candidates for _, r in ambiguous):
            return FighterResolution(
                None, METHOD_CONFLICT, detail="one name is shared by other profiles"
            )
        method = next(r.method for _, r in results if r.fighter_id == fighter_id)
        return FighterResolution(fighter_id, method)
    if ambiguous:
        name, r = ambiguous[0]
        return FighterResolution(
            None,
            METHOD_AMBIGUOUS,
            detail=f"{name!r} is shared by {len(r.candidates)} profiles and "
            f"the weight class {weight_class!r} does not settle it",
        )
    assert all(r.method == METHOD_UNKNOWN_NAME for _, r in results)
    return FighterResolution(
        provisional_id(fighter), METHOD_PROVISIONAL, provisional=True
    )


@dataclass(frozen=True, slots=True)
class BoutResolution:
    a: FighterResolution
    b: FighterResolution
    unresolved_reason: str | None

    @property
    def resolved(self) -> bool:
        return self.unresolved_reason is None


def resolve_bout(
    resolver: FighterResolver, bout: CardBout, bout_id: str
) -> BoutResolution:
    a = resolve_fighter(resolver, bout.fighter_a, bout_id, bout.weight_class)
    b = resolve_fighter(resolver, bout.fighter_b, bout_id, bout.weight_class)
    reasons: list[str] = []
    for side, fighter, r in (("A", bout.fighter_a, a), ("B", bout.fighter_b, b)):
        if r.ref is None:
            reasons.append(f"{fighter.name or side}: {r.method} ({r.detail})")
    if not reasons and a.ref == b.ref:
        reasons.append(f"both names resolve to one fighter ({a.ref})")
    return BoutResolution(a, b, "; ".join(reasons) or None)
