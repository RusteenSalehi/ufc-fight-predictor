"""Read-only views: the stored card and a hypothetical matchup.

Neither command writes. The database is opened read-only (so even a schema
migration cannot run), ``predictions`` is never inserted into, and
``data/publish/`` is never touched. ``show`` formats what ``publish.snapshot``
already reads from the tables; ``matchup`` builds features with
``build_matchup`` and predicts with the approved model, then discards the
result.
"""

from __future__ import annotations

import sqlite3
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ufc.features.data import BoutFacts, FeatureData, load_feature_data
from ufc.features.matchup import build_matchup
from ufc.ingest.identity import FighterResolver
from ufc.ingest.parsing import name_key
from ufc.models import registry
from ufc.predict import publish
from ufc.predict.backfill import merged_attributes
from ufc.predict.cards import CardFighter
from ufc.predict.explain import explain_bout
from ufc.predict.predict import feature_store, limited_reasons, model_inputs
from ufc.predict.resolve import resolve_fighter

HYPOTHETICAL_BANNER = "HYPOTHETICAL, NOT STORED"


class ShowError(Exception):
    """A problem to report to the user in one clear message."""


def connect_readonly(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise ShowError(f"No database at {path}; run the ingest load first.")
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# --- formatting ----------------------------------------------------------------


def _pct(p: float) -> str:
    return f"{100 * p:.1f}%"


def _wrap(text: str, width: int, indent: str, first: str | None = None) -> list[str]:
    return textwrap.wrap(
        text,
        width=width,
        initial_indent=first if first is not None else indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [first if first is not None else indent]


def format_prediction(
    name_a: str,
    name_b: str,
    p_a: float,
    p_b: float,
    labels: Sequence[str],
    limited: Sequence[str],
    why: Mapping | None,
    width: int,
    heading: str | None = None,
) -> list[str]:
    """One bout: fighters and probabilities, labels, limited-data reasons and
    the "why" summary, wrapped to ``width``."""
    lines: list[str] = []
    if heading:
        lines += _wrap(heading, width, "  ")
    lines += _wrap(
        f"{name_a} {_pct(p_a)}  vs.  {_pct(p_b)} {name_b}", width, "    ", "  "
    )
    if labels:
        lines += _wrap("[" + "] [".join(labels) + "]", width, "    ")
    for reason in limited:
        lines += _wrap(f"limited data: {reason}", width, "      ", "    ")
    if why is not None:
        lines += _wrap(why["summary"], width, "    ", "    Why: ")
    else:
        lines.append("    Why: not available (predicted before explanations existed)")
    return lines


def bout_labels(bout: Mapping) -> list[str]:
    labels = [bout["weight_class"] or "weight class unknown"]
    if bout["title_bout"]:
        labels.append("title bout")
    if bout["segment"]:
        labels.append(bout["segment"])
    for side in ("fighter_a", "fighter_b"):
        if bout[side]["provisional"]:
            labels.append(f"provisional: {bout[side]['name']}")
    labels.append("live" if bout["is_live"] else "made after the fact")
    if bout["limited_data"]:
        labels.append("limited data")
    if bout["stale_stats"]:
        labels.append("stale stats")
    return labels


def format_event(event: Mapping, width: int) -> list[str]:
    lines = [
        (
            f"{event['name']}, {event['date']} "
            f"(Wikipedia rev {event['wikipedia_revision']})"
        ),
        "=" * min(width, 72),
    ]
    for bout in event["bouts"]:
        lines += format_prediction(
            bout["fighter_a"]["name"],
            bout["fighter_b"]["name"],
            bout["p_fighter_a"],
            bout["p_fighter_b"],
            bout_labels(bout),
            bout["limited_data_reasons"],
            bout["why"],
            width,
        )
        lines += _wrap(
            f"model {bout['model_version']}, predicted {bout['predicted_at']}",
            width,
            "    ",
        )
        lines.append("")
    for bout in event["unresolved_bouts"]:
        lines += _wrap(
            f"UNPREDICTED {bout['fighter_a']['name']} vs. {bout['fighter_b']['name']}"
            f" [{bout['weight_class']}]: {bout['reason']}",
            width,
            "    ",
            "  ",
        )
        lines.append("")
    return lines


def select_events(events: Sequence[Mapping], event: str | None, all_: bool) -> list:
    """The next card, the event matching ``event``, or every event."""
    if not events:
        raise ShowError("No upcoming cards with predictions are stored.")
    if all_:
        return list(events)
    if event is None:
        return [events[0]]
    wanted = event.casefold().strip()
    exact = [e for e in events if e["name"].casefold() == wanted]
    found = exact or [e for e in events if wanted in e["name"].casefold()]
    if not found:
        known = ", ".join(e["name"] for e in events)
        raise ShowError(f"No stored event matches {event!r}. Stored: {known}")
    if len(found) > 1:
        names = ", ".join(e["name"] for e in found)
        raise ShowError(f"{event!r} matches more than one event: {names}")
    return found


def show(
    conn: sqlite3.Connection, today: date, event: str | None, all_: bool, width: int
) -> list[str]:
    record = publish.snapshot(
        conn, today=today, generated_at="", model=None, staleness={}, dataset={}
    )
    chosen = select_events(record["events"], event, all_)
    lines: list[str] = []
    for e in chosen:
        lines += format_event(e, width)
    return lines


# --- hypothetical matchup ---------------------------------------------------------


def resolve_name(
    resolver: FighterResolver,
    known: set[str],
    name: str,
    weight_class: str | None,
) -> str:
    """The fighter ID for ``name`` with the card resolution rules (D17, D49).
    Raises ``ShowError`` for an ambiguous name or one the dataset lacks."""
    r = resolve_fighter(
        resolver, CardFighter(name, None), f"matchup:{name_key(name)}", weight_class
    )
    if r.ref is None:
        hint = (
            ""
            if weight_class
            else " (a weight class given with --weight-class may settle it)"
        )
        raise ShowError(f"Cannot resolve {name!r}: {r.method}: {r.detail}{hint}")
    if r.provisional or r.ref not in known:
        raise ShowError(
            f"{name!r} was not found: no fighter with that name in the dataset "
            "(names match exactly, ignoring accents, case and punctuation)"
        )
    return r.ref


def _recent_classes(bouts: Sequence[BoutFacts], ref: str, before: date) -> list[str]:
    """Weight classes ``ref`` fought at before ``before``, most recent first."""
    seen: list[str] = []
    for b in reversed(bouts):
        if b.date >= before or ref not in (b.fighter_a_id, b.fighter_b_id):
            continue
        if b.weight_class and b.weight_class not in seen:
            seen.append(b.weight_class)
    return seen


def shared_division(data: FeatureData, a: str, b: str, on: date) -> str | None:
    """The division both fighters have fought at, the most recent for the pair:
    smallest sum of each fighter's recency rank (A's order breaks ties)."""
    ra, rb = (_recent_classes(data.bouts, r, on) for r in (a, b))
    shared = [c for c in ra if c in rb]
    if not shared:
        return None
    return min(shared, key=lambda c: (ra.index(c) + rb.index(c), ra.index(c)))


@dataclass(frozen=True, slots=True)
class Hypothetical:
    name_a: str
    name_b: str
    on: date
    weight_class: str
    weight_class_source: str
    model_version: str
    p_a: float
    p_b: float
    limited: list[str]
    why: dict


def predict_matchup(
    conn: sqlite3.Connection,
    resolver: FighterResolver,
    models_root: Path,
    name_a: str,
    name_b: str,
    on: date,
    weight_class: str | None,
) -> Hypothetical:
    data = load_feature_data(conn)
    known_classes = {b.weight_class for b in data.bouts if b.weight_class}
    if weight_class is not None and weight_class not in known_classes:
        raise ShowError(
            f"Unknown weight class {weight_class!r}. Known: {', '.join(sorted(known_classes))}"
        )
    known = {r[0] for r in conn.execute("SELECT fighter_id FROM fighters")}
    a = resolve_name(resolver, known, name_a, weight_class)
    b = resolve_name(resolver, known, name_b, weight_class)
    if a == b:
        raise ShowError(f"{name_a!r} and {name_b!r} are the same fighter")

    source = "given"
    if weight_class is None:
        weight_class = shared_division(data, a, b, on)
        source = "the division they share most recently"
        if weight_class is None:
            raise ShowError(
                "These fighters share no division in the dataset; pass --weight-class."
            )

    try:
        candidate, manifest = registry.load_approved(models_root)
    except registry.RegistryError as exc:
        raise ShowError(f"No approved model: {exc}") from exc
    version = manifest["version"]
    pipeline = registry.read_json(models_root, version, "feature_pipeline.json")
    merged, _ = merged_attributes(conn, data.fighters)
    store = feature_store(data, pipeline, merged)

    matchup = build_matchup(store, a, b, on, weight_class)
    x = model_inputs(matchup, candidate.spec.names)
    p_a, p_b = candidate.predict_pair(x)
    p_a, p_b = float(p_a[0]), float(p_b[0])
    names = (name_a, name_b)
    why = explain_bout(candidate, x, matchup, names, p_a, p_b)["why"]
    limited = limited_reasons(
        matchup, names, (store.fighters.get(a), store.fighters.get(b)), []
    )
    return Hypothetical(
        name_a, name_b, on, weight_class, source, version, p_a, p_b, limited, why
    )


def format_hypothetical(h: Hypothetical, width: int) -> list[str]:
    bar = "!" * min(width, 72)
    lines = [bar, f"  {HYPOTHETICAL_BANNER}", bar]
    lines += format_prediction(
        h.name_a,
        h.name_b,
        h.p_a,
        h.p_b,
        [h.weight_class, "hypothetical", *(["limited data"] if h.limited else [])],
        h.limited,
        h.why,
        width,
    )
    lines += _wrap(
        f"Date {h.on.isoformat()}; weight class {h.weight_class} ({h.weight_class_source}); "
        f"model {h.model_version}. Not written to the predictions table or "
        "data/publish/, and no stale-stats check is made.",
        width,
        "    ",
    )
    return lines
