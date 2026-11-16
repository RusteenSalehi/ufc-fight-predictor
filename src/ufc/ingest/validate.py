"""Checks on an assembled dataset: counts, keys, references, dates, stat sanity.

Pure. Returns issues; it never changes or drops data. Checks that can fire on
thousands of rows are reported as one issue with a count and a few examples.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import date

from ufc.ingest.models import Dataset, Issue, RoundStats

EARLIEST_PLAUSIBLE_EVENT = date(1993, 1, 1)
EARLIEST_PLAUSIBLE_BIRTH = date(1930, 1, 1)
MIN_FIGHTING_AGE = 17
MAX_FIGHTING_AGE = 55
_MAX_EXAMPLES = 5


def _examples(ids: Iterable[str]) -> str:
    unique = list(dict.fromkeys(ids))
    shown = ", ".join(unique[:_MAX_EXAMPLES])
    return shown + (
        f" and {len(unique) - _MAX_EXAMPLES} more"
        if len(unique) > _MAX_EXAMPLES
        else ""
    )


def _years_between(earlier: date, later: date) -> float:
    return (later - earlier).days / 365.25


def validate_dataset(dataset: Dataset, as_of: date) -> list[Issue]:
    """Run every dataset-level check. ``as_of`` is the snapshot's date."""
    issues: list[Issue] = []
    issues += _row_counts(dataset)
    issues += _duplicate_ids(dataset)
    issues += _references(dataset)
    issues += _orphans(dataset)
    issues += _date_sanity(dataset, as_of)
    issues += _results_breakdown(dataset)
    issues += _bout_sanity(dataset)
    issues += _stat_sanity(dataset.round_stats)
    return issues


def _row_counts(dataset: Dataset) -> list[Issue]:
    dropped = dataset.source_fights - len(dataset.bouts)
    with_stats = sum(bout.has_stats for bout in dataset.bouts)
    return [
        Issue(
            "info",
            "row_counts",
            f"events={len(dataset.events)} fighters={len(dataset.fighters)} "
            f"bouts={len(dataset.bouts)} (of {dataset.source_fights} distinct fights in the "
            f"source; {dropped} dropped) bouts_with_stats={with_stats} "
            f"round_stat_rows={len(dataset.round_stats)} fight_stat_rows={len(dataset.fight_stats)}",
        )
    ]


def _duplicate_ids(dataset: Dataset) -> list[Issue]:
    issues: list[Issue] = []
    keyed = {
        "event": [e.event_id for e in dataset.events],
        "fighter": [f.fighter_id for f in dataset.fighters],
        "bout": [b.bout_id for b in dataset.bouts],
        "round stats": [
            f"{r.bout_id}/{r.fighter_id}/{r.round}" for r in dataset.round_stats
        ],
        "fight stats": [f"{s.bout_id}/{s.fighter_id}" for s in dataset.fight_stats],
    }
    for label, keys in keyed.items():
        repeated = [key for key, count in Counter(keys).items() if count > 1]
        if repeated:
            issues.append(
                Issue(
                    "error",
                    "duplicate_id",
                    f"{len(repeated)} duplicate {label} keys: {_examples(repeated)}",
                )
            )
    return issues


def _references(dataset: Dataset) -> list[Issue]:
    """Every foreign key must point at a row in this dataset."""
    issues: list[Issue] = []
    event_ids = {e.event_id for e in dataset.events}
    fighter_ids = {f.fighter_id for f in dataset.fighters}
    bouts = {b.bout_id: b for b in dataset.bouts}

    for bout in dataset.bouts:
        if bout.event_id not in event_ids:
            issues.append(
                Issue(
                    "error",
                    "broken_reference",
                    "bout's event is missing",
                    "bout",
                    bout.bout_id,
                )
            )
        for fighter_id in (bout.fighter_a_id, bout.fighter_b_id):
            if fighter_id not in fighter_ids:
                issues.append(
                    Issue(
                        "error",
                        "broken_reference",
                        f"bout's fighter {fighter_id} is missing from the fighters table",
                        "bout",
                        bout.bout_id,
                    )
                )
    for row in [*dataset.round_stats, *dataset.fight_stats]:
        bout = bouts.get(row.bout_id)
        if bout is None or row.fighter_id not in (bout.fighter_a_id, bout.fighter_b_id):
            issues.append(
                Issue(
                    "error",
                    "broken_reference",
                    f"stats row for fighter {row.fighter_id} does not belong to a loaded bout",
                    "bout",
                    row.bout_id,
                )
            )
    return issues


def _orphans(dataset: Dataset) -> list[Issue]:
    issues: list[Issue] = []
    fighting = {
        fighter_id
        for b in dataset.bouts
        for fighter_id in (b.fighter_a_id, b.fighter_b_id)
    }
    idle = len(dataset.fighters) - len(
        fighting & {f.fighter_id for f in dataset.fighters}
    )
    if idle:
        issues.append(
            Issue(
                "info",
                "fighters_without_bouts",
                f"{idle} of {len(dataset.fighters)} fighter profiles appear in no loaded bout "
                "(the source lists fighters from other promotions too)",
            )
        )
    events_with_bouts = {b.event_id for b in dataset.bouts}
    empty = [e for e in dataset.events if e.event_id not in events_with_bouts]
    if empty:
        issues.append(
            Issue(
                "warning",
                "events_without_bouts",
                f"{len(empty)} events have no loaded bouts: {_examples(e.name for e in empty)}",
            )
        )
    return issues


def _date_sanity(dataset: Dataset, as_of: date) -> list[Issue]:
    issues: list[Issue] = []
    events = {e.event_id: e for e in dataset.events}
    fighters = {f.fighter_id: f for f in dataset.fighters}

    for event in dataset.events:
        if not EARLIEST_PLAUSIBLE_EVENT <= event.date <= as_of:
            issues.append(
                Issue(
                    "warning",
                    "date_sanity",
                    f"completed event dated {event.date}, outside {EARLIEST_PLAUSIBLE_EVENT} to "
                    f"{as_of}",
                    "event",
                    event.event_id,
                )
            )
    for fighter in dataset.fighters:
        if (
            fighter.dob is not None
            and not EARLIEST_PLAUSIBLE_BIRTH <= fighter.dob <= as_of
        ):
            issues.append(
                Issue(
                    "warning",
                    "date_sanity",
                    f"date of birth {fighter.dob} is implausible",
                    "fighter",
                    fighter.fighter_id,
                )
            )
    for bout in dataset.bouts:
        event = events.get(bout.event_id)
        for fighter_id in (bout.fighter_a_id, bout.fighter_b_id):
            fighter = fighters.get(fighter_id)
            if event is None or fighter is None or fighter.dob is None:
                continue
            age = _years_between(fighter.dob, event.date)
            if not MIN_FIGHTING_AGE <= age <= MAX_FIGHTING_AGE:
                issues.append(
                    Issue(
                        "warning",
                        "date_sanity",
                        f"{fighter.name} ({fighter_id}) would be {age:.1f} years old at this bout "
                        f"on {event.date}; wrong date of birth or wrong fighter",
                        "bout",
                        bout.bout_id,
                    )
                )
    if dataset.events:
        first = min(dataset.events, key=lambda e: e.date)
        last = max(dataset.events, key=lambda e: e.date)
        issues.append(
            Issue(
                "info",
                "date_range",
                f"events run from {first.date} ({first.name}) to {last.date} ({last.name})",
            )
        )
    return issues


def _results_breakdown(dataset: Dataset) -> list[Issue]:
    def tally(values: Iterable[str | None]) -> str:
        counts = Counter(str(v) for v in values)
        return " ".join(f"{name!r}={count}" for name, count in counts.most_common())

    if not dataset.bouts:
        return [Issue("error", "row_counts", "no bouts were loaded")]
    return [
        Issue(
            "info",
            "results_breakdown",
            "by result: " + tally(b.result for b in dataset.bouts),
        ),
        Issue(
            "info",
            "results_breakdown",
            "by method: " + tally(b.method for b in dataset.bouts),
        ),
    ]


def _bout_sanity(dataset: Dataset) -> list[Issue]:
    issues: list[Issue] = []
    odd_pairs = [
        b.bout_id
        for b in dataset.bouts
        if (b.result == "nc") != (b.method in ("Overturned", "Could Not Continue"))
        and not (b.result == "draw" and b.method == "Overturned")
    ]
    if odd_pairs:
        issues.append(
            Issue(
                "warning",
                "result_method_mismatch",
                f"{len(odd_pairs)} bouts pair a no-contest method with a decisive result or the "
                f"reverse: {_examples(odd_pairs)}",
            )
        )
    unclassed = sum(b.weight_class is None for b in dataset.bouts)
    if unclassed:
        issues.append(
            Issue(
                "info",
                "no_weight_class",
                f"{unclassed} bouts name no weight class (early tournament finals and superfights)",
            )
        )
    return issues


def _stat_sanity(rows: list[RoundStats]) -> list[Issue]:
    """Arithmetic that should hold within one fighter's round."""
    failures: dict[str, list[str]] = {}

    def fail(name: str, row: RoundStats) -> None:
        failures.setdefault(name, []).append(row.bout_id)

    for row in rows:
        s = row.stats
        pairs = (
            (s.sig_str_landed, s.sig_str_attempted),
            (s.total_str_landed, s.total_str_attempted),
            (s.td_landed, s.td_attempted),
            (s.head_landed, s.head_attempted),
            (s.body_landed, s.body_attempted),
            (s.leg_landed, s.leg_attempted),
            (s.distance_landed, s.distance_attempted),
            (s.clinch_landed, s.clinch_attempted),
            (s.ground_landed, s.ground_attempted),
        )
        if any(landed > attempted for landed, attempted in pairs):
            fail("more landed than attempted", row)
        if (
            s.sig_str_landed > s.total_str_landed
            or s.sig_str_attempted > s.total_str_attempted
        ):
            fail("more significant strikes than total strikes", row)
        by_target = (
            s.head_landed + s.body_landed + s.leg_landed,
            s.head_attempted + s.body_attempted + s.leg_attempted,
        )
        by_position = (
            s.distance_landed + s.clinch_landed + s.ground_landed,
            s.distance_attempted + s.clinch_attempted + s.ground_attempted,
        )
        if by_target != (s.sig_str_landed, s.sig_str_attempted):
            fail("head + body + leg differs from significant strikes", row)
        if by_position != (s.sig_str_landed, s.sig_str_attempted):
            fail("distance + clinch + ground differs from significant strikes", row)

    return [
        Issue(
            "warning",
            "stat_sanity",
            f"{len(bout_ids)} fighter-rounds have {name}; bouts: {_examples(bout_ids)}",
        )
        for name, bout_ids in failures.items()
    ]
