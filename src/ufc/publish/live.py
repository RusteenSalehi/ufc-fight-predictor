"""The live track record: predictions made before an event, scored once its
results reach the dataset (DECISIONS D66).

Rules, applied per upcoming bout that had at least one prediction and whose
event date is today or earlier:

1. **Before the first bout.** A prediction counts as live only if it was
   created before the event's first bout. Start times are not in the data,
   so the cut-off is the earliest moment the event's date begins anywhere
   on Earth: midnight at UTC+14, which is 10:00 UTC the day before. Every
   bout on that date starts later. Of the predictions created before the
   cut-off the latest is used. If it had been superseded before the cut-off
   (its bout was cancelled or became unresolved), nothing was on the card
   at the cut-off and the bout is "cancelled".
2. **Results.** Only when the dataset's latest event is on or after the
   event date; until then the bout is "pending". Fighters are matched by
   their dataset IDs (a provisional fighter by the ID it was linked to) and
   the event date: the completed bout on that date between the same two
   fighters, in either order. None found is "not_found" (cancelled late, or
   a date that differs between Wikipedia and the dataset; listed so that a
   person can check); a provisional fighter never linked is "unlinked".
3. **Outcomes.** A win is scored with the live prediction's calibrated
   probability. A draw or no contest is "no result": counted, not scored.

Metrics use ``ufc.models.metrics``, the functions behind the backtest's
numbers, so the two are computed the same way; they are never pooled.
"""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

import numpy as np

from ufc.models import metrics

EARLIEST_UTC_OFFSET = timedelta(hours=14)
"""UTC+14 (Kiribati's Line Islands) is the furthest-ahead time zone, so a
date begins there before anywhere else."""

STATUSES = (
    "scored",
    "no_result",
    "pending",
    "cancelled",
    "no_live_prediction",
    "not_found",
    "unlinked",
)

RULES = [
    (
        "A prediction is live only if it was created before the event's first "
        "bout. Start times are not in the data, so the cut-off is 10:00 UTC on "
        "the day before the event date (midnight at UTC+14, the earliest any "
        "date begins)."
    ),
    (
        "Of the live predictions for a bout, the latest one is scored; it is the "
        "one the published card showed at the cut-off. A bout taken off the card "
        "before the cut-off is not scored."
    ),
    (
        "Results come from the dataset only: the completed bout between the same "
        "two fighters (by dataset ID) on the event date."
    ),
    "Draws and no contests are counted as no result and not scored.",
    "Live results are never pooled with backtest results.",
]


def live_cutoff(event_date: date) -> datetime:
    return datetime.combine(event_date, time(0), UTC) - EARLIEST_UTC_OFFSET


def parse_moment(value: str) -> datetime:
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        raise ValueError(f"timestamp without a time zone: {value!r}")
    return moment.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class PredictionRow:
    prediction_id: int
    bout_id: str
    model_version: str
    created_at: datetime
    fighter_a_ref: str
    fighter_b_ref: str
    p_fighter_a: float
    limited_data: bool
    superseded_at: datetime | None
    superseded_reason: str | None


@dataclass(frozen=True, slots=True)
class CardBout:
    bout_id: str
    event_key: str
    event_name: str
    event_date: date
    name_a: str
    name_b: str


@dataclass(frozen=True, slots=True)
class Result:
    bout_id: str
    date: date
    fighter_a_id: str
    fighter_b_id: str
    result: str  # win, draw, nc
    winner_id: str | None


@dataclass(frozen=True, slots=True)
class LiveOutcome:
    bout: CardBout
    status: str
    detail: str | None = None
    prediction: PredictionRow | None = None
    result: Result | None = None
    label: float | None = None  # 1.0 if fighter A won, 0.0 if B won

    def to_dict(self) -> dict:
        p = self.prediction
        scored = self.status == "scored"
        y = np.array([self.label]) if scored else None
        pa = np.array([p.p_fighter_a]) if scored and p else None
        outcome = None
        if self.result is not None and self.status in ("scored", "no_result"):
            outcome = (
                ("fighter_a" if self.label == 1.0 else "fighter_b")
                if scored
                else self.result.result
            )
        return {
            "bout_id": self.bout.bout_id,
            "event_key": self.bout.event_key,
            "event_name": self.bout.event_name,
            "event_date": self.bout.event_date.isoformat(),
            "fighter_a": self.bout.name_a,
            "fighter_b": self.bout.name_b,
            "status": self.status,
            "detail": self.detail,
            "prediction_id": p.prediction_id if p else None,
            "model_version": p.model_version if p else None,
            "predicted_at": p.created_at.strftime("%Y-%m-%dT%H:%M:%SZ") if p else None,
            "p_fighter_a": p.p_fighter_a if p else None,
            "limited_data": p.limited_data if p else None,
            "result_bout_id": self.result.bout_id if self.result else None,
            "outcome": outcome,
            "log_loss": float(metrics.bout_log_loss(pa, y)[0]) if scored else None,
            "brier": float(metrics.bout_brier(pa, y)[0]) if scored else None,
            "correct": float(metrics.bout_correct(pa, y)[0]) if scored else None,
        }


def _dataset_id(ref: str, links: Mapping[str, str | None]) -> str | None:
    """A prediction's fighter reference as a dataset fighter ID."""
    if ref.startswith("wp"):
        return links.get(ref)
    return ref


def score_bout(
    bout: CardBout,
    predictions: Sequence[PredictionRow],
    results_on_date: Sequence[Result],
    links: Mapping[str, str | None],
    dataset_last_event: date | None,
) -> LiveOutcome:
    """One bout's live status. ``predictions`` are every prediction ever
    made for it; ``results_on_date`` the dataset's completed bouts on its
    event date."""
    cutoff = live_cutoff(bout.event_date)
    before = [p for p in predictions if p.created_at < cutoff]
    if not before:
        return LiveOutcome(
            bout, "no_live_prediction", "every prediction was made after the cut-off"
        )
    chosen = max(before, key=lambda p: (p.created_at, p.prediction_id))
    if chosen.superseded_at is not None and chosen.superseded_at < cutoff:
        return LiveOutcome(
            bout,
            "cancelled",
            f"off the card before the cut-off ({chosen.superseded_reason})",
            chosen,
        )
    if dataset_last_event is None or dataset_last_event < bout.event_date:
        return LiveOutcome(bout, "pending", "results not in the dataset yet", chosen)
    a_id = _dataset_id(chosen.fighter_a_ref, links)
    b_id = _dataset_id(chosen.fighter_b_ref, links)
    if a_id is None or b_id is None:
        ref = chosen.fighter_a_ref if a_id is None else chosen.fighter_b_ref
        return LiveOutcome(
            bout, "unlinked", f"provisional fighter {ref} has no dataset ID", chosen
        )
    matches = [
        r for r in results_on_date if {r.fighter_a_id, r.fighter_b_id} == {a_id, b_id}
    ]
    if len(matches) != 1:
        detail = (
            "no completed bout between these fighters on the event date"
            if not matches
            else f"{len(matches)} completed bouts between these fighters on the date"
        )
        return LiveOutcome(bout, "not_found", detail, chosen)
    (match,) = matches
    if match.result != "win" or match.winner_id is None:
        return LiveOutcome(bout, "no_result", match.result, chosen, match)
    label = 1.0 if match.winner_id == a_id else 0.0
    return LiveOutcome(bout, "scored", None, chosen, match, label)


def score_live(
    bouts: Iterable[CardBout],
    predictions: Iterable[PredictionRow],
    results: Iterable[Result],
    links: Mapping[str, str | None],
    dataset_last_event: date | None,
    today: date,
) -> list[LiveOutcome]:
    """Every bout with a prediction whose event date is ``today`` or earlier,
    in event-date and bout order."""
    by_bout: dict[str, list[PredictionRow]] = defaultdict(list)
    for p in predictions:
        by_bout[p.bout_id].append(p)
    by_date: dict[date, list[Result]] = defaultdict(list)
    for r in results:
        by_date[r.date].append(r)
    out = []
    for bout in sorted(bouts, key=lambda b: (b.event_date, b.event_key, b.bout_id)):
        if bout.event_date > today or not by_bout.get(bout.bout_id):
            continue
        out.append(
            score_bout(
                bout,
                by_bout[bout.bout_id],
                by_date.get(bout.event_date, []),
                links,
                dataset_last_event,
            )
        )
    return out


def _metrics(rows: Sequence[LiveOutcome]) -> dict | None:
    if not rows:
        return None
    p = np.array([r.prediction.p_fighter_a for r in rows])  # type: ignore[union-attr]
    y = np.array([r.label for r in rows])
    out = metrics.summary(p, y)
    out["ece"] = out["ece"] if np.isfinite(out["ece"]) else None
    return out


def live_record(outcomes: Sequence[LiveOutcome]) -> dict:
    """The live section of ``performance.json``."""
    scored = [o for o in outcomes if o.status == "scored"]
    counts = Counter(o.status for o in outcomes)
    p = np.array([o.prediction.p_fighter_a for o in scored])  # type: ignore[union-attr]
    y = np.array([o.label for o in scored])
    versions = sorted({o.prediction.model_version for o in scored})  # type: ignore[union-attr]
    return {
        "label": "Live: published before the event, scored after it",
        "rules": RULES,
        "counts": {s: counts.get(s, 0) for s in STATUSES},
        "metrics": _metrics(scored),
        "metrics_without_limited_data": _metrics(
            [o for o in scored if not o.prediction.limited_data]  # type: ignore[union-attr]
        ),
        "by_model_version": {
            v: _metrics([o for o in scored if o.prediction.model_version == v])
            for v in versions
        },
        "calibration": metrics.confidence_table(p, y),
        "calibration_buckets": "the favoured fighter's probability, in the "
        "backtest's confidence buckets",
        "bouts": [o.to_dict() for o in outcomes],
    }


# --- reading the database ------------------------------------------------------------


def _moment(value: str | None) -> datetime | None:
    return parse_moment(value) if value else None


def read_live_inputs(
    conn: sqlite3.Connection,
) -> tuple[list[CardBout], list[PredictionRow], list[Result], dict[str, str | None]]:
    """Card bouts, every prediction, completed results on the card dates, and
    provisional fighters' links. ``conn`` returns ``sqlite3.Row`` rows."""
    bouts = [
        CardBout(
            r["bout_id"],
            r["event_key"],
            r["event_name"],
            date.fromisoformat(r["event_date"]),
            r["name_a"],
            r["name_b"],
        )
        for r in conn.execute(
            """
            SELECT b.bout_id, b.event_key, b.name_a, b.name_b,
                   e.name AS event_name, e.date AS event_date
            FROM upcoming_bouts b JOIN upcoming_events e USING (event_key)
            WHERE b.bout_id IN (SELECT bout_id FROM predictions)
            """
        )
    ]
    predictions = [
        PredictionRow(
            r["prediction_id"],
            r["bout_id"],
            r["model_version"],
            parse_moment(r["created_at"]),
            r["fighter_a_ref"],
            r["fighter_b_ref"],
            r["p_fighter_a"],
            bool(r["limited_data"]),
            _moment(r["superseded_at"]),
            r["superseded_reason"],
        )
        for r in conn.execute("SELECT * FROM predictions ORDER BY prediction_id")
    ]
    dates = sorted({b.event_date.isoformat() for b in bouts})
    results = []
    if dates:
        marks = ",".join("?" * len(dates))
        results = [
            Result(
                r["bout_id"],
                date.fromisoformat(r["date"]),
                r["fighter_a_id"],
                r["fighter_b_id"],
                r["result"],
                r["winner_id"],
            )
            for r in conn.execute(
                f"""
                SELECT b.bout_id, e.date, b.fighter_a_id, b.fighter_b_id,
                       b.result, b.winner_id
                FROM bouts b JOIN events e USING (event_id)
                WHERE b.status = 'completed' AND e.date IN ({marks})
                """,
                dates,
            )
        ]
    links = {
        r["provisional_id"]: r["fighter_id"]
        for r in conn.execute(
            "SELECT provisional_id, fighter_id FROM provisional_fighters"
        )
    }
    return bouts, predictions, results, links
