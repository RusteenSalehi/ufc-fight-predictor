"""Live scoring: which prediction counts, how bouts are matched to results,
and what is scored. Pure functions on invented rows; no database."""

from datetime import UTC, date, datetime

import numpy as np
import pytest

from ufc.models import metrics
from ufc.publish.live import (
    CardBout,
    PredictionRow,
    Result,
    live_cutoff,
    live_record,
    score_bout,
    score_live,
)
from ufc.publish.schemas import LiveRecord

EVENT = date(2026, 11, 21)
A, B, C = "a" * 16, "b" * 16, "c" * 16
BOUT = CardBout("wb" + "1" * 16, "wp:1", "UFC Test", EVENT, "Ann", "Bea")


def at(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone(UTC)


def prediction(
    pid=1,
    created="2026-11-18T12:00:00Z",
    p_a=0.7,
    a=A,
    b=B,
    superseded=None,
    reason=None,
    bout=BOUT.bout_id,
    limited=False,
    version="m1",
):
    return PredictionRow(
        pid,
        bout,
        version,
        at(created),
        a,
        b,
        p_a,
        limited,
        at(superseded) if superseded else None,
        reason,
    )


def result(a=A, b=B, outcome="win", winner=A, on=EVENT, bout="f" * 16):
    return Result(bout, on, a, b, outcome, winner if outcome == "win" else None)


def score(predictions, results=(), links=None, last=EVENT, bout=BOUT):
    return score_bout(bout, predictions, list(results), links or {}, last)


# --- the before-the-first-bout rule ------------------------------------------------


def test_the_cutoff_is_midnight_at_utc_plus_14():
    assert live_cutoff(EVENT) == datetime(2026, 11, 20, 10, 0, tzinfo=UTC)


def test_a_prediction_counts_only_if_made_before_the_cutoff():
    made_after = prediction(created="2026-11-20T10:00:00Z")  # exactly at: not before
    out = score([made_after], [result()])
    assert out.status == "no_live_prediction" and out.prediction is None
    made_before = prediction(created="2026-11-20T09:59:59Z")
    assert score([made_before], [result()]).status == "scored"


def test_the_latest_prediction_before_the_cutoff_is_used():
    early = prediction(
        1,
        "2026-11-15T12:00:00Z",
        0.60,
        superseded="2026-11-19T08:00:00Z",
        reason="inputs changed",
    )
    late = prediction(
        2,
        "2026-11-19T08:00:00Z",
        0.80,
        superseded="2026-11-20T15:00:00Z",
        reason="inputs changed",
    )
    after = prediction(3, "2026-11-20T15:00:00Z", 0.30)  # fight week, too late
    out = score([after, early, late], [result()])
    assert out.status == "scored" and out.prediction.prediction_id == 2
    assert out.prediction.p_fighter_a == 0.80


def test_a_bout_cancelled_before_the_cutoff_is_not_scored():
    cancelled = prediction(superseded="2026-11-19T00:00:00Z", reason="bout cancelled")
    # Even if the same two fighters somehow fought on the date.
    out = score([cancelled], [result()])
    assert out.status == "cancelled" and "bout cancelled" in out.detail
    assert out.label is None


def test_a_bout_cancelled_after_the_cutoff_and_not_fought_is_not_found():
    late_cancel = prediction(superseded="2026-11-20T20:00:00Z", reason="bout cancelled")
    out = score([late_cancel], [result(a=A, b=C)])  # A fought someone else
    assert out.status == "not_found" and out.prediction is not None


def test_a_cancelled_and_restored_bout_uses_the_prediction_on_the_card():
    first = prediction(
        1,
        "2026-11-15T00:00:00Z",
        0.6,
        superseded="2026-11-16T00:00:00Z",
        reason="bout cancelled",
    )
    restored = prediction(2, "2026-11-17T00:00:00Z", 0.65)
    out = score([first, restored], [result()])
    assert out.status == "scored" and out.prediction.prediction_id == 2


# --- matching ---------------------------------------------------------------------


def test_results_match_by_fighter_ids_in_either_order_and_the_event_date():
    p = prediction(p_a=0.7)
    won = score([p], [result(a=B, b=A, winner=A)])
    assert won.status == "scored" and won.label == 1.0
    lost = score([p], [result(a=B, b=A, winner=B)])
    assert lost.status == "scored" and lost.label == 0.0
    # The same pair on another date is not this bout (score_live groups by date).
    outcomes = score_live(
        [BOUT], [p], [result(on=date(2026, 11, 22))], {}, EVENT, EVENT
    )
    assert [o.status for o in outcomes] == ["not_found"]


def test_draws_and_no_contests_are_no_result():
    for outcome in ("draw", "nc"):
        out = score([prediction()], [result(outcome=outcome)])
        assert out.status == "no_result" and out.label is None
        assert out.to_dict()["outcome"] == outcome and out.to_dict()["log_loss"] is None


def test_results_not_yet_in_the_dataset_are_pending():
    out = score([prediction()], [], last=date(2026, 11, 14))
    assert out.status == "pending"
    assert score([prediction()], [], last=None).status == "pending"


def test_provisional_fighters_match_through_their_link():
    p = prediction(b="wp:Newcomer Person")
    assert score([p], [result()]).status == "unlinked"
    linked = score([p], [result()], links={"wp:Newcomer Person": B})
    assert linked.status == "scored" and linked.label == 1.0


def test_only_past_bouts_with_a_prediction_are_listed():
    future = CardBout("wb" + "2" * 16, "wp:2", "Later", date(2026, 12, 5), "C", "D")
    never = CardBout("wb" + "3" * 16, "wp:1", "UFC Test", EVENT, "E", "F")
    outcomes = score_live(
        [future, never, BOUT],
        [prediction(), prediction(5, bout=future.bout_id)],
        [result()],
        {},
        EVENT,
        date(2026, 11, 22),
    )
    assert [o.bout.bout_id for o in outcomes] == [BOUT.bout_id]


# --- the record ---------------------------------------------------------------------


def _bout(i):
    return CardBout(f"wb{i:016x}", "wp:1", "UFC Test", EVENT, f"A{i}", f"B{i}")


def test_the_record_scores_with_the_backtest_metrics_and_validates():
    bouts = [_bout(i) for i in range(1, 9)]
    ps = [0.8, 0.7, 0.65, 0.55, 0.4, 0.3, 0.6, 0.9]
    winners = [A, A, B, A, B, B, None, A]
    preds, results = [], []
    for i, (bout, p, winner) in enumerate(zip(bouts, ps, winners, strict=True)):
        a, b = f"{i:016x}", f"{i + 100:016x}"
        preds.append(
            prediction(i + 1, p_a=p, a=a, b=b, bout=bout.bout_id, limited=i == 0)
        )
        w = {A: a, B: b, None: None}[winner]
        results.append(result(a=a, b=b, winner=w, outcome="win" if w else "draw"))
    # A cancelled bout and a pending one on a later date as well.
    cancelled = _bout(20)
    preds.append(
        prediction(
            30,
            bout=cancelled.bout_id,
            superseded="2026-11-18T13:00:00Z",
            reason="bout cancelled",
        )
    )
    outcomes = score_live([*bouts, cancelled], preds, results, {}, EVENT, EVENT)
    record = live_record(outcomes)
    assert record["counts"]["scored"] == 7 and record["counts"]["no_result"] == 1
    assert record["counts"]["cancelled"] == 1
    scored = [(p, w) for p, w in zip(ps, winners, strict=True) if w]
    p = np.array([x for x, _ in scored])
    y = np.array([1.0 if w == A else 0.0 for _, w in scored])
    assert record["metrics"]["log_loss"] == pytest.approx(metrics.log_loss(p, y))
    assert record["metrics"]["brier"] == pytest.approx(metrics.brier(p, y))
    assert record["metrics"]["accuracy"] == pytest.approx(metrics.accuracy(p, y))
    assert record["metrics_without_limited_data"]["n"] == 6
    assert sum(r["n"] for r in record["calibration"]) == 7
    assert [r["low"] for r in record["calibration"]] == list(
        metrics.CONFIDENCE_EDGES[:-1]
    )
    LiveRecord.model_validate(record)


def test_an_empty_record_is_valid_and_has_no_metrics():
    record = live_record([])
    assert record["metrics"] is None and record["bouts"] == []
    assert all(n == 0 for n in record["counts"].values())
    LiveRecord.model_validate(record)
