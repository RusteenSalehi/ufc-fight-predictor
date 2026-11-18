"""Pydantic models of every published artifact (DECISIONS D65).

Every artifact is validated against its model before anything is written
(``gate.py``), and the JSON on disk is the model's own dump, so a file that
exists always matches its schema. The models forbid unknown fields and
non-finite numbers, and check what a schema can check about the numbers:
probabilities in [0, 1] summing to 1, the predicted winner and the fair
moneyline agreeing with the probabilities, limited-data flags agreeing with
their reasons, counts agreeing with lists.

``SCHEMA_VERSION`` changes whenever a field is removed, renamed or changes
meaning. Adding an optional field does not change it; the market fields are
present (as nulls) from the first version so that M12 can fill them without
a schema change.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
Side = Literal["fighter_a", "fighter_b"]
IsoDate = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
IsoMoment = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")]
BoutId = Annotated[str, Field(pattern=r"^wb[0-9a-f]{16}$")]
EventKey = Annotated[str, Field(pattern=r"^wp:\d+$")]

SUM_TOLERANCE = 1e-12
MONEYLINE_TOLERANCE = 0.01


def fair_moneyline(p: float) -> float:
    """American odds with no margin for a win probability ``p``: negative
    for a favourite (stake to win 100), positive otherwise (win on a stake
    of 100); exactly 0.5 is +100. Rounded to cents."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"no finite moneyline for probability {p}")
    value = -100.0 * p / (1.0 - p) if p > 0.5 else 100.0 * (1.0 - p) / p
    return round(value, 2)


def winner_of(p_a: float, p_b: float) -> Side | None:
    if p_a > p_b:
        return "fighter_a"
    if p_b > p_a:
        return "fighter_b"
    return None


class Strict(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, frozen=True, populate_by_name=True
    )


class Artifact(Strict):
    schema_version: Literal[1] = SCHEMA_VERSION
    generated_at: IsoMoment


def _check_pair(p_a: float, p_b: float, winner: str | None) -> None:
    if abs(p_a + p_b - 1.0) > SUM_TOLERANCE:
        raise ValueError(f"probabilities {p_a} and {p_b} do not sum to 1")
    if winner != winner_of(p_a, p_b):
        raise ValueError(f"predicted_winner {winner!r} disagrees with {p_a}, {p_b}")


# --- shared parts --------------------------------------------------------------------


class ModelInfo(Strict):
    version: str
    status: str
    parent: str | None = None


class MissingEvent(Strict):
    name: str
    page: str
    date: IsoDate
    read: bool
    bouts: int | None = None


class Staleness(Strict):
    status: Literal["current", "behind", "unknown"]
    dataset_last_event: IsoDate | None
    missing_events: list[MissingEvent]

    @model_validator(mode="after")
    def _status_matches(self) -> Staleness:
        if (self.status == "current") != (not self.missing_events):
            raise ValueError("status 'current' exactly when no event is missing")
        return self


class Dataset(Strict):
    source_commit: str | None
    last_event: IsoDate | None


class FighterBrief(Strict):
    name: str
    ref: str | None
    provisional: bool
    wikipedia_page: str | None = None


class EventInfo(Strict):
    event_key: EventKey
    name: str
    date: IsoDate
    wikipedia_page: str
    wikipedia_revision: int | None


# --- the "why" and factors ---------------------------------------------------------


class Reason(Strict):
    reason: str
    component: str | None
    title: str
    toward: Side
    detail: str
    log_odds: float
    inputs: list[str]

    @model_validator(mode="after")
    def _points_its_way(self) -> Reason:
        if self.log_odds <= 0:
            raise ValueError(f"reason {self.reason!r} has no weight toward its side")
        return self


class Why(Strict):
    favoured: Side | None
    favoured_name: str | None = None
    for_: list[Reason] = Field(alias="for", max_length=3)
    against: Reason | None
    summary: str
    method: str

    @model_validator(mode="after")
    def _sides(self) -> Why:
        if self.favoured is None and (self.for_ or self.against):
            raise ValueError("an even bout has no reasons")
        if any(r.toward != self.favoured for r in self.for_):
            raise ValueError("a reason for the favourite points the other way")
        if self.against is not None and self.against.toward == self.favoured:
            raise ValueError("the reason against points toward the favourite")
        return self


class TopFactors(Strict):
    fighter_a: list[Reason]
    fighter_b: list[Reason]
    method: str

    @model_validator(mode="after")
    def _sides(self) -> TopFactors:
        for side in ("fighter_a", "fighter_b"):
            reasons = getattr(self, side)
            if any(r.toward != side for r in reasons):
                raise ValueError(f"a factor listed for {side} points the other way")
            weights = [r.log_odds for r in reasons]
            if weights != sorted(weights, reverse=True):
                raise ValueError(f"factors for {side} are not strongest first")
        return self


# --- bout artifact -------------------------------------------------------------------


class Moneyline(Strict):
    fighter_a: float
    fighter_b: float


class Prediction(Strict):
    prediction_id: int
    model_version: str
    predicted_at: IsoMoment
    p_fighter_a: Probability
    p_fighter_b: Probability
    predicted_winner: Side | None
    fair_moneyline: Moneyline
    fair_moneyline_note: str

    @model_validator(mode="after")
    def _consistent(self) -> Prediction:
        _check_pair(self.p_fighter_a, self.p_fighter_b, self.predicted_winner)
        for side, p in (
            ("fighter_a", self.p_fighter_a),
            ("fighter_b", self.p_fighter_b),
        ):
            got = getattr(self.fair_moneyline, side)
            if abs(got - fair_moneyline(p)) > MONEYLINE_TOLERANCE:
                raise ValueError(f"fair moneyline for {side} does not match {p}")
        return self


class Market(Strict):
    """Market odds for the bout. Null until odds are matched to bouts (M12);
    the fields exist now so that filling them is not a schema change."""

    source: str | None = None
    bookmaker: str | None = None
    captured_at: IsoMoment | None = None
    fighter_a_american: float | None = None
    fighter_b_american: float | None = None
    p_fighter_a_no_vig: Probability | None = None
    p_fighter_b_no_vig: Probability | None = None


class LimitedData(Strict):
    limited: bool
    reasons: list[str]
    stale_stats: bool

    @model_validator(mode="after")
    def _consistent(self) -> LimitedData:
        if self.limited != bool(self.reasons):
            raise ValueError("limited must be true exactly when there are reasons")
        if self.stale_stats != any(r.startswith("stale stats") for r in self.reasons):
            raise ValueError("stale_stats disagrees with the reasons")
        return self


class StatRow(Strict):
    key: str
    group: str
    label: str
    kind: str
    fighter_a: float | None
    fighter_b: float | None
    shrunk: bool


class Stats(Strict):
    note: str
    shrink_minutes: float | None
    stance: dict[Side, str | None]
    attribute_sources: dict[Side, dict[str, str | None]]
    rows: list[StatRow]


class PastBout(Strict):
    date: IsoDate
    event: str
    opponent: str
    opponent_id: str
    result: Literal["win", "loss", "draw", "nc"]
    method: str | None
    end_round: int | None
    end_time: str | None
    weight_class: str | None
    title_bout: bool
    deserved_win_probability: Probability | None


class RecentBouts(Strict):
    note: str
    performance_model: dict[str, str | int | None]
    fighter_a: list[PastBout] = Field(max_length=5)
    fighter_b: list[PastBout] = Field(max_length=5)

    @model_validator(mode="after")
    def _newest_first(self) -> RecentBouts:
        for side in ("fighter_a", "fighter_b"):
            dates = [b.date for b in getattr(self, side)]
            if dates != sorted(dates, reverse=True):
                raise ValueError(f"recent bouts of {side} are not newest first")
        return self


class Fighter(FighterBrief):
    resolution: str


class BoutArtifact(Artifact):
    kind: Literal["bout"] = "bout"
    bout_id: BoutId
    event: EventInfo
    order: int
    segment: str | None
    weight_class: str
    title_bout: bool
    fighter_a: Fighter
    fighter_b: Fighter
    prediction: Prediction
    limited_data: LimitedData
    why: Why | None
    top_factors: TopFactors | None
    stats: Stats
    recent_bouts: RecentBouts
    market: Market
    attribution: str

    @model_validator(mode="after")
    def _consistent(self) -> BoutArtifact:
        if not (self.fighter_a.ref and self.fighter_b.ref):
            raise ValueError("a predicted bout has both fighters resolved")
        if (
            self.why is not None
            and self.why.favoured != self.prediction.predicted_winner
        ):
            raise ValueError("the why favours a different fighter from the prediction")
        if (self.why is None) != (self.top_factors is None):
            raise ValueError("the why and the factors come from the same contributions")
        return self


# --- upcoming.json -------------------------------------------------------------------


class UpcomingBout(Strict):
    bout_id: BoutId
    order: int
    segment: str | None
    weight_class: str
    title_bout: bool
    fighter_a: FighterBrief
    fighter_b: FighterBrief
    status: Literal["predicted"] = "predicted"
    p_fighter_a: Probability
    p_fighter_b: Probability
    predicted_winner: Side | None
    model_version: str
    predicted_at: IsoMoment
    limited_data: bool
    limited_data_reasons: list[str]
    stale_stats: bool
    detail: str

    @model_validator(mode="after")
    def _consistent(self) -> UpcomingBout:
        _check_pair(self.p_fighter_a, self.p_fighter_b, self.predicted_winner)
        if self.limited_data != bool(self.limited_data_reasons):
            raise ValueError("limited_data must be true exactly when there are reasons")
        if self.detail != f"bouts/{self.bout_id}.json":
            raise ValueError("detail must name the bout's artifact")
        return self


class UnpredictedBout(Strict):
    bout_id: BoutId
    order: int
    segment: str | None
    weight_class: str
    title_bout: bool
    fighter_a: FighterBrief
    fighter_b: FighterBrief
    status: Literal["unresolved", "not_predicted"]
    reason: str


class UpcomingEvent(EventInfo):
    card_checked_at: IsoMoment | None
    bouts: list[UpcomingBout]
    unresolved_bouts: list[UnpredictedBout]


class Upcoming(Artifact):
    kind: Literal["upcoming"] = "upcoming"
    model: ModelInfo
    dataset: Dataset
    staleness: Staleness
    events: list[UpcomingEvent] = Field(min_length=1)
    attribution: str

    @model_validator(mode="after")
    def _consistent(self) -> Upcoming:
        dates = [e.date for e in self.events]
        if dates != sorted(dates):
            raise ValueError("events are not in date order")
        ids = [b.bout_id for e in self.events for b in (*e.bouts, *e.unresolved_bouts)]
        if len(ids) != len(set(ids)):
            raise ValueError("a bout is listed twice")
        if not any(e.bouts for e in self.events):
            raise ValueError("no predicted bouts")
        for event in self.events:
            for bout in event.bouts:
                if bout.model_version != self.model.version:
                    raise ValueError(f"{bout.bout_id} was predicted by another model")
        return self


# --- performance.json --------------------------------------------------------------


class Metrics(Strict):
    n: int = Field(ge=0)
    log_loss: float
    brier: float
    accuracy: float
    ece: float | None = None
    log_loss_low: float | None = None
    log_loss_high: float | None = None


class CalibrationRow(Strict):
    low: float
    high: float
    n: int = Field(ge=0)
    mean_confidence: float | None
    favourite_win_rate: float | None
    win_rate_low: float | None
    win_rate_high: float | None
    log_loss: float | None


class HeldOutPeriod(Strict):
    candidate_version: str
    first_date: IsoDate
    last_date: IsoDate
    n_scored: int
    n_no_result: int
    already_seen: bool
    already_seen_note: str | None
    metrics: dict[str, Metrics]
    calibration: list[CalibrationRow]
    calibration_buckets: str

    @model_validator(mode="after")
    def _counts(self) -> HeldOutPeriod:
        if self.metrics["model"].n != self.n_scored:
            raise ValueError("the model's metrics do not cover the scored bouts")
        if sum(r.n for r in self.calibration) != self.n_scored:
            raise ValueError("calibration buckets do not add up to the scored bouts")
        return self


class Fold(Strict):
    year: int
    in_test_period: bool
    n_scored: int
    metrics: dict[str, Metrics]


class Pooled(Strict):
    years: tuple[int, int]
    metrics: dict[str, Metrics]


class WalkForward(Strict):
    schedule: str
    folds: list[Fold]
    pooled_before_test_period: Pooled
    pooled_test_period: Pooled | None


class Backtest(Strict):
    label: str
    reports: str
    commands: list[str]
    candidate_version: str
    production_version: str
    production_note: str
    test_period: HeldOutPeriod
    walk_forward: WalkForward


class EraShift(Strict):
    text: str
    backtest_years: tuple[int, int]
    backtest_log_loss: float
    backtest_year_range: tuple[float, float]
    test_period_log_loss: float


LiveStatus = Literal[
    "scored",
    "no_result",
    "pending",
    "cancelled",
    "no_live_prediction",
    "not_found",
    "unlinked",
]


class LiveBout(Strict):
    bout_id: BoutId
    event_key: EventKey
    event_name: str
    event_date: IsoDate
    fighter_a: str
    fighter_b: str
    status: LiveStatus
    detail: str | None
    prediction_id: int | None
    model_version: str | None
    predicted_at: IsoMoment | None
    p_fighter_a: Probability | None
    limited_data: bool | None
    result_bout_id: str | None
    outcome: Literal["fighter_a", "fighter_b", "draw", "nc"] | None
    log_loss: float | None
    brier: float | None
    correct: float | None

    @model_validator(mode="after")
    def _consistent(self) -> LiveBout:
        scored = self.status == "scored"
        if scored != (self.log_loss is not None):
            raise ValueError("only a scored bout has a score")
        if scored and self.outcome not in ("fighter_a", "fighter_b"):
            raise ValueError("a scored bout has a winner")
        if self.status == "no_result" and self.outcome not in ("draw", "nc"):
            raise ValueError("a no-result bout is a draw or a no contest")
        if scored and (self.p_fighter_a is None or self.prediction_id is None):
            raise ValueError("a scored bout has its live prediction")
        return self


class LiveRecord(Strict):
    label: str
    rules: list[str]
    counts: dict[LiveStatus, int]
    metrics: Metrics | None
    metrics_without_limited_data: Metrics | None
    by_model_version: dict[str, Metrics]
    calibration: list[CalibrationRow]
    calibration_buckets: str
    bouts: list[LiveBout]

    @model_validator(mode="after")
    def _consistent(self) -> LiveRecord:
        counted = {s: 0 for s in self.counts}
        for b in self.bouts:
            counted[b.status] = counted.get(b.status, 0) + 1
        if counted != self.counts:
            raise ValueError("counts disagree with the bouts listed")
        n = self.counts.get("scored", 0)
        if (self.metrics is None) != (n == 0):
            raise ValueError("metrics exist exactly when a bout is scored")
        if self.metrics is not None and self.metrics.n != n:
            raise ValueError("metrics do not cover the scored bouts")
        if sum(r.n for r in self.calibration) != n:
            raise ValueError("calibration buckets do not add up to the scored bouts")
        return self


class Performance(Artifact):
    kind: Literal["performance"] = "performance"
    model: ModelInfo
    separation: str
    backtest: Backtest | None
    backtest_unavailable: str | None
    era_shift_caveat: EraShift | None
    live: LiveRecord

    @model_validator(mode="after")
    def _consistent(self) -> Performance:
        if (self.backtest is None) == (self.backtest_unavailable is None):
            raise ValueError("either the backtest or the reason it is missing")
        return self


# --- archive -------------------------------------------------------------------------


class ArchivedBout(Strict):
    bout_id: BoutId
    order: int
    weight_class: str
    fighter_a: FighterBrief
    fighter_b: FighterBrief
    prediction_id: int
    model_version: str
    predicted_at: IsoMoment
    inputs_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    p_fighter_a: Probability
    p_fighter_b: Probability
    predicted_winner: Side | None
    limited_data_reasons: list[str]
    why_summary: str | None

    @model_validator(mode="after")
    def _consistent(self) -> ArchivedBout:
        _check_pair(self.p_fighter_a, self.p_fighter_b, self.predicted_winner)
        return self


class ArchiveSnapshot(Strict):
    """The predictions published for one event at one moment. Written once,
    never overwritten; the evidence for the live track record."""

    schema_version: Literal[1] = SCHEMA_VERSION
    kind: Literal["prediction_archive"] = "prediction_archive"
    published_at: IsoMoment
    event: EventInfo
    model_version: str
    content_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    bouts: list[ArchivedBout] = Field(min_length=1)
    unresolved_bouts: list[UnpredictedBout]


def finite(value: float | None) -> float | None:
    """``None`` for a missing or non-finite number."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)
