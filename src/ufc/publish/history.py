"""Each fighter's last UFC bouts, with the M4.6 deserved-win probability.

This is context for a reader of the bout page, not anything the prediction
used. The deserved-win probability is the M4.6 performance model (DECISIONS
D63): a logistic regression on in-fight stat differences, fitted on
training-period decisions (through 2022-12-31) with the judged winner as the
label, and applied here to each past bout with stats. It describes how a
past fight went on the judges' criteria. Family 16, built on it, was not
adopted (D64), so no published prediction reads it.
"""

from __future__ import annotations

import sqlite3
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from ufc.features.data import BoutFacts
from ufc.features.performance import PerformanceModel, fit_performance_model

N_RECENT = 5
PERFORMANCE_TRAIN_END = date(2022, 12, 31)
"""The D63 fit: the training period's decisions, the transform whose
checks are reported in docs/MODEL_REPORT.md, "Model revision M4.6"."""

NOTE = (
    "A description of past fights, not a model input. Deserved-win "
    "probability: how likely the fighter was to win on the judges' criteria, "
    "estimated from the bout's in-fight statistics by the M4.6 performance "
    "model (fitted on 2022-and-earlier decisions). The prediction model does "
    "not read it. Null where the bout has no recorded statistics."
)


def _clock(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    return f"{seconds // 60}:{seconds % 60:02d}"


def _result(bout: BoutFacts, fighter_id: str) -> str:
    if bout.result == "win":
        return "win" if bout.winner_id == fighter_id else "loss"
    return bout.result  # draw or nc


@dataclass(slots=True)
class History:
    """Completed bouts by fighter, with names for display and the fitted
    performance model (``None`` if there were too few decisions to fit)."""

    bouts: Mapping[str, Sequence[BoutFacts]]
    dates: Mapping[str, Sequence[date]]
    events: Mapping[str, str]
    names: Mapping[str, str]
    model: PerformanceModel | None

    @classmethod
    def build(
        cls,
        bouts: Iterable[BoutFacts],
        events: Mapping[str, str],
        names: Mapping[str, str],
        train_end: date = PERFORMANCE_TRAIN_END,
    ) -> History:
        bouts = sorted(bouts, key=lambda b: (b.date, b.event_id, b.bout_id))
        by_fighter: dict[str, list[BoutFacts]] = defaultdict(list)
        for bout in bouts:
            by_fighter[bout.fighter_a_id].append(bout)
            by_fighter[bout.fighter_b_id].append(bout)
        return cls(
            by_fighter,
            {f: [b.date for b in bs] for f, bs in by_fighter.items()},
            events,
            names,
            fit_performance_model(bouts, train_end),
        )

    def performance_model_info(self) -> dict[str, str | int | None]:
        if self.model is None:
            return {"train_end": None, "n_decisions": None, "decisions": None}
        return {
            "train_end": self.model.train_end.isoformat(),
            "n_decisions": self.model.no_control.n_bouts,
            "decisions": "decisions with a winner, judged winner as label (D63)",
        }

    def recent(
        self, fighter_id: str | None, before: date, n: int = N_RECENT
    ) -> list[dict]:
        """The fighter's last ``n`` bouts on dates strictly before ``before``,
        newest first. Empty for a fighter without a dataset ID."""
        if fighter_id is None:
            return []
        bouts = self.bouts.get(fighter_id, ())
        end = bisect_left(self.dates.get(fighter_id, ()), before)
        out = []
        for bout in reversed(bouts[max(0, end - n) : end]):
            opponent = bout.opponent_of(fighter_id)
            deserved = (
                self.model.deserved(bout, fighter_id)
                if self.model is not None
                else None
            )
            out.append(
                {
                    "date": bout.date.isoformat(),
                    "event": self.events.get(bout.event_id, bout.event_id),
                    "opponent": self.names.get(opponent, opponent),
                    "opponent_id": opponent,
                    "result": _result(bout, fighter_id),
                    "method": bout.method,
                    "end_round": bout.end_round,
                    "end_time": _clock(bout.end_time_s),
                    "weight_class": bout.weight_class,
                    "title_bout": bout.is_title_bout,
                    "deserved_win_probability": deserved,
                }
            )
        return out


def read_names(conn: sqlite3.Connection) -> tuple[dict[str, str], dict[str, str]]:
    """Event names and fighter names by ID."""
    events = {r[0]: r[1] for r in conn.execute("SELECT event_id, name FROM events")}
    fighters = {
        r[0]: r[1] for r in conn.execute("SELECT fighter_id, name FROM fighters")
    }
    return events, fighters
