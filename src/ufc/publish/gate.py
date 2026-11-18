"""The validation gate and the writer (docs/ARCHITECTURE.md, "Validation
gate"; DECISIONS D66).

``publish`` builds every artifact in memory, checks the card (the M5 checks
in ``ufc.predict.publish.validate``) and validates each artifact against its
pydantic model. Only if everything passes does it write, in this order:

1. archive snapshots (``archive/predictions/{event}/{timestamp}.json``),
   created exclusively and made read-only, so an existing one can never be
   replaced; an event whose card is unchanged since its last snapshot gets
   no new one;
2. the bout artifacts (``public/bouts/{bout_id}.json``);
3. ``public/performance.json``;
4. ``public/upcoming.json``, last, so that every bout it links to exists;
5. then bout artifacts of bouts no longer on a card are removed.

Each public file is replaced atomically (written beside it, then renamed).
If any check fails nothing is written at all, and the last good artifacts
stay exactly as they were. The archive comes first because it is the
evidence for what the site shows: a set is never displayed without it.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ValidationError

from ufc.features.data import load_feature_data
from ufc.models import registry
from ufc.predict import publish as card
from ufc.publish import artifacts, live, performance
from ufc.publish.history import History, read_names
from ufc.publish.schemas import (
    ArchiveSnapshot,
    BoutArtifact,
    Performance,
    Upcoming,
)

logger = logging.getLogger(__name__)

PUBLIC = "public"
BOUTS = "bouts"
ARCHIVE = Path("archive") / "predictions"
UPCOMING_FILE = "upcoming.json"
PERFORMANCE_FILE = "performance.json"


class ArchiveError(RuntimeError):
    """An archive snapshot already exists under the name a new one needs."""


@dataclass(slots=True)
class Built:
    upcoming: Upcoming
    bouts: dict[str, BoutArtifact]
    performance: Performance
    archive: list[ArchiveSnapshot]


@dataclass(slots=True)
class PublishResult:
    ok: bool = False
    problems: list[str] = field(default_factory=list)
    written: list[Path] = field(default_factory=list)
    archived: list[Path] = field(default_factory=list)
    removed: list[Path] = field(default_factory=list)
    live_counts: dict[str, int] = field(default_factory=dict)


def _validated(model: type[BaseModel], raw: Mapping, label: str, problems: list[str]):
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        for error in exc.errors()[:5]:
            where = ".".join(str(p) for p in error["loc"]) or "(root)"
            problems.append(f"{label}: {where}: {error['msg']}")
        return None


def _model_details(
    models_root: Path, versions: set[str]
) -> dict[str, artifacts.ModelDetails]:
    out = {}
    for version in versions:
        pipeline = registry.read_json(models_root, version, "feature_pipeline.json")
        out[version] = artifacts.ModelDetails(
            float(pipeline["config"].get("shrink_minutes", 0.0)),
            {name.removeprefix("d_") for name in pipeline.get("shrunk_features", ())},
        )
    return out


def build(
    conn: sqlite3.Connection,
    *,
    today: date,
    now: str,
    model: dict | None,
    model_inputs: Sequence[str] | None,
    staleness: dict,
    dataset: dict,
    models_root: Path,
    reports_root: Path,
    history: History | None = None,
) -> tuple[Built | None, list[str], list[live.LiveOutcome]]:
    """Every artifact, validated; or ``None`` and the reasons it may not be
    published."""
    record = card.snapshot(
        conn,
        today=today,
        generated_at=now,
        model=model,
        staleness=staleness,
        dataset=dataset,
    )
    problems = card.validate(record, today, model_inputs)
    bouts_in, predictions, results, links = live.read_live_inputs(conn)
    last = dataset.get("last_event")
    outcomes = live.score_live(
        bouts_in,
        predictions,
        results,
        links,
        date.fromisoformat(last) if last else None,
        today,
    )
    if problems:
        return None, problems, outcomes

    predicted = [b for e in record["events"] for b in e["bouts"]]
    stored = artifacts.read_stored(conn, [b["prediction_id"] for b in predicted])
    details = _model_details(models_root, {b["model_version"] for b in predicted})
    if history is None:
        events, names = read_names(conn)
        history = History.build(load_feature_data(conn).bouts, events, names)

    raw_bouts = {}
    for event in record["events"]:
        for bout in event["bouts"]:
            raw_bouts[bout["bout_id"]] = artifacts.bout_artifact(
                event,
                bout,
                stored[bout["prediction_id"]],
                history,
                links,
                details[bout["model_version"]],
                now,
            )
    raw_upcoming = artifacts.upcoming_artifact(record)
    raw_performance = performance.performance(
        generated_at=now,
        model=model,  # type: ignore[arg-type]  # validate() refuses None
        reports_root=reports_root,
        live=live.live_record(outcomes),
    )
    raw_archive = artifacts.archive_snapshots(raw_upcoming, raw_bouts, stored)

    upcoming = _validated(Upcoming, raw_upcoming, UPCOMING_FILE, problems)
    bouts = {
        key: _validated(BoutArtifact, raw, f"{BOUTS}/{key}.json", problems)
        for key, raw in raw_bouts.items()
    }
    perf = _validated(Performance, raw_performance, PERFORMANCE_FILE, problems)
    archive = [
        _validated(
            ArchiveSnapshot, raw, f"archive {raw['event']['event_key']}", problems
        )
        for raw in raw_archive
    ]
    if upcoming is not None:
        listed = {b.bout_id for e in upcoming.events for b in e.bouts}
        if listed != set(bouts):
            problems.append("upcoming.json and the bout artifacts list different bouts")
        for e in upcoming.events:
            for b in e.bouts:
                detail = bouts.get(b.bout_id)
                if detail is not None and (
                    detail.prediction.p_fighter_a != b.p_fighter_a
                    or detail.prediction.model_version != b.model_version
                ):
                    problems.append(
                        f"{b.bout_id}: the bout artifact disagrees with the card"
                    )
    if problems:
        return None, problems, outcomes
    return Built(upcoming, bouts, perf, archive), [], outcomes  # type: ignore[arg-type]


# --- writing -------------------------------------------------------------------------


def _dump(model: BaseModel) -> str:
    return (
        json.dumps(
            model.model_dump(mode="json", by_alias=True), indent=2, ensure_ascii=False
        )
        + "\n"
    )


def _replace(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def archive_name(published_at: str) -> str:
    return published_at.replace(":", "-") + ".json"


def event_directory(event_key: str) -> str:
    return event_key.replace(":", "-")


def latest_snapshot(directory: Path) -> dict | None:
    names = sorted(p for p in directory.glob("*.json")) if directory.is_dir() else []
    return json.loads(names[-1].read_text(encoding="utf-8")) if names else None


def write_snapshot(root: Path, snapshot: ArchiveSnapshot) -> Path | None:
    """Write one archive snapshot, or nothing if the event's latest snapshot
    has the same content. Never replaces a file: the name is claimed with an
    exclusive hard link, and the file is made read-only."""
    directory = root / ARCHIVE / event_directory(snapshot.event.event_key)
    latest = latest_snapshot(directory)
    if latest is not None and latest.get("content_sha256") == snapshot.content_sha256:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / archive_name(snapshot.published_at)
    temporary = directory / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(_dump(snapshot), encoding="utf-8")
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise ArchiveError(
            f"{path} exists; archive snapshots are never replaced"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    path.chmod(0o444)
    return path


def check_archive(root: Path, snapshots: Sequence[ArchiveSnapshot]) -> list[str]:
    """Problems that would stop a snapshot from being written."""
    problems = []
    for snapshot in snapshots:
        directory = root / ARCHIVE / event_directory(snapshot.event.event_key)
        latest = latest_snapshot(directory)
        if (
            latest is not None
            and latest.get("content_sha256") == snapshot.content_sha256
        ):
            continue
        path = directory / archive_name(snapshot.published_at)
        if path.exists():
            problems.append(f"{path} exists; archive snapshots are never replaced")
        elif (
            latest is not None
            and latest.get("published_at", "") > snapshot.published_at
        ):
            problems.append(
                f"{directory} has a snapshot later than {snapshot.published_at}"
            )
    return problems


def write(built: Built, root: Path) -> PublishResult:
    result = PublishResult()
    for snapshot in built.archive:
        path = write_snapshot(root, snapshot)
        if path is not None:
            result.archived.append(path)
    public = root / PUBLIC
    for bout_id, artifact in sorted(built.bouts.items()):
        path = public / BOUTS / f"{bout_id}.json"
        _replace(path, _dump(artifact))
        result.written.append(path)
    for name, artifact in (
        (PERFORMANCE_FILE, built.performance),
        (UPCOMING_FILE, built.upcoming),
    ):
        path = public / name
        _replace(path, _dump(artifact))
        result.written.append(path)
    for stale in sorted((public / BOUTS).glob("*.json")):
        if stale.stem not in built.bouts:
            stale.unlink()
            result.removed.append(stale)
    result.ok = True
    return result


def publish(
    conn: sqlite3.Connection,
    root: Path,
    **kwargs,
) -> PublishResult:
    """Build, validate and, only if everything passes, write. ``kwargs`` are
    those of ``build``."""
    built, problems, outcomes = build(conn, **kwargs)
    counts = live.live_record(outcomes)["counts"] if outcomes else {}
    if built is not None:
        problems = check_archive(root, built.archive)
    if built is None or problems:
        for problem in problems:
            logger.error("not published: %s", problem)
        return PublishResult(problems=problems, live_counts=counts)
    result = write(built, root)
    result.live_counts = counts
    return result
