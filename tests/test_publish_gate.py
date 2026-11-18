# ruff: noqa: F811  (the refresh world's fixtures are imported by name)
"""The published artifacts end to end, on the invented league and Wikipedia
pages of ``test_predict_refresh``: every artifact validates, the gate never
replaces good artifacts with bad ones, the archive is never overwritten,
backtest numbers are copied from the reports, and live predictions are
scored once their results reach the dataset.
"""

import json
import os
import stat
from datetime import date
from pathlib import Path

import pytest

from tests.test_predict_refresh import (  # noqa: F401
    LATER,
    NOW,
    bout_template,
    card,
    league,
    model_root,
    names,
    query,
    resolver_for,
    run,
    world,
)
from ufc.db.connection import connect
from ufc.ingest.identity import FighterResolver, Profile
from ufc.models import registry
from ufc.predict.explain import COMPONENTS, parse
from ufc.publish import artifacts, gate, performance
from ufc.publish.artifacts import content_sha256
from ufc.publish.schemas import (
    ArchiveSnapshot,
    BoutArtifact,
    Performance,
    Upcoming,
    fair_moneyline,
)

PUBLIC = Path("public")


def files(world) -> dict[Path, tuple[bytes, int]]:
    """Every file under the publish directory: bytes and permission bits."""
    root = world["publish"]
    if not root.exists():
        return {}
    return {
        p.relative_to(root): (p.read_bytes(), stat.S_IMODE(p.stat().st_mode))
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def read(world, *parts):
    return json.loads((world["publish"].joinpath(*parts)).read_text())


def archive_files(world) -> list[Path]:
    root = world["publish"] / gate.ARCHIVE
    return sorted(root.rglob("*.json")) if root.exists() else []


# --- every artifact, valid ---------------------------------------------------------


def test_a_refresh_publishes_every_artifact_and_each_validates(world):
    result, _ = run(world)
    assert result.ok and result.published == world["publish"] / PUBLIC / "upcoming.json"
    upcoming = Upcoming.model_validate(read(world, "public", "upcoming.json"))
    assert upcoming.schema_version == 1 and upcoming.staleness.status == "behind"
    predicted = [b for e in upcoming.events for b in e.bouts]
    assert len(predicted) == 6
    pages = sorted((world["publish"] / PUBLIC / "bouts").glob("*.json"))
    assert {p.stem for p in pages} == {b.bout_id for b in predicted}
    stored = {
        r["bout_id"]: r
        for r in query(world, "SELECT * FROM predictions WHERE superseded_at IS NULL")
    }
    for bout in predicted:
        page = BoutArtifact.model_validate(read(world, "public", bout.detail))
        row = stored[bout.bout_id]
        assert page.prediction.p_fighter_a == row["p_fighter_a"] == bout.p_fighter_a
        assert page.prediction.prediction_id == row["prediction_id"]
        assert page.prediction.fair_moneyline.fighter_a == fair_moneyline(
            row["p_fighter_a"]
        )
        assert page.market.fighter_a_american is None  # filled in M12
        assert page.limited_data.reasons == json.loads(row["limited_reasons_json"])
        assert len(page.recent_bouts.fighter_a) <= 5
        assert "not a model input" in page.recent_bouts.note
        # The why's grouped reasons show a component that points their way.
        toward_a = json.loads(row["shap_json"])["log_odds_toward_a"]
        reasons = [*page.why.for_, *([page.why.against] if page.why.against else [])]
        for r in reasons:
            if r.component is None:
                continue
            s = 1.0 if r.toward == "fighter_a" else -1.0
            stems = COMPONENTS[r.reason][r.component]
            assert sum(s * toward_a[n] for n in r.inputs if parse(n).stem in stems) > 0
    newcomer = next(b for b in predicted if b.fighter_b.name == "Newcomer Person")
    page = read(world, "public", newcomer.detail)
    assert page["recent_bouts"]["fighter_b"] == []  # a debutant has none
    assert page["limited_data"]["limited"]

    perf = Performance.model_validate(read(world, "public", "performance.json"))
    assert perf.backtest is None and "no evaluation record" in perf.backtest_unavailable
    assert perf.live.metrics is None and perf.live.bouts == []

    snapshots = archive_files(world)
    assert len(snapshots) == 2  # one per event with predicted bouts
    for path in snapshots:
        snapshot = ArchiveSnapshot.model_validate_json(path.read_text())
        assert not os.access(path, os.W_OK)
        assert snapshot.published_at == NOW
        assert path.name == "2026-11-17T12-00-00Z.json"


def test_recent_bouts_carry_results_and_the_deserved_win_probability(world, league):
    run(world)
    upcoming = read(world, "public", "upcoming.json")
    bout = upcoming["events"][0]["bouts"][0]
    page = read(world, "public", bout["detail"])
    recent = page["recent_bouts"]["fighter_a"]
    assert recent, "a veteran of the invented league has past bouts"
    dates = [b["date"] for b in recent]
    assert dates == sorted(dates, reverse=True)
    assert all(d < page["event"]["date"] for d in dates)
    with_value = [b for b in recent if b["deserved_win_probability"] is not None]
    assert with_value and all(
        0 <= b["deserved_win_probability"] <= 1 for b in with_value
    )
    assert {b["result"] for b in recent} <= {"win", "loss", "draw", "nc"}


# --- backtest numbers come from the reports ----------------------------------------


def _metrics(n, loss):
    return {"n": n, "log_loss": loss, "brier": 0.2, "accuracy": 0.6, "ece": 0.03}


def write_reports(directory: Path, version: str) -> None:
    directory.mkdir(parents=True)
    rows = [
        {
            "low": lo,
            "high": hi,
            "n": n,
            "mean_confidence": (lo + hi) / 2 if n else None,
            "favourite_win_rate": 0.6 if n else None,
            "win_rate_low": 0.5 if n else None,
            "win_rate_high": 0.7 if n else None,
            "log_loss": 0.6 if n else None,
        }
        for lo, hi, n in ((0.5, 0.6, 40), (0.6, 0.8, 50), (0.8, 1.0, 10))
    ]
    test = {
        "version": version,
        "first_date": "2025-01-11",
        "last_date": "2026-11-14",
        "n_scored": 100,
        "n_no_result": 2,
        "already_seen": {"already_seen": False, "earlier_versions": [], "note": None},
        "models": {
            k: _metrics(100, v)
            for k, v in (("model", 0.61), ("elo", 0.68), ("logistic", 0.62))
        },
        "by_confidence": {"model": rows},
    }
    folds = [
        {
            "year": y,
            "in_test_period": y >= 2025,
            "n_scored": 50,
            "metrics": {"model": _metrics(50, loss)},
        }
        for y, loss in ((2023, 0.66), (2024, 0.64), (2025, 0.61), (2026, 0.62))
    ]
    walk = {
        "version": version,
        "schedule": "fit through Y-3; trees and calibrator on Y-2 and Y-1; scored on Y",
        "folds": folds,
        "pooled_before_test_period": {
            "years": [2023, 2024],
            "metrics": {"model": _metrics(100, 0.65)},
        },
        "pooled_test_period": {
            "years": [2025, 2026],
            "metrics": {"model": _metrics(100, 0.615)},
        },
    }
    (directory / "test_metrics.json").write_text(json.dumps(test))
    (directory / "walk_forward.json").write_text(json.dumps(walk))


def test_backtest_numbers_are_copied_from_the_evaluation_record(world):
    version = registry.approved_version(world["models"])
    write_reports(world["reports"] / "other", "xgboost-not-this-one")
    write_reports(world["reports"] / "eval", version)
    run(world)
    perf = read(world, "public", "performance.json")
    backtest = perf["backtest"]
    assert (
        backtest["reports"].endswith("eval")
        and backtest["candidate_version"] == version
    )
    assert backtest["test_period"]["metrics"]["model"]["log_loss"] == 0.61
    assert (
        backtest["walk_forward"]["pooled_before_test_period"]["metrics"]["model"][
            "log_loss"
        ]
        == 0.65
    )
    assert [f["year"] for f in backtest["walk_forward"]["folds"]] == [
        2023,
        2024,
        2025,
        2026,
    ]
    caveat = perf["era_shift_caveat"]
    assert (
        caveat["test_period_log_loss"] == 0.615 and caveat["backtest_log_loss"] == 0.65
    )
    assert caveat["backtest_year_range"] == [0.64, 0.66]
    assert "better than every earlier backtest year" in caveat["text"]
    # Kept apart from the live record.
    assert (
        perf["live"]["label"].startswith("Live")
        and "never pooled" in perf["separation"]
    )


def test_a_refit_is_reported_with_its_candidates_record(tmp_path):
    write_reports(tmp_path / "m45", "cand")
    model = {"version": "refit", "status": "production refit", "parent": "cand"}
    record = performance.performance(
        generated_at=NOW, model=model, reports_root=tmp_path, live={}
    )
    assert record["backtest"]["candidate_version"] == "cand"
    assert record["backtest"]["production_version"] == "refit"
    assert "no held-out test" in record["backtest"]["production_note"]
    assert record["backtest"]["commands"] == [
        "uv run python -m ufc.models evaluate --reports " + str(tmp_path / "m45")
    ]


# --- the gate: bad artifacts never replace good ones -----------------------------


def _tamper(kind, monkeypatch, world):
    if kind == "moneyline":
        monkeypatch.setattr(artifacts, "fair_moneyline", lambda p: 100.0)
    elif kind == "unknown field":
        original = artifacts.upcoming_artifact
        monkeypatch.setattr(
            artifacts, "upcoming_artifact", lambda r: {**original(r), "surprise": 1}
        )
    elif kind == "counts disagree":
        original = performance.performance

        def broken(**kwargs):
            out = original(**kwargs)
            out["live"] = {
                **out["live"],
                "counts": {**out["live"]["counts"], "scored": 1},
            }
            return out

        monkeypatch.setattr(performance, "performance", broken)
    elif kind == "probabilities":
        conn = connect(world["db"])
        with conn:
            conn.execute(
                "UPDATE predictions SET p_fighter_b = 0.99 WHERE prediction_id = "
                "(SELECT min(prediction_id) FROM predictions WHERE superseded_at IS NULL)"
            )
        conn.close()
    else:
        raise ValueError(kind)


@pytest.mark.parametrize(
    "kind", ["moneyline", "unknown field", "counts disagree", "probabilities"]
)
def test_a_failed_check_writes_nothing_and_keeps_the_last_good_artifacts(
    world, monkeypatch, kind
):
    run(world)
    good = files(world)
    assert good
    f = world["names"]["f"]
    # A real change, so that the next publish would differ.
    world["pages"].pages["UFC Test 2"] = card(
        bout_template(f[9], f[10]), bout_template(f[11], f[14])
    )
    _tamper(kind, monkeypatch, world)
    result, _ = run(world, LATER)
    assert result.publish_problems and result.published is None and not result.ok
    assert files(world) == good  # every file, byte for byte; nothing added
    log = query(world, "SELECT message FROM refresh_log WHERE run_at = ?", LATER)
    assert any(r[0].startswith("not published") for r in log)


def test_nothing_is_published_or_archived_when_the_first_run_fails(world, monkeypatch):
    monkeypatch.setattr(artifacts, "fair_moneyline", lambda p: 100.0)
    result, _ = run(world)
    assert result.publish_problems
    assert files(world) == {}


# --- the archive -------------------------------------------------------------------


def test_an_unchanged_card_gets_no_new_snapshot_and_a_change_adds_one(world):
    run(world)
    first = {p: p.read_bytes() for p in archive_files(world)}
    result, _ = run(world, LATER)
    assert result.ok and result.archived == []
    assert {p: p.read_bytes() for p in archive_files(world)} == first

    f = world["names"]["f"]
    world["pages"].pages["UFC Test 2"] = card(
        bout_template(f[9], f[14]), bout_template(f[11], f[12])
    )
    result, _ = run(world, "2026-11-19T12:00:00Z")
    assert result.ok and len(result.archived) == 1
    (new,) = result.archived
    assert new.name == "2026-11-19T12-00-00Z.json"
    # Earlier snapshots are untouched and still read-only.
    for path, content in first.items():
        assert path.read_bytes() == content and not os.access(path, os.W_OK)
    snapshot = json.loads(new.read_text())
    names = {
        (b["fighter_a"]["name"], b["fighter_b"]["name"]) for b in snapshot["bouts"]
    }
    assert (f[9], f[14]) in names and (f[9], f[10]) not in names
    content = {
        k: snapshot[k] for k in ("event", "model_version", "bouts", "unresolved_bouts")
    }
    content["event"] = {
        k: v for k, v in content["event"].items() if k != "wikipedia_revision"
    }
    assert snapshot["content_sha256"] == content_sha256(content)


def test_an_existing_snapshot_is_never_replaced(world, tmp_path):
    run(world)
    path = archive_files(world)[0]
    before = path.read_bytes()
    snapshot = ArchiveSnapshot.model_validate_json(before)
    changed = snapshot.model_copy(update={"content_sha256": "0" * 64})
    with pytest.raises(gate.ArchiveError):
        gate.write_snapshot(world["publish"], changed)
    assert path.read_bytes() == before
    assert gate.check_archive(world["publish"], [changed])


def test_a_new_set_at_an_archived_moment_is_refused_whole(world):
    run(world)
    good = files(world)
    f = world["names"]["f"]
    world["pages"].pages["UFC Test 2"] = card(
        bout_template(f[9], f[14]), bout_template(f[11], f[12])
    )
    result, _ = run(world, NOW)  # same timestamp as the first snapshot
    assert any("never replaced" in p for p in result.publish_problems)
    assert files(world) == good


# --- live scoring end to end -------------------------------------------------------

RESULT_EVENT = "e0e0e0e0e0e0e0e0"
NEWCOMER_ID = "0123456789abcdef"


def add_results(world, outcomes):
    """Put "UFC Test 1" (2026-11-21) into the dataset with ``outcomes``: bout
    ID to ("a" | "b" | "draw" | "nc" | "skip", reversed listing)."""
    conn = connect(world["db"])
    with conn:
        conn.execute(
            "INSERT INTO fighters (fighter_id, name) VALUES (?, 'Newcomer Person')",
            (NEWCOMER_ID,),
        )
        conn.execute(
            "INSERT INTO events VALUES (?, 'UFC Test 1', '2026-11-21', NULL, 'completed', ?, ?)",
            (RESULT_EVENT, NOW, NOW),
        )
        for i, (bout_id, (outcome, flip)) in enumerate(sorted(outcomes.items())):
            if outcome == "skip":
                continue
            row = conn.execute(
                "SELECT fighter_a_ref, fighter_b_ref FROM upcoming_bouts WHERE bout_id = ?",
                (bout_id,),
            ).fetchone()
            a, b = (
                NEWCOMER_ID if r.startswith("wp") else r
                for r in (row["fighter_a_ref"], row["fighter_b_ref"])
            )
            winner = {"a": a, "b": b}.get(outcome)
            first, second = (b, a) if flip else (a, b)
            conn.execute(
                """INSERT INTO bouts (bout_id, event_id, fighter_a_id, fighter_b_id,
                    weight_class, bout_type_raw, is_womens, is_title_bout, is_interim_title,
                    is_tournament_final, status, result, winner_id, method, has_stats,
                    first_seen_at, last_seen_at)
                VALUES (?, ?, ?, ?, 'Lightweight', 'Lightweight Bout', 0, 0, 0, 0,
                        'completed', ?, ?, 'KO/TKO', 0, ?, ?)""",
                (
                    f"{i:016x}",
                    RESULT_EVENT,
                    first,
                    second,
                    "win" if winner else outcome,
                    winner,
                    NOW,
                    NOW,
                ),
            )
    conn.close()


def test_live_predictions_are_scored_when_results_reach_the_dataset(world, league):
    run(world)  # 2026-11-17: predictions for UFC Test 1 (2026-11-21)
    f = world["names"]["f"]
    # 2026-11-18: one bout cancelled before the cut-off.
    world["pages"].pages["UFC Test 1"] = card(
        bout_template(f"[[{f[1]}]]", f[2]),
        bout_template(f"[[{world['names']['no_height']}]]", f[4]),
        bout_template(f[5], "[[Newcomer Person]]"),
        bout_template(f[6], "TBD"),
    )
    run(world, LATER)
    bouts = {
        (r["name_a"], r["name_b"]): r["bout_id"]
        for r in query(
            world,
            """SELECT b.* FROM upcoming_bouts b JOIN upcoming_events e USING (event_key)
            WHERE e.page_title = 'UFC Test 1'""",
        )
    }
    pair_1 = bouts[(f[1], f[2])]
    pair_2 = bouts[(world["names"]["no_height"], f[4])]
    newcomer = bouts[(f[5], "Newcomer Person")]
    cancelled = bouts[(world["names"]["has_height"], f[8])]
    add_results(
        world,
        {pair_1: ("b", True), pair_2: ("draw", False), newcomer: ("a", False)},
    )
    # The newcomer's profile is in the snapshot now.
    world["resolver"] = FighterResolver(
        [
            *[Profile(x.fighter_id, (x.name,), 155) for x in league.dataset.fighters],
            Profile(NEWCOMER_ID, ("Newcomer Person",), 155),
        ]
    )
    result, _ = run(world, "2026-11-25T12:00:00Z", today=date(2026, 11, 25))
    assert result.ok, result.publish_problems
    assert ("wp:Newcomer Person", NEWCOMER_ID) in result.linked
    live = read(world, "public", "performance.json")["live"]
    by_id = {b["bout_id"]: b for b in live["bouts"]}
    assert set(by_id) == {pair_1, pair_2, newcomer, cancelled}
    assert by_id[cancelled]["status"] == "cancelled"
    assert by_id[pair_2]["status"] == "no_result" and by_id[pair_2]["outcome"] == "draw"
    first = query(
        world,
        "SELECT * FROM predictions WHERE bout_id = ? ORDER BY prediction_id",
        pair_1,
    )[0]
    scored = by_id[pair_1]
    assert scored["status"] == "scored" and scored["outcome"] == "fighter_b"
    assert scored["prediction_id"] == first["prediction_id"]
    assert scored["p_fighter_a"] == first["p_fighter_a"]
    assert scored["predicted_at"] == NOW
    assert (
        by_id[newcomer]["status"] == "scored"
        and by_id[newcomer]["outcome"] == "fighter_a"
    )
    assert live["counts"]["scored"] == 2 and live["metrics"]["n"] == 2
    assert live["counts"]["no_result"] == 1 and live["counts"]["cancelled"] == 1
    Performance.model_validate(read(world, "public", "performance.json"))
    # The event has gone from the card; the next one is still published.
    upcoming = read(world, "public", "upcoming.json")
    assert [e["name"] for e in upcoming["events"]] == ["UFC Test 2"]
    assert not (world["publish"] / PUBLIC / "bouts" / f"{pair_1}.json").exists()
