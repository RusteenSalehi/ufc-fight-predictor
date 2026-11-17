"""The refresh end to end on an invented league and invented Wikipedia pages.

The league's winners are random, so nothing here checks that predictions are
good; it checks what is stored, labelled, superseded and published, and that
a failed or broken read changes nothing.
"""

import json
from datetime import date

import numpy as np
import pytest

from tests.feature_helpers import random_league
from tests.model_helpers import FAST
from tests.wiki_helpers import query_response
from ufc.db.connection import connect
from ufc.features.build import build_features, build_meta
from ufc.features.config import FeatureConfig
from ufc.features.data import FighterAttrs
from ufc.features.matchup import FEATURE_NAMES, LEVEL_NAMES, build_matchup
from ufc.ingest.identity import FighterResolver, Profile
from ufc.models import registry
from ufc.models.calibrate import PLATT
from ufc.models.design import FeatureSpec
from ufc.models.recipe import Recipe, fit_recipe
from ufc.models.splits import fit_rows, labelled
from ufc.predict import publish
from ufc.predict.backfill import backfill, merged_attributes
from ufc.predict.predict import feature_store, model_inputs
from ufc.predict.refresh import refresh
from ufc.predict.wikipedia import Wikipedia, WikipediaError

TODAY = date(2026, 11, 17)
NOW = "2026-11-17T12:00:00Z"
LATER = "2026-11-18T12:00:00Z"
CONFIG = FeatureConfig(train_end=date(2020, 12, 31), shrink_minutes=15.0)


@pytest.fixture(scope="module")
def league():
    return random_league(seed=11, n_fighters=40, n_events=110, start=date(2017, 1, 7))


@pytest.fixture(scope="module")
def model_root(league, tmp_path_factory):
    """A small model, saved and approved in a temporary registry."""
    root = tmp_path_factory.mktemp("models")
    data = league.data()
    build = build_features(data, CONFIG)
    frame = build.frame
    spec = FeatureSpec(FEATURE_NAMES, LEVEL_NAMES)
    recipe = Recipe(FAST["xgboost"], 8.0, spec, PLATT)
    train = fit_rows(frame[frame["date"] <= np.datetime64("2020-12-31")])
    valid = labelled(frame[frame["date"] > np.datetime64("2020-12-31")])
    candidate, _ = fit_recipe(recipe, train, valid, date(2020, 12, 31))
    saved = registry.save_candidate(
        root,
        candidate,
        config={
            "recipe": {
                k: recipe.to_dict()[k]
                for k in ("gbm", "half_life_years", "calibration")
            }
        },
        feature_pipeline=build_meta(build, None),
        metrics={},
        dataset={"training_sha256": registry.dataset_hash(train, spec)},
        git={},
        created_at=NOW,
    )
    (root / registry.APPROVED_FILE).write_text(json.dumps({"version": saved.version}))
    return root


def fighter_name(league, index):
    return league.dataset.fighters[index].name


@pytest.fixture(scope="module")
def names(league):
    fighters = league.dataset.fighters
    # Indices from 20 up: the cards below use fighters 1 to 14.
    no_height = next(
        i for i, f in enumerate(fighters) if i >= 20 and f.height_cm is None
    )
    has_height = next(
        i for i, f in enumerate(fighters) if i >= 20 and f.height_cm is not None
    )
    last = max(e.date for e in league.dataset.events)
    return {
        "no_height": fighters[no_height].name,
        "has_height": fighters[has_height].name,
        "has_height_cm": fighters[has_height].height_cm,
        "last": last,
        "f": [f.name for f in fighters],
    }


def resolver_for(league):
    return FighterResolver(
        [Profile(f.fighter_id, (f.name,), 155) for f in league.dataset.fighters]
    )


def listing(scheduled, past):
    def rows(events):
        return "".join(
            f"|-\n|[[{page}]]\n|{{{{dts|{d.year}|{d.month}|{d.day}}}}}\n|Arena\n"
            for page, d in events
        )

    return (
        '==Scheduled events==\n{| class="wikitable"\n! Event\n'
        + rows(scheduled)
        + '|}\n\n==Past events==\n{| class="wikitable"\n! Event\n'
        + rows(past)
        + "|}\n"
    )


def bout_template(a, b, weight="Lightweight", connector="vs."):
    return f"{{{{MMAevent bout\n|{weight}\n|{a}\n|{connector}\n|{b}\n|\n|\n|\n|\n}}}}\n"


def card(*bouts):
    return (
        "==Fight card==\n{{MMAevent}}\n{{MMAevent card|Main card}}\n"
        + "".join(bouts)
        + "{{MMAevent end}}\n"
    )


INFOBOX = (
    "{{Infobox martial artist\n| name = X\n| height = 6 ft 1 in\n| reach = 76 in\n"
    "| birth_date = {{Birth date and age|1996|4|2}}\n}}\nText."
)


class Pages:
    """A transport over invented pages; any title set is answered."""

    def __init__(self, pages):
        self.pages = dict(pages)
        self.fail = False
        self.requests = 0

    def __call__(self, params):
        self.requests += 1
        if self.fail:
            raise WikipediaError("network down")
        titles = params["titles"].split("|")
        body = query_response({t: self.pages.get(t) for t in titles})
        for page in body["query"]["pages"]:
            if "pageid" in page:
                page["pageid"] = 5000 + sorted(self.pages).index(page["title"])
        return body


@pytest.fixture
def world(league, names, model_root, tmp_path):
    f = names["f"]
    past_day = date(2026, 11, 14)
    pages = {
        "List of UFC events": listing(
            [("UFC Test 2", date(2026, 12, 5)), ("UFC Test 1", date(2026, 11, 21))],
            [("UFC Past", past_day)],
        ),
        "UFC Test 1": card(
            bout_template(f"[[{f[1]}]]", f[2]),
            bout_template(f"[[{names['no_height']}]]", f[4]),
            bout_template(f[5], "[[Newcomer Person]]"),
            bout_template(f[6], "TBD"),
            bout_template(f"[[{names['has_height']}]]", f[8]),
        ),
        "UFC Test 2": card(bout_template(f[9], f[10]), bout_template(f[11], f[12])),
        "UFC Past": card(bout_template(f[2], f[13], connector="def.")),
        names["no_height"]: INFOBOX,
        "Newcomer Person": INFOBOX,
        names["has_height"]: INFOBOX,
        f[1]: "An article without an infobox.",
    }
    db = tmp_path / "ufc.db"
    league.write_db(db)
    return {
        "pages": Pages(pages),
        "db": db,
        "models": model_root,
        "publish": tmp_path / "publish",
        "resolver": resolver_for(league),
        "names": names,
    }


def run(world, now=NOW, models=None):
    conn = connect(world["db"])
    try:
        return refresh(
            conn,
            Wikipedia(world["pages"]),
            world["resolver"],
            models_root=models or world["models"],
            publish_dir=world["publish"],
            today=TODAY,
            now=now,
        ), conn.execute(
            "SELECT * FROM upcoming_bouts ORDER BY event_key, card_order"
        ).fetchall()
    finally:
        conn.close()


def query(world, sql, *args):
    conn = connect(world["db"])
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def published(world):
    return json.loads((world["publish"] / publish.FILE_NAME).read_text())


def test_first_refresh_stores_predicts_labels_and_publishes(world):
    result, bouts = run(world)
    assert result.stopped is None and not result.event_errors
    assert result.published and not result.publish_problems and result.ok
    assert len(bouts) == 7
    f = world["names"]["f"]
    by_pair = {(b["name_a"], b["name_b"]): b for b in bouts}
    newcomer = by_pair[(f[5], "Newcomer Person")]
    assert (
        newcomer["fighter_b_ref"] == "wp:Newcomer Person"
        and newcomer["method_b"] == "provisional"
    )
    tbd = by_pair[(f[6], "TBD")]
    assert tbd["fighter_a_ref"] is None and "placeholder" in tbd["unresolved_reason"]

    predictions = query(world, "SELECT * FROM predictions WHERE superseded_at IS NULL")
    assert len(predictions) == 6 == len(result.predictions.made)
    for p in predictions:
        assert p["p_fighter_a"] + p["p_fighter_b"] == 1.0
        assert p["is_live"] == 1 and p["model_version"] == result.model_version
    reasons = {p["bout_id"]: json.loads(p["limited_reasons_json"]) for p in predictions}
    assert any(
        r.startswith("debut: Newcomer Person") for r in reasons[newcomer["bout_id"]]
    )
    stale = reasons[by_pair[(f[1], f[2])]["bout_id"]]
    assert any(r.startswith(f"stale stats: {f[2]} fought at UFC Past") for r in stale)

    record = published(world)
    assert [e["name"] for e in record["events"]] == ["UFC Test 1", "UFC Test 2"]
    first = record["events"][0]
    assert len(first["bouts"]) == 4 and len(first["unresolved_bouts"]) == 1
    assert record["staleness"]["status"] == "behind"
    assert record["staleness"]["missing_events"][0]["page"] == "UFC Past"
    assert record["model"]["version"] == result.model_version
    assert "Wikipedia" in record["attribution"]


def test_backfill_fills_gaps_only_and_never_overrides_upstream(world):
    run(world)
    names = world["names"]
    rows = query(
        world, "SELECT fighter_ref, field, value, page_title FROM attribute_backfill"
    )
    by_ref = {}
    for r in rows:
        by_ref.setdefault(r["fighter_ref"], {})[r["field"]] = r["value"]
    assert set(by_ref["wp:Newcomer Person"]) == {"height_cm", "reach_cm", "dob"}
    gap_id = query(
        world, "SELECT fighter_id FROM fighters WHERE name = ?", names["no_height"]
    )[0][0]
    assert "height_cm" in by_ref[gap_id]
    full_id = query(
        world, "SELECT fighter_id FROM fighters WHERE name = ?", names["has_height"]
    )[0][0]
    assert "height_cm" not in by_ref.get(
        full_id, {}
    )  # upstream has it: not even stored
    upstream_height = query(
        world, "SELECT height_cm FROM fighters WHERE fighter_id = ?", full_id
    )[0][0]
    assert upstream_height == names["has_height_cm"]  # fighters table untouched


def test_merged_attributes_prefer_upstream_even_over_an_older_backfill():
    conn = connect(":memory:")
    conn.execute(
        "INSERT INTO attribute_backfill VALUES ('x', 'height_cm', '190.0', 'wikipedia', 'X', 1, ?)",
        (NOW,),
    )
    conn.execute(
        "INSERT INTO attribute_backfill VALUES ('x', 'dob', '1990-01-02', 'wikipedia', 'X', 1, ?)",
        (NOW,),
    )
    upstream = {"x": FighterAttrs("x", height_cm=180.0, reach_cm=None, dob=None)}
    merged, sources = merged_attributes(conn, upstream)
    assert merged["x"].height_cm == 180.0 and merged["x"].dob == date(1990, 1, 2)
    assert sources == {"x": {"dob": "wikipedia"}}
    assert upstream["x"].dob is None  # the input is not modified


def test_backfill_looks_up_nothing_when_upstream_is_complete():
    conn = connect(":memory:")
    calls = []
    client = Wikipedia(lambda p: calls.append(p) or {})
    full = {"x": FighterAttrs("x", 180.0, 185.0, "Orthodox", date(1990, 1, 1))}
    result = backfill(conn, client, {"x": "X"}, full, NOW)
    assert result.pages_requested == 0 and not calls


def test_refreshing_again_with_nothing_changed_keeps_every_prediction(world):
    run(world)
    before = query(world, "SELECT prediction_id, inputs_sha256 FROM predictions")
    result, _ = run(world, LATER)
    assert result.predictions.made == [] and len(result.predictions.unchanged) == 6
    assert (
        query(world, "SELECT prediction_id, inputs_sha256 FROM predictions") == before
    )
    seen = query(world, "SELECT DISTINCT last_seen_at FROM upcoming_bouts")
    assert [r[0] for r in seen] == [LATER]


def test_cancellation_and_replacement_supersede_and_predict(world):
    run(world)
    f = world["names"]["f"]
    world["pages"].pages["UFC Test 2"] = card(
        bout_template(f[9], f[14])  # f[10] replaced; f[11] vs f[12] cancelled
    )
    result, bouts = run(world, LATER)
    change = next(c for c in result.changes if len(c.cancelled))
    assert len(change.cancelled) == 2 and len(change.added) == 1
    status = {(b["name_a"], b["name_b"]): b["status"] for b in bouts}
    assert (
        status[(f[9], f[10])] == "cancelled" and status[(f[11], f[12])] == "cancelled"
    )
    assert status[(f[9], f[14])] == "scheduled"
    superseded = query(
        world,
        "SELECT superseded_reason FROM predictions WHERE superseded_at = ?",
        LATER,
    )
    assert [r[0] for r in superseded] == ["bout cancelled", "bout cancelled"]
    event = next(e for e in published(world)["events"] if e["name"] == "UFC Test 2")
    assert [
        (b["fighter_a"]["name"], b["fighter_b"]["name"]) for b in event["bouts"]
    ] == [(f[9], f[14])]


def snapshot_state(world):
    return (
        query(world, "SELECT * FROM upcoming_bouts ORDER BY bout_id"),
        query(world, "SELECT * FROM predictions ORDER BY prediction_id"),
        query(world, "SELECT * FROM upcoming_events ORDER BY event_key"),
    )


def test_a_page_that_fails_to_parse_cancels_nothing(world):
    run(world)
    f = world["names"]["f"]
    bouts_before = query(world, "SELECT * FROM upcoming_bouts ORDER BY bout_id")
    preds_before = query(world, "SELECT * FROM predictions ORDER BY prediction_id")
    world["pages"].pages["UFC Test 2"] = card(
        bout_template(f[9], f[10], connector="beats")
    )
    result, _ = run(world, LATER)
    assert "UFC Test 2" in result.event_errors and not result.ok
    after = {b["bout_id"]: b for b in query(world, "SELECT * FROM upcoming_bouts")}
    for b in bouts_before:
        assert after[b["bout_id"]]["status"] == b["status"]
        assert after[b["bout_id"]]["cancelled_at"] is None
    assert (
        query(world, "SELECT * FROM predictions ORDER BY prediction_id") == preds_before
    )
    errors = query(world, "SELECT message FROM refresh_log WHERE level = 'error'")
    assert any("card not read" in r[0] for r in errors)


@pytest.mark.parametrize("broken", ["empty", "most_gone"])
def test_an_implausible_card_cancels_nothing(world, broken):
    run(world)
    f = world["names"]["f"]
    before = snapshot_state(world)
    world["pages"].pages["UFC Test 1"] = (
        "No card yet."
        if broken == "empty"
        else card(bout_template(f"[[{f[1]}]]", f[2]))
    )
    result, _ = run(world, LATER)
    assert "UFC Test 1" in result.event_errors
    after = snapshot_state(world)
    (key,) = query(
        world, "SELECT event_key FROM upcoming_events WHERE page_title = 'UFC Test 1'"
    )

    def of_event(rows):
        return [tuple(r) for r in rows if r["event_key"] == key[0]]

    assert of_event(after[0]) == of_event(before[0])  # not even last_seen_at moves
    assert [tuple(r) for r in after[1]] == [tuple(r) for r in before[1]]


def test_a_failed_schedule_read_changes_nothing_and_publishes_nothing(world):
    run(world)
    path = world["publish"] / publish.FILE_NAME
    good = path.read_text()
    before = snapshot_state(world)
    world["pages"].fail = True
    result, _ = run(world, LATER)
    assert result.stopped and not result.ok
    assert [[tuple(r) for r in t] for t in snapshot_state(world)] == [
        [tuple(r) for r in t] for t in before
    ]
    assert path.read_text() == good


def test_without_an_approved_model_nothing_is_predicted_or_published(world, tmp_path):
    empty = tmp_path / "no-models"
    empty.mkdir()
    result, bouts = run(world, models=empty)
    assert result.model_error and "approved.json" in result.model_error
    assert len(bouts) == 7
    assert not query(world, "SELECT * FROM predictions")
    assert (
        result.published is None and not (world["publish"] / publish.FILE_NAME).exists()
    )


def test_predictions_are_symmetric(world, league):
    run(world)
    candidate, manifest = registry.load_approved(world["models"])
    pipeline = registry.read_json(
        world["models"], manifest["version"], "feature_pipeline.json"
    )
    data = league.data()
    store = feature_store(data, pipeline, data.fighters)
    names = candidate.spec.names
    for row in query(
        world,
        """SELECT p.*, b.weight_class FROM predictions p JOIN upcoming_bouts b USING (bout_id)
        WHERE p.superseded_at IS NULL""",
    ):
        on = date.fromisoformat(row["event_date"])
        a, b = row["fighter_a_ref"], row["fighter_b_ref"]
        if a.startswith("wp") or b.startswith("wp"):
            continue  # provisional attributes are merged in the refresh's store
        ab = candidate.predict_pair(
            model_inputs(build_matchup(store, a, b, on, row["weight_class"]), names)
        )
        ba = candidate.predict_pair(
            model_inputs(build_matchup(store, b, a, on, row["weight_class"]), names)
        )
        assert ab[0][0] == ba[1][0] and ab[1][0] == ba[0][0]
        assert ab[0][0] + ab[1][0] == 1.0
