"""Name resolution for card fighters, and how card changes are stored."""

from datetime import date

import pytest

from ufc.db.connection import connect
from ufc.ingest.identity import FighterResolver, Override, Profile
from ufc.predict import cards_db
from ufc.predict.cards import CardBout, CardFighter, ListedEvent
from ufc.predict.resolve import (
    METHOD_AMBIGUOUS,
    METHOD_CONFLICT,
    METHOD_IN_DATASET,
    METHOD_PLACEHOLDER,
    METHOD_PROVISIONAL,
    check_provisional,
    cited_profiles,
    load_wikipedia_overrides,
    resolve_bout,
    resolve_fighter,
)
from ufc.predict.wikipedia import Page

PROFILES = [
    Profile("a" * 16, ("Alan Able",), 155),
    Profile("b" * 16, ("Bruno Same",), 125),  # two fighters share this name
    Profile("c" * 16, ("Bruno Same",), 185),
    Profile("d" * 16, ("Dan Dee", "Daniel Dee"), 170),  # renamed since capture
    Profile("e" * 16, ("Ed Override",), 145),
    Profile("f" * 16, ("Frank Other",), 145),
]
RESOLVER = FighterResolver(
    PROFILES, [Override("Eddie Override", None, "e" * 16, "spelled differently")]
)


def fighter(name, page=None):
    return CardFighter(name=name, page=page)


def bout(a, b, weight="Lightweight", order=0):
    return CardBout(order, "Main card", weight, weight, a, b, "vs.", "", False)


def test_exact_name_with_accents_and_either_recorded_name():
    assert (
        resolve_fighter(RESOLVER, fighter("Alán Able"), "wb", "Lightweight").ref
        == "a" * 16
    )
    assert resolve_fighter(RESOLVER, fighter("Daniel Dee"), "wb", None).ref == "d" * 16


def test_override_applies_to_wikipedia_names():
    r = resolve_fighter(RESOLVER, fighter("Eddie Override"), "wb", None)
    assert (r.ref, r.method) == ("e" * 16, "override")


def test_shared_name_settled_by_weight_class_or_left_unresolved():
    assert (
        resolve_fighter(RESOLVER, fighter("Bruno Same"), "wb", "Flyweight").ref
        == "b" * 16
    )
    unsettled = resolve_fighter(RESOLVER, fighter("Bruno Same"), "wb", "Catch Weight")
    assert unsettled.ref is None and unsettled.method == METHOD_AMBIGUOUS


def test_article_title_without_its_disambiguation_is_a_second_exact_name():
    r = resolve_fighter(RESOLVER, fighter("Al Able", "Alan Able (fighter)"), "wb", None)
    assert r.ref == "a" * 16
    clash = resolve_fighter(RESOLVER, fighter("Alan Able", "Frank Other"), "wb", None)
    assert clash.ref is None and clash.method == METHOD_CONFLICT


def test_unknown_name_is_a_provisional_debutant_keyed_by_article():
    linked = resolve_fighter(
        RESOLVER, fighter("New Guy", "New Guy (fighter)"), "wb", None
    )
    assert linked.provisional and linked.ref == "wp:New Guy (fighter)"
    assert linked.method == METHOD_PROVISIONAL
    unlinked = resolve_fighter(RESOLVER, fighter("Nuevo Tipo"), "wb", None)
    assert unlinked.ref == "wpname:nuevo tipo"


def test_placeholders_and_one_fighter_twice_leave_the_bout_unresolved():
    tbd = resolve_bout(RESOLVER, bout(fighter("Alan Able"), fighter("TBD")), "wb")
    assert not tbd.resolved and tbd.b.method == METHOD_PLACEHOLDER
    twice = resolve_bout(
        RESOLVER, bout(fighter("Alan Able"), fighter("Alan Able")), "wb"
    )
    assert not twice.resolved and "one fighter" in twice.unresolved_reason
    shared = resolve_bout(
        RESOLVER,
        bout(fighter("Bruno Same"), fighter("Alan Able"), "Catch Weight"),
        "wb",
    )
    assert not shared.resolved and "Bruno Same" in shared.unresolved_reason
    ok = resolve_bout(RESOLVER, bout(fighter("Alan Able"), fighter("New Guy")), "wb")
    assert ok.resolved and ok.b.provisional


# --- card changes ----------------------------------------------------------------

PAGEID = 4242
TODAY = date(2026, 11, 17)
NOW = "2026-11-17T12:00:00Z"
LATER = "2026-11-18T12:00:00Z"


@pytest.fixture
def conn():
    db = connect(":memory:")
    listed = ListedEvent("UFC Test", "UFC Test", date(2026, 11, 21))
    cards_db.upsert_event(
        db, listed, Page("UFC Test", "UFC Test", PAGEID, 7, "x"), NOW, TODAY
    )
    yield db
    db.close()


EVENT = cards_db.event_key(PAGEID)


def card(*pairs):
    bouts = [bout(fighter(a), fighter(b), order=i) for i, (a, b) in enumerate(pairs)]
    return [(b, resolve_bout(RESOLVER, b, "wb")) for b in bouts]


FULL = (
    ("Alan Able", "Dan Dee"),
    ("Ed Override", "Frank Other"),
    ("New Guy", "Nuevo Tipo"),
    ("Bruno Same", "Zed Zero"),
)


def bout_ids(*pairs):
    return [cards_db.bout_id(PAGEID, fighter(a), fighter(b)) for a, b in pairs]


def predict(conn, bout_id):
    conn.execute(
        """INSERT INTO predictions (bout_id, model_version, created_at, event_date,
        fighter_a_ref, fighter_b_ref, p_fighter_a, p_fighter_b, inputs_sha256,
        features_json, limited_data, limited_reasons_json, is_live)
        VALUES (?, 'v', ?, '2026-11-21', 'x', 'y', 0.6, 0.4, 'h', '{}', 0, '[]', 1)""",
        (bout_id, NOW),
    )


def rows(conn):
    return {
        r["bout_id"]: dict(r)
        for r in conn.execute("SELECT * FROM upcoming_bouts ORDER BY card_order")
    }


def current_predictions(conn):
    return {
        r["bout_id"]
        for r in conn.execute(
            "SELECT bout_id FROM predictions WHERE superseded_at IS NULL"
        )
    }


def test_bout_ids_are_stable_and_ignore_corner_and_spelling_of_accents():
    a = cards_db.bout_id(PAGEID, fighter("Jiří Procházka"), fighter("Navajo Stirling"))
    b = cards_db.bout_id(PAGEID, fighter("Navajo Stirling"), fighter("Jiri Prochazka"))
    assert a == b and a.startswith("wb") and len(a) == 18
    assert a != cards_db.bout_id(
        PAGEID + 1, fighter("Jiří Procházka"), fighter("Navajo Stirling")
    )


def test_first_card_is_stored_with_resolution_and_provisional_fighters(conn):
    change = cards_db.apply_card(conn, EVENT, PAGEID, card(*FULL), NOW)
    assert change.applied and len(change.added) == 4 and not change.cancelled
    stored = rows(conn)
    first = stored[bout_ids(FULL[0])[0]]
    assert (first["fighter_a_ref"], first["fighter_b_ref"]) == ("a" * 16, "d" * 16)
    assert first["first_seen_at"] == first["last_seen_at"] == NOW
    unresolved = stored[
        bout_ids(FULL[3])[0]
    ]  # shared name, Lightweight settles nothing
    assert unresolved["fighter_a_ref"] is None and unresolved["unresolved_reason"]
    assert change.unresolved == bout_ids(FULL[3])
    provisional = {
        r["provisional_id"] for r in conn.execute("SELECT * FROM provisional_fighters")
    }
    assert provisional == {"wpname:new guy", "wpname:nuevo tipo", "wpname:zed zero"}


def test_a_bout_dropped_from_the_card_is_cancelled_and_its_prediction_superseded(conn):
    cards_db.apply_card(conn, EVENT, PAGEID, card(*FULL), NOW)
    for key in bout_ids(*FULL[:3]):
        predict(conn, key)
    change = cards_db.apply_card(conn, EVENT, PAGEID, card(*FULL[:2], FULL[3]), LATER)
    (dropped,) = bout_ids(FULL[2])
    assert change.cancelled == [dropped]
    stored = rows(conn)
    assert (
        stored[dropped]["status"] == "cancelled"
        and stored[dropped]["cancelled_at"] == LATER
    )
    assert stored[bout_ids(FULL[0])[0]]["last_seen_at"] == LATER
    assert stored[bout_ids(FULL[0])[0]]["first_seen_at"] == NOW
    assert current_predictions(conn) == set(bout_ids(*FULL[:2]))
    reason = conn.execute(
        "SELECT superseded_reason FROM predictions WHERE bout_id = ?", (dropped,)
    ).fetchone()[0]
    assert reason == "bout cancelled"
    # Back on the card later: scheduled again, a new prediction will be made.
    back = cards_db.apply_card(conn, EVENT, PAGEID, card(*FULL), LATER)
    assert back.restored == [dropped] and rows(conn)[dropped]["cancelled_at"] is None


def test_a_replacement_opponent_is_a_new_bout_and_the_old_one_is_cancelled(conn):
    cards_db.apply_card(conn, EVENT, PAGEID, card(*FULL), NOW)
    replaced = (("Alan Able", "Frank Other"), FULL[1], FULL[2], FULL[3])
    change = cards_db.apply_card(conn, EVENT, PAGEID, card(*replaced), LATER)
    assert change.added == bout_ids(replaced[0])
    assert change.cancelled == bout_ids(FULL[0])


def test_an_implausible_card_changes_nothing(conn):
    cards_db.apply_card(conn, EVENT, PAGEID, card(*FULL), NOW)
    for key in bout_ids(*FULL[:3]):
        predict(conn, key)
    before = (rows(conn), current_predictions(conn))
    for broken in (card(), card(FULL[0])):  # no bouts; three of four gone at once
        change = cards_db.apply_card(conn, EVENT, PAGEID, broken, LATER)
        assert not change.applied and change.reason
        assert (rows(conn), current_predictions(conn)) == before
    errors = conn.execute(
        "SELECT message FROM refresh_log WHERE level = 'error'"
    ).fetchall()
    assert len(errors) == 2


def test_plausibility_rule():
    assert cards_db.implausible(set(), set()) is None  # an event with no card yet
    assert cards_db.implausible({"a"}, set())
    assert (
        cards_db.implausible({"a", "b", "c"}, {"x"}) is None
    )  # small cards: no share rule
    assert (
        cards_db.implausible({"a", "b", "c", "d"}, {"a", "b"}) is None
    )  # half is allowed
    assert cards_db.implausible({"a", "b", "c", "d"}, {"a"})


def test_a_provisional_fighter_is_linked_once_the_dataset_has_him(conn):
    cards_db.apply_card(conn, EVENT, PAGEID, card(*FULL[:3]), NOW)
    later = FighterResolver([*PROFILES, Profile("9" * 16, ("New Guy",), 155)])
    b = bout(fighter("New Guy"), fighter("Nuevo Tipo"), order=2)
    pairs = [*card(*FULL[:2]), (b, resolve_bout(later, b, "wb"))]
    cards_db.apply_card(conn, EVENT, PAGEID, pairs, LATER)
    row = conn.execute(
        "SELECT fighter_id, linked_at FROM provisional_fighters WHERE provisional_id = 'wpname:new guy'"
    ).fetchone()
    assert tuple(row) == ("9" * 16, LATER)
    assert rows(conn)[bout_ids(FULL[2])[0]]["fighter_a_ref"] == "9" * 16  # same bout ID


def test_events_retire_when_their_date_passes_or_they_leave_the_list(conn):
    listed = ListedEvent("UFC Gone", "UFC Gone", date(2026, 12, 1))
    other = cards_db.upsert_event(
        conn, listed, Page("UFC Gone", "UFC Gone", 99, 1, "x"), NOW, TODAY
    )
    cards_db.retire_events(conn, {EVENT}, TODAY, NOW)
    status = dict(
        conn.execute("SELECT event_key, status FROM upcoming_events").fetchall()
    )
    assert status == {EVENT: "scheduled", other: "unlisted"}
    cards_db.retire_events(conn, {EVENT}, date(2026, 11, 21), LATER)
    status = dict(
        conn.execute("SELECT event_key, status FROM upcoming_events").fetchall()
    )
    assert status[EVENT] == "past"


def test_an_article_citing_a_dataset_profile_unresolves_a_provisional_fighter():
    b = bout(fighter("Alan Able"), fighter("Al Long-Name", "Al Long-Name"))
    resolution = resolve_bout(RESOLVER, b, "wb")
    assert resolution.resolved and resolution.b.provisional
    text = "{{Infobox martial artist|reach = 70 in<ref>[http://ufcstats.com/fighter-details/AAAAAAAAAAAAAAAA x]</ref>}}"
    assert cited_profiles(text) == {"a" * 16}
    checked = check_provisional(
        resolution, b, {"Al Long-Name": cited_profiles(text)}, {"a" * 16}
    )
    assert not checked.resolved and checked.b.method == METHOD_IN_DATASET
    assert (
        "a" * 16 in checked.unresolved_reason
        and "override" in checked.unresolved_reason
    )
    # An ID the dataset does not have, or no citation, leaves him provisional.
    assert (
        check_provisional(resolution, b, {"Al Long-Name": {"9" * 16}}, {"a" * 16})
        is resolution
    )
    assert check_provisional(resolution, b, {}, {"a" * 16}) is resolution


def test_shipped_wikipedia_overrides_load_and_name_dataset_ids():
    overrides = load_wikipedia_overrides()
    assert overrides and all(len(o.fighter_id) == 16 and o.note for o in overrides)
    assert all(o.fight_id is None for o in overrides)
