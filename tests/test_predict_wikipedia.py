"""Wikipedia client and parsers, on saved API responses. No network."""

from datetime import date

import pytest

from tests.wiki_helpers import (
    FixtureTransport,
    legacy_wikitext,
    saved,
    wikitext_of,
)
from ufc.predict import wikipedia as wp
from ufc.predict.cards import (
    CardParseError,
    normalise_weight_class,
    parse_card,
    parse_event_list,
)
from ufc.predict.infobox import (
    parse_birth_date,
    parse_height,
    parse_infobox,
    parse_reach,
)
from ufc.predict.wikitext import plain, templates

# --- client ----------------------------------------------------------------------


def test_user_agent_names_the_client_and_a_contact(monkeypatch):
    monkeypatch.delenv(wp.CONTACT_ENV, raising=False)
    agent = wp.user_agent()
    assert agent.startswith("ufc-fight-predictor/")
    assert "(https://github.com/" in agent and "python-requests/" in agent
    monkeypatch.setenv(wp.CONTACT_ENV, "someone@example.org")
    assert "(someone@example.org)" in wp.user_agent()
    monkeypatch.setenv(wp.CONTACT_ENV, " ")
    with pytest.raises(wp.WikipediaError):
        wp.user_agent()


def test_pages_follow_redirects_and_report_missing_pages():
    record = saved("fighter_pages_query.json")
    requested = record["params"]["titles"].split("|")
    pages = wp.pages_from_query(record["response"], requested)
    assert pages["Jiri Prochazka"].title == "Jiří Procházka"
    assert pages["Jiri Prochazka"].exists and pages["Jiri Prochazka"].revid
    assert not pages["Tahir Abdullayev (fighter)"].exists
    assert pages["Navajo Stirling"].pageid


def test_pages_are_batched_cached_and_sent_with_api_etiquette(tmp_path):
    record = saved("fighter_pages_query.json")
    titles = record["params"]["titles"].split("|")
    transport = FixtureTransport({frozenset(titles): record["response"]})
    now = [1000.0]
    client = wp.Wikipedia(transport, tmp_path, clock=lambda: now[0])
    first = client.pages(reversed(titles), max_age_s=60)
    assert client.requests_made == 1
    params = transport.requests[0]
    assert params["maxlag"] == "5" and params["format"] == "json"
    assert params == record["params"]  # the fixture was saved by this client
    again = client.pages(titles, max_age_s=60)  # same set, any order: cached
    assert client.requests_made == 1 and again == first
    now[0] += 61
    client.pages(titles, max_age_s=60)  # expired
    assert client.requests_made == 2


def test_more_than_fifty_titles_are_split_into_batches():
    titles = [f"Page {i:02d}" for i in range(60)]
    seen = []

    def transport(params):
        batch = params["titles"].split("|")
        seen.append(len(batch))
        return {"query": {"pages": [{"title": t, "missing": True} for t in batch]}}

    pages = wp.Wikipedia(transport).pages(titles, 0)
    assert seen == [50, 10] and len(pages) == 60


class FakeResponse:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}

    def json(self):
        if self._body is None:
            raise ValueError
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses, self.headers, self.calls = list(responses), {}, []

    def get(self, url, params, timeout):
        self.calls.append((url, params, timeout))
        return self.responses.pop(0)


def test_transport_retries_rate_limits_and_maxlag_then_succeeds():
    ok = {"query": {"pages": []}}
    session = FakeSession(
        [
            FakeResponse(429, headers={"Retry-After": "7"}),
            FakeResponse(200, {"error": {"code": "maxlag", "info": "lagged"}}),
            FakeResponse(200, ok),
        ]
    )
    sleeps = []
    send = wp.http_transport(session, sleep=sleeps.append, clock=lambda: 0.0)
    assert send({"action": "query"}) == ok
    assert len(session.calls) == 3
    assert all(timeout == wp.TIMEOUT_S for _, _, timeout in session.calls)
    assert session.headers["User-Agent"] == wp.user_agent()
    assert 7.0 in sleeps  # Retry-After is honoured


def test_transport_does_not_retry_a_client_error():
    session = FakeSession([FakeResponse(403), FakeResponse(200, {})])
    send = wp.http_transport(session, sleep=lambda s: None, clock=lambda: 0.0)
    with pytest.raises(wp.WikipediaError, match="403"):
        send({})
    assert len(session.calls) == 1


def test_transport_gives_up_after_its_attempts():
    session = FakeSession([FakeResponse(503)] * wp.MAX_ATTEMPTS)
    send = wp.http_transport(session, sleep=lambda s: None, clock=lambda: 0.0)
    with pytest.raises(wp.WikipediaError, match="attempts"):
        send({})


# --- wikitext and cards ----------------------------------------------------------


def test_templates_split_on_top_level_pipes_only():
    text = "{{MMAevent bout|Flyweight|[[A B (fighter)|A B]]|vs.|{{sortname|C|D}}|x=1}}"
    (t,) = templates(text, "mmaevent_bout")
    assert t.positional == (
        "Flyweight",
        "[[A B (fighter)|A B]]",
        "vs.",
        "{{sortname|C|D}}",
    )
    assert t.named == {"x": "1"}
    assert plain("[[A B (fighter)|A B]]<ref>x</ref> {{small|y}}") == "A B"


def test_event_list_has_scheduled_and_past_events():
    text = wikitext_of("list_of_ufc_events_query.json", "List of UFC events")
    events = parse_event_list(text)
    assert len(events.scheduled) == 9
    assert events.scheduled[-1].page == "UFC Fight Night: Procházka vs. Stirling"
    assert events.scheduled[-1].date == date(2026, 11, 21)
    assert (
        events.past[0].page == "UFC 334"
        and events.past[0].name == "UFC 334: Gane vs. Hokit"
    )
    assert events.past[0].date == date(2026, 11, 14)
    assert len(events.past) == 12  # the fixture keeps 12 rows (see its README)
    assert all(a.date >= b.date for a, b in zip(events.past, events.past[1:]))
    with pytest.raises(CardParseError):
        parse_event_list("==Scheduled events==\nnothing here")


def test_upcoming_card_in_page_order_with_segments():
    card = parse_card(legacy_wikitext("upcoming_event_parse.json"))
    assert len(card.bouts) == 9
    first = card.bouts[0]
    assert (first.fighter_a.name, first.fighter_b.name) == (
        "Jiří Procházka",
        "Navajo Stirling",
    )
    assert first.segment == "Main card" and first.weight_class == "Light Heavyweight"
    assert not first.completed and first.connector == "vs."
    luna = card.bouts[5].fighter_b
    assert (luna.name, luna.page) == ("Santiago Luna", "Santiago Luna (fighter)")
    assert card.bouts[7].fighter_b.page is None  # Alessandro Costa has no article
    assert card.bouts[5].segment == "Preliminary card"
    assert [b.order for b in card.bouts] == list(range(9))
    assert card.announced[0] == "Lightweight bout: Manoel Sousa vs. TBD"
    # The same page, read through the query API in M5, gives the same card.
    again = parse_card(
        wikitext_of("event_pages_query.json", "UFC Fight Night: Procházka vs. Stirling")
    )
    assert again == card


def test_completed_card_has_results_titles_and_catchweights():
    card = parse_card(wikitext_of("event_pages_query.json", "UFC 334"))
    main = card.bouts[0]
    assert main.completed and main.connector == "def."
    assert main.fighter_a.name == "Ciryl Gane"  # "(c)" removed
    assert main.is_title_bout
    draw = next(b for b in card.bouts if b.fighter_a.name == "Uroš Medić")
    assert draw.connector == "vs." and draw.completed  # a draw has a result
    catch = [b for b in card.bouts if b.weight_class_raw.startswith("Catchweight")]
    assert catch and all(b.weight_class == "Catch Weight" for b in catch)


def test_weight_classes_map_to_the_datasets_labels():
    assert normalise_weight_class("Women's Strawweight") == "Women's Strawweight"
    assert normalise_weight_class("[[Light heavyweight]]") == "Light Heavyweight"
    assert normalise_weight_class("Catchweight (150 lb)") == "Catch Weight"
    assert normalise_weight_class("Super Heavyweight") is None


def test_a_bout_template_of_unexpected_shape_is_an_error():
    with pytest.raises(CardParseError):
        parse_card("{{MMAevent bout|Flyweight|A|beats|B}}")


# --- infobox ---------------------------------------------------------------------


def test_infobox_attributes_from_saved_fighter_pages():
    stirling = parse_infobox(wikitext_of("fighter_pages_query.json", "Navajo Stirling"))
    assert stirling.height_cm == 193.04 and stirling.reach_cm == 200.66
    assert stirling.dob == date(1997, 11, 7)
    luna = parse_infobox(
        wikitext_of("fighter_pages_query.json", "Santiago Luna (fighter)")
    )
    assert luna.height_cm == 172.72  # {{height|ft=5|in=8}}
    assert luna.dob == date(2004, 8, 6)
    assert parse_infobox("no infobox here").height_cm is None


@pytest.mark.parametrize(
    ("raw", "cm"),
    [
        ("6 ft 4 in", 193.04),
        ("5 ft 11 in (180 cm)", 180.34),
        ("{{height|ft=5|in=8}}", 172.72),
        ("{{height|m=1.80}}", 180.0),
        ("{{convert|6|ft|0|in|cm}}", 182.88),
        ("tall", None),
        ("9 ft 0 in", None),  # outside the plausible range
    ],
)
def test_height_forms(raw, cm):
    assert parse_height(raw) == cm


@pytest.mark.parametrize(
    ("raw", "cm"),
    [
        ("79 in", 200.66),
        ("72 in (183 cm)", 182.88),
        ("{{convert|72|in|cm}}", 182.88),
        ("", None),
    ],
)
def test_reach_forms(raw, cm):
    assert parse_reach(raw) == cm


def test_birth_date_forms():
    assert parse_birth_date("{{Birth date and age|1997|11|07|df=y}}") == date(
        1997, 11, 7
    )
    assert parse_birth_date("{{birth date and age|df=yes|1992|10|14}}") == date(
        1992, 10, 14
    )
    assert parse_birth_date("{{bda|1990|2|30}}") is None  # not a date
    assert parse_birth_date("May 1990") is None
