"""Unit tests for the pure field parsers."""

from datetime import date

import pytest

from ufc.ingest import parsing as p


def test_clean_collapses_whitespace():
    assert p.clean("  UFC 900:  Valcourt vs. Hollin ") == "UFC 900: Valcourt vs. Hollin"


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Jiří Procházka", "Jiri Prochazka"),
        ("Waldo Cortes-Acosta", "Waldo Cortes Acosta"),
        ("Raul Rosas Jr.", "raul rosas jr"),
        ("Jules Tomlin ", "Jules Tomlin"),
    ],
)
def test_name_key_matches_spelling_variants(left, right):
    assert p.name_key(left) == p.name_key(right)


def test_name_key_keeps_different_names_apart():
    assert p.name_key("Bruno Silva") != p.name_key("Bruna Silva")


def test_id_from_url():
    url = "http://ufcstats.com/fight-details/b000000000000001"
    assert p.id_from_url(url, "fight") == "b000000000000001"


@pytest.mark.parametrize(
    ("url", "kind"),
    [
        ("http://ufcstats.com/fight-details/b000000000000001", "event"),
        ("http://ufcstats.com/fight-details/b00000000000001", "fight"),
        ("http://example.com/fight-details/b000000000000001", "fight"),
        ("", "fighter"),
    ],
)
def test_id_from_url_rejects_wrong_kind_or_shape(url, kind):
    with pytest.raises(p.ParseError):
        p.id_from_url(url, kind)


def test_parse_event_date():
    assert p.parse_event_date("November 14, 2026") == date(2026, 11, 14)
    with pytest.raises(p.ParseError):
        p.parse_event_date("14/11/2026")


def test_parse_dob_present_and_missing():
    assert p.parse_dob("Jul 13, 1978") == date(1978, 7, 13)
    assert p.parse_dob("--") is None
    assert p.parse_dob("") is None
    with pytest.raises(p.ParseError):
        p.parse_dob("1978-07-13")


def test_parse_height_cm():
    assert p.parse_height_cm("5' 11\"") == 180.34
    assert p.parse_height_cm("6' 0\"") == 182.88
    assert p.parse_height_cm("--") is None
    with pytest.raises(p.ParseError):
        p.parse_height_cm("5' 12\"")
    with pytest.raises(p.ParseError):
        p.parse_height_cm("180cm")


def test_parse_reach_cm_present_and_missing():
    assert p.parse_reach_cm('72"') == 182.88
    assert p.parse_reach_cm("--") is None
    with pytest.raises(p.ParseError):
        p.parse_reach_cm("72")


def test_parse_weight_lbs():
    assert p.parse_weight_lbs("155 lbs.") == 155
    assert p.parse_weight_lbs("--") is None


def test_parse_stance():
    assert p.parse_stance("Southpaw") == "Southpaw"
    assert p.parse_stance("") is None


def test_parse_landed_attempted():
    assert p.parse_landed_attempted("41 of 65") == (41, 65)
    with pytest.raises(p.ParseError):
        p.parse_landed_attempted("---")


@pytest.mark.parametrize(
    ("text", "expected"), [("0", 0), ("2", 2), ("0.0", 0), ("3.0", 3)]
)
def test_parse_count_accepts_int_and_float_formatting(text, expected):
    assert p.parse_count(text) == expected


@pytest.mark.parametrize("text", ["", "1.5", "-1", "one"])
def test_parse_count_rejects_non_counts(text):
    with pytest.raises(p.ParseError):
        p.parse_count(text)


def test_parse_clock_and_control_time():
    assert p.parse_clock("4:21") == 261
    assert p.parse_clock("0:00") == 0
    assert p.parse_clock("12:00") == 720
    assert p.parse_control_time("--") is None
    assert p.parse_control_time("1:05") == 65
    with pytest.raises(p.ParseError):
        p.parse_clock("4:61")


def test_parse_round_label_and_number():
    assert p.parse_round_label("Round 5") == 5
    assert p.parse_round_number("3") == 3
    with pytest.raises(p.ParseError):
        p.parse_round_label("5")
    with pytest.raises(p.ParseError):
        p.parse_round_number("0")


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("W/L", ("win", "a")),
        ("L/W", ("win", "b")),
        ("D/D", ("draw", None)),
        ("NC/NC", ("nc", None)),
    ],
)
def test_parse_outcome(code, expected):
    assert p.parse_outcome(code) == expected


def test_parse_outcome_rejects_unknown_code():
    with pytest.raises(p.ParseError):
        p.parse_outcome("W/W")


def test_split_bout():
    assert p.split_bout("Orin Valcourt vs. Dax Hollin") == (
        "Orin Valcourt",
        "Dax Hollin",
    )
    with pytest.raises(p.ParseError):
        p.split_bout("Orin Valcourt")


@pytest.mark.parametrize(
    ("text", "weight_class", "womens", "title", "interim", "tournament"),
    [
        ("Lightweight Bout", "Lightweight", False, False, False, False),
        ("Light Heavyweight Bout", "Light Heavyweight", False, False, False, False),
        ("Super Heavyweight Bout", "Super Heavyweight", False, False, False, False),
        ("Women's Strawweight Bout", "Women's Strawweight", True, False, False, False),
        ("UFC Heavyweight Title Bout", "Heavyweight", False, True, False, False),
        (
            "UFC Women's Flyweight Title Bout",
            "Women's Flyweight",
            True,
            True,
            False,
            False,
        ),
        ("UFC Interim Heavyweight Title Bout", "Heavyweight", False, True, True, False),
        ("UFC Superfight Championship Bout", None, False, True, False, False),
        ("Catch Weight Bout", "Catch Weight", False, False, False, False),
        ("Open Weight Bout", "Open Weight", False, False, False, False),
        ("UFC 2 Tournament Title Bout", None, False, False, False, True),
        (
            "Ultimate Fighter 33 Welterweight Tournament Title Bout",
            "Welterweight",
            False,
            False,
            False,
            True,
        ),
        (
            "Road to UFC 3 Women's Strawweight Tournament TitleBout",
            "Women's Strawweight",
            True,
            False,
            False,
            True,
        ),
    ],
)
def test_parse_bout_type(text, weight_class, womens, title, interim, tournament):
    bout_type = p.parse_bout_type(text)
    assert bout_type.weight_class == weight_class
    assert bout_type.is_womens is womens
    assert bout_type.is_title_bout is title
    assert bout_type.is_interim_title is interim
    assert bout_type.is_tournament_final is tournament


def test_parse_time_format_standard():
    time_format = p.parse_time_format("3 Rnd (5-5-5)")
    assert time_format.scheduled_rounds == 3
    assert time_format.round_minutes == (5, 5, 5)
    assert not time_format.no_time_limit


def test_parse_time_format_overtime_counts_regulation_rounds_only():
    time_format = p.parse_time_format("1 Rnd + 2OT (15-3-3)")
    assert time_format.scheduled_rounds == 1
    assert time_format.round_minutes == (15, 3, 3)
    assert p.parse_time_format("3 Rnd + OT (5-5-5-5)").scheduled_rounds == 3


def test_parse_time_format_no_time_limit():
    time_format = p.parse_time_format("No Time Limit")
    assert time_format.no_time_limit
    assert time_format.scheduled_rounds is None


@pytest.mark.parametrize("text", ["", "3 Rounds", "3 Rnd (5-5)", "Rnd (5)"])
def test_parse_time_format_rejects_unknown_shapes(text):
    with pytest.raises(p.ParseError):
        p.parse_time_format(text)


def test_fight_time_s():
    three_rounds = p.parse_time_format("3 Rnd (5-5-5)")
    assert p.fight_time_s(three_rounds, 1, 87) == 87
    assert p.fight_time_s(three_rounds, 2, 226) == 300 + 226
    assert p.fight_time_s(three_rounds, 3, 300) == 900
    overtime = p.parse_time_format("1 Rnd + OT (30-3)")
    assert p.fight_time_s(overtime, 2, 180) == 30 * 60 + 180
    assert p.fight_time_s(three_rounds, 4, 10) is None


def test_fight_time_s_no_time_limit():
    no_limit = p.parse_time_format("No Time Limit")
    assert p.fight_time_s(no_limit, 1, 58) == 58
    assert p.fight_time_s(no_limit, 2, 58) is None
