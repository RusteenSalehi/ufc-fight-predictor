"""Fighter name resolution rules, on small hand-built profiles."""

import pytest

from ufc.ingest import identity
from ufc.ingest.identity import FighterResolver, Override, Profile, weight_fits

A, B, C, D = "a" * 16, "b" * 16, "c" * 16, "d" * 16
FIGHT, OTHER_FIGHT = "f" * 16, "e" * 16


def resolver(overrides=()):
    return FighterResolver(
        [
            Profile(A, ("Alex Stone",), 155),
            Profile(B, ("Sam Reed", "Samuel Reed"), 170),
            Profile(C, ("Chris Cole",), 135),
            Profile(D, ("Chris Cole",), 185),
        ],
        overrides,
    )


def test_unique_name_resolves():
    result = resolver().resolve("Alex Stone", FIGHT, "Lightweight")
    assert (result.fighter_id, result.method) == (A, identity.METHOD_NAME)


def test_unique_name_resolves_whatever_the_weight_class():
    assert resolver().resolve("Alex Stone", FIGHT, "Heavyweight").fighter_id == A
    assert resolver().resolve("Alex Stone", FIGHT, None).fighter_id == A


def test_either_recorded_name_resolves_ignoring_case_accents_and_punctuation():
    assert resolver().resolve("Samuel Reed", FIGHT, None).fighter_id == B
    assert resolver().resolve("SAM  REED", FIGHT, None).fighter_id == B
    assert resolver().resolve("Sám Réed.", FIGHT, None).fighter_id == B


def test_unknown_name_is_unresolved_never_fuzzy_matched():
    result = resolver().resolve("Alec Stone", FIGHT, "Lightweight")
    assert result.fighter_id is None
    assert result.method == identity.METHOD_UNKNOWN_NAME


def test_shared_name_is_settled_by_the_one_fitting_weight():
    result = resolver().resolve("Chris Cole", FIGHT, "Middleweight")
    assert (result.fighter_id, result.method) == (D, identity.METHOD_WEIGHT)
    assert result.candidates == (C, D)
    assert resolver().resolve("Chris Cole", FIGHT, "Bantamweight").fighter_id == C


@pytest.mark.parametrize(
    "weight_class", ["Featherweight", "Catch Weight", "Open Weight", None]
)
def test_shared_name_stays_unresolved_when_weight_does_not_single_one_out(weight_class):
    result = resolver().resolve("Chris Cole", FIGHT, weight_class)
    assert result.fighter_id is None
    assert result.method == identity.METHOD_AMBIGUOUS


def test_shared_name_stays_unresolved_when_both_profiles_fit():
    both_heavy = FighterResolver(
        [Profile(A, ("Big Man",), 240), Profile(B, ("Big Man",), 265)]
    )
    assert both_heavy.resolve("Big Man", FIGHT, "Heavyweight").fighter_id is None


def test_fight_specific_override_beats_name_override_and_rules():
    overrides = [
        Override("Chris Cole", None, C, "default"),
        Override("Chris Cole", FIGHT, D, "this fight"),
    ]
    subject = resolver(overrides)
    assert subject.resolve("Chris Cole", FIGHT, "Bantamweight").fighter_id == D
    assert subject.resolve("Chris Cole", OTHER_FIGHT, "Middleweight").fighter_id == C
    assert subject.resolve("Chris Cole", FIGHT, None).method == identity.METHOD_OVERRIDE


def test_override_maps_an_unknown_spelling():
    subject = resolver([Override("Alec Stone", None, A, "misspelling")])
    assert subject.resolve("Alec Stone", FIGHT, None).fighter_id == A


def test_override_for_another_fight_does_not_apply():
    subject = resolver([Override("Chris Cole", OTHER_FIGHT, D, "")])
    assert subject.resolve("Chris Cole", FIGHT, "Featherweight").fighter_id is None


def test_override_pointing_at_an_unknown_fighter_is_rejected():
    bad = Override("Alex Stone", None, "9" * 16, "")
    subject = resolver([bad])
    assert subject.invalid_overrides == [bad]
    assert subject.resolve("Alex Stone", FIGHT, None).fighter_id == A


def test_unused_overrides_are_reported():
    used, unused = Override("Alec Stone", None, A, ""), Override("Nobody", None, B, "")
    subject = resolver([used, unused])
    subject.resolve("Alec Stone", FIGHT, None)
    assert subject.unused_overrides() == [unused]


@pytest.mark.parametrize(
    ("weight", "weight_class", "expected"),
    [
        (155, "Lightweight", True),
        (145, "Lightweight", False),
        (115, "Women's Strawweight", True),
        (135, "Women's Flyweight", False),
        (265, "Heavyweight", True),
        (240, "Heavyweight", True),
        (205, "Heavyweight", False),
        (205, "Light Heavyweight", True),
        (None, "Lightweight", False),
        (155, None, False),
        (155, "Catch Weight", False),
        (155, "Open Weight", False),
        (300, "Super Heavyweight", False),
    ],
)
def test_weight_fits(weight, weight_class, expected):
    assert weight_fits(weight, weight_class) is expected


def test_shipped_overrides_load_and_carry_a_reason():
    overrides = identity.load_overrides()
    assert overrides
    for override in overrides:
        assert len(override.fighter_id) == 16
        assert override.fight_id is None or len(override.fight_id) == 16
        assert len(override.note) > 20


def test_overrides_file_with_wrong_columns_is_rejected(tmp_path):
    path = tmp_path / "overrides.csv"
    path.write_text("name,fighter_id\nA B,abc\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        identity.load_overrides(path)
