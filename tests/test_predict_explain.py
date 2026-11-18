"""The plain-language "why": grouping, templates, wording, and the mirror rule.

The league's winners are random, so nothing here checks that a reason is
right about fighting; it checks that the text is a faithful, deterministic
rendering of what the model weighs, and that swapping the fighters gives the
mirrored explanation.
"""

import math
import re
from datetime import date

import numpy as np
import pytest

from tests.feature_helpers import random_league
from tests.model_helpers import FAST
from ufc.features.build import build_features
from ufc.features.config import FeatureConfig
from ufc.features.matchup import (
    ALL_LEVEL_NAMES,
    FEATURE_NAMES,
    FEATURES,
    LEVEL_NAMES,
    Matchup,
    build_matchup,
)
from ufc.models.calibrate import PLATT
from ufc.models.design import FeatureSpec, matrix, swapped
from ufc.models.gbm import LIGHTGBM, XGBOOST
from ufc.models.recipe import Recipe, fit_recipe
from ufc.models.splits import fit_rows, labelled
from ufc.predict.explain import (
    COMPONENTS,
    STATS,
    TITLES,
    contributions,
    detail,
    explain,
    explain_bout,
    explain_values,
    factors,
    group_of,
    group_totals,
    parse,
)
from ufc.predict.predict import model_inputs

CONFIG = FeatureConfig(train_end=date(2020, 12, 31), shrink_minutes=15.0)
SPEC = FeatureSpec(FEATURE_NAMES, ALL_LEVEL_NAMES)
ON = date(2026, 6, 1)
CAUSAL = re.compile(
    r"\b(because|causes?|caused|too old|too young|will win|will lose|guarantee"
    r"|is better|is worse|past (his|her) prime|washed)\b",
    re.IGNORECASE,
)


@pytest.fixture(scope="module")
def league():
    return random_league(seed=17, n_fighters=40, n_events=110, start=date(2017, 1, 7))


@pytest.fixture(scope="module")
def build(league):
    return build_features(league.data(), CONFIG)


def _fit(build, library):
    frame = build.frame
    recipe = Recipe(FAST[library], 8.0, SPEC, PLATT)
    train = fit_rows(frame[frame["date"] <= np.datetime64("2020-12-31")])
    valid = labelled(frame[frame["date"] > np.datetime64("2020-12-31")])
    candidate, _ = fit_recipe(recipe, train, valid, date(2020, 12, 31))
    return candidate


@pytest.fixture(scope="module", params=[XGBOOST, LIGHTGBM])
def candidate(build, request):
    return _fit(build, request.param)


def pairs(league, n=12):
    fighters = sorted(league.data().fighters)
    return [(fighters[i], fighters[(i * 7 + 3) % len(fighters)]) for i in range(n)]


def _explained(candidate, store, a, b, names):
    m = build_matchup(store, a, b, ON, "Lightweight")
    x = model_inputs(m, candidate.spec.names)
    p_a, p_b = candidate.predict_pair(x)
    return explain_bout(candidate, x, m, names, float(p_a[0]), float(p_b[0])), m, x


# --- grouping ----------------------------------------------------------------------


def test_every_model_input_has_a_reason_with_a_title():
    for name in (*FEATURE_NAMES, *ALL_LEVEL_NAMES):
        group = group_of(name)
        assert group in TITLES, name
        p = parse(name)
        assert p.stem in STATS or p.stem in {
            "stance_southpaw_vs_orthodox",
            "matchup_td",
            "matchup_strike",
            "matchup_kd",
            "matchup_ctrl",
            "matchup_sub",
        }, name
    with pytest.raises(KeyError):
        group_of("d_something_new")


def test_correlated_inputs_share_one_reason():
    age = {
        group_of(n)
        for n in (
            "d_age",
            "d_age_past_32",
            "d_age_past_class_median",
            "lvl_age",
            "lvl_age_rel_class",
            "d_n_fights",
            "lvl_n_fights",
        )
    }
    assert age == {"age_experience"}
    rating = {
        group_of(n) for n in ("d_elo", "lvl_elo", "d_opp_elo_career", "d_opp_elo_l3")
    }
    assert rating == {"rating"}
    for stat in ("slpm", "sapm", "sig_acc", "td_def", "kd_per15", "sub_att_per15"):
        names = [f"d_{stat}_career", f"d_{stat}_l3", f"d_adj_{stat}_career"]
        names += [f"d_adj_{stat}_decay"]
        assert {group_of(n) for n in names} == {stat}
    # Different stats stay apart.
    assert group_of("d_slpm_career") != group_of("d_sapm_career")
    families = {f.name: f.family for f in FEATURES}
    assert all(
        group_of(n) == "record"
        for n in families
        if families[n] == "record" and n not in ("d_n_fights", "d_cage_min")
    )


def test_every_model_input_can_be_written_as_a_reason(build, league):
    a, b = pairs(league, 1)[0]
    m = build_matchup(build.store, a, b, ON, "Lightweight")
    stored = [
        {k: (None if math.isnan(v) else v) for k, v in side.items()}
        for side in (m.a, m.b)
    ]
    for name in (*FEATURE_NAMES, *ALL_LEVEL_NAMES):
        group = group_of(name)
        for f, o in ((m.a, m.b), tuple(stored)):
            text = detail(group, [(name, 1.0)], f, o)
            assert text and "nan" not in text.lower().split(), name


# --- contributions -----------------------------------------------------------------


def test_contributions_add_up_to_the_symmetric_log_odds(candidate, build, league):
    store = build.store
    for a, b in pairs(league):
        m = build_matchup(store, a, b, ON, "Lightweight")
        x = model_inputs(m, candidate.spec.names)
        toward_a = contributions(candidate, x)
        u = candidate.model.one_sided(matrix(x, candidate.spec))[0]
        v = candidate.model.one_sided(
            swapped(matrix(x, candidate.spec), candidate.spec)
        )[0]
        logit = lambda p: math.log(p / (1 - p))
        assert sum(toward_a.values()) == pytest.approx(
            (logit(u) - logit(v)) / 2, abs=1e-4
        )


# --- the mirror rule ---------------------------------------------------------------


def test_swapping_the_fighters_gives_the_mirrored_summary(candidate, build, league):
    store = build.store
    flipped = {"fighter_a": "fighter_b", "fighter_b": "fighter_a", None: None}
    seen_favourite = 0
    for a, b in pairs(league):
        ab, _, _ = _explained(candidate, store, a, b, ("Ann Alpha", "Bea Beta"))
        ba, _, _ = _explained(candidate, store, b, a, ("Bea Beta", "Ann Alpha"))
        # Every contribution changes sign exactly.
        assert ab["log_odds_toward_a"] == {
            k: -v for k, v in ba["log_odds_toward_a"].items()
        }
        x, y = ab["why"], ba["why"]
        # The same words about the same people, with the sides swapped.
        assert x["summary"] == y["summary"]
        assert x["favoured"] == flipped[y["favoured"]]
        assert len(x["for"]) == len(y["for"])
        for r, s in zip(x["for"], y["for"], strict=True):
            assert r["toward"] == flipped[s["toward"]]
            assert {k: v for k, v in r.items() if k != "toward"} == {
                k: v for k, v in s.items() if k != "toward"
            }
        if x["against"] is None:
            assert y["against"] is None
        else:
            assert x["against"]["toward"] == flipped[y["against"]["toward"]]
            assert x["against"]["detail"] == y["against"]["detail"]
            assert x["against"]["log_odds"] == y["against"]["log_odds"]
        seen_favourite += x["favoured"] is not None
    assert seen_favourite > 0


def test_reasons_are_ranked_and_point_the_right_way(candidate, build, league):
    store = build.store
    for a, b in pairs(league):
        out, _, _ = _explained(candidate, store, a, b, ("Ann Alpha", "Bea Beta"))
        why = out["why"]
        if why["favoured"] is None:
            continue
        sign = 1.0 if why["favoured"] == "fighter_a" else -1.0
        totals = group_totals(out["log_odds_toward_a"])
        toward = sorted(
            (sign * t for t in totals.values() if sign * t > 0), reverse=True
        )
        assert [r["log_odds"] for r in why["for"]] == toward[:3]
        assert all(r["toward"] == why["favoured"] for r in why["for"])
        if why["against"] is not None:
            assert why["against"]["log_odds"] == max(-sign * t for t in totals.values())
            assert why["against"]["toward"] != why["favoured"]
        assert len({r["reason"] for r in why["for"]}) == len(why["for"])


# --- templates and wording ---------------------------------------------------------


def _matchup(a: dict, b: dict) -> Matchup:
    return Matchup("fa", "fb", ON, "Lightweight", {}, {}, {}, a, b, {})


def test_template_values_are_written_from_the_favoured_side():
    a = {"age": 27.6, "n_fights": 6.0, "elo": 1642.0, "opp_elo_career": 1560.0}
    b = {"age": 40.0, "n_fights": 15.0, "elo": 1485.0, "opp_elo_career": 1475.0}
    b["slpm_career"], a["slpm_career"] = 4.61, 3.1
    toward_a = {"d_age": 0.40, "d_n_fights": 0.10, "d_elo": 0.30, "d_slpm_career": -0.2}
    why = explain(toward_a, _matchup(a, b), ("Ann Alpha", "Bea Beta"), 0.71, 0.29)
    assert why["favoured"] == "fighter_a"
    first, second = why["for"]
    assert first["reason"] == "age_experience" and first["component"] == "age"
    assert first["title"] == "the age gap"
    assert first["detail"] == "12.4 years younger (27.6 vs 40.0)"
    assert first["log_odds"] == pytest.approx(0.5)  # the whole reason ranks it
    assert second["component"] == "rating"
    assert second["detail"] == "a 157-point rating edge (1,642 vs 1,485)"
    assert why["against"]["toward"] == "fighter_b"
    assert why["against"]["detail"] == "career: 4.61 vs 3.10"
    assert why["summary"] == (
        "The model favours Ann Alpha (71%). What it weighs most toward Ann Alpha: "
        "(1) the age gap, 12.4 years younger (27.6 vs 40.0); (2) the rating gap, a "
        "157-point rating edge (1,642 vs 1,485). What it weighs most toward Bea "
        "Beta: significant strikes landed per minute, career: 4.61 vs 3.10."
    )


def test_a_grouped_reason_shows_only_its_strongest_component():
    # Much younger and much less experienced. Age carries the reason toward
    # A; experience weighs slightly toward B. The text must not credit A
    # with "19 fewer UFC fights".
    a = {"age": 25.0, "n_fights": 4.0, "elo": 1693.0, "opp_elo_career": 1583.0}
    b = {"age": 37.3, "n_fights": 23.0, "elo": 1850.0, "opp_elo_career": 1664.0}
    toward_a = {
        "d_age": 0.50,
        "d_age_past_32": 0.20,
        "lvl_age_rel_class": 0.05,
        "d_n_fights": -0.10,
        "d_cage_min": -0.05,
        "d_elo": -0.30,
        "d_opp_elo_career": -0.02,
    }
    why = explain(toward_a, _matchup(a, b), ("A", "B"), 0.56, 0.44)
    (reason,) = why["for"]
    assert reason["component"] == "age" and reason["title"] == "the age gap"
    assert reason["detail"] == "12.3 years younger (25.0 vs 37.3)"
    assert "fewer" not in why["summary"] and "UFC fights" not in why["summary"]
    # The reason toward B: own rating carries it, not the opposition.
    assert why["against"]["component"] == "rating"
    assert why["against"]["detail"] == "a 157-point rating edge (1,850 vs 1,693)"
    assert "past opponents" not in why["summary"]

    # When experience carries the reason, it is the part that is shown.
    toward_a = {"d_age": -0.05, "d_n_fights": -0.40, "d_elo": 0.1}
    why = explain(toward_a, _matchup(a, b), ("A", "B"), 0.30, 0.70)
    first = why["for"][0]
    assert first["toward"] == "fighter_b" and first["component"] == "experience"
    assert first["title"] == "UFC experience"
    assert first["detail"] == "19 more UFC fights (23 vs 4)"
    assert "younger" not in why["summary"] and "older" not in why["summary"]

    # Opposition faced alone carries a rating reason.
    toward_a = {"d_elo": -0.05, "d_opp_elo_l3": 0.2, "d_opp_elo_career": 0.1}
    why = explain(toward_a, _matchup(b, a), ("B", "A"), 0.6, 0.4)
    assert why["for"][0]["component"] == "opposition"
    assert why["for"][0]["detail"] == (
        "past opponents rated 81 points higher on average (1,664 vs 1,583)"
    )


def test_every_grouped_reason_names_a_component_that_points_its_way(
    candidate, build, league
):
    """On real model output: the component a reason shows carries weight in
    the reason's direction (never a component the model weighs against the
    fighter) and is the largest one; only reasons with components name one."""
    store = build.store
    seen = 0
    for a, b in pairs(league, n=20):
        out, m, _ = _explained(candidate, store, a, b, ("Ann Alpha", "Bea Beta"))
        toward_a = out["log_odds_toward_a"]
        every = factors(toward_a, m.a, m.b, n=len(TITLES))
        for r in every["fighter_a"] + every["fighter_b"]:
            if r["reason"] not in COMPONENTS:
                assert r["component"] is None
                continue
            seen += 1
            s = 1.0 if r["toward"] == "fighter_a" else -1.0
            parts = {
                k: sum(
                    s * toward_a[n] for n in sorted(r["inputs"]) if parse(n).stem in v
                )
                for k, v in COMPONENTS[r["reason"]].items()
            }
            assert parts[r["component"]] > 0, r
            assert parts[r["component"]] == max(parts.values())
    if not seen:
        pytest.skip("this small fit gives no weight to a reason with components")


def test_missing_values_say_so_and_an_even_bout_favours_nobody():
    a = {"age": math.nan, "n_fights": 0.0, "elo": 1500.0, "opp_elo_career": math.nan}
    b = {"age": 30.0, "n_fights": 4.0, "elo": 1530.0, "opp_elo_career": 1490.0}
    toward_a = {"d_age": 0.2, "d_opp_elo_career": -0.1}
    why = explain(toward_a, _matchup(a, b), ("A", "B"), 0.6, 0.4)
    assert why["for"][0]["detail"] == "age unknown for one fighter"
    assert why["against"]["detail"] == "no past UFC opponents for one fighter"
    # Stored values have None where the matchup had NaN.
    stored = {
        k: None if isinstance(v, float) and math.isnan(v) else v for k, v in a.items()
    }
    assert explain_values(toward_a, stored, b, ("A", "B"), 0.6, 0.4) == why
    even = explain({"d_age": 0.0}, _matchup(a, b), ("A", "B"), 0.5, 0.5)
    assert even["favoured"] is None and even["for"] == [] and even["against"] is None


def test_factors_list_each_sides_reasons_in_the_why_wording():
    a = {"age": 27.6, "n_fights": 6.0, "elo": 1642.0, "opp_elo_career": 1560.0}
    b = {"age": 40.0, "n_fights": 15.0, "elo": 1485.0, "opp_elo_career": 1475.0}
    b["slpm_career"], a["slpm_career"] = 4.61, 3.1
    toward_a = {"d_age": 0.40, "d_n_fights": 0.10, "d_elo": 0.30, "d_slpm_career": -0.2}
    why = explain(toward_a, _matchup(a, b), ("A", "B"), 0.71, 0.29)
    out = factors(toward_a, a, b, n=5)
    assert out["fighter_a"] == why["for"]
    assert out["fighter_b"] == [why["against"]]
    assert factors(toward_a, a, b, n=1)["fighter_a"] == why["for"][:1]
    mirrored = factors({k: -v for k, v in toward_a.items()}, b, a, n=5)
    flip = {"fighter_a": "fighter_b", "fighter_b": "fighter_a"}
    for side in out:
        for r, s in zip(out[side], mirrored[flip[side]], strict=True):
            assert {**r, "toward": flip[r["toward"]]} == s


def test_wording_describes_what_the_model_weighs_never_causes(candidate, build, league):
    store = build.store
    for a, b in pairs(league):
        out, _, _ = _explained(candidate, store, a, b, ("Ann Alpha", "Bea Beta"))
        why = out["why"]
        text = why["summary"]
        assert not CAUSAL.search(text), text
        assert "nan" not in text.lower().split()
        if why["favoured"] is not None:
            assert text.startswith("The model favours ")
            assert "What it weighs most toward" in text
        assert "not what causes a result" in why["method"]


def test_a_favourite_with_no_reason_toward_them_is_said_plainly():
    a, b = {"age": 30.0, "n_fights": 5.0}, {"age": 33.0, "n_fights": 9.0}
    why = explain({"d_age": 0.3}, _matchup(a, b), ("A", "B"), 0.4, 0.6)
    assert why["favoured"] == "fighter_b" and why["for"] == []
    assert "No reason the model weighs points toward B on its own." in why["summary"]
    assert why["against"]["toward"] == "fighter_a"


def test_level_inputs_are_explained_too():
    assert set(LEVEL_NAMES) < set(ALL_LEVEL_NAMES)
    a = {"age": 30.0, "n_fights": 5.0}
    b = {"age": 33.0, "n_fights": 9.0}
    why = explain({"lvl_n_fights": 0.3}, _matchup(a, b), ("A", "B"), 0.6, 0.4)
    assert why["for"][0]["reason"] == "age_experience"
