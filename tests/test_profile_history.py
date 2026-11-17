"""The profile-history comparison, on invented profiles. No network."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "profile_history.py"


def _load():
    spec = importlib.util.spec_from_file_location("profile_history", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules["profile_history"] = loaded
    spec.loader.exec_module(loaded)
    return loaded


history = _load()


def _profile(height=180.0, reach=185.0, stance="Orthodox", dob=date(1990, 1, 1)):
    return {"height_cm": height, "reach_cm": reach, "stance": stance, "dob": dob}


OLD = {
    "same": _profile(),
    "filled": _profile(height=None, reach=None),
    "moved": _profile(dob=date(1990, 1, 1)),
    "emptied": _profile(),
    "gone": _profile(),
}
NEW = {
    "same": _profile(),
    "filled": _profile(),
    "moved": _profile(dob=date(1991, 1, 1)),
    "emptied": _profile(stance=None),
    "added": _profile(),
}


def test_compare_counts_each_kind_of_change() -> None:
    result = history.compare(OLD, NEW)
    assert result["profiles_in_both"] == 4
    assert result["profiles_only_in_newer"] == 1
    assert result["any_attribute_differs"] == 3
    assert result["attributes"]["height_cm"] == {
        "same": 3,
        "empty_in_both": 0,
        "filled_later": 1,
        "emptied": 0,
        "value_changed": 0,
    }
    assert result["attributes"]["dob"]["value_changed"] == 1
    assert result["attributes"]["stance"]["emptied"] == 1


def test_compare_can_be_limited_to_some_fighters() -> None:
    result = history.compare(OLD, NEW, only={"same", "filled", "added"})
    assert result["profiles_in_both"] == 2
    assert result["profiles_only_in_newer"] == 1
    assert result["any_attribute_differs"] == 1


def test_change_state() -> None:
    assert history.change_state("added", OLD, NEW) == "absent"
    assert history.change_state("same", OLD, NEW) == "same"
    assert history.change_state("filled", OLD, NEW) == "changed"
    assert history.change_state("filled", OLD, NEW, "height_cm") == "filled"
    assert history.change_state("filled", OLD, NEW, "dob") == "same"
    assert history.change_state("moved", OLD, NEW, "dob") == "changed"


def test_dob_changes_measure_how_far_dates_moved() -> None:
    result = history.dob_changes(OLD, NEW)
    assert result["n"] == 1
    assert 0.99 < result["median_years"] < 1.01
    assert result["under_one_year"] == 1
    assert history.dob_changes({"same": _profile()}, {"same": _profile()}) == {"n": 0}


def test_read_profiles_parses_and_keeps_the_last_row(tmp_path: Path) -> None:
    fighter = "0123456789abcdef"
    url = f"http://ufcstats.com/fighter-details/{fighter}"
    path = tmp_path / "ufc_fighter_tott.csv"
    path.write_text(
        "FIGHTER,HEIGHT,WEIGHT,REACH,STANCE,DOB,URL\n"
        f"Invented Person,--,155 lbs.,--,,--,{url}\n"
        f'Invented Person,"5\' 10""",155 lbs.,"70""",Southpaw,"Jan 02, 1990",{url}\n',
        encoding="utf-8",
    )
    profiles = history.read_profiles(path)
    assert list(profiles) == [fighter]
    assert profiles[fighter]["dob"] == date(1990, 1, 2)
    assert profiles[fighter]["stance"] is not None
    assert round(profiles[fighter]["height_cm"], 1) == 177.8
    assert round(profiles[fighter]["reach_cm"], 1) == 177.8
