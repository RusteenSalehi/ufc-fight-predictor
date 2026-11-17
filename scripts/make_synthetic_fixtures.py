"""Write the synthetic dataset fixtures under tests/fixtures/greco/.

Usage:
    uv run python scripts/make_synthetic_fixtures.py [FIXTURES_DIR]

Nothing here is copied from the real dataset. Fighters, events, referees and
numbers are invented; only the six files' column layout and the *kinds* of
rows (draws, no contests, renamed events, shared names, missing attributes,
early events without stats, ...) mirror the published CSVs. The output is
deterministic: the same script always writes the same bytes.

IDs are readable on purpose, so a test can say what it means:
    e0..  events        f1..  fighters        b0..  fights
"""

from __future__ import annotations

import csv
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from ufc.ingest import source

SNAPSHOT_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SNAPSHOT_DATE = "2026-11-15T18:03:06Z"


def event_id(n: int) -> str:
    return f"e{n:015x}"


def fighter_id(n: int) -> str:
    return f"f1{n:014x}"


def fight_id(n: int) -> str:
    return f"b0{n:014x}"


# --- events ----------------------------------------------------------------

# key -> (name, date, location). Dates fall on or before the snapshot date.
EVENTS: dict[str, tuple[str, date, str]] = {
    "e01": ("UFC 900: Valcourt vs. Hollin", date(2026, 11, 14), "Lakeside, Ohio, USA"),
    "e02": (
        "UFC Fight Night: Dufort vs. Garrow",
        date(2026, 10, 17),
        "Calder, Alberta, Canada",
    ),
    "e03": (
        "UFC Fight Night: Atlas vs. Beksul",
        date(2026, 9, 26),
        "Harbor City, Nevada, USA",
    ),
    "e04": (
        "Noche UFC: Marlow vs. Tellin",
        date(2026, 9, 12),
        "Mesa Verde, Arizona, USA",
    ),
    "e05": (
        "UFC Fight Night: Pemberton vs. Montero",
        date(2026, 8, 8),
        "Harbor City, Nevada, USA",
    ),
    "e06": (
        "UFC Fight Night: Karpov vs. Fairweather",
        date(2026, 7, 25),
        "Qasr Dunes, Dunya, Emirates",
    ),
    "e07": ("UFC Liberty 300", date(2026, 6, 14), "Capital City, DC, USA"),
    "e08": (
        "UFC Fight Night: Larkin vs. Chavarria",
        date(2026, 6, 6),
        "Harbor City, Nevada, USA",
    ),
    "e09": (
        "UFC Fight Night: Kovalenko vs. Tellin",
        date(2026, 2, 28),
        "Valle Alto, Distrito Central, Mexico",
    ),
    "e10": ("Noche UFC: Lorca vs. Pruitt", date(2025, 9, 13), "Alamo Park, Texas, USA"),
    "e11": (
        "UFC 850: Lemaitre vs. Larkin",
        date(2025, 5, 10),
        "Fleuve, Quebec, Canada",
    ),
    "e12": (
        "UFC 840: Aslanov vs. Cervera",
        date(2024, 10, 26),
        "Qasr Dunes, Dunya, Emirates",
    ),
    "e13": (
        "UFC Fight Night: Dorsey vs. Pruitt",
        date(2024, 7, 13),
        "Highmount, Colorado, USA",
    ),
    "e14": (
        "Noche UFC: Calderon vs. Ostrova 2",
        date(2023, 9, 16),
        "Harbor City, Nevada, USA",
    ),
    "e15": (
        "UFC 700: Tahir vs. Larkin",
        date(2019, 10, 5),
        "Port Marlow, Victoria, Australia",
    ),
    "e16": ("UFC - Ultimate Osaka", date(1997, 12, 21), "Osaka, Honshu, Japan"),
    "e17": (
        "UFC 7: The Riverfront Brawl",
        date(1995, 9, 8),
        "Brookford, New York, USA",
    ),
    "e18": (
        "UFC 4: Return of the Warriors",
        date(1994, 12, 16),
        "Tulsa Flats, Oklahoma, USA",
    ),
    "e19": ("UFC 2: No Exit", date(1994, 3, 11), "Denver Heights, Colorado, USA"),
}
EVENT_IDS = {key: event_id(i) for i, key in enumerate(EVENTS, start=1)}
# A fight listed under a name that the events file no longer uses.
OLD_EVENT_NAMES = {
    "e04": "UFC Fight Night: Lorca vs. Pruitt",
    "e14": "UFC Fight Night: Calderon vs. Ostrova 2",
}
ORPHAN_EVENT = "UFC - Road to UFC 9.9"


# --- fighters --------------------------------------------------------------


@dataclass
class Fighter:
    first: str  # as listed in the details file
    last: str
    nickname: str
    height: str  # "--" when unknown
    weight: str
    reach: str
    stance: str
    dob: str  # "--" when unknown
    tott_name: str | None = None  # the name recorded in the tale of the tape
    in_details: bool = True
    in_tott: bool = True
    bout_name: str | None = None  # how fight rows spell it, if not "first last"

    @property
    def name(self) -> str:
        return f"{self.first} {self.last}"


F = Fighter
FIGHTERS: dict[str, Fighter] = {
    # Complete profiles.
    "valcourt": F(
        "Orin",
        "Valcourt",
        "Lightning",
        "6' 4\"",
        "245 lbs.",
        '81"',
        "Orthodox",
        "Feb 03, 1991",
    ),
    "hollin": F(
        "Dax", "Hollin", "", "6' 1\"", "238 lbs.", '73"', "Orthodox", "Nov 12, 1997"
    ),
    "kessler": F(
        "Mira", "Kessler", "", "5' 8\"", "135 lbs.", '66"', "Southpaw", "Jul 02, 1990"
    ),
    "arnault": F(
        "Tess",
        "Arnault",
        "The Falcon",
        "5' 8\"",
        "135 lbs.",
        '69"',
        "Orthodox",
        "May 30, 1988",
    ),
    "brandt": F(
        "Pavel",
        "Brandt",
        "The Surgeon",
        "6' 1\"",
        "170 lbs.",
        '71"',
        "Southpaw",
        "Apr 25, 1993",
    ),
    "doyle": F(
        "Kane",
        "Doyle",
        "Wildcard",
        "6' 3\"",
        "170 lbs.",
        '81"',
        "Orthodox",
        "Nov 05, 1992",
    ),
    "corrin": F(
        "Lyle",
        "Corrin",
        "Vise",
        "5' 8\"",
        "185 lbs.",
        '74"',
        "Southpaw",
        "Jan 25, 1999",
    ),
    "reyes": F(
        "Tomas",
        "Reyes",
        "Chispa",
        "6' 1\"",
        "185 lbs.",
        '74"',
        "Orthodox",
        "Mar 28, 1988",
    ),
    "dellamore": F(
        "Jared", "Dellamore", "", "5' 11\"", "170 lbs.", '73"', "Switch", "Sep 10, 1996"
    ),
    "lorca": F(
        "Dario", "Lorca", "", "5' 11\"", "145 lbs.", '72"', "Orthodox", "Dec 30, 1994"
    ),
    "ransom": F(
        "Nate",
        "Ransom",
        "Savage",
        "6' 4\"",
        "185 lbs.",
        '77"',
        "Switch",
        "Mar 24, 1994",
        tott_name="Nathan Ransom",
    ),
    "delaney": F(
        "Marquis",
        "Delaney",
        "The Reaper",
        "6' 2\"",
        "185 lbs.",
        '79"',
        "Orthodox",
        "Aug 06, 1995",
    ),
    "oduya": F(
        "Rafe", "Oduya", "", "5' 9\"", "170 lbs.", '71"', "Orthodox", "Dec 11, 1997"
    ),
    "volkan": F(
        "Dmitri", "Volkan", "", "5' 11\"", "170 lbs.", '71"', "Orthodox", "Aug 04, 2001"
    ),
    "calderon": F(
        "Ines", "Calderon", "", "5' 5\"", "125 lbs.", '66"', "Orthodox", "Aug 09, 1993"
    ),
    "ostrova": F(
        "Valeria",
        "Ostrova",
        "Comet",
        "5' 5\"",
        "125 lbs.",
        '66"',
        "Southpaw",
        "Mar 07, 1988",
    ),
    "chavarria": F(
        "Edwin",
        "Chavarria",
        "Chicali",
        "5' 7\"",
        "125 lbs.",
        '71"',
        "Orthodox",
        "Jan 27, 1996",
    ),
    "leite": F(
        "Danilo",
        "Leite",
        "Noodle",
        "5' 6\"",
        "125 lbs.",
        '70"',
        "Switch",
        "Jun 05, 1996",
    ),
    "karpov": F(
        "Ruslan", "Karpov", "", "6' 4\"", "240 lbs.", '76"', "Orthodox", "Mar 10, 1992"
    ),
    "pereda": F(
        "Aleks",
        "Pereda",
        "Wolf",
        "6' 4\"",
        "205 lbs.",
        '79"',
        "Orthodox",
        "Jul 07, 1987",
    ),
    "kovalenko": F(
        "Edita",
        "Kovalenko",
        "Heavy-Handed",
        "5' 9\"",
        "125 lbs.",
        '71"',
        "Orthodox",
        "Jul 05, 1998",
    ),
    "lemaitre": F(
        "Marc",
        "Lemaitre",
        "Powerhouse",
        "6' 1\"",
        "185 lbs.",
        '74"',
        "Orthodox",
        "Feb 18, 1990",
    ),
    "aslanov": F(
        "Ilker",
        "Aslanov",
        "The Last Sultan",
        "6' 3\"",
        "205 lbs.",
        '77"',
        "Orthodox",
        "Apr 23, 1996",
    ),
    "dorsey": F(
        "Dean", "Dorsey", "", "5' 8\"", "155 lbs.", '70"', "Southpaw", "Oct 19, 1988"
    ),
    "tahir": F(
        "Karim",
        "Tahir",
        "The Warrior",
        "5' 5\"",
        "135 lbs.",
        '69"',
        "Orthodox",
        "Feb 15, 1992",
    ),
    "brandon": F(
        "Kurt", "Brandon", "", "6' 1\"", "205 lbs.", '72"', "Orthodox", "Feb 11, 1964"
    ),
    "tarasov": F(
        "Igor",
        "Tarasov",
        "The Bear",
        "6' 0\"",
        "210 lbs.",
        "--",
        "Orthodox",
        "Aug 26, 1967",
    ),
    "ferrante": F(
        "Hugo", "Ferrante", "", "6' 1\"", "175 lbs.", "--", "Southpaw", "Dec 12, 1966"
    ),
    "sullivan": F(
        "Pat", "Sullivan", "", "6' 2\"", "225 lbs.", "--", "Orthodox", "Aug 28, 1963"
    ),
    "pemberton": F(
        "Gus",
        "Pemberton",
        "Vanilla",
        "6' 3\"",
        "264 lbs.",
        '76"',
        "Orthodox",
        "Feb 08, 1994",
    ),
    "marlow": F(
        "KJ", "Marlow", "", "5' 5\"", "125 lbs.", '67"', "Southpaw", "Sep 29, 1992"
    ),
    "dufort": F("Remy", "Dufort", "", "--", "185 lbs.", "--", "", "Jan 21, 1992"),
    "wei": F("Li", "Mei", "", "5' 2\"", "115 lbs.", '60"', "Orthodox", "Nov 14, 1994"),
    "ribeiro": F(
        "Bianca", "Ribeiro", "", "5' 6\"", "115 lbs.", '65"', "Orthodox", "Sep 07, 1993"
    ),
    # Reach unknown.
    "sakamoto": F(
        "Kenji",
        "Sakamoto",
        "The Hunter",
        "6' 0\"",
        "185 lbs.",
        "--",
        "Southpaw",
        "Jul 14, 1970",
    ),
    "teixeira": F(
        "Marcos", "Teixeira", "Conan", "6' 3\"", "242 lbs.", "--", "Orthodox", "--"
    ),
    # Date of birth unknown (and reach).
    "rosen": F("Barney", "Rosen", "", "6' 4\"", "275 lbs.", "--", "Orthodox", "--"),
    "carlisle": F(
        "Jeb", "Carlisle", "The Brawler", "6' 1\"", "260 lbs.", "--", "Southpaw", "--"
    ),
    "mercer": F("Sam", "Mercer", "", "5' 10\"", "210 lbs.", "--", "Orthodox", "--"),
    "dougherty": F(
        "Sid", "Dougherty", "", "6' 0\"", "175 lbs.", "--", "", "Dec 04, 1975"
    ),
    # Everything missing but the date of birth; appears in no bout.
    "abbott": F("Todd", "Abbott", "", "--", "155 lbs.", "--", "", "Jul 13, 1978"),
    # Name with trailing whitespace in the tale of the tape; no reach or birth date; no bout.
    "tomlin": F(
        "Jules",
        "Tomlin",
        "The Nice Guy",
        "5' 10\"",
        "170 lbs.",
        "--",
        "",
        "--",
        tott_name="Jules Tomlin ",
    ),
    # Only in the tale-of-the-tape file; no bout.
    "linh": F(
        "Mai",
        "Linh",
        "",
        "--",
        "115 lbs.",
        "--",
        "",
        "Nov 22, 1996",
        tott_name="Mai Linh",
        in_details=False,
    ),
    # Different names in the two fighter files that share a word: the same person.
    "tashev": F(
        "Rahim",
        "Tashev",
        "The Ineffable",
        "--",
        "135 lbs.",
        "--",
        "",
        "Aug 05, 2004",
        tott_name="Rahimjon Tashev",
    ),
    "beksul": F(
        "Ilmar",
        "Beksul",
        "",
        "--",
        "135 lbs.",
        "--",
        "",
        "Sep 01, 2003",
        tott_name="Ilmar Beksul Uulu",
    ),
    "tellin": F(
        "Rosa",
        "Tellin",
        "Kill Bill",
        "--",
        "135 lbs.",
        "--",
        "",
        "Oct 19, 2004",
        tott_name="Rosa Valdez Tellin",
    ),
    # A woman's profile whose stored attributes were recorded under another
    # name entirely (the two names share no word): attributes are dropped.
    "fairweather": F(
        "Tyron",
        "Fairweather",
        "",
        "--",
        "125 lbs.",
        "--",
        "",
        "Oct 19, 2004",
        tott_name="Marta Quispe",
    ),
    # Two profiles whose names are crossed between the two files.
    "montero": F(
        "Jorge",
        "Montero",
        "",
        "--",
        "265 lbs.",
        "--",
        "",
        "Aug 25, 1996",
        tott_name="Hector Da Costa",
    ),
    "dacosta": F(
        "Hector",
        "Da Costa",
        "Montanha",
        "6' 3\"",
        "240 lbs.",
        "--",
        "",
        "Mar 08, 1985",
        tott_name="Jorge Montero",
    ),
    # Shared names. Larkin: 125 lbs and 185 lbs. Pruitt: 145 lbs and 160 lbs.
    "larkin_fly": F(
        "Beau",
        "Larkin",
        "Bulldog",
        "5' 4\"",
        "125 lbs.",
        '65"',
        "Orthodox",
        "Mar 16, 1990",
    ),
    "larkin_mid": F(
        "Beau",
        "Larkin",
        "Anchor",
        "6' 0\"",
        "185 lbs.",
        '74"',
        "Orthodox",
        "Jul 13, 1989",
    ),
    "pruitt_fw": F(
        "Jonas",
        "Pruitt",
        "Lord",
        "5' 7\"",
        "145 lbs.",
        '69"',
        "Orthodox",
        "Dec 27, 1996",
    ),
    "pruitt_old": F(
        "Jonas",
        "Pruitt",
        "White Bear",
        "5' 6\"",
        "160 lbs.",
        "--",
        "Orthodox",
        "Oct 08, 1977",
    ),
    # Names the bout rows spell differently from the profile.
    "cervera": F(
        "Anselmo",
        "Cervera",
        "The Lion",
        "6' 3\"",
        "205 lbs.",
        '76"',
        "Southpaw",
        "Mar 15, 1990",
        bout_name="Anselmo Cerveira",
    ),
    "garrow": F(
        "Nicholas",
        "Garrow",
        "",
        "--",
        "185 lbs.",
        "--",
        "",
        "Jun 18, 1998",
        bout_name="Nick Garrow",
    ),
}
FIGHTER_IDS = {key: fighter_id(i) for i, key in enumerate(FIGHTERS, start=1)}


def bout_name(key: str) -> str:
    fighter = FIGHTERS[key]
    return fighter.bout_name or fighter.name


# --- fights ----------------------------------------------------------------

STATS_FULL = "full"  # round stats with control time
STATS_NO_CTRL = "no_ctrl"  # round stats, control time recorded as "--"
STATS_NONE = "none"  # blank rows
STATS_SHARED = "shared"  # one stats block for two fights with the same bout text


@dataclass
class Fight:
    key: str
    event: str
    a: str
    b: str
    outcome: str  # as in the results file
    weight_class: str
    method: str
    end_round: int
    end_time: str
    time_format: str
    referee: str
    details: str
    stats: str = STATS_FULL
    listed_under_old_name: bool = False
    in_events_file: bool = True


T3 = "3 Rnd (5-5-5)"
T5 = "5 Rnd (5-5-5-5-5)"
T3OT = "3 Rnd + OT (5-5-5-5)"
NTL = "No Time Limit"

# Judges' scores are invented too.
FIGHTS: list[Fight] = [
    Fight(
        "title",
        "e01",
        "valcourt",
        "hollin",
        "W/L",
        "UFC Heavyweight Title Bout",
        "Decision - Unanimous ",
        5,
        "5:00",
        T5,
        "Alan Mercer",
        "Judge One 48 - 47.Judge Two 48 - 47.Judge Three 49 - 46.",
    ),
    Fight(
        "womens_title",
        "e01",
        "kessler",
        "arnault",
        "W/L",
        "UFC Women's Bantamweight Title Bout",
        "Submission ",
        4,
        "1:29",
        T5,
        "Beth Quill",
        "Rear Naked Choke ",
    ),
    Fight(
        "majority_draw",
        "e01",
        "brandt",
        "doyle",
        "D/D",
        "Welterweight Bout",
        "Decision - Majority ",
        3,
        "5:00",
        T3,
        "Alan Mercer",
        "Judge One 27 - 29.Judge Two 28 - 28.Judge Three 28 - 28.",
    ),
    Fight(
        "catch_weight",
        "e01",
        "corrin",
        "reyes",
        "L/W",
        "Catch Weight Bout",
        "KO/TKO ",
        1,
        "3:20",
        T3,
        "Carl Dunmore",
        "Punch to Head At Distance ",
    ),
    Fight(
        "shortened_name",
        "e02",
        "dufort",
        "garrow",
        "L/W",
        "Middleweight Bout",
        "Submission ",
        1,
        "1:26",
        T3,
        "Vera Holloway",
        "Neck Crank From Back Control ",
    ),
    Fight(
        "dq",
        "e03",
        "tashev",
        "beksul",
        "L/W",
        "Bantamweight Bout",
        "DQ ",
        1,
        "2:19",
        T3OT,
        "Keith Marrow",
        "Illegal Knee to downed opponent",
    ),
    Fight(
        "womens_flyweight",
        "e04",
        "marlow",
        "tellin",
        "L/W",
        "Women's Flyweight Bout",
        "Decision - Unanimous ",
        3,
        "5:00",
        T3,
        "Gail Burnside",
        "Judge Four 28 - 29.Judge Two 28 - 29.Judge Five 28 - 29.",
    ),
    Fight(
        "two_heavyweights",
        "e05",
        "pemberton",
        "montero",
        "L/W",
        "Heavyweight Bout",
        "Submission ",
        1,
        "1:50",
        T3,
        "Tom Ledger",
        "Neck Crank From Back Control ",
    ),
    Fight(
        "shared_featherweight",
        "e04",
        "lorca",
        "pruitt_fw",
        "W/L",
        "Featherweight Bout",
        "KO/TKO ",
        2,
        "4:48",
        T5,
        "Mike Pelham",
        "Punches to Head On Ground ",
        listed_under_old_name=True,
    ),
    Fight(
        "nc_could_not_continue",
        "e04",
        "ransom",
        "delaney",
        "NC/NC",
        "Middleweight Bout",
        "Could Not Continue ",
        1,
        "0:51",
        T3,
        "Mike Pelham",
        "Kick to Groin by Ransom",
        listed_under_old_name=True,
    ),
    Fight(
        "tuf_final",
        "e04",
        "oduya",
        "volkan",
        "L/W",
        "Ultimate Fighter 99 Welterweight Tournament Title Bout",
        "KO/TKO ",
        1,
        "4:27",
        T3OT,
        "Keith Marrow",
        "Punches to Head At Distance ",
        listed_under_old_name=True,
    ),
    Fight(
        "split_draw",
        "e14",
        "calderon",
        "ostrova",
        "D/D",
        "UFC Women's Flyweight Title Bout",
        "Decision - Split ",
        5,
        "5:00",
        T5,
        "Hal Dorrance",
        "Judge Six 47 - 48.Judge Two 48 - 47.Judge Three 47 - 47.",
        listed_under_old_name=True,
    ),
    Fight(
        "split_second_wins",
        "e14",
        "doyle",
        "dellamore",
        "L/W",
        "Welterweight Bout",
        "Decision - Split ",
        3,
        "5:00",
        T3,
        "Jay Hertz",
        "Judge Seven 28 - 29.Judge Two 29 - 28.Judge Three 28 - 29.",
        listed_under_old_name=True,
    ),
    Fight(
        "nc_overturned",
        "e14",
        "chavarria",
        "leite",
        "NC/NC",
        "Flyweight Bout",
        "Overturned ",
        1,
        "3:47",
        T3,
        "Tom Ledger",
        "Guillotine win for Chavarria overturned on replay",
        listed_under_old_name=True,
    ),
    Fight(
        "conflicting_profile",
        "e06",
        "karpov",
        "fairweather",
        "W/L",
        "Heavyweight Bout",
        "KO/TKO ",
        3,
        "1:12",
        T3,
        "Jim Parrish",
        "Knee to Head In Clinch ",
    ),
    Fight(
        "interim_title",
        "e07",
        "pereda",
        "valcourt",
        "L/W",
        "UFC Interim Heavyweight Title Bout",
        "KO/TKO ",
        2,
        "1:27",
        T5,
        "Hal Dorrance",
        "Punches to Head At Distance ",
    ),
    Fight(
        "flyweight_shared_name",
        "e08",
        "larkin_fly",
        "chavarria",
        "L/W",
        "Flyweight Bout",
        "Submission ",
        1,
        "4:13",
        T3,
        "Tom Ledger",
        "Rear Naked Choke ",
    ),
    Fight(
        "womens_catch_weight",
        "e09",
        "tellin",
        "kovalenko",
        "W/L",
        "Catch Weight Bout",
        "Decision - Unanimous ",
        3,
        "5:00",
        T3,
        "Rita Pascoe",
        "Judge Eight 27 - 30.Judge Nine 27 - 30.Judge Ten 28 - 29.",
    ),
    Fight(
        "orphan",
        "e99",
        "wei",
        "ribeiro",
        "L/W",
        "Women's Strawweight Bout",
        "Decision - Unanimous ",
        3,
        "5:00",
        T3,
        "Alan Mercer",
        "Judge One 27 - 30.Judge Two 27 - 30.Judge Three 27 - 30.",
        in_events_file=False,
    ),
    Fight(
        "middleweight_shared_name",
        "e11",
        "lemaitre",
        "larkin_mid",
        "W/L",
        "Middleweight Bout",
        "KO/TKO ",
        1,
        "1:27",
        T3,
        "Luc Barbier",
        "Elbows to Head In Clinch ",
    ),
    Fight(
        "misspelled_name",
        "e12",
        "aslanov",
        "cervera",
        "W/L",
        "Light Heavyweight Bout",
        "KO/TKO ",
        1,
        "0:51",
        T3,
        "Alan Mercer",
        "Punches to Head At Distance ",
    ),
    Fight(
        "doctor_stoppage",
        "e13",
        "dorsey",
        "pruitt_fw",
        "L/W",
        "Lightweight Bout",
        "TKO - Doctor's Stoppage ",
        3,
        "1:28",
        T3,
        "Hal Dorrance",
        "Cut to Dorsey's eye",
    ),
    Fight(
        "nc_drug_test",
        "e15",
        "tahir",
        "larkin_fly",
        "NC/NC",
        "Bantamweight Bout",
        "Overturned ",
        3,
        "3:00",
        T3,
        "Greg Kessel",
        "Failed Drug Test by Tahir",
    ),
    Fight(
        "pair_win",
        "e16",
        "sakamoto",
        "teixeira",
        "W/L",
        "Osaka Heavyweight Tournament Title Bout",
        "Submission ",
        1,
        "3:44",
        "1 Rnd + OT (12-3)",
        "John Marsh",
        "Armbar From Side Control ",
        stats=STATS_SHARED,
    ),
    Fight(
        "pair_nc",
        "e16",
        "sakamoto",
        "teixeira",
        "NC/NC",
        "Heavyweight Bout",
        "Overturned ",
        1,
        "1:51",
        "1 Rnd + OT (12-3)",
        "John Marsh",
        "Early Stoppage",
        stats=STATS_SHARED,
    ),
    Fight(
        "superfight_draw",
        "e17",
        "brandon",
        "tarasov",
        "D/D",
        "UFC Superfight Championship Bout",
        "Other ",
        2,
        "3:00",
        "1 Rnd + OT (30-3)",
        "John Marsh",
        "Time Expired",
        stats=STATS_NO_CTRL,
    ),
    Fight(
        "no_stats",
        "e18",
        "carlisle",
        "rosen",
        "W/L",
        "Open Weight Bout",
        "Submission ",
        1,
        "0:14",
        NTL,
        "John Marsh",
        "Armbar ",
        stats=STATS_NONE,
    ),
    Fight(
        "early_tournament_final",
        "e19",
        "ferrante",
        "sullivan",
        "W/L",
        "UFC 2 Tournament Title Bout",
        "KO/TKO ",
        1,
        "1:17",
        NTL,
        "John Marsh",
        "Punches to Head From MountSubmission to Strikes",
        stats=STATS_NO_CTRL,
    ),
    Fight(
        "early_no_ctrl",
        "e19",
        "mercer",
        "dougherty",
        "W/L",
        "Open Weight Bout",
        "Submission ",
        1,
        "0:20",
        NTL,
        "John Marsh",
        "Guillotine Choke From Mount ",
        stats=STATS_NO_CTRL,
    ),
]
FIGHT_IDS = {fight.key: fight_id(i) for i, fight in enumerate(FIGHTS, start=1)}

# Entries for tests/fixtures/greco/identity_overrides.csv: (name, fight key or
# "", fighter key, note). Every note cites evidence inside this fixture.
OVERRIDES = [
    (
        "Anselmo Cerveira",
        "",
        "cervera",
        "Bout rows spell the surname Cerveira; the only similar profile is Anselmo Cervera (205 lbs), and the bout is light heavyweight.",
    ),
    (
        "Nick Garrow",
        "",
        "garrow",
        "Only profile with this surname is Nicholas Garrow (185 lbs); the bout is middleweight.",
    ),
    (
        "Beau Larkin",
        "nc_drug_test",
        "larkin_fly",
        "Bantamweight bout on 2019-10-05. Two profiles share the name: 125 lbs and 185 lbs. Bantamweight is one class above the 125 lbs profile and 50 lbs below the other, whose first bout is in 2025.",
    ),
    (
        "Jonas Pruitt",
        "doctor_stoppage",
        "pruitt_fw",
        "Lightweight bout on 2024-07-13. Two profiles share the name: 145 lbs born 1996 with featherweight bouts either side of this date, and 160 lbs born 1977 with no other bouts.",
    ),
]


# --- statistics ------------------------------------------------------------

STAT_COLUMNS = [
    "EVENT", "BOUT", "ROUND", "FIGHTER", "KD", "SIG.STR.", "SIG.STR. %", "TOTAL STR.",
    "TD", "TD %", "SUB.ATT", "REV.", "CTRL", "HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND",
]  # fmt: skip


def split_total(total: int, caps: list[int], rng: random.Random) -> list[int]:
    """Spread ``total`` over parts, each at most its cap (sum of caps >= total)."""
    parts = [0] * len(caps)
    remaining = total
    while remaining:
        i = rng.randrange(len(caps))
        if parts[i] < caps[i]:
            parts[i] += 1
            remaining -= 1
    return parts


def split_free(total: int, n: int, rng: random.Random) -> list[int]:
    cuts = sorted(rng.randint(0, total) for _ in range(n - 1))
    return [b - a for a, b in zip([0, *cuts], [*cuts, total])]


def of(landed: int, attempted: int) -> str:
    return f"{landed} of {attempted}"


def pct(landed: int, attempted: int) -> str:
    return "---" if attempted == 0 else f"{round(100 * landed / attempted)}%"


def clock(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


@dataclass
class RoundLine:
    """One fighter's numbers for one round, before formatting."""

    kd: int
    sig: tuple[int, int]
    total: tuple[int, int]
    td: tuple[int, int]
    sub: int
    rev: int
    ctrl: int | None
    target: list[tuple[int, int]] = field(default_factory=list)  # head, body, leg
    position: list[tuple[int, int]] = field(
        default_factory=list
    )  # distance, clinch, ground


def random_line(rng: random.Random, length: int, with_ctrl: bool) -> RoundLine:
    sig_att = rng.randint(0, max(1, length // 5))
    sig_land = rng.randint(0, sig_att)
    total_att = sig_att + rng.randint(0, 6)
    total_land = min(total_att, sig_land + rng.randint(0, 4))
    td_att = rng.choice([0, 0, 0, 1, 1, 2, 4])
    td_land = rng.randint(0, td_att)
    target_att = split_free(sig_att, 3, rng)
    target_land = split_total(sig_land, target_att, rng)
    position_att = split_free(sig_att, 3, rng)
    position_land = split_total(sig_land, position_att, rng)
    return RoundLine(
        kd=rng.choice([0, 0, 0, 0, 1]),
        sig=(sig_land, sig_att),
        total=(total_land, total_att),
        td=(td_land, td_att),
        sub=rng.choice([0, 0, 0, 1]),
        rev=rng.choice([0, 0, 0, 0, 1]),
        ctrl=rng.randint(0, min(length, 150)) if with_ctrl else None,
        target=list(zip(target_land, target_att)),
        position=list(zip(position_land, position_att)),
    )


def fixed_line(sig, ctrl, target, position, total, td, kd=0, sub=0, rev=0) -> RoundLine:
    return RoundLine(kd, sig, total, td, sub, rev, ctrl, target, position)


# Hand-written rounds for the title fight, so that tests can state exact
# expectations. Each tuple holds the numbers for (Valcourt, Hollin).
TITLE_ROUNDS = [
    (
        fixed_line(
            (30, 52),
            3,
            [(24, 44), (5, 6), (1, 2)],
            [(28, 49), (2, 3), (0, 0)],
            (31, 54),
            (0, 0),
        ),
        fixed_line(
            (12, 31),
            0,
            [(8, 24), (3, 5), (1, 2)],
            [(10, 27), (2, 4), (0, 0)],
            (14, 34),
            (1, 2),
        ),
    ),
    (
        fixed_line(
            (44, 70),
            0,
            [(36, 60), (6, 8), (2, 2)],
            [(40, 64), (4, 6), (0, 0)],
            (47, 75),
            (0, 1),
        ),
        fixed_line(
            (15, 38),
            21,
            [(10, 30), (4, 6), (1, 2)],
            [(13, 33), (2, 5), (0, 0)],
            (16, 40),
            (0, 0),
        ),
    ),
    (
        fixed_line(
            (25, 48),
            0,
            [(20, 40), (4, 6), (1, 2)],
            [(22, 43), (3, 5), (0, 0)],
            (26, 50),
            (0, 0),
        ),
        fixed_line(
            (18, 39),
            0,
            [(14, 31), (3, 6), (1, 2)],
            [(16, 34), (2, 5), (0, 0)],
            (19, 41),
            (0, 0),
        ),
    ),
    (
        fixed_line(
            (33, 55),
            1,
            [(27, 46), (5, 8), (1, 1)],
            [(30, 51), (3, 4), (0, 0)],
            (34, 57),
            (0, 0),
        ),
        fixed_line(
            (20, 41),
            0,
            [(15, 33), (4, 6), (1, 2)],
            [(18, 37), (2, 4), (0, 0)],
            (21, 43),
            (0, 0),
        ),
    ),
    (
        fixed_line(
            (38, 66),
            0,
            [(30, 54), (6, 10), (2, 2)],
            [(34, 60), (4, 6), (0, 0)],
            (40, 69),
            (0, 0),
        ),
        fixed_line(
            (16, 42),
            0,
            [(11, 33), (4, 7), (1, 2)],
            [(14, 36), (2, 6), (0, 0)],
            (17, 44),
            (0, 0),
        ),
    ),
]


# Rounds fixed by hand, by fight key. Fights not listed get seeded random rounds.
PINNED_ROUNDS = {
    "title": TITLE_ROUNDS,
    "early_no_ctrl": [
        (
            fixed_line(
                (1, 1),
                None,
                [(1, 1), (0, 0), (0, 0)],
                [(0, 0), (1, 1), (0, 0)],
                (2, 2),
                (1, 1),
                sub=1,
            ),
            fixed_line(
                (0, 4),
                None,
                [(0, 2), (0, 0), (0, 2)],
                [(0, 3), (0, 1), (0, 0)],
                (1, 5),
                (0, 0),
            ),
        ),
    ],
}


def round_lengths(end_round: int, end_time: str) -> list[int]:
    minutes, seconds = end_time.split(":")
    last = int(minutes) * 60 + int(seconds)
    return [300] * (end_round - 1) + [last]


def stat_rows(fight: Fight, event_name: str, rng: random.Random) -> list[list[str]]:
    bout = f"{bout_name(fight.a)} vs. {bout_name(fight.b)}"
    if fight.stats == STATS_NONE:
        return [[event_name, bout] + [""] * 17 for _ in range(2)]
    with_ctrl = fight.stats == STATS_FULL
    rows = []
    # Older events print kills as "0.0"; the newest card prints "0".
    float_style = fight.event != "e01"
    for number, length in enumerate(round_lengths(fight.end_round, fight.end_time), 1):
        for side, key in enumerate((fight.a, fight.b)):
            if fight.key in PINNED_ROUNDS:
                line = PINNED_ROUNDS[fight.key][number - 1][side]
            else:
                line = random_line(rng, length, with_ctrl)
            count = (lambda n: f"{n}.0") if float_style else str
            rows.append(
                [event_name, bout, f"Round {number}", bout_name(key)]
                + [count(line.kd), of(*line.sig), pct(*line.sig), of(*line.total)]
                + [of(*line.td), pct(*line.td), count(line.sub), count(line.rev)]
                + ["--" if line.ctrl is None else clock(line.ctrl)]
                + [of(*part) for part in line.target + line.position]
            )
    return rows


# --- writing ---------------------------------------------------------------


def write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"{path.name}: {len(rows)} rows")


def fight_url(key: str) -> str:
    return f"http://ufcstats.com/fight-details/{FIGHT_IDS[key]}"


def build(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261115)

    events = [
        [
            name,
            f"http://ufcstats.com/event-details/{EVENT_IDS[key]}",
            f"{when:%B %d, %Y}",
            location,
        ]
        for key, (name, when, location) in EVENTS.items()
    ]

    def event_names(fight: Fight) -> list[str]:
        if fight.event == "e99":
            return [ORPHAN_EVENT]
        names = [EVENTS[fight.event][0]]
        if fight.listed_under_old_name:
            names.append(OLD_EVENT_NAMES[fight.event])
        return names

    details, results, stats = [], [], []
    stats_done: set[tuple[str, str]] = set()
    for fight in FIGHTS:
        bout = f"{bout_name(fight.a)} vs. {bout_name(fight.b)}"
        for name in event_names(fight):
            details.append([name, bout, fight_url(fight.key)])
            results.append(
                [
                    name + " ",
                    bout,
                    fight.outcome,
                    fight.weight_class,
                    fight.method,
                    str(fight.end_round),
                    fight.end_time,
                    fight.time_format,
                    fight.referee,
                    fight.details,
                    fight_url(fight.key),
                ]
            )
            # One block of stats per (event, bout text), however many fights share it.
            if (name, bout) not in stats_done:
                stats_done.add((name, bout))
                stats.extend(stat_rows(fight, name, rng))
            elif fight.stats != STATS_SHARED:
                raise SystemExit(f"duplicate stats block for {bout}")
            # A second fight with the same bout text on one card adds its own rows.
            elif fight.key == "pair_nc":
                stats.extend(stat_rows(fight, name, rng))

    fighter_details = [
        [
            f.first,
            f.last,
            f.nickname,
            f"http://ufcstats.com/fighter-details/{FIGHTER_IDS[key]}",
        ]
        for key, f in FIGHTERS.items()
        if f.in_details
    ]
    tale_of_the_tape = [
        [
            f.tott_name or f.name,
            f.height,
            f.weight,
            f.reach,
            f.stance,
            f.dob,
            f"http://ufcstats.com/fighter-details/{FIGHTER_IDS[key]}",
        ]
        for key, f in FIGHTERS.items()
        if f.in_tott
    ]

    columns = source.EXPECTED_COLUMNS
    write_csv(directory / source.EVENTS_FILE, columns[source.EVENTS_FILE], events)
    write_csv(
        directory / source.FIGHT_DETAILS_FILE,
        columns[source.FIGHT_DETAILS_FILE],
        details,
    )
    write_csv(
        directory / source.FIGHT_RESULTS_FILE,
        columns[source.FIGHT_RESULTS_FILE],
        results,
    )
    write_csv(directory / source.FIGHT_STATS_FILE, STAT_COLUMNS, stats)
    write_csv(
        directory / source.FIGHTER_DETAILS_FILE,
        columns[source.FIGHTER_DETAILS_FILE],
        fighter_details,
    )
    write_csv(
        directory / source.FIGHTER_TOTT_FILE,
        columns[source.FIGHTER_TOTT_FILE],
        tale_of_the_tape,
    )

    write_csv(
        directory / "identity_overrides.csv",
        ["name", "fight_id", "fighter_id", "note"],
        [
            [
                name,
                FIGHT_IDS[fight_key] if fight_key else "",
                FIGHTER_IDS[fighter_key],
                note,
            ]
            for name, fight_key, fighter_key, note in OVERRIDES
        ],
    )
    (directory / source.SOURCE_INFO_FILE).write_text(
        json.dumps(
            {
                "repo": "Greco1899/scrape_ufc_stats",
                "commit": SNAPSHOT_COMMIT,
                "commit_date": SNAPSHOT_DATE,
                "subset": "synthetic test fixture written by scripts/make_synthetic_fixtures.py",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(fixtures_dir: Path) -> None:
    build(fixtures_dir / "greco")
    print("IDs: see FIGHTER_IDS, FIGHT_IDS and EVENT_IDS in this script.")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures"))
