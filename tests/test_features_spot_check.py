"""Spot check: Khabib Nurmagomedov's stats before UFC 160, worked by hand.

The fixture in tests/fixtures/greco_spot_check holds the published rows for
his first four UFC bouts, unchanged. The test runs them through ingestion,
SQLite and the feature pipeline, and compares his as-of values before the
fourth bout (2013-05-25, against Abel Trujillo) with arithmetic done here
from the three earlier bouts. Every number below can be read off
``ufc_fight_stats.csv`` and ``ufc_fight_results.csv`` in that directory.

    bout                         time       sig. strikes   takedowns  KD  sub  control
    2012-01-20 Shalorus  sub R3  2:08   K   35 of  77      3 of  9    1   1    359 s
                                        opp 13 of  44      0 of  2             5 s
    2012-07-07 Tibau     dec R3  5:00   K   25 of 101      0 of 13    0   0    265 s
                                        opp 28 of  87      1 of  6            17 s
    2013-01-19 Tavares   KO  R1  1:55   K   22 of  31      0 of  0    1   0     13 s
                                        opp  2 of  10      0 of  0             0 s

His opponents' other bouts are not in the excerpt, so values that depend on
them (opponent-adjusted stats, the opponents' real ratings) are not checked
here; tests/test_features_adjust.py covers that arithmetic.
"""

import math
from datetime import date
from pathlib import Path

import pytest

from tests.feature_helpers import NO_PRIOR, SEEN_AT, plain_params
from ufc.db import loaders
from ufc.db.connection import connect
from ufc.features.build import build_features
from ufc.features.data import load_feature_data
from ufc.features.matchup import build_matchup
from ufc.ingest.identity import load_overrides
from ufc.ingest.source import load_raw_tables
from ufc.ingest.transform import build_dataset

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "greco_spot_check"
KHABIB = "032cc3922d871c7f"
TRUJILLO = "e14b5aa06eea9653"
UFC_160_BOUT = "e1d33aa47b35d021"
UFC_160_DATE = date(2013, 5, 25)

SECONDS = (2 * 300 + 128) + 900 + 115  # 1743
MINUTES = SECONDS / 60  # 29.05


@pytest.fixture(scope="module")
def built():
    dataset = build_dataset(load_raw_tables(FIXTURE_DIR), load_overrides())
    assert dataset.count_issues("error") == 0
    conn = connect(":memory:")
    loaders.load_dataset(conn, dataset, seen_at=SEEN_AT)
    data = load_feature_data(conn)
    conn.close()
    assert [b.date for b in data.bouts] == [
        date(2012, 1, 20),
        date(2012, 7, 7),
        date(2013, 1, 19),
        UFC_160_DATE,
    ]
    return build_features(data, NO_PRIOR, plain_params(k=32))


@pytest.fixture(scope="module")
def khabib(built):
    """His values as of UFC 160, asked of the finished store."""
    return build_matchup(built.store, KHABIB, TRUJILLO, UFC_160_DATE, "Lightweight").a


def test_record_before_ufc_160(khabib):
    assert khabib["n_fights"] == 3
    assert (khabib["wins"], khabib["losses"]) == (3, 0)
    assert (khabib["ko_wins"], khabib["sub_wins"]) == (1, 1)
    assert khabib["finish_win_share"] == pytest.approx(2 / 3)
    assert khabib["win_rate_career"] == 1.0
    assert (khabib["win_streak"], khabib["loss_streak"]) == (3, 0)
    assert khabib["title_fights"] == 0
    assert khabib["cage_min"] == pytest.approx(29.05)
    assert khabib["debut"] == 0


def test_striking_before_ufc_160(khabib):
    landed, attempted = 35 + 25 + 22, 77 + 101 + 31  # 82 of 209
    absorbed, faced = 13 + 28 + 2, 44 + 87 + 10  # 43 of 141
    assert khabib["slpm_career"] == pytest.approx(landed / MINUTES)  # 2.82
    assert khabib["sapm_career"] == pytest.approx(absorbed / MINUTES)  # 1.48
    assert khabib["sig_acc_career"] == pytest.approx(82 / 209)  # 0.392
    assert khabib["sig_def_career"] == pytest.approx(1 - 43 / 141)  # 0.695
    assert khabib["sig_att_pm_career"] == pytest.approx(attempted / MINUTES)
    assert khabib["kd_per15_career"] == pytest.approx(2 / MINUTES * 15)
    # Where his 82 landed: head 32 + 13 + 22, body 3 + 7 + 0, leg 0 + 5 + 0.
    assert khabib["head_share"] == pytest.approx(67 / 82)
    assert khabib["body_share"] == pytest.approx(10 / 82)
    assert khabib["leg_share"] == pytest.approx(5 / 82)
    # Distance 15 + 15 + 5, clinch 3 + 10 + 0, ground 17 + 0 + 17.
    assert khabib["distance_share"] == pytest.approx(35 / 82)
    assert khabib["clinch_share"] == pytest.approx(13 / 82)
    assert khabib["ground_share"] == pytest.approx(34 / 82)
    assert landed == 67 + 10 + 5 == 35 + 13 + 34 and faced == 141


def test_grappling_before_ufc_160(khabib):
    assert khabib["td_per15_career"] == pytest.approx(3 / MINUTES * 15)  # 1.55
    assert khabib["td_acc_career"] == pytest.approx(3 / (9 + 13 + 0))
    assert khabib["td_def_career"] == pytest.approx(1 - 1 / (2 + 6 + 0))  # 0.875
    assert khabib["sub_att_per15_career"] == pytest.approx(1 / MINUTES * 15)
    assert khabib["ctrl_share_career"] == pytest.approx((359 + 265 + 13) / SECONDS)
    assert khabib["ctrl_against_share_career"] == pytest.approx((5 + 17 + 0) / SECONDS)


def test_durability_and_activity_before_ufc_160(khabib):
    assert khabib["ko_losses"] == khabib["sub_losses"] == khabib["kd_abs"] == 0
    assert khabib["kd_abs_per15_career"] == 0.0
    # Head strikes absorbed: 9 from Shalorus, 15 from Tibau, 0 from Tavares.
    assert khabib["head_abs_pm_career"] == pytest.approx(24 / MINUTES)
    # 2013-01-19 to 2013-05-25: 12 + 28 + 31 + 30 + 25 days.
    assert khabib["days_since_last"] == 126
    assert khabib["fights_last_2y"] == 3
    assert (khabib["moved_up"], khabib["moved_down"]) == (0, 0)
    assert khabib["fights_at_weight"] == 3


def test_physical_attributes_at_ufc_160(khabib):
    # Born 1988-09-20: 25 years (9131 days) less the 118 days to his birthday.
    assert khabib["age"] == pytest.approx((9131 - 118) / 365.25)  # 24.68
    assert khabib["height_cm"] == pytest.approx(70 * 2.54)  # 5 ft 10 in
    assert khabib["reach_cm"] == pytest.approx(70 * 2.54)
    assert (khabib["stance_southpaw"], khabib["stance_switch"]) == (0, 0)


def test_windows_before_ufc_160(khabib):
    # With three bouts, the last-3 and last-5 windows are the whole history.
    for stat in ("slpm", "sig_acc", "td_def", "ctrl_share", "win_rate"):
        assert khabib[f"{stat}_l3"] == khabib[f"{stat}_l5"] == khabib[f"{stat}_career"]
    # Time decay, half-life 730 days. The bouts were 491, 322 and 126 days
    # before UFC 160.
    w1, w2, w3 = (0.5 ** (days / 730) for days in (491, 322, 126))
    assert khabib["slpm_decay"] == pytest.approx(
        (w1 * 35 + w2 * 25 + w3 * 22) / ((w1 * 728 + w2 * 900 + w3 * 115) / 60)
    )
    assert khabib["td_acc_decay"] == pytest.approx(
        (w1 * 3) / (w1 * 9 + w2 * 13 + w3 * 0)
    )


def test_cardio_before_ufc_160(khabib):
    """Rounds 3+ against round 1, over the two bouts that reached round 3.

    Round 1 (300 s each): he landed 24 of 54 and 8 of 27; opponents 9 of 34
    and 10 of 27. Round 3 (128 s and 300 s): he landed 3 of 8 and 8 of 41;
    opponents 2 of 6 and 9 of 29. The round-1 knockout does not count.
    """
    r1_seconds, late_seconds = 600, 128 + 300
    assert khabib["cardio_fights"] == 2
    assert khabib["cardio_slpm_ratio_career"] == pytest.approx(
        (11 / late_seconds) / (32 / r1_seconds)  # 0.48
    )
    assert khabib["cardio_sig_att_ratio_career"] == pytest.approx(
        (49 / late_seconds) / (81 / r1_seconds)
    )
    assert khabib["cardio_sapm_ratio_career"] == pytest.approx(
        (11 / late_seconds) / (19 / r1_seconds)
    )
    assert khabib["cardio_sig_acc_delta"] == pytest.approx(11 / 49 - 32 / 81)
    assert khabib["cardio_sig_def_delta"] == pytest.approx(
        (1 - 11 / 35) - (1 - 19 / 61)
    )


def test_elo_before_ufc_160(khabib):
    """K = 32, everyone starts at 1500, and in this excerpt each opponent is
    in his first bout."""
    rating = 1500.0
    for _ in range(3):
        rating += 32 * (1 - 1 / (1 + 10 ** ((1500 - rating) / 400)))
    assert khabib["elo"] == pytest.approx(rating)
    assert khabib["opp_elo_career"] == 1500.0


def test_the_built_row_for_ufc_160_matches(built, khabib):
    row = built.frame[built.frame["bout_id"] == UFC_160_BOUT].iloc[0]
    assert (row["fighter_a_id"], row["fighter_b_id"]) == (KHABIB, TRUJILLO)
    assert row["label"] == 1.0
    assert not row["a_debut"] and not row["a_no_stats"]
    # In this excerpt Trujillo has no earlier bout, so he is a debutant here:
    # record differences exist, rate differences are missing.
    assert row["b_debut"]
    assert row["d_n_fights"] == 3
    assert row["d_wins"] == 3
    assert row["d_elo"] == pytest.approx(khabib["elo"] - 1500)
    assert math.isnan(row["d_slpm_career"])
    assert math.isnan(row["d_adj_td_def_career"])


def test_his_debut_row_has_no_stats_at_all(built):
    row = built.frame[built.frame["date"] == "2012-01-20"].iloc[0]
    khabib_side = "a" if row["fighter_a_id"] == KHABIB else "b"
    assert row[f"{khabib_side}_debut"] and row[f"{khabib_side}_no_stats"]
    assert row["d_n_fights"] == 0
    assert math.isnan(row["d_slpm_career"]) and math.isnan(row["d_win_rate_career"])
