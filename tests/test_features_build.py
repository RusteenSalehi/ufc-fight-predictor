"""The feature table, its metadata, and the command line."""

import json
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.feature_helpers import League, random_league
from ufc.db.connection import connect
from ufc.features import __main__ as cli
from ufc.features.build import (
    CONTEXT_COLUMNS,
    FEATURES_FILE,
    META_FILE,
    build_features,
    feature_summary,
    write_features,
)
from ufc.features.config import TRAIN_END, FeatureConfig
from ufc.features.data import load_feature_data
from ufc.features.matchup import (
    FAMILIES,
    FEATURE_NAMES,
    FEATURES,
    FLAG_NAMES,
    LEVEL_NAMES,
)
from ufc.features.store import FittedParams


@pytest.fixture(scope="module")
def league():
    return random_league(seed=3)


@pytest.fixture(scope="module")
def data(league):
    return league.data()


@pytest.fixture(scope="module")
def built(data):
    return build_features(data, FeatureConfig(train_end=data.bouts[150].date))


def test_training_period_ends_where_the_requirements_say():
    assert TRAIN_END == date(2022, 12, 31)
    assert FeatureConfig().train_end == TRAIN_END


def test_one_row_per_bout_in_date_order_with_the_documented_columns(data, built):
    frame = built.frame
    assert list(frame.columns) == [
        *CONTEXT_COLUMNS,
        *FLAG_NAMES,
        *FEATURE_NAMES,
        *LEVEL_NAMES,
    ]
    assert list(frame["bout_id"]) == [b.bout_id for b in data.bouts]
    assert frame["date"].is_monotonic_increasing
    assert frame["bout_id"].is_unique
    assert all(frame[name].dtype == "float64" for name in FEATURE_NAMES)
    assert all(frame[name].dtype == "float64" for name in LEVEL_NAMES)
    assert all(frame[name].dtype == bool for name in FLAG_NAMES)


def test_label_is_from_fighter_a_side_and_missing_without_a_winner(data, built):
    frame = built.frame.set_index("bout_id")
    seen = set()
    for bout in data.bouts:
        label = frame.loc[bout.bout_id, "label"]
        if bout.result == "win":
            assert label == (1.0 if bout.winner_id == bout.fighter_a_id else 0.0)
        else:
            assert math.isnan(label)
        seen.add(bout.result)
    # Draws and no contests stay in the table; they are excluded at training time.
    assert seen == {"win", "draw", "nc"}


def test_every_family_in_the_requirements_has_features():
    families = {f.family for f in FEATURES}
    assert families == set(FAMILIES)
    assert len(FAMILIES) == 13
    by_family = {name: [f for f in FEATURES if f.family == name] for name in FAMILIES}
    assert all(by_family.values())
    assert {f.name for f in by_family["cardio"]} >= {
        "d_cardio_slpm_ratio_career",
        "d_cardio_sapm_ratio_career",
    }
    # Family 13: the fighter's own rating, apart from who he has fought.
    assert [f.name for f in by_family["rating"]] == ["d_elo"]
    assert all("opp_elo" in f.name for f in by_family["opponent_quality"])


def test_missing_stats_are_nan_and_flagged_never_zero(built):
    frame = built.frame
    debut = frame["a_debut"] | frame["b_debut"]
    no_stats = frame["a_no_stats"] | frame["b_no_stats"]
    assert debut.any() and (no_stats & ~debut).any()
    assert (frame.loc[frame["a_debut"], "a_no_stats"]).all()
    # A rate difference needs both fighters' rates.
    for name in ("d_slpm_career", "d_sig_acc_l3", "d_adj_slpm_career", "d_kd_abs"):
        assert frame.loc[no_stats, name].isna().all(), name
        assert frame.loc[~no_stats, name].notna().any(), name
    for name in ("d_win_rate_career", "d_days_since_last", "d_opp_elo_career"):
        assert frame.loc[debut, name].isna().all(), name
    # Record counts are true zeros for a debutant, so their differences exist.
    assert frame[["d_n_fights", "d_wins", "d_elo", "d_debut"]].notna().all().all()


def test_written_files_round_trip_and_rebuild_identically(data, built, tmp_path):
    features_path, meta_path = write_features(built, tmp_path / "out", "abc123")
    assert features_path.name == FEATURES_FILE and meta_path.name == META_FILE
    pd.testing.assert_frame_equal(pd.read_parquet(features_path), built.frame)

    meta = json.loads(meta_path.read_text())
    assert meta["source_commit"] == "abc123"
    assert meta["n_bouts"] == len(built.frame)
    assert [f["name"] for f in meta["features"]] == list(FEATURE_NAMES)
    assert list(meta["families"]) == list(FAMILIES)
    assert sum(len(names) for names in meta["families"].values()) == len(FEATURES)
    assert meta["families"]["rating"] == ["d_elo"]
    assert meta["level_features"] == list(LEVEL_NAMES)
    assert not set(LEVEL_NAMES) & {f["name"] for f in meta["features"]}
    assert meta["config"]["shrink_minutes"] == 0.0
    assert meta["config"]["train_end"] == data.bouts[150].date.isoformat()
    assert meta["elo_fit"]["last_date"] <= meta["config"]["train_end"]
    restored = FittedParams.from_dict(meta["params"])
    assert restored.elo == built.params.elo
    np.testing.assert_array_equal(restored.league, built.params.league)
    assert restored.rate_priors.to_dict() == built.params.rate_priors.to_dict()
    np.testing.assert_array_equal(
        restored.rate_priors.overall, built.params.rate_priors.overall
    )

    # Rerunning from scratch replaces the files with the same content.
    again = build_features(data, FeatureConfig(train_end=data.bouts[150].date))
    write_features(again, tmp_path / "again", "abc123")
    assert (tmp_path / "again" / META_FILE).read_text() == meta_path.read_text()
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "again" / FEATURES_FILE),
        pd.read_parquet(features_path),
        check_exact=True,
    )
    write_features(again, tmp_path / "out", "abc123")
    assert not list((tmp_path / "out").glob("*.tmp"))


def test_feature_summary_counts_missing_values(built):
    summary = feature_summary(built.frame).set_index("feature")
    frame = built.frame
    established = ~(frame["a_debut"] | frame["b_debut"])
    assert list(summary.index) == [*FEATURE_NAMES, *LEVEL_NAMES]
    assert set(summary.loc[list(LEVEL_NAMES), "family"]) == {"level"}
    for name in ("d_slpm_career", "d_td_def_l3", "d_age", "d_n_fights", "lvl_age"):
        assert summary.loc[name, "missing"] == frame[name].isna().sum()
        assert (
            summary.loc[name, "missing_established"]
            == frame.loc[established, name].isna().sum()
        )
    assert (summary["missing_established"] <= summary["missing"]).all()


def test_reader_returns_completed_bouts_in_order_with_their_stats(league, tmp_path):
    db_path = tmp_path / "ufc.db"
    league.write_db(db_path)
    conn = connect(db_path)
    conn.execute(
        "UPDATE bouts SET status = 'cancelled' WHERE bout_id = ?",
        (league.dataset.bouts[0].bout_id,),
    )
    data = load_feature_data(conn)
    conn.close()

    assert len(data.bouts) == len(league.dataset.bouts) - 1
    assert league.dataset.bouts[0].bout_id not in {b.bout_id for b in data.bouts}
    assert [b.date for b in data.bouts] == sorted(b.date for b in data.bouts)
    assert data.source_commit is None  # the league was loaded without source files
    assert len(data.fighters) == len(league.dataset.fighters)
    with_stats = next(b for b in data.bouts if b.has_stats)
    assert set(with_stats.totals) == {with_stats.fighter_a_id, with_stats.fighter_b_id}
    lines = with_stats.rounds[with_stats.fighter_a_id]
    assert len(lines) == with_stats.end_round
    assert (
        sum(r["sig_str_landed"] for r in lines)
        == (with_stats.totals[with_stats.fighter_a_id]["sig_str_landed"])
    )
    without = next(b for b in data.bouts if not b.has_stats)
    assert not without.totals and not without.rounds


def test_command_line_builds_and_summarises(league, tmp_path, capsys):
    db_path, out_dir = tmp_path / "ufc.db", tmp_path / "features"
    league.write_db(db_path)
    assert cli.main(["build", "--db", str(db_path), "--out", str(out_dir)]) == 0
    assert (out_dir / FEATURES_FILE).exists() and (out_dir / META_FILE).exists()
    output = capsys.readouterr().out
    assert f"Features: {len(FEATURES)}" in output
    assert "Rate shrinkage: 0 cage minutes" in output

    assert cli.main(["summary", "--out", str(out_dir)]) == 0
    output = capsys.readouterr().out
    assert "d_cardio_slpm_ratio_career" in output
    for family in (*FAMILIES, "level"):
        assert f"\n{family} (" in output

    shrunk_dir = tmp_path / "shrunk"
    assert (
        cli.main(
            ["build", "--db", str(db_path), "--out", str(shrunk_dir)]
            + ["--shrink-minutes", "15"]
        )
        == 0
    )
    assert "Rate shrinkage: 15 cage minutes" in capsys.readouterr().out
    meta = json.loads((shrunk_dir / META_FILE).read_text())
    assert meta["config"]["shrink_minutes"] == 15.0


def test_command_line_fails_cleanly_without_inputs(tmp_path, capsys):
    assert cli.main(["build", "--db", str(tmp_path / "none.db")]) == 1
    assert cli.main(["summary", "--out", str(tmp_path / "none")]) == 1
    League().write_db(tmp_path / "empty.db")
    assert cli.main(["build", "--db", str(tmp_path / "empty.db")]) == 1
    assert "no completed bouts" in capsys.readouterr().err
