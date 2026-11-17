"""Walk-forward backtest and the command line, end to end on an invented league.

The league's winners are random, so nothing here checks that a model is good;
it checks that the three commands run the real code path, respect the test
lock, and leave the files they promise.
"""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.feature_helpers import random_league
from tests.model_helpers import FAST, SMALL
from ufc.features.build import FEATURES_FILE, build_features, fit_params
from ufc.features.config import FeatureConfig
from ufc.features.matchup import FEATURE_NAMES
from ufc.models import __main__ as cli
from ufc.models import registry
from ufc.models.calibrate import PLATT
from ufc.models.design import FeatureSpec
from ufc.models.recipe import Recipe, fit_recipe
from ufc.models.report import BEGIN, END
from ufc.models.splits import (
    SPLIT_DATES,
    Splits,
    TestPeriodLocked,
    attribute_gap,
    fit_rows,
    labelled,
)
from ufc.models.walkforward import FIRST_YEAR, fold_dates, run_fold, walk_forward


@pytest.fixture(scope="module")
def league():
    # About 150 events from 2018 into 2026: all three periods are populated.
    return random_league(seed=21, n_fighters=60, n_events=150, start=date(2018, 1, 6))


@pytest.fixture(scope="module")
def data(league):
    return league.data()


RECIPE = Recipe(FAST["xgboost"], 4.0, FeatureSpec(FEATURE_NAMES), PLATT)
CONFIG = FeatureConfig(shrink_minutes=15.0)


def test_fold_schedule_is_three_years_back_then_two_years_of_validation():
    dates = fold_dates(2025)
    assert dates.train_end == date(2022, 12, 31)
    assert (dates.valid_start, dates.valid_end) == (
        date(2023, 1, 1),
        date(2024, 12, 31),
    )
    assert dates == SPLIT_DATES  # the first test-period fold is the real split
    assert fold_dates(2013).train_end == date(2010, 12, 31)
    assert FIRST_YEAR == 2013


def test_a_fold_that_touches_the_test_period_cannot_run_while_locked(data):
    # 2024: fitted through 2021, validated on 2022 and 2023. No test rows.
    fold = run_fold(data, RECIPE, CONFIG, 0.01, 2024, None)
    assert fold["year"] == 2024 and not fold["in_test_period"]
    assert fold["train_end"] == "2021-12-31"
    assert fold["n_scored"] == len(fold["_bout_ids"]) > 0
    # 2025 scores test-period bouts; 2026 also calibrates on them.
    for year in (2025, 2026):
        with pytest.raises(TestPeriodLocked):
            run_fold(data, RECIPE, CONFIG, 0.01, year, None)


def test_fold_refits_feature_parameters_on_its_own_training_period(data):
    fold = run_fold(data, RECIPE, CONFIG, 0.01, 2024, None)
    own_params, _ = fit_params(
        list(data.bouts),
        FeatureConfig(train_end=date(2021, 12, 31), shrink_minutes=15.0),
    )
    assert fold["elo_params"] == own_params.elo.to_dict()
    assert set(fold["metrics"]) == {"model", "model_uncalibrated", "elo", "logistic"}
    assert all(m["n"] == fold["n_scored"] for m in fold["metrics"].values())


def test_first_test_period_fold_is_the_model_the_main_split_produces(data):
    """Same schedule, same recipe, same data: identical probabilities."""
    fold = run_fold(data, RECIPE, CONFIG, 0.01, 2025, "v")
    frame = build_features(data, CONFIG).frame
    splits = Splits(frame)
    train = fit_rows(splits.train())
    assert attribute_gap(labelled(splits.train())).any()  # the league has such bouts
    assert not attribute_gap(train).any() and fold["n_train"] == len(train)
    candidate, _ = fit_recipe(
        RECIPE, train, labelled(splits.valid()), SPLIT_DATES.train_end
    )
    splits.unlock_test("v")
    test = labelled(splits.test())
    in_2025 = test[test["date"].dt.year == 2025]
    assert list(in_2025["bout_id"]) == fold["_bout_ids"]
    np.testing.assert_array_equal(
        candidate.predict(in_2025), fold["_predictions"]["model"]
    )
    assert fold["n_trees"] == candidate.model.n_trees
    assert fold["in_test_period"]


def test_walk_forward_pools_test_period_folds_apart_from_earlier_ones(data):
    record, raw = walk_forward(
        data, RECIPE, CONFIG, 0.01, [2023, 2024, 2025, 2026], unlocked_for="v", n_jobs=2
    )
    assert [f["year"] for f in record["folds"]] == [2023, 2024, 2025, 2026]
    assert [f["in_test_period"] for f in record["folds"]] == [False, False, True, True]
    # The training window expands.
    sizes = [f["n_train"] for f in record["folds"]]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]
    assert record["pooled_before_test_period"]["years"] == [2023, 2024]
    assert record["pooled_test_period"]["years"] == [2025, 2026]
    before = sum(f["n_scored"] for f in record["folds"][:2])
    assert record["pooled_before_test_period"]["metrics"]["model"]["n"] == before
    assert set(raw) == {2023, 2024, 2025, 2026}
    json.dumps(record)
    assert not any(key.startswith("_") for fold in record["folds"] for key in fold)
    # Locked, the same call fails as soon as it reaches 2025.
    with pytest.raises(TestPeriodLocked):
        walk_forward(data, RECIPE, CONFIG, 0.01, [2024, 2025])


@pytest.fixture(scope="module")
def workspace(league, tmp_path_factory):
    """Run train, then report, then evaluate, then report again."""
    root = tmp_path_factory.mktemp("pipeline")
    db = root / "ufc.db"
    league.write_db(db)
    paths = {
        "root": root,
        "db": db,
        "features": root / "features",
        "models": root / "models",
        "reports": root / "reports",
        "doc": root / "MODEL_REPORT.md",
    }
    common = ["--models", str(paths["models"]), "--reports", str(paths["reports"])]
    inputs = ["--db", str(db), "--features", str(paths["features"]), "--jobs", "2"]
    paths["train_args"] = ["train", *common, *inputs]
    paths["evaluate_args"] = ["evaluate", *common, *inputs]
    paths["report_args"] = ["report", *common, "--out", str(paths["doc"])]

    patch = pytest.MonkeyPatch()
    patch.setattr(cli, "SETTINGS", SMALL)
    patch.setattr(cli, "FIRST_YEAR", 2024)
    try:
        assert cli.main(paths["train_args"]) == 0
        paths["selection"] = json.loads(
            (paths["reports"] / cli.SELECTION_FILE).read_text()
        )
        assert cli.main(paths["report_args"]) == 0
        paths["report_before"] = paths["doc"].read_text()
        paths["approved_after_train"] = (
            paths["models"] / registry.APPROVED_FILE
        ).exists()
        paths["test_files_after_train"] = [
            p.name
            for p in paths["reports"].rglob("*")
            if "test" in p.name or "walk_forward" in p.name
        ]
        assert cli.main(paths["evaluate_args"]) == 0
        assert cli.main(paths["report_args"]) == 0
        yield paths
    finally:
        patch.undo()


def test_train_saves_a_candidate_and_writes_the_selection_record(workspace):
    selection = workspace["selection"]
    version = selection["version"]
    assert registry.list_versions(workspace["models"]) == [version]
    candidate, manifest = registry.load_candidate(workspace["models"], version)
    assert manifest["dataset"]["source_commit"] is None  # the league has no snapshot
    gap = selection["attribute_gap"]
    assert manifest["dataset"]["n_training_bouts"] == gap["n_train_fitted"]
    assert gap["n_train_fitted"] == selection["n_train_scored"] - gap["n_train_gap"]
    assert 0 < gap["n_train_one_sided"] <= gap["n_train_gap"]
    assert candidate.recipe.to_dict() == selection["selected"]["recipe"]
    assert candidate.model.n_trees == selection["selected"]["n_trees"]

    config = registry.read_json(workspace["models"], version, "config.json")
    assert config["splits"] == SPLIT_DATES.to_dict()
    assert config["shrink_minutes"] == selection["shrinkage"]["chosen"]
    assert config["selection"]["seeds"] == list(SMALL.seeds)
    # The feature file on disk is the chosen build, and the registry keeps
    # the settings and fitted parameters needed to rebuild its inputs.
    meta = json.loads((workspace["features"] / "features_meta.json").read_text())
    assert meta["config"]["shrink_minutes"] == selection["shrinkage"]["chosen"]
    pipeline = registry.read_json(workspace["models"], version, "feature_pipeline.json")
    assert pipeline == meta
    assert cli.feature_config_from(pipeline).shrink_minutes == config["shrink_minutes"]
    frame = pd.read_parquet(workspace["features"] / FEATURES_FILE)
    assert (
        registry.dataset_hash(frame, candidate.spec)
        == manifest["dataset"]["feature_table_sha256"]
    )
    plots = workspace["reports"] / "plots" / "validation"
    assert {p.name for p in plots.glob("*.png")} >= {
        "reliability.png",
        "by_confidence.png",
        "by_year.png",
        "ablation.png",
    }
    assert set(selection["ablation"]["families"][0]) >= {"family", "delta", "helped"}
    assert len(selection["ablation"]["families"]) == 13


def test_train_does_not_evaluate_the_test_period_or_approve_anything(workspace):
    assert workspace["test_files_after_train"] == []
    assert not workspace["approved_after_train"]
    assert "Not evaluated yet" in workspace["report_before"]
    assert "## Validation period" in workspace["report_before"]
    selection = workspace["selection"]
    assert selection["validation"]["last_date"] <= "2024-12-31"
    assert "2025" not in selection["validation"]["by_year"]
    # Still not approved after evaluation: promotion is manual.
    assert not (workspace["models"] / registry.APPROVED_FILE).exists()


def test_evaluate_scores_the_saved_candidate_on_the_test_period(workspace):
    version = workspace["selection"]["version"]
    record = json.loads((workspace["reports"] / cli.TEST_FILE).read_text())
    assert record["version"] == version
    assert record["first_date"] >= "2025-01-01"
    assert record["n_scored"] + record["n_no_result"] == record["n_bouts"] > 0
    assert set(record["models"]) == {"model", "model_uncalibrated", "elo", "logistic"}
    assert record["symmetry"] == {
        "n_bouts": record["n_bouts"],
        "sum_is_exactly_one": True,
        "swap_is_exact": True,
    }
    # The same record is attached to the registry version, and it reloads.
    stored = registry.read_json(workspace["models"], version, registry.TEST_METRICS)
    assert stored == record
    registry.load_candidate(workspace["models"], version)

    # Recomputed here from the saved model and the feature file.
    candidate, _ = registry.load_candidate(workspace["models"], version)
    frame = pd.read_parquet(workspace["features"] / FEATURES_FILE)
    test = labelled(frame[frame["date"] >= pd.Timestamp("2025-01-01")])
    p = candidate.predict(test)
    y = test["label"].to_numpy()
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert record["models"]["model"]["log_loss"] == pytest.approx(loss)

    backtest = json.loads((workspace["reports"] / cli.WALK_FORWARD_FILE).read_text())
    assert [f["year"] for f in backtest["folds"]] == [2024, 2025, 2026]
    assert backtest["first_test_year_check"] == {
        "year": 2025,
        "same_bouts": True,
        "identical_to_registered_model": True,
    }
    assert backtest["folds"][1]["metrics"]["model"] == pytest.approx(
        record["by_year"]["2025"]["model"]
    )
    assert (workspace["reports"] / "plots" / "walk_forward.png").exists()
    assert (workspace["reports"] / "plots" / "test" / "reliability.png").exists()
    assert len(registry.read_ledger(workspace["models"])) == 1


def test_report_is_generated_between_markers_and_keeps_hand_written_text(workspace):
    doc = workspace["doc"]
    text = doc.read_text()
    assert text.count(BEGIN) == 1 and text.count(END) == 1
    assert "## Test period" in text and "Not evaluated yet" not in text
    assert (
        "## Walk-forward backtest" in text and "### 7. Feature-family ablation" in text
    )
    assert workspace["selection"]["version"] in text
    assert "identical to the registered model's, bout for bout" in text
    assert "exactly 1 for every bout: yes" in text

    doc.write_text(text.replace(BEGIN, "## Findings\n\nWritten by hand.\n\n" + BEGIN))
    assert cli.main(workspace["report_args"]) == 0
    again = doc.read_text()
    assert "Written by hand." in again
    assert again.split(BEGIN)[1] == text.split(BEGIN)[1]  # same numbers, regenerated
    doc.write_text("no markers here")
    with pytest.raises(ValueError, match="markers"):
        cli.main(workspace["report_args"])
    doc.write_text(text)


def test_running_again_reproduces_the_same_version_and_metrics(
    workspace, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "SETTINGS", SMALL)
    monkeypatch.setattr(cli, "FIRST_YEAR", 2024)
    before = (workspace["reports"] / cli.TEST_FILE).read_text()
    assert cli.main(workspace["train_args"]) == 0
    assert "already there, unchanged" in capsys.readouterr().out
    assert registry.list_versions(workspace["models"]) == [
        workspace["selection"]["version"]
    ]
    # Evaluating the same version again gives byte-identical metrics, and the
    # ledger shows that the test period was looked at twice.
    assert cli.main(workspace["evaluate_args"]) == 0
    assert (workspace["reports"] / cli.TEST_FILE).read_text() == before
    assert len(registry.read_ledger(workspace["models"])) == 2
    assert "evaluations recorded so far: 2" in capsys.readouterr().out


def test_evaluate_refuses_a_feature_file_the_model_was_not_trained_from(
    workspace, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "FIRST_YEAR", 2024)
    path = workspace["features"] / FEATURES_FILE
    original = path.read_bytes()
    frame = pd.read_parquet(path)
    frame.loc[frame.index[-1], "d_age"] = 99.0  # one test-period value
    frame.to_parquet(path, index=False)
    try:
        assert cli.main(workspace["evaluate_args"]) == 1
        assert "not the table" in capsys.readouterr().err
        assert len(registry.read_ledger(workspace["models"])) == 2  # nothing recorded
    finally:
        path.write_bytes(original)


def test_commands_fail_cleanly_without_their_inputs(tmp_path, capsys):
    common = ["--models", str(tmp_path / "m"), "--reports", str(tmp_path / "r")]
    assert cli.main(["train", *common, "--db", str(tmp_path / "none.db")]) == 1
    assert "run the ingest load first" in capsys.readouterr().err
    assert cli.main(["evaluate", *common]) == 1
    assert cli.main(["report", *common, "--out", str(tmp_path / "x.md")]) == 1
    assert "run train first" in capsys.readouterr().err


# --- production refit (D47) ------------------------------------------------------


@pytest.fixture(scope="module")
def refitted(workspace):
    """Run refit after train and evaluate, as the project does."""
    args = ["refit", *workspace["train_args"][1:]]
    assert cli.main(args) == 0
    record = json.loads((workspace["reports"] / cli.REFIT_FILE).read_text())
    return {**workspace, "refit_args": args, "refit": record}


def test_refit_saves_a_new_version_with_lineage_and_approves_nothing(refitted):
    parent = refitted["selection"]["version"]
    record = refitted["refit"]
    version = record["version"]
    assert version != parent
    assert set(registry.list_versions(refitted["models"])) == {parent, version}
    candidate, manifest = registry.load_candidate(refitted["models"], version)
    assert manifest["lineage"]["parent"] == parent
    assert manifest["lineage"]["status"] == (
        "production refit; no held-out test; recipe validated by the candidate"
    )
    assert registry.is_refit(manifest)
    old, old_manifest = registry.load_candidate(refitted["models"], parent)
    assert not registry.is_refit(old_manifest)
    assert candidate.recipe == old.recipe  # the recipe is not changed
    config = registry.read_json(refitted["models"], version, "config.json")
    assert config["production_refit"]["parent"] == parent
    pipeline = registry.read_json(refitted["models"], version, "feature_pipeline.json")
    assert pipeline["config"]["train_end"] == record["fitted_through"]
    assert pipeline["last_date"] == record["fitted_through"]
    assert record["checks_passed"] and record["checks"]["schema"]["all"]
    assert record["checks"]["symmetry"]["swap_is_exact"]
    assert manifest["dataset"]["n_training_bouts"] == record["n_training_bouts"]
    assert not (refitted["models"] / registry.APPROVED_FILE).exists()
    # The feature file is still the candidate's, so evaluate's check still holds.
    frame = pd.read_parquet(refitted["features"] / FEATURES_FILE)
    assert (
        registry.dataset_hash(frame, old.spec)
        == old_manifest["dataset"]["feature_table_sha256"]
    )


def test_a_refit_cannot_be_evaluated_or_refitted_again(refitted, capsys):
    version = refitted["refit"]["version"]
    ledger = len(registry.read_ledger(refitted["models"]))
    assert cli.main([*refitted["evaluate_args"], "--version", version]) == 1
    assert "no held-out test" in capsys.readouterr().err
    assert len(registry.read_ledger(refitted["models"])) == ledger
    assert cli.main([*refitted["refit_args"], "--candidate", version]) == 1
    assert "already a refit" in capsys.readouterr().err


def test_refit_again_gives_the_same_version(refitted, capsys):
    assert cli.main(refitted["refit_args"]) == 0
    assert "already there, unchanged" in capsys.readouterr().out
