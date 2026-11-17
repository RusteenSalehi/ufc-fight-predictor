"""The model registry: save, load, integrity, and the approval pointer."""

import json

import numpy as np
import pytest

from tests.model_helpers import FAST, FEATURES, LEVELS, synthetic_frame
from ufc.models import registry
from ufc.models.calibrate import ISOTONIC, PLATT
from ufc.models.design import FeatureSpec
from ufc.models.gbm import LIBRARIES, MODEL_FILES
from ufc.models.recipe import Recipe, fit_recipe
from ufc.models.registry import RegistryError
from ufc.models.splits import SPLIT_DATES, Splits, labelled

GIT = {"commit": "0123abc", "dirty": False}
CREATED = "2026-11-17T00:00:00+00:00"


@pytest.fixture(scope="module")
def frame():
    return synthetic_frame(seed=8)


@pytest.fixture(scope="module")
def rows(frame):
    splits = Splits(frame)
    return labelled(splits.train()), labelled(splits.valid())


def fitted(rows, library="xgboost", calibration=PLATT, half_life=4.0):
    recipe = Recipe(
        FAST[library], half_life, FeatureSpec(FEATURES, LEVELS), calibration
    )
    return fit_recipe(recipe, *rows, SPLIT_DATES.train_end)[0]


def save(root, candidate, rows, **overrides):
    recipe = candidate.recipe.to_dict()
    arguments = {
        "config": {
            "recipe": {k: recipe[k] for k in ("gbm", "half_life_years", "calibration")},
            "n_trees": candidate.model.n_trees,
            "shrink_minutes": 15.0,
        },
        "feature_pipeline": {"config": {"shrink_minutes": 15.0}},
        "metrics": {"validation": {"log_loss": 0.66}},
        "dataset": {
            "training_sha256": registry.dataset_hash(rows[0], candidate.spec),
            "source_commit": "abc123",
        },
        "git": GIT,
        "created_at": CREATED,
    }
    arguments.update(overrides)
    return registry.save_candidate(root, candidate, **arguments)


@pytest.mark.parametrize("library", LIBRARIES)
@pytest.mark.parametrize("calibration", [PLATT, ISOTONIC])
def test_saved_model_loads_back_with_identical_predictions(
    rows, frame, tmp_path, library, calibration
):
    candidate = fitted(rows, library, calibration)
    saved = save(tmp_path, candidate, rows)
    assert saved.created and saved.directory == tmp_path / saved.version
    assert saved.version.startswith(f"{library}-")
    assert {p.name for p in saved.directory.iterdir()} == {
        MODEL_FILES[library],
        "calibrator.json",
        "features.json",
        "config.json",
        "feature_pipeline.json",
        "metrics.json",
        "manifest.json",
    }
    loaded, manifest = registry.load_candidate(tmp_path, saved.version)
    assert loaded.recipe == candidate.recipe
    assert loaded.calibrator == candidate.calibrator
    for calibrated in (True, False):
        for original, restored in zip(
            candidate.predict_pair(frame, calibrated),
            loaded.predict_pair(frame, calibrated),
            strict=True,
        ):
            np.testing.assert_array_equal(restored, original)
    p_a, p_b = loaded.predict_pair(frame)
    assert np.all(p_a + p_b == 1.0)
    assert manifest == saved.manifest
    assert not list(tmp_path.glob(".*.tmp"))


def test_manifest_records_what_the_requirements_ask_for(rows, tmp_path):
    candidate = fitted(rows)
    saved = save(tmp_path, candidate, rows)
    manifest = saved.manifest
    assert manifest["git_commit"] == "0123abc" and manifest["git_dirty"] is False
    assert manifest["created_at"] == CREATED
    assert manifest["dataset"]["training_sha256"] == registry.dataset_hash(
        rows[0], candidate.spec
    )
    assert manifest["n_trees"] == candidate.model.n_trees
    assert {"xgboost", "lightgbm", "numpy", "python"} <= set(
        manifest["library_versions"]
    )
    # Ordered feature list, with which columns flip sign on a swap.
    features = registry.read_json(tmp_path, saved.version, "features.json")
    assert features["features"] == [*FEATURES, *LEVELS]
    assert features["antisymmetric"] == list(FEATURES)
    assert features["symmetric"] == list(LEVELS)
    config = registry.read_json(tmp_path, saved.version, "config.json")
    assert config["recipe"]["half_life_years"] == 4.0
    assert config["recipe"]["gbm"]["seed"] == 0
    assert registry.read_json(tmp_path, saved.version, "metrics.json")["validation"]
    for name, digest in manifest["files"].items():
        assert registry.file_sha256(saved.directory / name) == digest
    assert "manifest.json" not in manifest["files"]


def test_dataset_hash_follows_the_rows_and_features_a_model_saw(rows):
    train, _ = rows
    spec = FeatureSpec(FEATURES)
    first = registry.dataset_hash(train, spec)
    assert first == registry.dataset_hash(train.copy(), spec)
    assert len(first) == 64
    assert first != registry.dataset_hash(train.iloc[:-1], spec)
    assert first != registry.dataset_hash(train, FeatureSpec(FEATURES[:-1]))
    changed = train.copy()
    changed.iloc[0, changed.columns.get_loc("d_age")] += 1e-9
    assert first != registry.dataset_hash(changed, spec)
    flipped = train.copy()
    flipped.iloc[0, flipped.columns.get_loc("label")] = 1 - flipped["label"].iloc[0]
    assert first != registry.dataset_hash(flipped, spec)
    # A column the model does not read does not matter.
    other = train.copy()
    other["lvl_age"] = 0.0
    assert first == registry.dataset_hash(other, spec)


def test_a_version_is_never_overwritten(rows, tmp_path):
    candidate = fitted(rows)
    saved = save(tmp_path, candidate, rows)
    # Same model again: nothing happens, the first manifest stands.
    again = save(tmp_path, candidate, rows, created_at="2030-01-01T00:00:00+00:00")
    assert not again.created and again.version == saved.version
    assert again.manifest["created_at"] == CREATED
    # Same name, different contents: refused.
    with pytest.raises(RegistryError, match="never overwritten"):
        save(tmp_path, candidate, rows, metrics={"validation": {"log_loss": 0.1}})
    assert registry.read_manifest(tmp_path, saved.version) == saved.manifest
    # A different configuration or different data is a different version.
    other = fitted(rows, half_life=None)
    assert save(tmp_path, other, rows).version != saved.version
    assert registry.list_versions(tmp_path) == sorted(
        [saved.version, save(tmp_path, other, rows).version]
    )


def test_a_changed_file_is_detected_on_load(rows, tmp_path):
    saved = save(tmp_path, fitted(rows), rows)
    calibrator = saved.directory / "calibrator.json"
    original = calibrator.read_text()
    calibrator.write_text(original.replace('"platt"', '"none"'))
    with pytest.raises(
        RegistryError, match=r"calibrator\.json is missing or has been changed"
    ):
        registry.load_candidate(tmp_path, saved.version)
    calibrator.write_text(original)
    registry.load_candidate(tmp_path, saved.version)
    (saved.directory / "features.json").unlink()
    with pytest.raises(RegistryError, match=r"features\.json"):
        registry.load_candidate(tmp_path, saved.version)
    with pytest.raises(RegistryError, match="no model version"):
        registry.load_candidate(tmp_path, "xgboost-nope")


def test_approval_is_a_hand_written_pointer_that_this_code_only_reads(rows, tmp_path):
    saved = save(tmp_path, fitted(rows), rows)
    # Saving a model does not approve it.
    assert not (tmp_path / registry.APPROVED_FILE).exists()
    with pytest.raises(RegistryError, match="No model has been approved"):
        registry.approved_version(tmp_path)

    (tmp_path / registry.APPROVED_FILE).write_text(
        json.dumps({"version": saved.version})
    )
    assert registry.approved_version(tmp_path) == saved.version
    candidate, manifest = registry.load_approved(tmp_path)
    assert manifest["version"] == saved.version
    assert candidate.model.n_trees == saved.manifest["n_trees"]

    (tmp_path / registry.APPROVED_FILE).write_text(json.dumps({"version": "gone"}))
    with pytest.raises(RegistryError, match="not in the registry"):
        registry.approved_version(tmp_path)
    assert registry.list_versions(tmp_path / "nowhere") == []


def test_nothing_in_the_package_writes_the_approval_pointer():
    """Promotion is a manual edit. No source file writes approved.json."""
    from pathlib import Path

    import ufc.models

    for path in Path(ufc.models.__file__).parent.glob("*.py"):
        source = path.read_text()
        if "approved" not in source.lower():
            continue
        assert path.name == "registry.py", path
        writers = [
            line
            for line in source.splitlines()
            if "APPROVED_FILE" in line and ("write" in line or "open(" in line)
        ]
        assert writers == [], writers


def test_test_metrics_are_attached_once_and_the_ledger_only_grows(rows, tmp_path):
    saved = save(tmp_path, fitted(rows), rows)
    metrics = {"models": {"model": {"log_loss": 0.67}}}
    path = registry.record_test_metrics(tmp_path, saved.version, metrics)
    assert json.loads(path.read_text()) == metrics
    # The same numbers again are fine; different numbers are not.
    registry.record_test_metrics(tmp_path, saved.version, metrics)
    with pytest.raises(RegistryError, match="differ"):
        registry.record_test_metrics(
            tmp_path, saved.version, {"models": {"model": {"log_loss": 0.66}}}
        )
    # The model files are untouched, so the version still loads.
    registry.load_candidate(tmp_path, saved.version)

    assert registry.read_ledger(tmp_path) == []
    assert (
        registry.append_ledger(tmp_path, {"version": saved.version, "log_loss": 0.67})
        == 1
    )
    assert (
        registry.append_ledger(tmp_path, {"version": saved.version, "log_loss": 0.67})
        == 2
    )
    assert [entry["version"] for entry in registry.read_ledger(tmp_path)] == [
        saved.version
    ] * 2


def test_git_state_reports_commit_and_dirtiness(tmp_path):
    state = registry.git_state()
    assert set(state) == {"commit", "dirty"}
    assert state["commit"] is None or len(state["commit"]) == 40
    outside = registry.git_state(tmp_path)
    assert outside == {"commit": None, "dirty": None}
