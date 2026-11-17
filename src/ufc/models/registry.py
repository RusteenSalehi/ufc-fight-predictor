"""The model registry: one directory per trained model under ``data/models``.

docs/ML_REQUIREMENTS.md, "Model registry". A version directory holds:

- ``model.json`` or ``model.txt``: the booster, in its library's own format
- ``calibrator.json``: the calibration method and its numbers
- ``features.json``: the ordered feature list, and which columns change sign
  when the fighters swap
- ``config.json``: the training configuration (settings, seed, half-life,
  splits, shrinkage, how it was selected)
- ``feature_pipeline.json``: the feature settings and fitted parameters the
  model's inputs were built with, so prediction can rebuild them exactly
- ``metrics.json``: validation metrics
- ``manifest.json``: version, creation time, git commit, hashes of the
  training and validation rows, library versions, SHA-256 of every file above,
  and for a production refit its ``lineage``: the candidate it refits and a
  status line saying it has no held-out test
- ``test_metrics.json``: added once by the final evaluation

``approved.json`` next to the version directories names the approved version.
Promotion and rollback are manual edits of that file; nothing in this module
writes it.

A version is never overwritten. Its name is derived from the training data
and the configuration, so the same inputs give the same name, and saving the
same model again is a no-op.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pandas as pd

from ufc.models.calibrate import Calibrator
from ufc.models.design import FeatureSpec, matrix
from ufc.models.gbm import MODEL_FILES, GbmModel
from ufc.models.recipe import Candidate, Recipe

DEFAULT_ROOT = Path("data/models")
APPROVED_FILE = "approved.json"
LEDGER_FILE = "test_evaluations.jsonl"
MANIFEST = "manifest.json"
TEST_METRICS = "test_metrics.json"
_PACKAGES = ("xgboost", "lightgbm", "numpy", "pandas", "scikit-learn")


class RegistryError(RuntimeError):
    """A version is missing, damaged, or would be overwritten."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def dataset_hash(frame: pd.DataFrame, spec: FeatureSpec) -> str:
    """SHA-256 of the rows a model saw: bout IDs, labels, and the values of
    its features, in order. Independent of file format and column order."""
    digest = hashlib.sha256()
    digest.update("\n".join(spec.names).encode())
    digest.update("\n".join(frame["bout_id"].astype(str)).encode())
    digest.update(np.ascontiguousarray(frame["label"].to_numpy(dtype="<f8")).tobytes())
    digest.update(matrix(frame, spec).astype("<f8").tobytes())
    return digest.hexdigest()


def git_state(repo: Path | None = None) -> dict:
    """The current commit and whether the working tree has uncommitted
    changes. Both are None outside a git checkout."""

    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args],
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit.strip() if commit else None,
        "dirty": bool(status.strip()) if status is not None else None,
    }


def library_versions() -> dict[str, str]:
    versions = {name: package_version(name) for name in _PACKAGES}
    versions["python"] = platform.python_version()
    return versions


def version_name(library: str, training_sha256: str, config: Mapping) -> str:
    config_sha = hashlib.sha256(_dump(config).encode()).hexdigest()
    return f"{library}-{training_sha256[:8]}-{config_sha[:8]}"


@dataclass(frozen=True, slots=True)
class SavedModel:
    version: str
    directory: Path
    manifest: dict
    created: bool  # False when this exact model was already in the registry


def save_candidate(
    root: Path,
    candidate: Candidate,
    *,
    config: Mapping,
    feature_pipeline: Mapping,
    metrics: Mapping,
    dataset: Mapping,
    git: Mapping,
    created_at: str,
    lineage: Mapping | None = None,
) -> SavedModel:
    """Write a new version directory. ``dataset`` must hold
    ``training_sha256``; ``config`` must hold the recipe under ``recipe``.
    ``lineage`` (a production refit only) names the version whose recipe it
    refits and must hold ``parent`` and ``status``."""
    if lineage is not None and not {"parent", "status"} <= set(lineage):
        raise ValueError("lineage must name the parent version and a status")
    library = candidate.model.library
    version = version_name(library, dataset["training_sha256"], config)
    directory = root / version
    staging = root / f".{version}.tmp"
    if staging.exists():
        for leftover in staging.iterdir():
            leftover.unlink()
    staging.mkdir(parents=True, exist_ok=True)

    candidate.model.save(staging)
    (staging / "calibrator.json").write_text(_dump(candidate.calibrator.to_dict()))
    (staging / "features.json").write_text(_dump(candidate.spec.to_dict()))
    (staging / "config.json").write_text(_dump(config))
    (staging / "feature_pipeline.json").write_text(_dump(feature_pipeline))
    (staging / "metrics.json").write_text(_dump(metrics))
    files = {p.name: file_sha256(p) for p in sorted(staging.iterdir())}
    manifest = {
        "version": version,
        "created_at": created_at,
        "library": library,
        "n_trees": candidate.model.n_trees,
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "dataset": dict(dataset),
        "library_versions": library_versions(),
        "files": files,
    }
    if lineage is not None:
        manifest["lineage"] = dict(lineage)

    if directory.exists():
        existing = read_manifest(root, version)
        for leftover in staging.iterdir():
            leftover.unlink()
        staging.rmdir()
        if existing["files"] != files:
            raise RegistryError(
                f"{version} already exists with different contents; a saved "
                "version is never overwritten"
            )
        return SavedModel(version, directory, existing, created=False)

    (staging / MANIFEST).write_text(_dump(manifest))
    staging.rename(directory)
    return SavedModel(version, directory, manifest, created=True)


def read_manifest(root: Path, version: str) -> dict:
    path = root / version / MANIFEST
    if not path.exists():
        raise RegistryError(f"no model version {version!r} under {root}")
    return json.loads(path.read_text())


def read_json(root: Path, version: str, name: str) -> dict:
    return json.loads((root / version / name).read_text())


def load_candidate(root: Path, version: str) -> tuple[Candidate, dict]:
    """Load a version after checking every file against the manifest."""
    manifest = read_manifest(root, version)
    directory = root / version
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.exists() or file_sha256(path) != expected:
            raise RegistryError(f"{version}: {name} is missing or has been changed")
    config = read_json(root, version, "config.json")
    spec = FeatureSpec.from_dict(read_json(root, version, "features.json"))
    recipe = Recipe.from_dict({**config["recipe"], **spec.to_dict()})
    model = GbmModel.load(
        manifest["library"], directory / MODEL_FILES[manifest["library"]]
    )
    if model.n_trees != manifest["n_trees"]:
        raise RegistryError(
            f"{version}: the model file does not hold the recorded trees"
        )
    calibrator = Calibrator.from_dict(read_json(root, version, "calibrator.json"))
    return Candidate(recipe, model, calibrator), manifest


def is_refit(manifest: Mapping) -> bool:
    """A production refit: fitted on every period, so it has no test."""
    return "lineage" in manifest


def list_versions(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(p.parent.name for p in root.glob(f"*/{MANIFEST}"))


def approved_version(root: Path) -> str:
    """The version named by ``approved.json``.

    The file is written by hand: ``{"version": "<version name>"}``.
    """
    path = root / APPROVED_FILE
    if not path.exists():
        raise RegistryError(
            f"{path} does not exist. No model has been approved. To approve "
            'one, create it by hand with {"version": "<name>"}; available: '
            f"{list_versions(root) or 'none'}"
        )
    version = json.loads(path.read_text()).get("version")
    if not version or version not in list_versions(root):
        raise RegistryError(f"{path} names {version!r}, which is not in the registry")
    return version


def load_approved(root: Path) -> tuple[Candidate, dict]:
    return load_candidate(root, approved_version(root))


def record_test_metrics(root: Path, version: str, metrics: Mapping) -> Path:
    """Attach the test-period metrics to a version, once.

    Evaluating the same version again must reproduce the same numbers; if it
    does not, something is not deterministic and this raises.
    """
    path = root / version / TEST_METRICS
    text = _dump(metrics)
    if path.exists():
        if path.read_text() != text:
            raise RegistryError(
                f"{version} already has test metrics and the new ones differ"
            )
        return path
    path.write_text(text)
    return path


def append_ledger(root: Path, entry: Mapping) -> int:
    """Record that the test period was looked at. Returns how many times it
    has been, over all versions. The ledger only ever grows."""
    path = root / LEDGER_FILE
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(entry), sort_keys=True) + "\n")
    return len(read_ledger(root))


def read_ledger(root: Path) -> list[dict]:
    path = root / LEDGER_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
