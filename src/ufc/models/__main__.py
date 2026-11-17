"""Command line for model selection, evaluation and the report.

uv run python -m ufc.models train       select on validation, save the candidate
uv run python -m ufc.models train --recipe-of VERSION
                                        the same, with VERSION's recipe fixed and
                                        the current feature families (D58)
uv run python -m ufc.models evaluate    score the saved candidate on the test period
uv run python -m ufc.models report      write the generated part of docs/MODEL_REPORT.md
uv run python -m ufc.models refit       refit the candidate's recipe on all data (D47)

``train`` never reads a test-period row. ``evaluate`` is the only command
that unlocks the test period, and it scores the candidate exactly as it was
saved in the registry.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ufc.db.connection import connect
from ufc.features.build import (
    FEATURES_FILE,
    build_features,
    build_meta,
    mirror,
    write_features,
)
from ufc.features.config import FeatureConfig
from ufc.features.data import FeatureData, load_feature_data
from ufc.features.matchup import (
    FAMILY_LEVEL_NAMES,
    FEATURE_NAMES,
    LEVEL_NAMES,
    SHRUNK_FEATURE_NAMES,
)
from ufc.models import plots, registry
from ufc.models.design import FeatureSpec
from ufc.models.evaluate import evaluate_period
from ufc.models.gbm import MAX_TREES, PATIENCE
from ufc.models.refit import STATUS as REFIT_STATUS
from ufc.models.refit import compare_predictions, refit, schema_check, symmetry_check
from ufc.models.report import write_report
from ufc.models.select import (
    Selection,
    SelectionSettings,
    add_levels,
    baseline_predictions,
    select,
    select_fixed,
)
from ufc.models.splits import (
    SPLIT_DATES,
    Splits,
    development_only,
    fit_rows,
    labelled,
)
from ufc.models.tune import SEARCH_SPACE
from ufc.models.walkforward import FIRST_YEAR, walk_forward

DEFAULT_DB = Path("data/ufc.db")
DEFAULT_FEATURES = Path("data/features")
DEFAULT_REPORTS = Path("data/reports/eval")
DEFAULT_REPORT_DOC = Path("docs/MODEL_REPORT.md")

SETTINGS = SelectionSettings()

SELECTION_FILE = "selection.json"
TEST_FILE = "test_metrics.json"
WALK_FORWARD_FILE = "walk_forward.json"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_data(db_path: Path) -> FeatureData | None:
    if not db_path.exists():
        print(f"No database at {db_path}; run the ingest load first.", file=sys.stderr)
        return None
    conn = connect(db_path)
    try:
        return load_feature_data(conn)
    finally:
        conn.close()


def feature_config_from(meta: dict) -> FeatureConfig:
    """The feature settings a saved model's inputs were built with."""
    raw = dict(meta["config"])
    raw["train_end"] = date.fromisoformat(raw["train_end"])
    return FeatureConfig(**raw)


# --- train ---------------------------------------------------------------------


def run_train(args: argparse.Namespace) -> int:
    if args.recipe_of:
        return run_train_fixed(args)
    data = _load_data(args.db)
    if data is None:
        return 1

    # One build per shrinkage strength. The fitted parameters do not depend on
    # the strength, so they are fitted once and reused.
    grid = sorted(SETTINGS.shrink_grid)
    first = build_features(data, FeatureConfig(shrink_minutes=grid[0]))
    builds = {grid[0]: first}
    for minutes in grid[1:]:
        builds[minutes] = replace(
            build_features(data, FeatureConfig(shrink_minutes=minutes), first.params),
            elo_fit=first.elo_fit,
        )
    # Selection is handed tables with every test-period row removed.
    splits = {m: Splits(development_only(b.frame)) for m, b in builds.items()}
    meta = build_meta(first, data.source_commit)

    selection = select(
        lambda minutes: splits[minutes],
        FEATURE_NAMES,
        meta["families"],
        LEVEL_NAMES,
        SHRUNK_FEATURE_NAMES,
        family_levels=FAMILY_LEVEL_NAMES,
        settings=SETTINGS,
        n_jobs=args.jobs,
    )
    return _save_selection(
        args,
        data,
        selection,
        builds[selection.shrink_minutes],
        splits[selection.shrink_minutes],
        meta,
    )


def run_train_fixed(args: argparse.Namespace) -> int:
    """A candidate with a recorded candidate's recipe and today's feature
    families: nothing re-tuned, trees and calibrator fitted on validation
    (DECISIONS D58). Stages 1 to 3 of the selection record are carried over
    from the parent's, read from ``--parent-reports`` (default
    ``--reports``)."""
    parent_version = args.recipe_of
    parent, parent_manifest = registry.load_candidate(args.models, parent_version)
    if registry.is_refit(parent_manifest):
        print(f"{parent_version} is a refit; name a candidate.", file=sys.stderr)
        return 1
    carried_path = (args.parent_reports or args.reports) / SELECTION_FILE
    carried = json.loads(carried_path.read_text())
    if carried["version"] != parent_version:
        print(
            f"{carried_path} records {carried['version']}, not {parent_version}.",
            file=sys.stderr,
        )
        return 1
    parent_config = registry.read_json(args.models, parent_version, "config.json")
    parent_pipeline = registry.read_json(
        args.models, parent_version, "feature_pipeline.json"
    )
    data = _load_data(args.db)
    if data is None:
        return 1

    build = build_features(data, feature_config_from(parent_pipeline))
    splits = Splits(development_only(build.frame))
    # The parent's inputs must come out of today's build unchanged.
    upto = build.frame[
        build.frame["date"] <= pd.Timestamp(parent_pipeline["last_date"])
    ]
    if (
        registry.dataset_hash(upto, parent.spec)
        != parent_manifest["dataset"]["feature_table_sha256"]
    ):
        print(
            f"Today's build does not reproduce {parent_version}'s inputs.",
            file=sys.stderr,
        )
        return 1
    meta = build_meta(build, data.source_commit)
    base = FeatureSpec(FEATURE_NAMES, FAMILY_LEVEL_NAMES)
    spec = add_levels(base, LEVEL_NAMES) if carried["levels"]["kept"] else base
    selection = select_fixed(
        splits,
        parent.recipe,
        spec,
        meta["families"],
        carried,
        parent_version,
        seeds=SETTINGS.seeds,
        n_jobs=args.jobs,
    )
    assert selection.shrink_minutes == parent_config["shrink_minutes"]
    return _save_selection(args, data, selection, build, splits, meta)


def _save_selection(
    args: argparse.Namespace,
    data: FeatureData,
    selection: Selection,
    chosen,
    kept: Splits,
    meta: dict,
) -> int:
    features_path, meta_path = write_features(chosen, args.features, data.source_commit)
    chosen_meta = json.loads(meta_path.read_text())

    kept.assert_test_locked("saving the candidate")
    spec = selection.recipe.spec
    train, valid = fit_rows(kept.train()), labelled(kept.valid())
    recipe = selection.recipe.to_dict()
    config = {
        "recipe": {k: recipe[k] for k in ("gbm", "half_life_years", "calibration")},
        "n_trees": selection.candidate.model.n_trees,
        "splits": SPLIT_DATES.to_dict(),
        "shrink_minutes": selection.shrink_minutes,
        "early_stopping": {"max_trees": MAX_TREES, "patience": PATIENCE},
        "baseline_logistic_c": selection.logistic_c,
        "selection": {**SETTINGS.to_dict(), "search_space": SEARCH_SPACE},
    }
    if "fixed_recipe" in selection.report:
        config["fixed_recipe"] = selection.report["fixed_recipe"]
    report = selection.report
    saved = registry.save_candidate(
        args.models,
        selection.candidate,
        config=config,
        feature_pipeline=chosen_meta,
        metrics={
            "validation": report["validation"],
            "versus_baselines": report["versus_baselines"],
            "calibration": report["calibration"],
        },
        dataset={
            "training_sha256": registry.dataset_hash(train, spec),
            "validation_sha256": registry.dataset_hash(valid, spec),
            "feature_table_sha256": registry.dataset_hash(chosen.frame, spec),
            "n_training_bouts": len(train),
            "n_validation_bouts": len(valid),
            "source_commit": data.source_commit,
        },
        git=registry.git_state(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )

    report["version"] = saved.version
    report["source_commit"] = data.source_commit
    report["feature_meta"] = {
        "n_features": len(spec.names),
        "families": {name: len(members) for name, members in meta["families"].items()},
        "elo": chosen_meta["params"]["elo"],
        "elo_fit": chosen_meta.get("elo_fit"),
        "rate_prior_classes": chosen_meta["params"]["rate_priors"]["n_bouts"],
    }
    _write_json(args.reports / SELECTION_FILE, report)
    plots.period_plots(
        report["validation"], args.reports / "plots" / "validation", "Validation period"
    )
    plots.ablation(
        report["ablation"], args.reports / "plots" / "validation" / "ablation.png"
    )

    selected, versus = report["selected"], report["versus_baselines"]
    print(f"Snapshot commit: {data.source_commit}")
    print(f"Rate shrinkage chosen: {selection.shrink_minutes:g} cage minutes")
    print(f"Level features kept: {report['levels']['kept']}")
    print(
        f"Selected: {selected['recipe']['gbm']['library']}, half-life "
        f"{selected['recipe']['half_life_years']}, {selected['n_trees']} trees, "
        f"calibration {selected['recipe']['calibration']}"
    )
    print(
        f"Validation log loss (seed mean, uncalibrated): {versus['candidate_log_loss']:.4f}"
    )
    for name, row in versus["baselines"].items():
        print(f"  {name}: {row['log_loss']:.4f}")
    print(f"Beats both baselines on validation: {versus['beats_all']}")
    print(
        f"Saved {saved.version} to {saved.directory}"
        + ("" if saved.created else " (already there, unchanged)")
    )
    print(f"Features: {features_path}")
    print(f"Selection record: {args.reports / SELECTION_FILE}")
    return 0


# --- evaluate ------------------------------------------------------------------


def _symmetry_check(candidate, frame: pd.DataFrame) -> dict:
    """The specified symmetry test, on the bouts being evaluated: P(A) + P(B)
    is exactly 1, and presenting every bout the other way round swaps the two
    probabilities exactly."""
    p_a, p_b = candidate.predict_pair(frame)
    q_a, q_b = candidate.predict_pair(mirror(frame))
    return {
        "n_bouts": len(frame),
        "sum_is_exactly_one": bool(np.all(p_a + p_b == 1.0)),
        "swap_is_exact": bool(np.array_equal(p_a, q_b) and np.array_equal(p_b, q_a)),
    }


def run_evaluate(args: argparse.Namespace) -> int:
    selection_path = args.reports / SELECTION_FILE
    if not selection_path.exists():
        print(f"No {selection_path}; run train first.", file=sys.stderr)
        return 1
    version = args.version or json.loads(selection_path.read_text())["version"]
    candidate, manifest = registry.load_candidate(args.models, version)
    if registry.is_refit(manifest):
        print(
            f"{version} is a production refit, fitted on the test period; it has "
            "no held-out test to evaluate on.",
            file=sys.stderr,
        )
        return 1
    config = registry.read_json(args.models, version, "config.json")
    feature_meta = registry.read_json(args.models, version, "feature_pipeline.json")
    spec = candidate.spec

    frame = pd.read_parquet(args.features / FEATURES_FILE)
    if (
        registry.dataset_hash(frame, spec)
        != manifest["dataset"]["feature_table_sha256"]
    ):
        print(
            f"{args.features / FEATURES_FILE} is not the table {version} was trained "
            "from. Run train again, or restore the feature file.",
            file=sys.stderr,
        )
        return 1
    data = _load_data(args.db)
    if data is None:
        return 1

    splits = Splits(frame)
    train = fit_rows(splits.train())
    assert registry.dataset_hash(train, spec) == manifest["dataset"]["training_sha256"]

    # From here on the test period is open, for this saved version only.
    splits.unlock_test(version)
    test = splits.test()
    scored = labelled(test)
    predictions = {
        "model": candidate.predict(scored),
        "model_uncalibrated": candidate.predict(scored, calibrated=False),
        **baseline_predictions(train, scored, spec, config["baseline_logistic_c"]),
    }
    record = evaluate_period(test, predictions, primary="model")
    record["version"] = version
    record["symmetry"] = _symmetry_check(candidate, test)

    last_year = int(frame["date"].dt.year.max())
    first_year = SPLIT_DATES.test_start.year
    backtest, raw = walk_forward(
        data,
        candidate.recipe,
        feature_config_from(feature_meta),
        config["baseline_logistic_c"],
        list(range(min(FIRST_YEAR, first_year), last_year + 1)),
        unlocked_for=version,
        n_jobs=args.jobs,
    )
    # The first test-period fold repeats the registered model's own schedule,
    # so it must reproduce the registered model's probabilities.
    in_year = (scored["date"].dt.year == first_year).to_numpy()
    same_bouts = list(scored.loc[in_year, "bout_id"]) == raw[first_year]["bout_ids"]
    backtest["first_test_year_check"] = {
        "year": first_year,
        "same_bouts": same_bouts,
        "identical_to_registered_model": bool(
            same_bouts
            and np.array_equal(
                raw[first_year]["predictions"]["model"], predictions["model"][in_year]
            )
        ),
    }
    backtest["version"] = version

    registry.record_test_metrics(args.models, version, record)
    _write_json(args.reports / TEST_FILE, record)
    _write_json(args.reports / WALK_FORWARD_FILE, backtest)
    plots.period_plots(record, args.reports / "plots" / "test", "Test period")
    plots.walk_forward(backtest, args.reports / "plots" / "walk_forward.png")
    looks = registry.append_ledger(
        args.models,
        {
            "version": version,
            "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **{f"git_{k}": v for k, v in registry.git_state().items()},
            "log_loss": record["models"]["model"]["log_loss"],
        },
    )

    print(f"Evaluated {version} on {record['n_scored']} test-period bouts")
    print(
        f"({record['first_date']} to {record['last_date']}; {record['n_no_result']} no result)"
    )
    for name, row in record["models"].items():
        print(
            f"  {name:20} log loss {row['log_loss']:.4f}  Brier {row['brier']:.4f}  "
            f"accuracy {row['accuracy']:.3f}  ECE {row['ece']:.3f}"
        )
    print(f"Symmetry: {record['symmetry']}")
    print(f"Walk-forward check: {backtest['first_test_year_check']}")
    print(f"Test-period evaluations recorded so far: {looks}")
    print(f"Metrics and plots: {args.reports}")
    return 0


# --- refit ---------------------------------------------------------------------

REFIT_FILE = "refit.json"


def run_refit(args: argparse.Namespace) -> int:
    selection_path = args.reports / SELECTION_FILE
    if args.candidate is None and not selection_path.exists():
        print(f"No {selection_path}; run train first.", file=sys.stderr)
        return 1
    parent = args.candidate or json.loads(selection_path.read_text())["version"]
    old, old_manifest = registry.load_candidate(args.models, parent)
    if registry.is_refit(old_manifest):
        print(f"{parent} is already a refit; refit a candidate.", file=sys.stderr)
        return 1
    old_config = registry.read_json(args.models, parent, "config.json")
    old_pipeline = registry.read_json(args.models, parent, "feature_pipeline.json")
    data = _load_data(args.db)
    if data is None:
        return 1

    # The candidate's own inputs, rebuilt from its stored settings. Rows up to
    # its last bout must be the table it was trained from (later bouts, if the
    # snapshot is newer, are added rows and change nothing before them).
    old_build = build_features(data, feature_config_from(old_pipeline))
    old_frame = old_build.frame
    upto = old_frame[old_frame["date"] <= pd.Timestamp(old_pipeline["last_date"])]
    if (
        registry.dataset_hash(upto, old.spec)
        != old_manifest["dataset"]["feature_table_sha256"]
    ):
        print(
            f"The database does not rebuild the table {parent} was trained from.",
            file=sys.stderr,
        )
        return 1

    reason = f"production refit of {parent}"
    result = refit(data, old.recipe, feature_config_from(old_pipeline), reason)
    frame = result.final.frame
    meta = build_meta(result.final, data.source_commit)
    last = frame["date"].max()
    recent = frame[frame["date"] > last - pd.Timedelta(days=365)]
    checks = {
        "symmetry": symmetry_check(result.candidate, recent),
        "schema": schema_check(
            result.candidate.spec.to_dict(),
            old.spec.to_dict(),
            meta,
            old_pipeline,
            frame,
            old_frame,
        ),
        "versus_candidate": compare_predictions(
            result.candidate, frame, old, old_frame, int(last.year)
        ),
    }
    passed = (
        checks["symmetry"]["sum_is_exactly_one"]
        and checks["symmetry"]["swap_is_exact"]
        and checks["schema"]["all"]
    )
    record = {
        "status": REFIT_STATUS,
        "parent": parent,
        "fitted_through": last.date().isoformat(),
        "source_commit": data.source_commit,
        "n_training_bouts": len(result.final_rows),
        "trees": result.trees,
        "calibration": result.calibration,
        "sharpness": result.sharpness,
        "checks": checks,
        "checks_passed": passed,
    }
    if not passed:
        _write_json(args.reports / REFIT_FILE, record)
        print("A sanity check failed; nothing was saved.", file=sys.stderr)
        print(json.dumps(checks, indent=2), file=sys.stderr)
        return 1

    spec = result.candidate.spec
    config = {
        "recipe": old_config["recipe"],
        "n_trees": result.candidate.model.n_trees,
        "shrink_minutes": old_config["shrink_minutes"],
        "early_stopping": old_config["early_stopping"],
        "baseline_logistic_c": old_config["baseline_logistic_c"],
        "production_refit": {
            "status": REFIT_STATUS,
            "parent": parent,
            "fitted_through": record["fitted_through"],
            "trees": result.trees,
            "calibration": result.calibration,
        },
    }
    saved = registry.save_candidate(
        args.models,
        result.candidate,
        config=config,
        feature_pipeline=meta,
        metrics={
            "validation": None,
            "note": (
                "No held-out metrics: every period was used for fitting. The "
                f"recipe was validated and tested as {parent}."
            ),
            **{k: record[k] for k in ("calibration", "sharpness", "checks")},
        },
        dataset={
            "training_sha256": registry.dataset_hash(result.final_rows, spec),
            "feature_table_sha256": registry.dataset_hash(frame, spec),
            "n_training_bouts": len(result.final_rows),
            "last_event": record["fitted_through"],
            "source_commit": data.source_commit,
        },
        git=registry.git_state(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        lineage={
            "parent": parent,
            "parent_training_sha256": old_manifest["dataset"]["training_sha256"],
            "status": REFIT_STATUS,
        },
    )
    record["version"] = saved.version
    _write_json(args.reports / REFIT_FILE, record)

    cal, versus = record["calibration"], checks["versus_candidate"]
    print(f"Refit of {parent} on bouts through {record['fitted_through']}")
    print(
        f"  final booster: {saved.manifest['n_trees']} trees on "
        f"{record['n_training_bouts']} bouts"
    )
    print(
        f"  Platt slope {cal['slope']:.4f} from a model fitted through "
        f"{cal['model_fitted_through']} ({cal['model_n_trees']} trees), on "
        f"{cal['n_bouts']} later bouts"
    )
    print(f"  symmetry: {checks['symmetry']}")
    print(f"  schema matches the candidate: {checks['schema']['all']}")
    print(
        f"  versus the candidate on {versus['year']}: correlation "
        f"{versus['correlation']:.4f}, mean absolute difference "
        f"{versus['mean_abs_difference']:.4f}"
    )
    print(
        f"Saved {saved.version}"
        + ("" if saved.created else " (already there, unchanged)")
        + f"; promotion is a manual step. Record: {args.reports / REFIT_FILE}"
    )
    return 0


# --- report --------------------------------------------------------------------


def run_report(args: argparse.Namespace) -> int:
    selection_path = args.reports / SELECTION_FILE
    if not selection_path.exists():
        print(f"No {selection_path}; run train first.", file=sys.stderr)
        return 1

    def optional(name: str) -> dict | None:
        path = args.reports / name
        return json.loads(path.read_text()) if path.exists() else None

    selection = json.loads(selection_path.read_text())
    version = selection["version"]
    test, backtest = optional(TEST_FILE), optional(WALK_FORWARD_FILE)
    write_report(
        args.out,
        selection=selection,
        manifest=registry.read_manifest(args.models, version),
        config=registry.read_json(args.models, version, "config.json"),
        test=test,
        backtest=backtest,
        ledger=registry.read_ledger(args.models),
    )
    # Plots are a function of the same records, so they are redrawn here and
    # always match the numbers in the report.
    directory = args.reports / "plots"
    plots.period_plots(
        selection["validation"], directory / "validation", "Validation period"
    )
    plots.ablation(selection["ablation"], directory / "validation" / "ablation.png")
    if test is not None:
        plots.period_plots(test, directory / "test", "Test period")
    if backtest is not None:
        plots.walk_forward(backtest, directory / "walk_forward.png")
    print(f"Wrote the generated block of {args.out}")
    print(f"Plots: {directory}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ufc.models", description=__doc__.split("\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--models", type=Path, default=registry.DEFAULT_ROOT)
        sub.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)

    def inputs(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--db", type=Path, default=DEFAULT_DB)
        sub.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
        sub.add_argument(
            "--jobs",
            type=int,
            default=min(8, os.cpu_count() or 1),
            help="parallel fits; results do not depend on it",
        )

    train = commands.add_parser(
        "train", help="select on validation and save a candidate"
    )
    common(train)
    inputs(train)
    train.add_argument(
        "--recipe-of",
        metavar="VERSION",
        help="keep this candidate's recipe and add today's feature families; "
        "nothing is re-tuned (D58)",
    )
    train.add_argument(
        "--parent-reports",
        type=Path,
        help="where the parent's selection.json is (default: --reports)",
    )
    evaluate = commands.add_parser(
        "evaluate", help="score the saved candidate on the test period, once"
    )
    common(evaluate)
    inputs(evaluate)
    evaluate.add_argument("--version", help="default: the candidate train last saved")
    refit_cmd = commands.add_parser(
        "refit", help="refit the candidate's recipe on all data"
    )
    common(refit_cmd)
    inputs(refit_cmd)
    refit_cmd.add_argument(
        "--candidate", help="default: the candidate train last saved"
    )
    report = commands.add_parser("report", help="write docs/MODEL_REPORT.md")
    common(report)
    report.add_argument("--out", type=Path, default=DEFAULT_REPORT_DOC)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)
    handlers = {
        "train": run_train,
        "evaluate": run_evaluate,
        "refit": run_refit,
        "report": run_report,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
