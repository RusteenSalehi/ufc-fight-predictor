"""Write the generated part of docs/MODEL_REPORT.md from the metrics records.

The block between the markers is rebuilt from ``data/reports/eval/*.json``
and the registry manifest, and from nothing else, so every number in it was
produced by ``python -m ufc.models``. Prose outside the markers is kept.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

BEGIN = "<!-- BEGIN GENERATED: python -m ufc.models report -->"
END = "<!-- END GENERATED -->"

MODEL_LABELS = {
    "model": "Model (calibrated)",
    "model_uncalibrated": "Model before calibration",
    "logistic": "Logistic regression",
    "elo": "Elo",
}
MODEL_ORDER = ("model", "model_uncalibrated", "logistic", "elo")
LIBRARY_LABELS = {"xgboost": "XGBoost", "lightgbm": "LightGBM"}


def f4(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def signed(value: float) -> str:
    return f"{value:+.4f}"


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def interval(low: float, high: float) -> str:
    return f"{signed(low)} to {signed(high)}"


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def cell(value: object) -> str:
        return (
            f"{value:,}"
            if isinstance(value, int) and not isinstance(value, bool)
            else str(value)
        )

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _half_life(value: float | None) -> str:
    return "none" if value is None else f"{value:g} years"


def _params(params: Mapping[str, float]) -> str:
    return ", ".join(f"{name} {value:g}" for name, value in params.items())


def _metric_rows(
    models: Mapping[str, Mapping], with_interval: bool
) -> list[list[object]]:
    rows = []
    for name in MODEL_ORDER:
        if name not in models:
            continue
        m = models[name]
        row: list[object] = [MODEL_LABELS[name], m["n"], f4(m["log_loss"])]
        if with_interval:
            row.append(f"{f4(m['log_loss_low'])} to {f4(m['log_loss_high'])}")
        row += [f4(m["brier"]), pct(m["accuracy"]), f4(m["ece"])]
        rows.append(row)
    return rows


def _group_table(groups: Mapping[str, Mapping[str, Mapping]], label: str) -> str:
    names = [n for n in ("model", "logistic", "elo")]
    headers = [label, "bouts"]
    for name in names:
        headers += [f"{MODEL_LABELS[name]} log loss", "accuracy"]
    rows = []
    for group, models in groups.items():
        row: list[object] = [group, models[names[0]]["n"]]
        for name in names:
            row += [f4(models[name]["log_loss"]), pct(models[name]["accuracy"])]
        rows.append(row)
    return table(headers, rows)


# --- sections ------------------------------------------------------------------


def _header(selection: Mapping, manifest: Mapping, config: Mapping) -> list[str]:
    splits = selection["splits"]
    recipe = selection["selected"]["recipe"]
    dataset = manifest["dataset"]
    meta = selection["feature_meta"]
    elo, elo_fit = meta["elo"], meta["elo_fit"]
    lines = [
        "## Candidate",
        "",
        f"- Registry version: `{manifest['version']}` (created {manifest['created_at']})",
        f"- Code: git commit `{manifest['git_commit']}`"
        + (" with uncommitted changes" if manifest["git_dirty"] else ""),
        (
            f"- Data: upstream snapshot commit `{dataset['source_commit']}`; training rows "
            f"SHA-256 `{dataset['training_sha256'][:16]}...`"
        ),
        (
            f"- Model: {LIBRARY_LABELS[recipe['gbm']['library']]}, {manifest['n_trees']} trees, "
            f"{selection['selected']['n_features']} features, seed {recipe['gbm']['seed']}"
        ),
        f"- Settings: {_params(recipe['gbm']['params'])}",
        f"- Recency half-life: {_half_life(recipe['half_life_years'])}",
        f"- Rate shrinkage: {selection['selected']['shrink_minutes']:g} cage minutes",
        f"- Calibration: {recipe['calibration']}",
        "- Libraries: "
        + ", ".join(
            f"{k} {v}" for k, v in sorted(manifest["library_versions"].items())
        ),
        "",
        "## Splits",
        "",
        table(
            ["period", "dates", "bouts", "with a winner"],
            [
                [
                    "train",
                    f"through {splits['train_end']}",
                    selection["n_train_bouts"],
                    selection["n_train_scored"],
                ],
                [
                    "validation",
                    f"{splits['valid_start']} to {splits['valid_end']}",
                    selection["n_validation_bouts"],
                    selection["n_validation_scored"],
                ],
                ["test", f"from {splits['test_start']}", "see below", "see below"],
            ],
        ),
        "",
        "Draws and no contests have no winner; they are not trained on and not scored.",
        "",
        "## Features and fitted parameters",
        "",
        f"{meta['n_features']} difference features in {len(meta['families'])} families: "
        + ", ".join(f"{name} {count}" for name, count in meta["families"].items())
        + ".",
        "",
        f"Elo parameters, fit on the training period only: K = {elo['k']:g}, finish "
        f"multiplier {elo['finish_mult']:g}, early-career multiplier "
        f"{elo['provisional_mult']:g} for the first {elo['provisional_bouts']} rated bouts"
        + (
            f" (training log loss {f4(elo_fit['train_log_loss'])} over "
            f"{elo_fit['n_bouts']:,} bouts)."
            if elo_fit
            else "."
        ),
        "",
    ]
    return lines


def _shrinkage(selection: Mapping) -> list[str]:
    stage = selection["shrinkage"]
    rows = [
        [
            f"{r['shrink_minutes']:g}",
            f4(r["xgboost"]),
            f4(r["lightgbm"]),
            f4(r["criterion"]),
            signed(r["delta"]),
            interval(r["delta_low"], r["delta_high"]),
            f4(r["logistic_log_loss"]),
            f"{r['tails']['largest_abs_sd']:.1f}",
            f"{r['tails']['median_excess_kurtosis']:.1f}",
        ]
        for r in stage["grid"]
    ]
    classes = selection["feature_meta"]["rate_prior_classes"]
    return [
        "## Selection on validation",
        "",
        (
            f"Every comparison below is validation log loss, averaged over "
            f"{len(selection['seeds'])} seeds unless it says otherwise. Intervals are 95% "
            "paired bootstrap intervals over validation bouts: the bouts are resampled and "
            "both sides of a comparison keep their loss on each resampled bout. An interval "
            "that contains zero means the comparison did not separate the two options."
        ),
        "",
        "### 1. Rate shrinkage strength",
        "",
        (
            f"Criterion: {stage['criterion']}. Delta: {stage['delta_is']}. The last two "
            "columns describe the shrunk features on the training period: the largest "
            "absolute value of any of them in standard deviations, and their median excess "
            "kurtosis. The logistic regression column is that baseline's validation log "
            "loss at its best C; it is shown for information and is not part of the criterion."
        ),
        "",
        table(
            [
                "cage minutes",
                "XGBoost",
                "LightGBM",
                "criterion",
                "delta",
                "95% interval",
                "logistic regression",
                "largest SD",
                "median kurtosis",
            ],
            rows,
        ),
        "",
        (
            f"Chosen: **{stage['chosen']:g} cage minutes** (lowest criterion; a tie goes to "
            "the smaller value)."
        ),
        "",
        f"Weight classes with their own training mean ({len(classes)}): "
        + ", ".join(f"{name} ({count:,} bouts)" for name, count in classes.items())
        + ". Other bouts use the mean over all training bouts.",
        "",
    ]


def _levels(selection: Mapping) -> list[str]:
    stage = selection["levels"]
    rows = [
        [
            label,
            f4(stage[key]["xgboost"]),
            f4(stage[key]["lightgbm"]),
            f4(stage[key]["criterion"]),
        ]
        for label, key in (
            ("without level features", "without"),
            ("with level features", "with"),
        )
    ]
    verdict = (
        "**Kept**: the criterion is lower with them."
        if stage["kept"]
        else "**Not kept**: the criterion is not lower with them."
    )
    return [
        "### 2. Swap-invariant level features",
        "",
        "Candidates: " + ", ".join(f"`{name}`" for name in stage["features"]) + ". "
        "Same criterion as above.",
        "",
        table(["", "XGBoost", "LightGBM", "criterion"], rows),
        "",
        (
            f"Delta (with minus without): {signed(stage['delta'])}, 95% interval "
            f"{interval(stage['delta_low'], stage['delta_high'])}. {verdict}"
        ),
        "",
    ]


def _search(selection: Mapping, config: Mapping) -> list[str]:
    lines = [
        "### 3. Hyperparameter search",
        "",
        (
            f"{config['selection']['n_trials']} settings per library (the reference settings "
            "plus random draws from the search space in `src/ufc/models/tune.py`), one seed "
            f"each, no recency weighting. The best {config['selection']['top_k']} were scored "
            f"again on all {len(selection['seeds'])} seeds and the lowest mean chosen. The "
            "number of trees is chosen by early stopping on validation in every fit."
        ),
        "",
    ]
    for library, stage in selection["search"].items():
        losses = sorted(t["log_loss"] for t in stage["trials"])
        lines += [
            (
                f"**{LIBRARY_LABELS[library]}.** Reference settings: {f4(stage['reference_log_loss'])} "
                f"(one seed). Across the {len(losses)} settings, one-seed log loss ranged from "
                f"{f4(losses[0])} to {f4(losses[-1])} (median {f4(losses[len(losses) // 2])}). "
                "Finalists:"
            ),
            "",
            table(
                ["trial", "settings", "one seed", "trees", "mean over seeds"],
                [
                    [
                        t["trial"],
                        _params(t["params"]),
                        f4(t["log_loss"]),
                        t["n_trees"],
                        f4(t["seed_mean_log_loss"]),
                    ]
                    for t in stage["finalists"]
                ],
            ),
            "",
            f"Chosen: trial {stage['best_trial']}"
            + (" (the reference settings)" if stage["best_trial"] == 0 else "")
            + f", {f4(stage['best_seed_mean_log_loss'])}.",
            "",
        ]
    return lines


def _half_life_section(selection: Mapping) -> list[str]:
    stage = selection["half_life"]
    rows = [
        [
            LIBRARY_LABELS[r["library"]],
            _half_life(r["half_life_years"]),
            f4(r["log_loss"]),
            signed(r["delta"]),
            interval(r["delta_low"], r["delta_high"]),
            f"{min(r['n_trees'])} to {max(r['n_trees'])}",
        ]
        for r in stage["table"]
    ]
    return [
        "### 4. Recency half-life and library",
        "",
        "Each library's chosen settings at each half-life. Delta: "
        + stage["delta_is"]
        + ".",
        "",
        table(
            ["library", "half-life", "log loss", "delta", "95% interval", "trees"], rows
        ),
        "",
        (
            f"Selected: **{LIBRARY_LABELS[stage['library']]}, half-life "
            f"{_half_life(stage['half_life_years'])}** ({f4(stage['log_loss'])}), the lowest "
            "of the eight."
        ),
        "",
    ]


def _baselines(selection: Mapping) -> list[str]:
    versus = selection["versus_baselines"]
    logistic = selection["logistic"]
    rows = [
        [
            MODEL_LABELS[name],
            f4(row["log_loss"]),
            signed(row["delta"]),
            interval(row["delta_low"], row["delta_high"]),
        ]
        for name, row in versus["baselines"].items()
    ]
    verdict = (
        "The candidate has lower validation log loss than both baselines."
        if versus["beats_all"]
        else "**The candidate does not have lower validation log loss than both "
        "baselines.** By the rule in ML_REQUIREMENTS it is not better than them."
    )
    return [
        "### 5. Against the baselines",
        "",
        (
            f"Candidate ({versus['candidate_is']}): **{f4(versus['candidate_log_loss'])}**. "
            f"Delta: {versus['delta_is']}."
        ),
        "",
        table(["baseline", "log loss", "delta", "95% interval"], rows),
        "",
        verdict,
        "",
        f"Logistic regression: C = {logistic['c']:g}, chosen on validation from "
        + ", ".join(f"{r['c']:g} ({f4(r['log_loss'])})" for r in logistic["grid"])
        + ". Elo has nothing to tune: its parameters were fit on the training period.",
        "",
    ]


def _calibration(selection: Mapping) -> list[str]:
    stage = selection["calibration"]
    losses = stage["cv_log_loss"]
    detail = (
        f"Fitted on all of validation, the slope is {stage['slope']:.3f} "
        "(below 1 pulls probabilities toward 50%, above 1 pushes them away)."
        if stage["chosen"] == "platt"
        else f"Fitted on all of validation, it has {stage['isotonic_steps']} steps."
    )
    return [
        "### 6. Calibration",
        "",
        (
            f"The kept model is one fit (seed {selection['selected']['recipe']['gbm']['seed']}) "
            f"with {selection['selected']['n_trees']} trees. Calibration methods were compared "
            f"by {stage['n_folds']}-fold cross-validation inside validation "
            f"({stage['n_bouts']:,} bouts, folds made of whole event dates):"
        ),
        "",
        table(
            ["method", "cross-validated log loss"],
            [
                ["none (uncalibrated)", f4(losses["none"])],
                ["Platt scaling (slope only)", f4(losses["platt"])],
                ["isotonic regression", f4(losses["isotonic"])],
            ],
        ),
        "",
        f"Chosen: **{stage['chosen']}** (the lower of the two methods). {detail}",
        "",
    ]


def _period(record: Mapping, title: str, note: str) -> list[str]:
    lines = [
        title,
        "",
        f"{record['n_bouts']:,} bouts from {record['first_date']} to {record['last_date']}: "
        f"{record['n_scored']:,} with a winner, {record['n_no_result']} no result"
        + (
            " ("
            + ", ".join(
                f"{count} {kind}" for kind, count in record["no_result"].items()
            )
            + ")"
            if record["no_result"]
            else ""
        )
        + f". {note}",
        "",
        table(
            ["", "bouts", "log loss", "95% interval", "Brier", "accuracy", "ECE"],
            _metric_rows(record["models"], with_interval=True),
        ),
        "",
        (
            "Accuracy counts a bout where a model gives exactly 50% as half right. Elo's "
            "accuracy is the higher-rated-fighter baseline; "
            f"{record['equal_elo']} of these bouts have equal ratings. ECE is over "
            "ten equal-width bins of the favoured fighter's probability."
        ),
        "",
        (
            "Paired differences in log loss, calibrated model minus the other "
            "(negative: the model is better):"
        ),
        "",
        table(
            ["model minus", "difference", "95% interval"],
            [
                [
                    MODEL_LABELS[name],
                    signed(row["mean"]),
                    interval(row["low"], row["high"]),
                ]
                for name, row in record["primary_minus"].items()
            ],
        ),
        "",
    ]
    return lines


def _breakdowns(record: Mapping) -> list[str]:
    lines = ["**By year**", "", _group_table(record["by_year"], "year"), ""]
    lines += [
        "**By confidence of the calibrated model**",
        "",
        table(
            [
                "favoured fighter's probability",
                "bouts",
                "mean probability",
                "pick won",
                "95% interval",
            ],
            [
                [
                    f"{pct(r['low'])} to {pct(r['high'])}",
                    r["n"],
                    pct(r["mean_confidence"]),
                    pct(r["favourite_win_rate"]),
                    f"{pct(r['win_rate_low'])} to {pct(r['win_rate_high'])}"
                    if r["n"]
                    else "n/a",
                ]
                for r in record["by_confidence"]["model"]
            ],
        ),
        "",
        (
            "**Reliability, before and after calibration** (favoured fighter's probability, "
            "ten bins; bins with no bouts omitted)"
        ),
        "",
    ]
    before = {
        (r["low"], r["high"]): r for r in record["reliability"]["model_uncalibrated"]
    }
    after = {(r["low"], r["high"]): r for r in record["reliability"]["model"]}
    rows = []
    for key, b in before.items():
        a = after[key]
        if not (b["n"] or a["n"]):
            continue
        rows.append(
            [
                f"{pct(key[0])} to {pct(key[1])}",
                b["n"],
                pct(b["mean_confidence"]),
                pct(b["favourite_win_rate"]),
                a["n"],
                pct(a["mean_confidence"]),
                pct(a["favourite_win_rate"]),
            ]
        )
    lines += [
        table(
            [
                "bin",
                "before: bouts",
                "mean probability",
                "pick won",
                "after: bouts",
                "mean probability",
                "pick won",
            ],
            rows,
        ),
        "",
        '**Debut bouts** (a debutant has no earlier UFC bout; "any" is one or two)',
        "",
        _group_table(record["by_debut"], "bout has"),
        "",
        (
            f"**By weight class** (classes with at least {record['weight_class_min_bouts']} "
            "scored bouts in the period)"
        ),
        "",
    ]
    if record["by_weight_class"]:
        lines += [_group_table(record["by_weight_class"], "weight class"), ""]
    else:
        lines += ["No weight class has enough bouts in this period.", ""]
    if record["weight_classes_too_small"]:
        lines += [
            "Not shown, too few bouts: "
            + ", ".join(
                f"{k} ({v})" for k, v in record["weight_classes_too_small"].items()
            )
            + ".",
            "",
        ]
    lines += [
        "**By completeness of physical attributes**",
        "",
        _group_table(record["by_attributes"], "bouts where"),
        "",
    ]
    return lines


def _ablation(selection: Mapping) -> list[str]:
    stage = selection["ablation"]
    rows = sorted(stage["families"], key=lambda r: -r["delta"])
    did_not_help = [r["family"] for r in rows if not r["helped"]]
    return [
        "### 7. Feature-family ablation",
        "",
        f"The selected recipe refitted without each family in turn, {stage['n_seeds']} "
        f"seeds each. With every family: {f4(stage['full_log_loss'])}. Delta: "
        + stage["delta_is"]
        + ".",
        "",
        table(
            [
                "family removed",
                "features",
                "log loss without",
                "delta",
                "95% interval",
                "helped",
            ],
            [
                [
                    r["family"],
                    r["n_features"],
                    f4(r["log_loss_without"]),
                    signed(r["delta"]),
                    interval(r["delta_low"], r["delta_high"]),
                    "yes" if r["helped"] else "no",
                ]
                for r in rows
            ],
        ),
        "",
        (
            "Families whose removal did not raise validation log loss: "
            + ", ".join(did_not_help)
            + "."
            if did_not_help
            else "Removing any family raised validation log loss."
        ),
        "",
    ]


def _backtest(backtest: Mapping) -> list[str]:
    rows = []
    for fold in backtest["folds"]:
        m = fold["metrics"]
        rows.append(
            [
                f"{fold['year']}{' *' if fold['in_test_period'] else ''}",
                fold["n_train"],
                fold["n_scored"],
                fold["n_trees"],
                f4(m["model"]["log_loss"]),
                f4(m["logistic"]["log_loss"]),
                f4(m["elo"]["log_loss"]),
                pct(m["model"]["accuracy"]),
                pct(m["logistic"]["accuracy"]),
                pct(m["elo"]["accuracy"]),
            ]
        )
    lines = [
        "## Walk-forward backtest",
        "",
        f"Schedule for each year Y: {backtest['schedule']}. Refitted in every fold: "
        + "; ".join(backtest["refit_per_fold"])
        + ". Fixed from the selection above: "
        + "; ".join(backtest["fixed_from_selection"])
        + ". Years marked * are in the test period; the others were scored with a "
        "recipe chosen on 2023 to 2024 and are a backtest, not a held-out result.",
        "",
        table(
            [
                "year",
                "training bouts",
                "scored",
                "trees",
                "model log loss",
                "logistic",
                "Elo",
                "model accuracy",
                "logistic",
                "Elo",
            ],
            rows,
        ),
        "",
    ]
    pooled_rows = []
    for label, key in (
        ("before the test period", "pooled_before_test_period"),
        ("test period", "pooled_test_period"),
    ):
        pooled = backtest.get(key)
        if not pooled:
            continue
        for name in ("model", "logistic", "elo"):
            m = pooled["metrics"][name]
            pooled_rows.append(
                [
                    f"{label} ({pooled['years'][0]} to {pooled['years'][1]})",
                    MODEL_LABELS[name],
                    m["n"],
                    f4(m["log_loss"]),
                    f4(m["brier"]),
                    pct(m["accuracy"]),
                    f4(m["ece"]),
                ]
            )
    check = backtest["first_test_year_check"]
    lines += [
        "Pooled over folds:",
        "",
        table(
            ["years", "", "bouts", "log loss", "Brier", "accuracy", "ECE"], pooled_rows
        ),
        "",
        f"Check: the {check['year']} fold repeats the registered model's own schedule. Its "
        "probabilities are "
        + (
            "identical to the registered model's, bout for bout."
            if check["identical_to_registered_model"]
            else "**not** identical to the registered model's."
        ),
        "",
    ]
    return lines


def generated_block(
    *,
    selection: Mapping,
    manifest: Mapping,
    config: Mapping,
    test: Mapping | None,
    backtest: Mapping | None,
    ledger: Sequence[Mapping],
) -> str:
    lines: list[str] = []
    lines += _header(selection, manifest, config)
    lines += _shrinkage(selection)
    lines += _levels(selection)
    lines += _search(selection, config)
    lines += _half_life_section(selection)
    lines += _baselines(selection)
    lines += _calibration(selection)
    lines += _ablation(selection)
    lines += _period(
        selection["validation"],
        "## Validation period",
        "These bouts were used for every choice above, so these numbers are optimistic. "
        "The calibrated row uses calibrators fitted by cross-validation inside "
        "validation; the final calibrator is fitted on all of it.",
    )
    lines += _breakdowns(selection["validation"])
    if test is None:
        lines += [
            "## Test period",
            "",
            "Not evaluated yet. Run `uv run python -m ufc.models evaluate`.",
            "",
        ]
    else:
        versions = sorted({entry["version"] for entry in ledger})
        lines += _period(
            test,
            "## Test period",
            "No choice above used these bouts.",
        )
        symmetry = test["symmetry"]
        lines += [
            (
                f"Symmetry test on all {symmetry['n_bouts']:,} test-period bouts: P(A) + P(B) "
                f"is exactly 1 for every bout: {'yes' if symmetry['sum_is_exactly_one'] else 'NO'}. "
                "Presenting every bout with the fighters swapped swaps the two probabilities "
                f"exactly: {'yes' if symmetry['swap_is_exact'] else 'NO'}."
            ),
            "",
            f"Test-period evaluations recorded in the registry ledger: {len(ledger)}, of "
            f"{len(versions)} version(s): "
            + ", ".join(f"`{v}`" for v in versions)
            + ".",
            "",
        ]
        lines += _breakdowns(test)
    if backtest is not None:
        lines += _backtest(backtest)
    lines += [
        "## Files",
        "",
        "Written by the commands above, under `data/` (not committed):",
        "",
        "- `data/reports/eval/selection.json`: every comparison in the selection sections",
        (
            "- `data/reports/eval/test_metrics.json`, `walk_forward.json`: the test-period "
            "and backtest sections"
        ),
        (
            "- `data/reports/eval/plots/validation/` and `plots/test/`: `reliability.png`, "
            "`by_confidence.png`, `by_year.png`, `debut.png`; `plots/validation/ablation.png`; "
            "`plots/walk_forward.png`"
        ),
        (
            f"- `data/models/{manifest['version']}/`: the model, calibrator, feature list, "
            "configuration, metrics and manifest"
        ),
    ]
    return "\n".join(lines).rstrip() + "\n"


SKELETON = """# Model report (M4)

Reproduce the generated part of this report with:

    uv run python -m ufc.ingest fetch
    uv run python -m ufc.ingest load
    uv run python -m ufc.models train
    uv run python -m ufc.models evaluate
    uv run python -m ufc.models report

{begin}
{end}
"""


def write_report(path: Path, **records) -> None:
    """Replace the block between the markers in ``path``, creating the file
    with a skeleton if it does not exist."""
    block = f"{BEGIN}\n\n{generated_block(**records)}\n{END}"
    text = path.read_text() if path.exists() else SKELETON.format(begin=BEGIN, end=END)
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"{path} has no generated-block markers")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pattern.sub(lambda _: block, text, count=1))
