"""Era-shift diagnostics: why do 2025 and 2026 score better than earlier years?

    uv run python -m ufc.models.diagnostics

Analysis only. Nothing here selects, saves or changes a model, and nothing
here may be used to change a selection decision. It refits the registered
recipe in walk-forward folds (as ``evaluate`` does) with groups of inputs
removed, and reads the test period to describe it. Because it reads
test-period rows it unlocks the split lock under its own name, and every run
is appended to ``data/models/test_diagnostics.jsonl`` (DECISIONS D45).

Checks, each written to ``data/reports/diagnostics/era_shift.json``:

1. single-feature rules by year ("pick the younger fighter", and so on)
2. matchup difficulty by year: Elo gaps and the model's own confidence
3. the backtest without the physical family and without each attribute that
   comes from a fighter profile
4. test-period results by when the fighters' profiles were captured
5. other paths: truncation and same-card tests of the feature build on the
   real data, each fold's model scored on the test period, one-sided gaps
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ufc.db.connection import connect
from ufc.features.build import FEATURES_FILE, build_features
from ufc.features.config import FeatureConfig
from ufc.features.data import FeatureData, load_feature_data
from ufc.models import registry
from ufc.models.metrics import bootstrap_mean_interval, bout_log_loss, summary
from ufc.models.recipe import Recipe, fit_recipe
from ufc.models.select import baseline_predictions
from ufc.models.splits import SPLIT_DATES, Splits, fit_rows, labelled
from ufc.models.walkforward import FIRST_YEAR, fold_dates

logger = logging.getLogger(__name__)

DEFAULT_DB = Path("data/ufc.db")
DEFAULT_FEATURES = Path("data/features")
DEFAULT_EVAL = Path("data/reports/eval")
DEFAULT_OUT = Path("data/reports/diagnostics")
OUT_FILE = "era_shift.json"
BOUTS_FILE = "era_shift_bouts.parquet"
LEDGER_FILE = "test_diagnostics.jsonl"

PROFILE_REFRESH = date(2025, 11, 22)
"""The date after which debutants' profiles are mostly incomplete
(docs/DATA_QUALITY.md). Taken as upstream's last capture of every existing
profile: a fighter who debuted before it has a profile from that capture or
earlier; one who debuted on or after it was captured when first listed."""

ERAS: tuple[tuple[str, int, int], ...] = (
    ("2013 to 2018", 2013, 2018),
    ("2019 to 2022", 2019, 2022),
    ("2023 to 2024", 2023, 2024),
    ("2025 to 2026", 2025, 2026),
)
BEFORE, INSIDE = "2013 to 2024", "2025 to 2026"

STANCE = ("d_stance_southpaw", "d_stance_switch", "d_stance_southpaw_vs_orthodox")
PHYSICAL = ("d_age", "d_height_cm", "d_reach_cm", *STANCE)
VARIANTS: dict[str, tuple[str, ...]] = {
    "full": (),
    "no_physical": PHYSICAL,
    "no_height": ("d_height_cm",),
    "no_reach": ("d_reach_cm",),
    "no_stance": STANCE,
    "no_dob": ("d_age", "lvl_age"),
    "no_profile": (*PHYSICAL, "lvl_age"),
}
"""Inputs removed in each rerun of the backtest. ``no_physical`` is the
family as the ablation removes it; ``no_dob`` also removes the mean age;
``no_profile`` removes everything that is read from a fighter profile."""

CONTEXT = [
    "bout_id",
    "date",
    "fighter_a_id",
    "fighter_b_id",
    "label",
    "a_debut",
    "b_debut",
    "a_attr_gap",
    "b_attr_gap",
    "d_elo",
    "d_age",
]

REPORTED_RULES = (
    "d_age",
    "d_height_cm",
    "d_reach_cm",
    "d_elo",
    "d_n_fights",
    "d_win_rate_career",
    "d_win_streak",
    "d_loss_streak",
    "d_days_since_last",
    "d_fights_last_2y",
    "d_opp_elo_career",
    "d_debut",
)


# --- pure helpers --------------------------------------------------------------


def era_masks(years: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean masks for each era, for the two pooled periods and each year."""
    years = np.asarray(years)
    masks = {name: (years >= low) & (years <= high) for name, low, high in ERAS}
    masks[BEFORE] = (years >= FIRST_YEAR) & (years < SPLIT_DATES.test_start.year)
    for year in sorted(set(years.tolist())):
        if year >= FIRST_YEAR:
            masks[str(year)] = years == year
    return masks


def rule_direction(values: np.ndarray, y: np.ndarray) -> int:
    """+1 if the fighter with the larger value won more often than not among
    bouts where the value differs, otherwise -1."""
    values, y = np.asarray(values, dtype=np.float64), np.asarray(y, dtype=np.float64)
    usable = np.isfinite(values) & (values != 0)
    if not usable.any():
        return 1
    larger_won = np.where(values[usable] > 0, y[usable], 1.0 - y[usable])
    return 1 if larger_won.mean() >= 0.5 else -1


def rule_accuracy(values: np.ndarray, y: np.ndarray, direction: int) -> dict:
    """Accuracy of "pick the fighter with the larger value" (or the smaller,
    for direction -1) over bouts where the value is known and differs."""
    values, y = np.asarray(values, dtype=np.float64), np.asarray(y, dtype=np.float64)
    usable = np.isfinite(values) & (values != 0)
    n = int(usable.sum())
    if n == 0:
        return {"n": 0, "accuracy": None, "coverage": 0.0}
    picked_a = (values[usable] * direction) > 0
    correct = np.where(picked_a, y[usable], 1.0 - y[usable])
    return {
        "n": n,
        "accuracy": float(correct.mean()),
        "coverage": float(usable.mean()),
    }


def single_feature_rules(
    frame: pd.DataFrame,
    features: Sequence[str],
    direction_through: date,
    fit_only: bool = False,
) -> dict:
    """Each feature as a one-line rule, scored by year and era.

    The direction of each rule (larger wins or smaller wins) is set on bouts
    through ``direction_through`` and then held fixed, so later years are
    scored with a rule that did not see them. ``fit_only`` restricts scoring
    to bouts without an attribute gap.
    """
    rows = labelled(frame)
    if fit_only:
        rows = rows[~(rows["a_attr_gap"] | rows["b_attr_gap"])]
    y = rows["label"].to_numpy(dtype=np.float64)
    early = (rows["date"] <= pd.Timestamp(direction_through)).to_numpy()
    masks = era_masks(rows["date"].dt.year.to_numpy())
    out = {}
    for name in features:
        values = rows[name].to_numpy(dtype=np.float64)
        direction = rule_direction(values[early], y[early])
        out[name] = {
            "direction": "larger wins" if direction > 0 else "smaller wins",
            "periods": {
                label: rule_accuracy(values[mask], y[mask], direction)
                for label, mask in masks.items()
            },
        }
    return out


def favoured(p: np.ndarray) -> np.ndarray:
    """The favoured fighter's probability."""
    p = np.asarray(p, dtype=np.float64)
    return np.maximum(p, 1.0 - p)


def entropy(p: np.ndarray) -> np.ndarray:
    """Per-bout log loss a model expects of itself if it is calibrated."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def difficulty(bouts: pd.DataFrame, elo_scale: float = 400.0) -> dict:
    """Per year and era: how lopsided the bouts are by Elo, and how confident
    the model was against how often its pick won."""
    y = bouts["label"].to_numpy(dtype=np.float64)
    gap = bouts["d_elo"].abs().to_numpy(dtype=np.float64)
    age = bouts["d_age"].abs().to_numpy(dtype=np.float64)
    p = bouts["p_full"].to_numpy(dtype=np.float64)
    raw = bouts["p_full_uncalibrated"].to_numpy(dtype=np.float64)
    conf, picked_right = favoured(p), np.where(p > 0.5, y, 1 - y)
    picked_right = np.where(p == 0.5, 0.5, picked_right)
    debut = (bouts["a_debut"] | bouts["b_debut"]).to_numpy()
    one_gap = (bouts["a_attr_gap"] | bouts["b_attr_gap"]).to_numpy()
    out = {}
    for label, mask in era_masks(bouts["date"].dt.year.to_numpy()).items():
        if not mask.any():
            continue
        out[label] = {
            "n": int(mask.sum()),
            "abs_elo_mean": float(gap[mask].mean()),
            "abs_elo_median": float(np.median(gap[mask])),
            "abs_elo_p90": float(np.quantile(gap[mask], 0.9)),
            "elo_favourite_mean": float(
                (1 / (1 + 10 ** (-gap[mask] / elo_scale))).mean()
            ),
            "abs_age_mean": float(np.nanmean(age[mask])),
            "abs_age_5y_share": float((age[mask][np.isfinite(age[mask])] >= 5).mean()),
            "any_debutant_share": float(debut[mask].mean()),
            "attribute_gap_share": float(one_gap[mask].mean()),
            "confidence_mean": float(conf[mask].mean()),
            "confidence_median": float(np.median(conf[mask])),
            "confidence_70_share": float((conf[mask] >= 0.70).mean()),
            "confidence_mean_uncalibrated": float(favoured(raw[mask]).mean()),
            "pick_won": float(picked_right[mask].mean()),
            "expected_log_loss": float(entropy(p[mask]).mean()),
            "log_loss": float(bout_log_loss(p[mask], y[mask]).mean()),
        }
    return out


def attribution(bouts: pd.DataFrame, variants: Sequence[str]) -> dict:
    """Log loss by era for each rerun, and each rerun minus the full model on
    the same bouts. ``shift`` is the before-period loss minus the test-period
    loss: how much better the test period scored."""
    y = bouts["label"].to_numpy(dtype=np.float64)
    masks = era_masks(bouts["date"].dt.year.to_numpy())
    no_gap = ~(bouts["a_attr_gap"] | bouts["b_attr_gap"]).to_numpy()
    full = bout_log_loss(bouts["p_full"].to_numpy(dtype=np.float64), y)
    out = {}
    for name in variants:
        loss = bout_log_loss(bouts[f"p_{name}"].to_numpy(dtype=np.float64), y)
        delta = loss - full
        by = {
            label: float(loss[mask].mean())
            for label, mask in masks.items()
            if mask.any()
        }
        record = {
            "log_loss": by,
            "shift": by[BEFORE] - by[INSIDE],
            "shift_no_gap": float(
                loss[masks[BEFORE] & no_gap].mean()
                - loss[masks[INSIDE] & no_gap].mean()
            ),
            "log_loss_no_gap": {
                label: float(loss[masks[label] & no_gap].mean())
                for label in (BEFORE, INSIDE)
            },
            "minus_full": {
                label: bootstrap_mean_interval(delta[masks[label]])
                for label in (BEFORE, INSIDE)
            },
        }
        out[name] = record
    return out


def group_summaries(
    bouts: pd.DataFrame, groups: Mapping[str, np.ndarray], models: Sequence[str]
) -> dict:
    y = bouts["label"].to_numpy(dtype=np.float64)
    out = {}
    for label, mask in groups.items():
        mask = np.asarray(mask, dtype=bool)
        out[label] = {"n": int(mask.sum())}
        if mask.any():
            for name in models:
                p = bouts[f"p_{name}"].to_numpy(dtype=np.float64)
                out[label][name] = summary(p[mask], y[mask])
    return out


def debut_dates(data: FeatureData) -> dict[str, date]:
    """Each fighter's first bout in the data."""
    first: dict[str, date] = {}
    for bout in data.bouts:
        for fighter in (bout.fighter_a_id, bout.fighter_b_id):
            if fighter not in first or bout.date < first[fighter]:
                first[fighter] = bout.date
    return first


def profile_capture(bouts: pd.DataFrame, debuts: Mapping[str, date]) -> dict:
    """Test-period results by when the two profiles were captured, taking a
    fighter's debut date against ``PROFILE_REFRESH`` as the capture time."""
    test = bouts[bouts["date"] >= pd.Timestamp(SPLIT_DATES.test_start)]
    refresh = pd.Timestamp(PROFILE_REFRESH)
    first_a = pd.to_datetime(test["fighter_a_id"].map(debuts))
    first_b = pd.to_datetime(test["fighter_b_id"].map(debuts))
    later_debut = np.maximum(first_a.to_numpy(), first_b.to_numpy())
    both_at_refresh = later_debut < np.datetime64(refresh)
    after = (test["date"] >= refresh).to_numpy()
    no_gap = ~(test["a_attr_gap"] | test["b_attr_gap"]).to_numpy()
    veteran = later_debut < np.datetime64("2024-01-01")
    groups = {
        "all test-period bouts": np.ones(len(test), dtype=bool),
        "bout before the refresh (profiles captured after the bout)": ~after,
        "bout after the refresh": after,
        "bout after the refresh, both debuted before it "
        "(profiles captured before the bout)": after & both_at_refresh,
        "bout after the refresh, either debuted after it "
        "(captured when first listed)": after & ~both_at_refresh,
        "both debuted before the refresh": both_at_refresh,
        "both debuted before 2024": veteran,
        "either debuted in 2024 or later": ~veteran,
    }
    groups |= {f"{k}; no attribute gap": v & no_gap for k, v in list(groups.items())}
    return group_summaries(test, groups, ("full", "no_profile", "logistic", "elo"))


def one_sided_gaps(frame: pd.DataFrame) -> dict:
    """How often the fighter with the complete profile won, by era."""
    rows = labelled(frame)
    rows = rows[rows["a_attr_gap"] != rows["b_attr_gap"]]
    won = np.where(rows["a_attr_gap"], 1.0 - rows["label"], rows["label"])
    years = rows["date"].dt.year.to_numpy()
    out = {}
    for label, low, high in (("through 2012", 0, 2012), *ERAS):
        mask = (years >= low) & (years <= high)
        out[label] = {
            "n": int(mask.sum()),
            "complete_profile_won": float(won[mask].mean()) if mask.any() else None,
        }
    return out


def truncation_check(
    data: FeatureData, config: FeatureConfig, cutoffs: Sequence[date]
) -> dict:
    """Rebuild the feature table from the real data with every bout after a
    cutoff deleted. Rows dated on or before the cutoff must not change.
    Profiles cannot be truncated: they are one capture per fighter."""
    full = build_features(data, config).frame.set_index("bout_id")
    out = {}
    for cutoff in cutoffs:
        cut = replace(data, bouts=[b for b in data.bouts if b.date <= cutoff])
        short = build_features(cut, config).frame.set_index("bout_id")
        same_rows = full.loc[short.index]
        differing = [
            column
            for column in short.columns
            if not short[column].equals(same_rows[column])
        ]
        out[cutoff.isoformat()] = {
            "n_bouts_compared": len(short),
            "n_columns": len(short.columns),
            "columns_that_differ": differing,
        }
    return out


def same_card_check(data: FeatureData, config: FeatureConfig, n_dates: int = 6) -> dict:
    """For event dates spread over the test period: delete every other bout
    held on that date, rebuild, and compare the one bout kept. Its features
    must not change (leakage rule 1, on the real data)."""
    full = build_features(data, config).frame.set_index("bout_id")
    days = sorted({b.date for b in data.bouts if b.date >= SPLIT_DATES.test_start})
    chosen = [days[i] for i in np.linspace(0, len(days) - 1, n_dates).astype(int)]
    out = {}
    for day in chosen:
        kept = next(b for b in data.bouts if b.date == day)
        cut = replace(data, bouts=[b for b in data.bouts if b.date != day or b is kept])
        short = build_features(cut, config).frame.set_index("bout_id")
        mine, original = short.loc[[kept.bout_id]], full.loc[[kept.bout_id]]
        out[day.isoformat()] = {
            "bouts_removed": sum(b.date == day for b in data.bouts) - 1,
            "columns_that_differ": [
                c for c in short.columns if not mine[c].equals(original[c])
            ],
        }
    return out


# --- folds ---------------------------------------------------------------------


def run_fold(
    data: FeatureData,
    recipe: Recipe,
    feature_config: FeatureConfig,
    logistic_c: float,
    year: int,
    variants: Mapping[str, Sequence[str]],
    reason: str,
) -> dict:
    """One walk-forward fold on the registered model's schedule, fitted once
    per variant from the same feature table. Folds before the test period
    also score the test period with the full model, to tell an easier era
    from a better-trained model."""
    dates = fold_dates(year)
    build = build_features(data, replace(feature_config, train_end=dates.train_end))
    splits = Splits(build.frame, dates, test_dates=SPLIT_DATES)
    splits.unlock_test(reason)
    train, valid = fit_rows(splits.train()), labelled(splits.valid())
    scored = labelled(
        splits.between(date(year, 1, 1), date(year, 12, 31), f"diagnostic {year}")
    )
    out = scored[CONTEXT].copy()
    later, info = None, {"year": year, "n_train": len(train), "variants": {}}
    for name, drop in variants.items():
        missing = set(drop) - set(recipe.spec.names)
        if missing:
            raise ValueError(f"{name}: not model inputs: {sorted(missing)}")
        variant = replace(recipe, spec=recipe.spec.without(drop))
        candidate, _ = fit_recipe(variant, train, valid, dates.train_end)
        out[f"p_{name}"] = candidate.predict(scored)
        info["variants"][name] = {
            "n_trees": candidate.model.n_trees,
            "calibrator_slope": candidate.calibrator.slope,
        }
        if name != "full":
            continue
        out["p_full_uncalibrated"] = candidate.predict(scored, calibrated=False)
        for model, p in baseline_predictions(
            train, scored, recipe.spec, logistic_c
        ).items():
            out[f"p_{model}"] = p
        if year < SPLIT_DATES.test_start.year:
            test = labelled(
                splits.between(SPLIT_DATES.test_start, None, "old model, test period")
            )
            later = summary(
                candidate.predict(test), test["label"].to_numpy(dtype=np.float64)
            )
            later["own_year"] = summary(
                out["p_full"].to_numpy(), scored["label"].to_numpy(dtype=np.float64)
            )
    return {"bouts": out, "info": info, "later": later}


def reproduces(bouts: pd.DataFrame, backtest: dict) -> dict:
    """The full variant must repeat the saved backtest, fold for fold."""
    worst = 0.0
    for fold in backtest["folds"]:
        rows = bouts[bouts["date"].dt.year == fold["year"]]
        y = rows["label"].to_numpy(dtype=np.float64)
        for model in ("model", "logistic", "elo"):
            column = "p_full" if model == "model" else f"p_{model}"
            mine = float(bout_log_loss(rows[column].to_numpy(), y).mean())
            worst = max(worst, abs(mine - fold["metrics"][model]["log_loss"]))
    return {"largest_log_loss_difference": worst, "identical": worst < 1e-12}


# --- command -------------------------------------------------------------------


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        prog="python -m ufc.models.diagnostics", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--models", type=Path, default=registry.DEFAULT_ROOT)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args(argv)

    version = json.loads((args.eval / "selection.json").read_text())["version"]
    backtest = json.loads((args.eval / "walk_forward.json").read_text())
    candidate, _ = registry.load_candidate(args.models, version)
    config = registry.read_json(args.models, version, "config.json")
    raw = dict(
        registry.read_json(args.models, version, "feature_pipeline.json")["config"]
    )
    raw["train_end"] = date.fromisoformat(raw["train_end"])
    feature_config = FeatureConfig(**raw)
    conn = connect(args.db)
    try:
        data = load_feature_data(conn)
    finally:
        conn.close()
    frame = pd.read_parquet(args.features / FEATURES_FILE)

    reason = f"era-shift diagnostic of {version} (analysis only)"
    ledger = args.models / LEDGER_FILE
    with ledger.open("a", encoding="utf-8") as handle:
        entry = {
            "what": reason,
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **{f"git_{k}": v for k, v in registry.git_state().items()},
        }
        handle.write(json.dumps(entry, sort_keys=True) + "\n")

    years = [fold["year"] for fold in backtest["folds"]]
    logger.info("refitting %d folds x %d variants", len(years), len(VARIANTS))
    folds = Parallel(n_jobs=args.jobs)(
        delayed(run_fold)(
            data,
            candidate.recipe,
            feature_config,
            config["baseline_logistic_c"],
            year,
            VARIANTS,
            reason,
        )
        for year in years
    )
    bouts = pd.concat([fold["bouts"] for fold in folds], ignore_index=True)

    features = [n for n in candidate.spec.antisymmetric if n in frame.columns]
    record = {
        "version": version,
        "profile_refresh": PROFILE_REFRESH.isoformat(),
        "variants": {name: list(drop) for name, drop in VARIANTS.items()},
        "reproduces_saved_backtest": reproduces(bouts, backtest),
        "single_feature_rules": {
            "direction_set_through": SPLIT_DATES.train_end.isoformat(),
            "reported": list(REPORTED_RULES),
            "all_bouts": single_feature_rules(frame, features, SPLIT_DATES.train_end),
            "no_attribute_gap": single_feature_rules(
                frame, features, SPLIT_DATES.train_end, fit_only=True
            ),
        },
        "difficulty": difficulty(bouts),
        "attribution": attribution(bouts, list(VARIANTS)),
        "folds": [fold["info"] for fold in folds],
        "profile_capture": profile_capture(bouts, debut_dates(data)),
        "old_models_on_test_period": {
            str(fold["info"]["year"]): {
                "n_train": fold["info"]["n_train"],
                "own_year": fold["later"]["own_year"],
                "test_period": {
                    k: v for k, v in fold["later"].items() if k != "own_year"
                },
            }
            for fold in folds
            if fold["later"] is not None
        },
        "one_sided_gaps": one_sided_gaps(frame),
        "truncation": truncation_check(
            data, feature_config, [date(2024, 12, 31), date(2025, 11, 21)]
        ),
        "same_card": same_card_check(data, feature_config),
    }
    _write_json(args.out / OUT_FILE, record)
    bouts.to_parquet(args.out / BOUTS_FILE, index=False)
    print(f"Wrote {args.out / OUT_FILE} and {args.out / BOUTS_FILE}")
    print(f"Saved backtest reproduced: {record['reproduces_saved_backtest']}")
    for name, row in record["attribution"].items():
        print(
            f"  {name:12} {BEFORE}: {row['log_loss'][BEFORE]:.4f}  "
            f"{INSIDE}: {row['log_loss'][INSIDE]:.4f}  shift {row['shift']:+.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
