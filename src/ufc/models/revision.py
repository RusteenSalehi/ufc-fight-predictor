"""Model revision M4.5: does the model underrate elite veterans against much
younger opponents, and do candidate features fix it?

    uv run python -m ufc.models.revision run --variants base
    uv run python -m ufc.models.revision run --variants base,schedule,age_shape,...

Analysis only: nothing here saves a model, writes the registry or changes
which model is live. Two kinds of evidence, kept apart in the output:

- **Validation (2023 to 2024), candidate pipeline.** Feature parameters and
  models fitted through 2022, trees chosen by early stopping on validation,
  Platt calibration by cross-validation inside validation (as the selection
  report does), five seeds. This is where variants are compared and where
  the adoption rule is applied (DECISIONS D54).
- **Walk-forward backtest.** The registered schedule for each year (fit
  through Y-3, trees and calibrator on Y-2 and Y-1, score Y), one seed, as
  in ``evaluate``. Years 2013 to 2024 do not read the test period. Years 2025
  and 2026 do: they are reported separately, as a secondary check on bouts
  that have already been looked at, and a run that includes them is logged
  in ``data/models/test_diagnostics.jsonl`` (DECISIONS D45). They are not an
  input to any decision.

The slice under study: bouts where one fighter is at least ``AGE_GAP`` years
older than the other and his Elo rating, as of the bout, is higher by at
least 50 or 100 points; and the mirror image, where the older fighter's
rating is lower by as much. Everything is expressed from the older
fighter's side.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ufc.db.connection import connect
from ufc.features.build import build_features
from ufc.features.config import FeatureConfig
from ufc.features.data import FeatureData, load_feature_data
from ufc.features.matchup import CANDIDATE_FEATURES, FAMILY_LEVELS, FEATURES
from ufc.features.store import FittedParams
from ufc.models import registry
from ufc.models.baselines import ELO_SPEC, EloModel
from ufc.models.calibrate import (
    IDENTITY_CALIBRATOR,
    PLATT,
    cross_validated,
    fit_calibrator,
    fold_ids,
)
from ufc.models.design import FeatureSpec, predict
from ufc.models.metrics import (
    CLIP,
    bootstrap_mean_interval,
    bout_log_loss,
    wilson_interval,
)
from ufc.models.recipe import Candidate, Recipe, fit_recipe
from ufc.models.refit import fit_fixed, tree_count
from ufc.models.splits import SPLIT_DATES, SplitDates, Splits, fit_rows, labelled
from ufc.models.tune import SEEDS
from ufc.models.walkforward import FIRST_YEAR, fold_dates

logger = logging.getLogger(__name__)

AGE_GAP = 6.0
"""Years by which the older fighter is older, at least."""
ELO_GAPS: tuple[float, ...] = (50.0, 100.0)
"""Rating margins tried for "the older fighter is the higher rated one"."""

DEFAULT_DB = Path("data/ufc.db")
DEFAULT_OUT = Path("data/reports/revision")
LEDGER_FILE = "test_diagnostics.jsonl"
N_BOOT = 2000

VALIDATION = "validation"
BACKTEST = "backtest_2013_2024"
SECONDARY = "secondary_2025_2026_already_seen"


# --- variants ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Variant:
    """The candidate recipe with extra inputs and, optionally, different
    feature settings. ``base`` adds nothing and is the registered recipe."""

    name: str
    antisymmetric: tuple[str, ...] = ()
    symmetric: tuple[str, ...] = ()
    config: Mapping[str, object] = field(default_factory=dict)

    def recipe(self, base: Recipe) -> Recipe:
        spec = FeatureSpec(
            (*base.spec.antisymmetric, *self.antisymmetric),
            (*base.spec.symmetric, *self.symmetric),
        )
        return replace(base, spec=spec)

    def feature_config(self, base: FeatureConfig) -> FeatureConfig:
        return replace(base, **self.config)

    @property
    def config_key(self) -> tuple:
        return tuple(sorted(self.config.items()))


def _family(name: str) -> tuple[str, ...]:
    """Antisymmetric columns of a family, adopted (``age_shape``, D58) or
    still a candidate (``schedule``)."""
    return tuple(f.name for f in (*FEATURES, *CANDIDATE_FEATURES) if f.family == name)


_SCHEDULE = _family("schedule")
_AGE_SHAPE = _family("age_shape")
_AGE_SHAPE_LEVELS = tuple(lv.name for lv in FAMILY_LEVELS if lv.family == "age_shape")
_CATCH = {"catch_weight_adjacent": True}

VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in (
        Variant("base"),
        Variant("schedule", _SCHEDULE),  # (a) family 14
        Variant("age_shape", _AGE_SHAPE, _AGE_SHAPE_LEVELS),  # (b)
        Variant("catch_weight", config=_CATCH),  # (c)
        Variant(
            "combined", _SCHEDULE + _AGE_SHAPE, _AGE_SHAPE_LEVELS, _CATCH
        ),  # a + b + c
    )
}
"""The variants registered in DECISIONS D54, before any of them was run."""


# --- the slice -----------------------------------------------------------------


def older_view(frame: pd.DataFrame, p_a: np.ndarray) -> pd.DataFrame:
    """Each bout from the older fighter's side.

    ``age_gap`` is how much older he is (NaN if either age is unknown),
    ``elo_gap`` his rating minus the other's, ``p`` the probability he wins
    and ``y`` whether he did. A bout between fighters of exactly equal age
    has no older fighter; it gets an age gap of 0 and so falls in no slice.
    """
    d_age = frame["d_age"].to_numpy(dtype=np.float64)
    d_elo = frame["d_elo"].to_numpy(dtype=np.float64)
    label = frame["label"].to_numpy(dtype=np.float64)
    p_a = np.asarray(p_a, dtype=np.float64)
    a_older = d_age > 0
    return pd.DataFrame(
        {
            "age_gap": np.abs(d_age),
            "elo_gap": np.where(a_older, d_elo, -d_elo),
            "p": np.where(a_older, p_a, 1.0 - p_a),
            "y": np.where(a_older, label, 1.0 - label),
        },
        index=frame.index,
    )


def slice_masks(view: pd.DataFrame) -> dict[str, np.ndarray]:
    """Named slices, from the older fighter's side.

    ``elite_veteran_+50``: at least ``AGE_GAP`` years older and rated at least
    50 points higher; ``mirror_-50``: as much older and rated at least 50
    points lower. ``age_gap`` is every bout with that age gap, ``all`` every
    bout with both ages known.
    """
    age_ok = (view["age_gap"] >= AGE_GAP).to_numpy()
    elo = view["elo_gap"].to_numpy()
    masks = {
        "all": view["age_gap"].notna().to_numpy(),
        "age_gap": age_ok,
    }
    for gap in ELO_GAPS:
        masks[f"elite_veteran_+{gap:g}"] = age_ok & (elo >= gap)
    for gap in ELO_GAPS:
        masks[f"mirror_-{gap:g}"] = age_ok & (elo <= -gap)
    return masks


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, CLIP, 1.0 - CLIP)
    return np.log(p) - np.log1p(-p)


def logit_shift(p: np.ndarray, y: np.ndarray) -> float:
    """The constant ``d`` that minimises log loss of ``sigmoid(logit(p) + d)``.

    Positive means the outcomes favour the older fighter by more than the
    model says, on the log-odds scale: the model underrates him. It is the
    calibration-in-the-large of a slice, in one number that does not depend
    on how confident the predictions are.
    """
    z = _logit(np.asarray(p, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64)
    if len(z) == 0 or y.min() == y.max():
        return math.nan  # all wins or all losses: no finite answer
    d = 0.0
    for _ in range(100):
        s = 1.0 / (1.0 + np.exp(-(z + d)))
        gradient = float(np.sum(s - y))
        hessian = float(np.sum(s * (1.0 - s)))
        step = gradient / hessian
        d -= step
        if abs(step) < 1e-12:
            break
    return float(d)


def _bootstrap_shift(
    p: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(p)
    shifts = []
    for _ in range(n_boot):
        i = rng.integers(0, n, size=n)
        value = logit_shift(p[i], y[i])
        if math.isfinite(value):
            shifts.append(value)
    if len(shifts) < n_boot / 2:
        return (math.nan, math.nan)
    low, high = np.quantile(shifts, [0.025, 0.975])
    return float(low), float(high)


def slice_metrics(p: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """Calibration of one slice, from the older fighter's side.

    ``gap`` is mean predicted probability minus actual win rate (negative:
    the model underrates the older fighter), with a bootstrap interval over
    bouts. ``win_rate_interval`` is a Wilson interval. ``log_loss`` has a
    bootstrap interval. ``logit_shift`` is described in ``logit_shift``.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(p)
    if n == 0:
        return {"n": 0}
    gap = bootstrap_mean_interval(p - y, n_boot=N_BOOT, seed=seed)
    loss = bootstrap_mean_interval(bout_log_loss(p, y), n_boot=N_BOOT, seed=seed)
    low, high = wilson_interval(float(y.sum()), n)
    shift_low, shift_high = _bootstrap_shift(p, y, seed=seed)
    return {
        "n": n,
        "mean_predicted": float(p.mean()),
        "win_rate": float(y.mean()),
        "win_rate_interval": [low, high],
        "gap": gap["mean"],
        "gap_interval": [gap["low"], gap["high"]],
        "log_loss": loss["mean"],
        "log_loss_interval": [loss["low"], loss["high"]],
        "logit_shift": logit_shift(p, y),
        "logit_shift_interval": [shift_low, shift_high],
    }


def slice_table(frame: pd.DataFrame, columns: Mapping[str, str]) -> dict:
    """``slice_metrics`` for every slice and every probability column.

    ``frame`` holds ``d_age``, ``d_elo``, ``label`` and the probability
    columns named in ``columns`` (display name to column), each P(A wins).
    """
    out: dict = {}
    for name, column in columns.items():
        view = older_view(frame, frame[column].to_numpy())
        out[name] = {
            label: slice_metrics(view["p"].to_numpy()[mask], view["y"].to_numpy()[mask])
            for label, mask in slice_masks(view).items()
        }
    return out


def paired_slice_change(
    frame: pd.DataFrame, column: str, reference: str, seed: int = 0
) -> dict:
    """A variant against the reference on each slice, paired over bouts.

    ``log_loss_delta``: variant minus reference (negative is better).
    ``shift_toward_older``: mean of (variant - reference) probability for the
    older fighter, so how far the variant moved toward him. Whether that is
    an improvement depends on the sign of the reference's gap, which is why
    both gaps are reported as well.
    """
    ref_view = older_view(frame, frame[reference].to_numpy())
    var_view = older_view(frame, frame[column].to_numpy())
    out = {}
    for label, mask in slice_masks(ref_view).items():
        p_ref, p_var = ref_view["p"].to_numpy()[mask], var_view["p"].to_numpy()[mask]
        y = ref_view["y"].to_numpy()[mask]
        if len(y) == 0:
            out[label] = {"n": 0}
            continue
        loss = bootstrap_mean_interval(
            bout_log_loss(p_var, y) - bout_log_loss(p_ref, y), n_boot=N_BOOT, seed=seed
        )
        move = bootstrap_mean_interval(p_var - p_ref, n_boot=N_BOOT, seed=seed)
        out[label] = {
            "n": len(y),
            "log_loss_delta": loss["mean"],
            "log_loss_delta_interval": [loss["low"], loss["high"]],
            "shift_toward_older": move["mean"],
            "shift_toward_older_interval": [move["low"], move["high"]],
            "gap_reference": float((p_ref - y).mean()),
            "gap_variant": float((p_var - y).mean()),
        }
    return out


# --- validation: the candidate pipeline ------------------------------------------


def _fit_validation(
    recipe: Recipe, seed: int, train: pd.DataFrame, valid: pd.DataFrame
) -> np.ndarray:
    seeded = replace(recipe, gbm=recipe.gbm.with_seed(seed))
    _, p_valid = fit_recipe(seeded, train, valid, SPLIT_DATES.train_end)
    return p_valid


def run_validation(
    data: FeatureData,
    recipe: Recipe,
    config: FeatureConfig,
    variants: Sequence[Variant],
    seeds: Sequence[int] = SEEDS,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Per validation bout: identity, ``d_age``, ``d_elo``, the Elo
    baseline, and for each variant and seed the uncalibrated probability and
    the probability with Platt calibration cross-validated inside validation.

    Uses the candidate's schedule (fitted through 2022) with the test period
    locked. Feature tables are built once per distinct feature setting.
    """
    builds: dict[tuple, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for variant in variants:
        if variant.config_key in builds:
            continue
        frame = build_features(data, variant.feature_config(config)).frame
        splits = Splits(frame)  # locked
        splits.assert_test_locked("revision: validation")
        builds[variant.config_key] = (
            fit_rows(splits.train()),
            labelled(splits.valid()),
        )

    reference = builds[variants[0].config_key][1]
    for train, valid in builds.values():
        if list(valid["bout_id"]) != list(reference["bout_id"]):
            raise ValueError("validation bouts differ between feature settings")

    tasks = [(v, s) for v in variants for s in seeds]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_validation)(v.recipe(recipe), s, *builds[v.config_key])
        for v, s in tasks
    )
    out = reference[["bout_id", "date", "label", "d_age", "d_elo"]].copy()
    y = reference["label"].to_numpy(dtype=np.float64)
    folds = fold_ids(reference["date"].to_numpy())
    out["p_elo"] = predict(EloModel(), reference, ELO_SPEC)
    for (variant, seed), p in zip(tasks, results, strict=True):
        out[f"{variant.name}|{seed}|raw"] = p
        out[f"{variant.name}|{seed}|cal"] = cross_validated(PLATT, p, y, folds)
    return out


def seed_columns(frame: pd.DataFrame, variant: str, kind: str) -> list[str]:
    return [
        c
        for c in frame.columns
        if c.startswith(f"{variant}|") and c.endswith(f"|{kind}")
    ]


def seed_mean_loss(frame: pd.DataFrame, variant: str, kind: str) -> np.ndarray:
    """Per-bout log loss averaged over seeds (the selection convention)."""
    y = frame["label"].to_numpy(dtype=np.float64)
    columns = seed_columns(frame, variant, kind)
    return np.mean([bout_log_loss(frame[c].to_numpy(), y) for c in columns], axis=0)


def seed_mean_probability(frame: pd.DataFrame, variant: str, kind: str) -> np.ndarray:
    return frame[seed_columns(frame, variant, kind)].to_numpy().mean(axis=1)


# --- walk-forward backtest -------------------------------------------------------


def run_fold(
    data: FeatureData,
    recipe: Recipe,
    config: FeatureConfig,
    variants: Sequence[Variant],
    year: int,
    unlocked_for: str | None,
) -> pd.DataFrame:
    """One backtest year, every variant, seed 0, on the registered schedule.

    The test period stays locked unless ``unlocked_for`` is given, so a fold
    for 2025 or later cannot run by accident.
    """
    dates = fold_dates(year)
    out: pd.DataFrame | None = None
    built: dict[tuple, tuple] = {}
    for variant in variants:
        if variant.config_key not in built:
            frame = build_features(
                data, replace(variant.feature_config(config), train_end=dates.train_end)
            ).frame
            splits = Splits(frame, dates, test_dates=SPLIT_DATES)
            if unlocked_for is not None:
                splits.unlock_test(unlocked_for)
            built[variant.config_key] = (
                fit_rows(splits.train()),
                labelled(splits.valid()),
                labelled(
                    splits.between(
                        date(year, 1, 1), date(year, 12, 31), f"revision {year}"
                    )
                ),
            )
        train, valid, scored = built[variant.config_key]
        candidate, _ = fit_recipe(variant.recipe(recipe), train, valid, dates.train_end)
        if out is None:
            out = scored[["bout_id", "date", "label", "d_age", "d_elo"]].copy()
            out["year"] = year
            out["p_elo"] = predict(EloModel(), scored, ELO_SPEC)
        elif list(scored["bout_id"]) != list(out["bout_id"]):
            raise ValueError(f"{year}: scored bouts differ between feature settings")
        out[variant.name] = candidate.predict(scored)
    assert out is not None
    return out


def run_backtest(
    data: FeatureData,
    recipe: Recipe,
    config: FeatureConfig,
    variants: Sequence[Variant],
    years: Sequence[int],
    unlocked_for: str | None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    folds = Parallel(n_jobs=n_jobs)(
        delayed(run_fold)(
            data,
            recipe,
            config,
            variants,
            year,
            unlocked_for if year >= SPLIT_DATES.test_start.year else None,
        )
        for year in years
    )
    return pd.concat(folds, ignore_index=True)


def reproduces_backtest(bouts: pd.DataFrame, saved: Mapping) -> dict:
    """The ``base`` variant must repeat the saved backtest, fold for fold."""
    worst = 0.0
    compared = []
    for fold in saved["folds"]:
        rows = bouts[bouts["year"] == fold["year"]]
        if rows.empty:
            continue
        y = rows["label"].to_numpy(dtype=np.float64)
        for column, model in (("base", "model"), ("p_elo", "elo")):
            mine = float(bout_log_loss(rows[column].to_numpy(), y).mean())
            worst = max(worst, abs(mine - fold["metrics"][model]["log_loss"]))
        compared.append(fold["year"])
    return {
        "years": compared,
        "largest_log_loss_difference": worst,
        "identical": worst < 1e-12,
    }


# --- summaries -------------------------------------------------------------------


def validation_summary(frame: pd.DataFrame, variants: Sequence[str]) -> dict:
    """Validation log loss of each variant (mean over seeds), paired against
    ``base``, uncalibrated (the selection convention) and calibrated; and
    the slices on seed-averaged calibrated probabilities."""
    out: dict = {"n": len(frame), "variants": {}}
    base = {kind: seed_mean_loss(frame, "base", kind) for kind in ("raw", "cal")}
    probabilities = frame[["d_age", "d_elo", "label", "p_elo"]].copy()
    for name in variants:
        row: dict = {}
        for kind in ("raw", "cal"):
            losses = seed_mean_loss(frame, name, kind)
            delta = bootstrap_mean_interval(losses - base[kind], n_boot=N_BOOT)
            per_seed = [
                float(
                    bout_log_loss(frame[c].to_numpy(), frame["label"].to_numpy()).mean()
                )
                for c in seed_columns(frame, name, kind)
            ]
            row[kind] = {
                "log_loss": float(losses.mean()),
                "seed_log_losses": per_seed,
                "delta": delta["mean"],
                "delta_interval": [delta["low"], delta["high"]],
            }
        out["variants"][name] = row
        probabilities[name] = seed_mean_probability(frame, name, "cal")
    out["slices"] = slice_table(
        probabilities, {"elo": "p_elo", **{v: v for v in variants}}
    )
    out["paired_slices"] = {
        v: paired_slice_change(probabilities, v, "base")
        for v in variants
        if v != "base"
    }
    out["note"] = (
        "log_loss: mean over seeds of per-bout loss; raw = uncalibrated, cal = Platt "
        "cross-validated inside validation; slices use the seed-mean calibrated "
        "probability"
    )
    return out


def backtest_summary(bouts: pd.DataFrame, variants: Sequence[str]) -> dict:
    y = bouts["label"].to_numpy(dtype=np.float64)
    base = bout_log_loss(bouts["base"].to_numpy(), y)
    out: dict = {
        "years": sorted(int(v) for v in bouts["year"].unique()),
        "n": len(bouts),
        "variants": {},
    }
    for name in variants:
        losses = bout_log_loss(bouts[name].to_numpy(), y)
        delta = bootstrap_mean_interval(losses - base, n_boot=N_BOOT)
        out["variants"][name] = {
            "log_loss": float(losses.mean()),
            "delta": delta["mean"],
            "delta_interval": [delta["low"], delta["high"]],
        }
    out["slices"] = slice_table(bouts, {"elo": "p_elo", **{v: v for v in variants}})
    out["paired_slices"] = {
        v: paired_slice_change(bouts, v, "base") for v in variants if v != "base"
    }
    return out


# --- the adoption rule (DECISIONS D54) ---------------------------------------------

RULE_SLICE = "elite_veteran_+50"


def adoption(record: Mapping, valid: pd.DataFrame) -> dict:
    """Apply the rule registered in D54 to every variant but ``base``.

    1. validation log loss (uncalibrated, mean over seeds): the paired 95%
       interval of variant minus base does not lie entirely above zero;
    2. backtest 2013 to 2024, slice ``RULE_SLICE``: smaller absolute gap and
       lower log loss than base;
    3. validation, same slice (seed-mean, cross-validated Platt): log loss
       not higher than base's.

    Among variants that pass, the lowest validation log loss is recommended,
    unless its paired interval against a passing variant with fewer added
    inputs contains zero, in which case the one with fewer inputs is.
    """
    val, back = record[VALIDATION], record[BACKTEST]
    base_back = back["slices"]["base"][RULE_SLICE]
    base_val = val["slices"]["base"][RULE_SLICE]
    rows: dict = {}
    for name in val["variants"]:
        if name == "base":
            continue
        low = val["variants"][name]["raw"]["delta_interval"][0]
        b, v = back["slices"][name][RULE_SLICE], val["slices"][name][RULE_SLICE]
        criteria = {
            "1_validation_not_worse_beyond_noise": bool(low <= 0.0),
            "2_backtest_slice_better_calibrated": bool(
                abs(b["gap"]) < abs(base_back["gap"])
                and b["log_loss"] < base_back["log_loss"]
            ),
            "3_validation_slice_not_worse": bool(v["log_loss"] <= base_val["log_loss"]),
        }
        rows[name] = {
            "criteria": criteria,
            "passes": all(criteria.values()),
            "n_inputs_added": len(record["variants"][name]["antisymmetric_added"])
            + len(record["variants"][name]["symmetric_added"])
            + len(record["variants"][name]["feature_settings"]),
            "validation_log_loss": val["variants"][name]["raw"]["log_loss"],
        }
    passing = sorted(
        (n for n, r in rows.items() if r["passes"]),
        key=lambda n: rows[n]["validation_log_loss"],
    )
    recommended, tie_breaks = None, []
    if passing:
        recommended = passing[0]
        for other in passing[1:]:
            if rows[other]["n_inputs_added"] >= rows[recommended]["n_inputs_added"]:
                continue
            diff = bootstrap_mean_interval(
                seed_mean_loss(valid, recommended, "raw")
                - seed_mean_loss(valid, other, "raw"),
                n_boot=N_BOOT,
            )
            within = diff["low"] <= 0.0 <= diff["high"]
            tie_breaks.append(
                {
                    "lower": recommended,
                    "fewer_inputs": other,
                    "delta": diff["mean"],
                    "delta_interval": [diff["low"], diff["high"]],
                    "inside_noise": bool(within),
                }
            )
            if within:
                recommended = other
    return {
        "rule": "DECISIONS D54",
        "slice": RULE_SLICE,
        "variants": rows,
        "passing": passing,
        "tie_breaks": tie_breaks,
        "recommended": recommended,
    }


# --- (d) calibration window of the production refit ---------------------------------


def spread(p_a: np.ndarray) -> dict:
    """How confident a set of probabilities is: the favourite's probability."""
    fav = np.maximum(p_a, 1.0 - p_a)
    q10, q50, q90 = np.quantile(fav, [0.1, 0.5, 0.9])
    return {
        "n": len(fav),
        "mean_favourite": float(fav.mean()),
        "p10": float(q10),
        "median": float(q50),
        "p90": float(q90),
        "share_at_least_70": float((fav >= 0.7).mean()),
    }


def _oot(
    candidate_model, recipe: Recipe, frame: pd.DataFrame, start: date, end: date
) -> pd.DataFrame:
    """Uncalibrated predictions of ``candidate_model`` for labelled bouts
    dated ``start`` to ``end`` without an attribute gap (as D47 fits the
    slope), with their labels and dates."""
    rows = frame[
        (frame["date"] >= pd.Timestamp(start))
        & (frame["date"] <= pd.Timestamp(end))
        & frame["label"].notna()
        & ~(frame["a_attr_gap"] | frame["b_attr_gap"])
    ]
    p = Candidate(recipe, candidate_model, IDENTITY_CALIBRATOR).predict(
        rows, calibrated=False
    )
    return pd.DataFrame(
        {"date": rows["date"].to_numpy(), "y": rows["label"].to_numpy(), "p": p}
    )


def calibration_window(args: argparse.Namespace) -> int:
    """Platt slopes for the production refit fitted on 2023 to 2026 instead of
    2025 to 2026, and what each does to confidence. Nothing is saved but a
    JSON record; no slope is adopted here (it cannot be scored on
    validation)."""
    version = json.loads((args.eval / "selection.json").read_text())["version"]
    candidate, _ = registry.load_candidate(args.models, version)
    refits = [
        v
        for v in registry.list_versions(args.models)
        if (registry.read_manifest(args.models, v).get("lineage") or {}).get("parent")
        == version
    ]
    if len(refits) != 1:
        raise ValueError(f"expected one production refit of {version}, found {refits}")
    (production_version,) = refits
    production, _ = registry.load_candidate(args.models, production_version)
    recipe = candidate.recipe
    config = _feature_config(args.models, version)
    conn = connect(args.db)
    try:
        data = load_feature_data(conn)
    finally:
        conn.close()
    reason = (
        f"model revision M4.5: calibration window for the refit of {version} "
        "(analysis only; nothing saved or adopted)"
    )
    _log_test_look(args.models, reason)
    last = max(b.date for b in data.bouts)
    valid_start, test_start = SPLIT_DATES.valid_start, SPLIT_DATES.test_start

    # Model fitted through 2022: the candidate itself, on its own table.
    t2022 = build_features(data, config).frame
    # Model fitted through 2024: the refit's calibration model (D47).
    count, _, _ = tree_count(data, recipe, config, SPLIT_DATES.valid_end, reason)
    t2024 = build_features(data, replace(config, train_end=SPLIT_DATES.valid_end))
    splits = Splits(t2024.frame, SplitDates(SPLIT_DATES.valid_end, last))
    splits.unlock_test(reason)
    m2024 = fit_fixed(
        recipe, fit_rows(splits.train()), count.n_trees, SPLIT_DATES.valid_end
    )

    a = _oot(candidate.model, recipe, t2022, valid_start, SPLIT_DATES.valid_end)
    b = _oot(candidate.model, recipe, t2022, test_start, last)
    c = _oot(m2024, recipe, t2024.frame, test_start, last)
    windows = {
        "current_2025_2026_from_2024_model": c,
        "2023_2026_from_2022_model": pd.concat([a, b]),
        "2023_2026_rolling_2022_then_2024_model": pd.concat([a, c]),
        "reference_2023_2024_from_2022_model": a,
    }

    # The production booster's uncalibrated probabilities on 2026 bouts, from
    # its own feature pipeline. It was fitted on these bouts, so only their
    # spread is reported, never a score.
    pipeline = registry.read_json(
        args.models, production_version, "feature_pipeline.json"
    )
    final = build_features(
        data,
        _config_from(pipeline["config"]),
        FittedParams.from_dict(pipeline["params"]),
    ).frame
    recent = labelled(final[final["date"] >= pd.Timestamp(date(last.year, 1, 1))])
    raw = production.predict(recent, calibrated=False)

    out: dict = {
        "candidate": version,
        "production": production_version,
        "production_slope": production.calibrator.slope,
        "calibration_model_2024_trees": count.n_trees,
        "spread_on": f"{len(recent)} bouts of {last.year} (in the production booster's training data)",
        "uncalibrated": spread(raw),
        "windows": {},
    }
    for name, frame in windows.items():
        cal = fit_calibrator(PLATT, frame["p"].to_numpy(), frame["y"].to_numpy())
        out["windows"][name] = {
            "n_bouts": len(frame),
            "dates": [
                str(pd.Timestamp(frame["date"].min()).date()),
                str(pd.Timestamp(frame["date"].max()).date()),
            ],
            "slope": cal.slope,
            "production_spread": spread(cal.apply(raw)),
            "out_of_time_spread_calibrated": spread(cal.apply(frame["p"].to_numpy())),
        }
    _write_json(args.out / "calibration_window.json", out)
    print(f"Wrote {args.out / 'calibration_window.json'}")
    for name, row in out["windows"].items():
        print(
            f"  {name:42} n={row['n_bouts']:5d} slope={row['slope']:.3f} "
            f"mean favourite {row['production_spread']['mean_favourite']:.3f}"
        )
    return 0


def _config_from(raw: Mapping) -> FeatureConfig:
    raw = dict(raw)
    raw["train_end"] = date.fromisoformat(raw["train_end"])
    return FeatureConfig(**raw)


# --- command ---------------------------------------------------------------------


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _feature_config(models: Path, version: str) -> FeatureConfig:
    pipeline = registry.read_json(models, version, "feature_pipeline.json")
    return _config_from(pipeline["config"])


def _log_test_look(models: Path, what: str) -> None:
    entry = {
        "what": what,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **{f"git_{k}": v for k, v in registry.git_state().items()},
    }
    with (models / LEDGER_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    names = [n.strip() for n in args.variants.split(",") if n.strip()]
    if names[0] != "base":
        names = ["base", *[n for n in names if n != "base"]]
    unknown = set(names) - set(VARIANTS)
    if unknown:
        print(f"unknown variants: {sorted(unknown)}; known: {sorted(VARIANTS)}")
        return 2
    variants = [VARIANTS[n] for n in names]

    version = json.loads((args.eval / "selection.json").read_text())["version"]
    candidate, _ = registry.load_candidate(args.models, version)
    if registry.is_refit(registry.read_manifest(args.models, version)):
        raise ValueError("the revision starts from the tested candidate, not a refit")
    config = _feature_config(args.models, version)
    conn = connect(args.db)
    try:
        data = load_feature_data(conn)
    finally:
        conn.close()

    record: dict = {
        "candidate": version,
        "variants": {
            v.name: {
                "antisymmetric_added": list(v.antisymmetric),
                "symmetric_added": list(v.symmetric),
                "feature_settings": {k: v_ for k, v_ in v.config.items()},
            }
            for v in variants
        },
        "slice_definition": {
            "age_gap_years_at_least": AGE_GAP,
            "elo_gaps": list(ELO_GAPS),
        },
        "seeds": list(args.seeds),
    }

    logger.info("validation: %d variants x %d seeds", len(variants), len(args.seeds))
    valid = run_validation(
        data, candidate.recipe, config, variants, args.seeds, args.jobs
    )
    seed0 = f"base|{SEEDS[0]}|raw"
    registered = candidate.predict(
        build_features(data, config).frame.pipe(lambda f: labelled(Splits(f).valid())),
        calibrated=False,
    )
    record["validation_reproduces_candidate"] = bool(
        np.array_equal(valid[seed0].to_numpy(), registered)
    )
    record[VALIDATION] = validation_summary(valid, names)

    years = list(range(FIRST_YEAR, SPLIT_DATES.test_start.year))
    reason = None
    if args.secondary:
        last = max(b.date for b in data.bouts).year
        years += list(range(SPLIT_DATES.test_start.year, last + 1))
        reason = (
            f"model revision M4.5 of {version} (analysis only; 2025-2026 reported "
            "as a secondary, already-seen check, not used for any decision)"
        )
        _log_test_look(args.models, reason)
    logger.info("backtest: %d years x %d variants", len(years), len(variants))
    bouts = run_backtest(
        data, candidate.recipe, config, variants, years, reason, args.jobs
    )
    saved = json.loads((args.eval / "walk_forward.json").read_text())
    record["backtest_reproduces_saved"] = reproduces_backtest(bouts, saved)
    before = bouts[bouts["year"] < SPLIT_DATES.test_start.year]
    record[BACKTEST] = backtest_summary(before, names)
    record["backtest_by_year"] = {
        str(year): {
            name: float(
                bout_log_loss(
                    rows[name].to_numpy(), rows["label"].to_numpy(dtype=np.float64)
                ).mean()
            )
            for name in names
        }
        for year, rows in bouts.groupby("year")
    }
    if args.secondary:
        after = bouts[bouts["year"] >= SPLIT_DATES.test_start.year]
        record[SECONDARY] = backtest_summary(after, names)
    if len(names) > 1:
        record["adoption"] = adoption(record, valid)

    args.out.mkdir(parents=True, exist_ok=True)
    tag = "-".join(names)
    _write_json(args.out / f"revision_{tag}.json", record)
    valid.to_parquet(args.out / f"validation_{tag}.parquet", index=False)
    bouts.to_parquet(args.out / f"backtest_{tag}.parquet", index=False)
    print(f"Wrote {args.out / f'revision_{tag}.json'}")
    print(
        f"validation reproduces candidate: {record['validation_reproduces_candidate']}"
    )
    print(f"backtest reproduces saved: {record['backtest_reproduces_saved']}")
    if "adoption" in record:
        for name, row in record["adoption"]["variants"].items():
            print(
                f"  {name:13} {'passes' if row['passes'] else 'fails'} {row['criteria']}"
            )
        print(f"recommended under D54: {record['adoption']['recommended']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ufc.models.revision", description=__doc__.split("\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="validation and backtest for named variants")
    p.add_argument("--variants", default="base")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument(
        "--no-secondary",
        dest="secondary",
        action="store_false",
        help="leave out the 2025 and 2026 folds (the test period)",
    )
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--models", type=Path, default=registry.DEFAULT_ROOT)
    p.add_argument("--eval", type=Path, default=Path("data/reports/eval"))
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    p.set_defaults(func=run)

    c = sub.add_parser(
        "calibration-window", help="(d) Platt slope fitted on 2023-2026 for the refit"
    )
    c.add_argument("--db", type=Path, default=DEFAULT_DB)
    c.add_argument("--models", type=Path, default=registry.DEFAULT_ROOT)
    c.add_argument("--eval", type=Path, default=Path("data/reports/eval"))
    c.add_argument("--out", type=Path, default=DEFAULT_OUT)
    c.set_defaults(func=calibration_window)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
