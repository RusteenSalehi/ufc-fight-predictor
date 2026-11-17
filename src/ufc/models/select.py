"""Model selection on validation, in a fixed order of stages.

1. Rate shrinkage strength and the level features, with reference settings.
2. Hyperparameter search for XGBoost and LightGBM.
3. Recency half-life, and which library.
4. Calibration method, by cross-validation inside validation.
5. Feature-family ablation of the selected recipe.

Feature choices come first and use settings fixed in advance, so the search
cannot tilt them toward the option it happened to be tuned on. Every stage
asserts that the test period is still locked, and gets its rows from
``Splits``, which would refuse a test-period row anyway.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ufc.models.baselines import (
    C_GRID,
    ELO_SPEC,
    EloModel,
    LogisticModel,
    fit_logistic,
)
from ufc.models.calibrate import IDENTITY, choose_method
from ufc.models.design import FeatureSpec, labels, matrix, predict
from ufc.models.evaluate import evaluate_period
from ufc.models.gbm import LIBRARIES, GbmConfig
from ufc.models.metrics import bout_log_loss, log_loss
from ufc.models.recipe import Candidate, Recipe, fit_data, fit_recipe
from ufc.models.splits import Splits, labelled
from ufc.models.tune import (
    HALF_LIVES,
    N_TRIALS,
    REFERENCE,
    SEEDS,
    TOP_K,
    Score,
    Variant,
    difference,
    pooled,
    sample_params,
    score_variants,
)

logger = logging.getLogger(__name__)

SHRINK_GRID: tuple[float, ...] = (0.0, 5.0, 15.0, 30.0, 60.0)
"""Prior strengths in cage minutes: none, one round, one three-round fight,
two of them, four of them."""

FINAL_SEED = 0
LEVEL_GROUP = "level"


def _rows(splits: Splits, stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and validation bouts that have a winner."""
    splits.assert_test_locked(stage)
    return labelled(splits.train()), labelled(splits.valid())


def _half_life_label(half_life: float | None) -> str:
    return "none" if half_life is None else f"{half_life:g}"


# --- baselines ---------------------------------------------------------------


def tune_logistic(
    train: pd.DataFrame, valid: pd.DataFrame, spec: FeatureSpec
) -> tuple[LogisticModel, dict]:
    """Fit the logistic baseline at every C and keep the best on validation."""
    logistic_spec = FeatureSpec(spec.antisymmetric)
    x, y = matrix(train, logistic_spec), labels(train)
    valid_y = labels(valid)
    best: tuple[float, LogisticModel] | None = None
    table = []
    for c in C_GRID:
        model = fit_logistic(x, y, c, logistic_spec)
        loss = log_loss(predict(model, valid, logistic_spec), valid_y)
        table.append({"c": c, "log_loss": loss})
        if best is None or loss < best[0]:
            best = (loss, model)
    assert best is not None
    return best[1], {"grid": table, "c": best[1].c, "log_loss": best[0]}


def baseline_predictions(
    train: pd.DataFrame, frame: pd.DataFrame, spec: FeatureSpec, c: float
) -> dict[str, np.ndarray]:
    """P(A wins) from both baselines for the bouts in ``frame``. The
    logistic regression is fitted on ``train``; Elo needs no fitting."""
    logistic_spec = FeatureSpec(spec.antisymmetric)
    model = fit_logistic(matrix(train, logistic_spec), labels(train), c, logistic_spec)
    return {
        "elo": predict(EloModel(), frame, ELO_SPEC),
        "logistic": predict(model, frame, logistic_spec),
    }


# --- stage 1: feature options ------------------------------------------------


def _tail_summary(train: pd.DataFrame, names: Sequence[str]) -> dict:
    """How heavy the tails of ``names`` are on the training period: the
    largest absolute value in standard deviations, and excess kurtosis."""
    values = train[list(names)]
    extreme = (values.abs().max() / values.std()).sort_values(ascending=False)
    return {
        "largest_abs_sd": float(extreme.iloc[0]),
        "largest_abs_sd_feature": str(extreme.index[0]),
        "median_abs_sd": float(extreme.median()),
        "median_excess_kurtosis": float(values.kurt().median()),
        "n_features": len(names),
    }


def choose_shrinkage(
    splits_for: Callable[[float], Splits],
    spec: FeatureSpec,
    shrunk_features: Sequence[str],
    grid: Sequence[float] = SHRINK_GRID,
    seeds: Sequence[int] = SEEDS,
    n_jobs: int = 1,
) -> dict:
    """Score each prior strength with both libraries at reference settings.

    The criterion is validation log loss averaged over the two libraries and
    the seeds. The lowest wins; a tie goes to the smaller strength.
    """
    if 0.0 not in grid:
        raise ValueError("the grid must include 0 (no shrinkage)")
    variants: list[Variant] = []
    rows: dict[float, dict] = {}
    bout_ids = None
    for minutes in grid:
        train, valid = _rows(splits_for(minutes), "shrinkage selection")
        if bout_ids is None:
            bout_ids = list(valid["bout_id"])
        elif list(valid["bout_id"]) != bout_ids:
            raise ValueError("validation bouts differ between shrinkage builds")
        data = fit_data(train, valid, spec, None, splits_for(minutes).dates.train_end)
        variants += [
            Variant(
                f"{library}@{minutes:g}", GbmConfig(library, REFERENCE[library]), data
            )
            for library in LIBRARIES
        ]
        rows[minutes] = {
            "shrink_minutes": minutes,
            "logistic_log_loss": tune_logistic(train, valid, spec)[1]["log_loss"],
            "tails": _tail_summary(train, shrunk_features),
        }
    scores = score_variants(variants, seeds, n_jobs)

    def names(minutes: float) -> list[str]:
        return [f"{library}@{minutes:g}" for library in LIBRARIES]

    reference = pooled(scores, names(0.0))
    chosen = None
    for minutes in sorted(grid):
        losses = pooled(scores, names(minutes))
        rows[minutes].update(
            {
                library: scores[f"{library}@{minutes:g}"].log_loss
                for library in LIBRARIES
            }
        )
        rows[minutes]["criterion"] = float(losses.mean())
        rows[minutes].update(difference(losses, reference))
        if chosen is None or rows[minutes]["criterion"] < rows[chosen]["criterion"]:
            chosen = minutes
    logger.info("shrinkage: chose %g cage minutes", chosen)
    return {
        "grid": [rows[m] for m in sorted(grid)],
        "chosen": chosen,
        "criterion": "validation log loss, mean of XGBoost and LightGBM at "
        "reference settings, mean over seeds",
        "delta_is": "criterion minus the criterion at 0 minutes, paired over bouts",
    }


def try_level_features(
    splits: Splits,
    spec: FeatureSpec,
    level_names: Sequence[str],
    seeds: Sequence[int] = SEEDS,
    n_jobs: int = 1,
) -> dict:
    """Do the swap-invariant level features lower validation log loss?

    Same criterion as the shrinkage choice. They are kept only if it is
    strictly lower with them than without.
    """
    train, valid = _rows(splits, "level-feature test")
    variants = []
    for label, option in (
        ("without", spec),
        ("with", spec.with_symmetric(level_names)),
    ):
        data = fit_data(train, valid, option, None, splits.dates.train_end)
        variants += [
            Variant(f"{library}/{label}", GbmConfig(library, REFERENCE[library]), data)
            for library in LIBRARIES
        ]
    scores = score_variants(variants, seeds, n_jobs)
    without = pooled(scores, [f"{library}/without" for library in LIBRARIES])
    with_levels = pooled(scores, [f"{library}/with" for library in LIBRARIES])
    result = {
        "features": list(level_names),
        "without": {
            "criterion": float(without.mean()),
            **{library: scores[f"{library}/without"].log_loss for library in LIBRARIES},
        },
        "with": {
            "criterion": float(with_levels.mean()),
            **{library: scores[f"{library}/with"].log_loss for library in LIBRARIES},
        },
        **difference(with_levels, without),
        "delta_is": "criterion with level features minus without, paired over bouts",
    }
    result["kept"] = bool(result["with"]["criterion"] < result["without"]["criterion"])
    logger.info("level features: %s", "kept" if result["kept"] else "not kept")
    return result


# --- stage 2 and 3: search, half-life, library --------------------------------


def search(
    splits: Splits,
    spec: FeatureSpec,
    n_trials: int = N_TRIALS,
    top_k: int = TOP_K,
    seeds: Sequence[int] = SEEDS,
    n_jobs: int = 1,
) -> dict:
    """Random search per library with no recency weighting.

    Every setting is scored on one seed; the best ``top_k`` are scored again
    on all seeds and the lowest mean wins. Re-scoring guards against picking
    a setting that was merely lucky on one seed.
    """
    train, valid = _rows(splits, "hyperparameter search")
    data = fit_data(train, valid, spec, None, splits.dates.train_end)
    out: dict = {}
    for library in LIBRARIES:
        params = sample_params(library, n_trials)
        first = score_variants(
            [
                Variant(f"{library}#{i}", GbmConfig(library, p), data)
                for i, p in enumerate(params)
            ],
            seeds[:1],
            n_jobs,
        )
        trials = [
            {
                "trial": i,
                "params": p,
                "log_loss": first[f"{library}#{i}"].log_loss,
                "n_trees": first[f"{library}#{i}"].n_trees[0],
            }
            for i, p in enumerate(params)
        ]
        finalists = sorted(trials, key=lambda t: (t["log_loss"], t["trial"]))[:top_k]
        again = score_variants(
            [
                Variant(
                    f"{library}#{t['trial']}", GbmConfig(library, t["params"]), data
                )
                for t in finalists
            ],
            seeds,
            n_jobs,
        )
        for t in finalists:
            t["seed_mean_log_loss"] = again[f"{library}#{t['trial']}"].log_loss
        best = min(finalists, key=lambda t: (t["seed_mean_log_loss"], t["trial"]))
        out[library] = {
            "trials": trials,
            "finalists": finalists,
            "best_trial": best["trial"],
            "best_params": best["params"],
            "best_seed_mean_log_loss": best["seed_mean_log_loss"],
            "reference_log_loss": trials[0]["log_loss"],
        }
        logger.info("search %s: trial %d", library, best["trial"])
    return out


def choose_half_life(
    splits: Splits,
    spec: FeatureSpec,
    params: Mapping[str, Mapping[str, float]],
    half_lives: Sequence[float | None] = HALF_LIVES,
    seeds: Sequence[int] = SEEDS,
    n_jobs: int = 1,
) -> tuple[dict, dict[str, Score]]:
    """Score each library's best settings at each recency half-life and
    choose the library and half-life with the lowest validation log loss.
    A tie goes to the earlier library, then to less weighting."""
    train, valid = _rows(splits, "half-life selection")
    variants = []
    for half_life in half_lives:
        data = fit_data(train, valid, spec, half_life, splits.dates.train_end)
        variants += [
            Variant(
                f"{library}|{_half_life_label(half_life)}",
                GbmConfig(library, params[library]),
                data,
            )
            for library in LIBRARIES
        ]
    scores = score_variants(variants, seeds, n_jobs)
    table = []
    best = None
    for library in LIBRARIES:
        unweighted = scores[f"{library}|none"].bout_losses
        for half_life in half_lives:
            score = scores[f"{library}|{_half_life_label(half_life)}"]
            row = {
                "library": library,
                "half_life_years": half_life,
                **score.to_dict(),
                **difference(score.bout_losses, unweighted),
            }
            table.append(row)
            if best is None or row["log_loss"] < best["log_loss"]:
                best = row
    assert best is not None
    logger.info("selected %s, half-life %s", best["library"], best["half_life_years"])
    return {
        "table": table,
        "library": best["library"],
        "half_life_years": best["half_life_years"],
        "log_loss": best["log_loss"],
        "delta_is": "log loss minus the same library with no weighting, paired over bouts",
    }, scores


# --- stage 5: ablation --------------------------------------------------------


def ablate(
    splits: Splits,
    recipe: Recipe,
    families: Mapping[str, Sequence[str]],
    seeds: Sequence[int] = SEEDS,
    n_jobs: int = 1,
) -> dict:
    """Remove each feature family in turn and refit the selected recipe.

    ``delta`` is validation log loss without the family minus with it:
    positive means the family helps, zero or negative means it did not.
    """
    train, valid = _rows(splits, "ablation")
    used = set(recipe.spec.names)
    groups = {
        name: [f for f in members if f in used] for name, members in families.items()
    }
    if recipe.spec.symmetric:
        groups[LEVEL_GROUP] = list(recipe.spec.symmetric)
    variants = [
        Variant(
            "full",
            recipe.gbm,
            fit_data(
                train,
                valid,
                recipe.spec,
                recipe.half_life_years,
                splits.dates.train_end,
            ),
        )
    ]
    for name, members in groups.items():
        if not members:
            continue
        variants.append(
            Variant(
                name,
                recipe.gbm,
                fit_data(
                    train,
                    valid,
                    recipe.spec.without(members),
                    recipe.half_life_years,
                    splits.dates.train_end,
                ),
            )
        )
    scores = score_variants(variants, seeds, n_jobs)
    full = scores["full"]
    rows = [
        {
            "family": name,
            "n_features": len(groups[name]),
            "log_loss_without": scores[name].log_loss,
            **difference(scores[name].bout_losses, full.bout_losses),
            "helped": bool(scores[name].log_loss > full.log_loss),
        }
        for name in groups
        if name in scores
    ]
    return {
        "full_log_loss": full.log_loss,
        "families": rows,
        "delta_is": "validation log loss without the family minus with it; "
        "positive means the family helps",
        "n_seeds": len(seeds),
    }


# --- the whole selection ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SelectionSettings:
    """How much searching the selection does. The defaults are the real run;
    tests use smaller values to exercise the same code quickly."""

    shrink_grid: tuple[float, ...] = SHRINK_GRID
    seeds: tuple[int, ...] = SEEDS
    n_trials: int = N_TRIALS
    top_k: int = TOP_K
    half_lives: tuple[float | None, ...] = HALF_LIVES

    def to_dict(self) -> dict:
        return {
            "shrink_grid": list(self.shrink_grid),
            "seeds": list(self.seeds),
            "n_trials": self.n_trials,
            "top_k": self.top_k,
            "half_lives": list(self.half_lives),
        }


@dataclass(frozen=True, slots=True)
class Selection:
    recipe: Recipe
    candidate: Candidate
    shrink_minutes: float
    logistic_c: float
    report: dict


def select(
    splits_for: Callable[[float], Splits],
    features: Sequence[str],
    families: Mapping[str, Sequence[str]],
    level_names: Sequence[str],
    shrunk_features: Sequence[str],
    *,
    settings: SelectionSettings | None = None,
    n_jobs: int = 1,
) -> Selection:
    """Run every stage and return the frozen recipe, the fitted candidate and
    a JSON-ready record of every comparison that was made."""
    settings = settings or SelectionSettings()
    shrink_grid, seeds, half_lives = (
        settings.shrink_grid,
        settings.seeds,
        settings.half_lives,
    )
    n_trials, top_k = settings.n_trials, settings.top_k
    base = FeatureSpec(tuple(features))
    report: dict = {"seeds": list(seeds), "reference_params": REFERENCE}

    report["shrinkage"] = choose_shrinkage(
        splits_for, base, shrunk_features, shrink_grid, seeds, n_jobs
    )
    shrink_minutes = report["shrinkage"]["chosen"]
    splits = splits_for(shrink_minutes)
    train, valid = _rows(splits, "selection")
    report["splits"] = splits.dates.to_dict()
    for name, rows in (("train", splits.train()), ("validation", splits.valid())):
        report[f"n_{name}_bouts"] = len(rows)
        report[f"n_{name}_scored"] = int(rows["label"].notna().sum())

    report["levels"] = try_level_features(splits, base, level_names, seeds, n_jobs)
    spec = base.with_symmetric(level_names) if report["levels"]["kept"] else base

    report["search"] = search(splits, spec, n_trials, top_k, seeds, n_jobs)
    params = {
        library: report["search"][library]["best_params"] for library in LIBRARIES
    }
    report["half_life"], scores = choose_half_life(
        splits, spec, params, half_lives, seeds, n_jobs
    )
    library = report["half_life"]["library"]
    half_life = report["half_life"]["half_life_years"]
    selected = scores[f"{library}|{_half_life_label(half_life)}"]

    logistic, report["logistic"] = tune_logistic(train, valid, spec)
    valid_y = labels(valid)
    baselines = baseline_predictions(train, valid, spec, logistic.c)
    comparison = {}
    for name, p in baselines.items():
        comparison[name] = {
            "log_loss": log_loss(p, valid_y),
            **difference(selected.bout_losses, bout_log_loss(p, valid_y)),
        }
    report["versus_baselines"] = {
        "candidate_log_loss": selected.log_loss,
        "candidate_is": "selected library and half-life, mean over seeds, uncalibrated",
        "baselines": comparison,
        "delta_is": "candidate log loss minus the baseline's, paired over bouts",
        "beats_all": bool(
            all(selected.log_loss < b["log_loss"] for b in comparison.values())
        ),
    }

    # The model that is kept: one seed, trees by early stopping on validation.
    gbm = GbmConfig(library, params[library], FINAL_SEED)
    draft = Recipe(gbm, half_life, spec, IDENTITY)
    _, p_valid = fit_recipe(draft, train, valid, splits.dates.train_end)
    calibration = choose_method(p_valid, valid_y, valid["date"].to_numpy())
    cross_validated = calibration.pop("predictions")
    recipe = Recipe(gbm, half_life, spec, calibration["chosen"])
    candidate, p_again = fit_recipe(recipe, train, valid, splits.dates.train_end)
    assert np.array_equal(p_again, p_valid), "refitting the recipe changed the model"
    calibration["slope"] = candidate.calibrator.slope
    calibration["isotonic_steps"] = len(candidate.calibrator.x)
    report["calibration"] = calibration
    report["selected"] = {
        "recipe": recipe.to_dict(),
        "n_trees": candidate.model.n_trees,
        "shrink_minutes": shrink_minutes,
        "n_features": len(spec.names),
    }

    # Validation scores. The calibrated row uses cross-validated calibration,
    # because the final calibrator was fitted on these same bouts.
    report["validation"] = evaluate_period(
        splits.valid(),
        {
            "model": cross_validated[recipe.calibration],
            "model_uncalibrated": p_valid,
            **baselines,
        },
        primary="model",
    )
    report["validation"]["note"] = (
        "model = calibrated by cross-validation inside validation; "
        "model_uncalibrated = the kept single-seed model before calibration"
    )

    report["ablation"] = ablate(splits, recipe, families, seeds, n_jobs)
    splits.assert_test_locked("selection")
    return Selection(recipe, candidate, shrink_minutes, logistic.c, report)
