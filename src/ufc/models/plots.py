"""Static plots of the evaluation records.

Every plot is drawn from a record that ``evaluate.py``, ``select.py`` or
``walkforward.py`` produced, never from raw predictions, so a number on a
plot is a number in the metrics JSON.

Conventions: one colour per model everywhere (never by rank), thin marks,
hairline grid, a legend whenever there are two or more series, text in ink
colours and never in a series colour, and one y-scale per panel: two measures
get two panels, not two axes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display: files only

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

COLORS: dict[str, str] = {
    "model": "#2a78d6",
    "logistic": "#eb6834",
    "elo": "#1baf7a",
    "model_uncalibrated": MUTED,
}
"""Fixed colour per model. The three hues were checked for separation under
colour-vision deficiency; the uncalibrated model is a de-emphasised gray."""

LABELS: dict[str, str] = {
    "model": "Model (calibrated)",
    "model_uncalibrated": "Model before calibration",
    "logistic": "Logistic regression",
    "elo": "Elo",
}
BASELINE_ORDER: tuple[str, ...] = ("model", "logistic", "elo")

LINE = 2.0
MARKER = 7.0  # points; about 9 px at the saved resolution
BAR = 0.22  # bar thickness in axis units, well under the slot width
DPI = 160


def _style(ax: Axes, *, grid: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis=grid, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.xaxis.label.set_size(9.5)
    ax.yaxis.label.set_size(9.5)


def _title(ax: Axes, text: str) -> None:
    ax.set_title(text, loc="left", color=INK, fontsize=11, fontweight="bold", pad=10)


def _legend(ax: Axes, **kwargs) -> None:
    legend = ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, **kwargs)
    legend.set_zorder(5)


def _figure(n_cols: int, width: float = 5.2, height: float = 4.0, n_rows: int = 1):
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(width * n_cols, height * n_rows), squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def _save(fig: Figure, path: Path, caption: str) -> Path:
    fig.text(0.01, 0.005, caption, color=MUTED, fontsize=8, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path


def _points(
    ax: Axes, x: Sequence[float], y: Sequence[float], name: str, **kwargs
) -> None:
    ax.plot(
        x,
        y,
        color=COLORS[name],
        linewidth=LINE,
        marker="o",
        markersize=MARKER,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        solid_capstyle="round",
        solid_joinstyle="round",
        label=LABELS[name],
        **kwargs,
    )


# --- reliability --------------------------------------------------------------


def _reliability_panel(
    ax: Axes, tables: Mapping[str, list[dict]], names: Sequence[str]
) -> None:
    ax.plot([0.5, 1.0], [0.5, 1.0], color=AXIS, linewidth=1.0, zorder=1)
    ax.annotate(
        "perfectly calibrated",
        xy=(0.93, 0.93),
        xytext=(-6, -14),
        textcoords="offset points",
        color=MUTED,
        fontsize=8,
        ha="right",
    )
    for name in names:
        rows = [r for r in tables[name] if r["n"]]
        x = [r["mean_confidence"] for r in rows]
        y = [r["favourite_win_rate"] for r in rows]
        ax.vlines(
            x,
            [r["win_rate_low"] for r in rows],
            [r["win_rate_high"] for r in rows],
            color=COLORS[name],
            linewidth=1.0,
            alpha=0.45,
            zorder=2,
        )
        _points(ax, x, y, name, zorder=3)
    ax.set_xlim(0.48, 1.0)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("Probability given to the favoured fighter")
    ax.set_ylabel("How often the favoured fighter won")
    _style(ax, grid="both")
    _legend(ax, loc="upper left")


def reliability(record: Mapping, path: Path, period: str) -> Path:
    """Reliability before and after calibration, and for the baselines, with
    the number of bouts behind each point of the calibrated model."""
    tables = record["reliability"]
    fig, axes = _figure(2, height=5.6, n_rows=2)
    fig.set_size_inches(10.4, 6.6)
    grid = fig.add_gridspec(2, 2, height_ratios=(4.2, 1.0))
    for ax in axes.ravel():
        ax.remove()
    top_left, top_right = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    bottom = fig.add_subplot(grid[1, :])

    _reliability_panel(top_left, tables, ("model_uncalibrated", "model"))
    _title(top_left, "Before and after calibration")
    _reliability_panel(top_right, tables, ("logistic", "elo"))
    _title(top_right, "Baselines")

    rows = tables["model"]
    centres = [(r["low"] + r["high"]) / 2 for r in rows]
    bottom.bar(
        centres,
        [r["n"] for r in rows],
        width=0.022,
        color=COLORS["model"],
        linewidth=0,
    )
    for centre, row in zip(centres, rows, strict=True):
        if row["n"]:
            bottom.annotate(
                f"{row['n']}",
                xy=(centre, row["n"]),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                color=INK_SECONDARY,
                fontsize=8,
            )
    bottom.set_xlim(0.48, 1.0)
    bottom.set_xlabel("Probability given to the favoured fighter (calibrated model)")
    bottom.set_ylabel("Bouts")
    bottom.margins(y=0.25)
    _style(bottom)
    _title(bottom, "Bouts per bin")
    return _save(
        fig,
        path,
        f"{period}: {record['n_scored']} bouts with a winner. Whiskers are 95% "
        "intervals for the win rate in a bin.",
    )


# --- breakdowns ---------------------------------------------------------------


def _grouped_bars(
    ax: Axes,
    groups: Mapping[str, Mapping[str, Mapping]],
    metric: str,
    names: Sequence[str],
) -> None:
    labels = list(groups)
    for i, name in enumerate(names):
        offset = (i - (len(names) - 1) / 2) * (BAR + 0.03)
        values = [groups[label][name][metric] for label in labels]
        ax.bar(
            [j + offset for j in range(len(labels))],
            values,
            width=BAR,
            color=COLORS[name],
            linewidth=0,
            label=LABELS[name],
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        [f"{label}\nn = {groups[label][names[0]]['n']}" for label in labels],
        color=INK_SECONDARY,
    )
    ax.set_xlim(-0.6, len(labels) - 0.4)
    _style(ax)


def _zoom(ax: Axes, values: Sequence[float], pad: float = 0.25) -> None:
    """Bars of log loss all sit near 0.65 to 0.70; start the axis just below
    the smallest so differences are visible, and say so on the axis."""
    low, high = min(values), max(values)
    span = max(high - low, 0.01)
    ax.set_ylim(low - pad * span - 0.01, high + pad * span)


def by_group(
    groups: Mapping[str, Mapping[str, Mapping]],
    path: Path,
    title: str,
    caption: str,
    names: Sequence[str] = BASELINE_ORDER,
) -> Path:
    """Log loss and accuracy per group (year, debut status), one panel each."""
    if not groups:
        raise ValueError("nothing to plot")
    fig, axes = _figure(2, width=5.4)
    for ax, metric, label in (
        (
            axes[0, 0],
            "log_loss",
            "Log loss (lower is better; axis does not start at 0)",
        ),
        (axes[0, 1], "accuracy", "Accuracy"),
    ):
        _grouped_bars(ax, groups, metric, names)
        ax.set_ylabel(label)
        values = [groups[g][n][metric] for g in groups for n in names]
        if metric == "log_loss":
            _zoom(ax, values)
            ax.axhline(0.6931, color=AXIS, linewidth=1.0)
            if ax.get_ylim()[0] < 0.6931 < ax.get_ylim()[1]:
                ax.annotate(
                    "coin flip",
                    xy=(ax.get_xlim()[1], 0.6931),
                    xytext=(-2, 3),
                    textcoords="offset points",
                    ha="right",
                    color=MUTED,
                    fontsize=8,
                )
        else:
            ax.set_ylim(0.4, max(0.75, max(values) + 0.03))
            ax.axhline(0.5, color=AXIS, linewidth=1.0)
        _title(ax, f"{title}: {'log loss' if metric == 'log_loss' else 'accuracy'}")
    _legend(axes[0, 1], loc="upper right")
    return _save(fig, path, caption)


def by_confidence(record: Mapping, path: Path, period: str) -> Path:
    """Per confidence bucket of the calibrated model: how often its pick won,
    against the probability it gave."""
    rows = [r for r in record["by_confidence"]["model"] if r["n"]]
    fig, axes = _figure(1, width=7.2)
    ax = axes[0, 0]
    positions = range(len(rows))
    ax.bar(
        positions,
        [r["favourite_win_rate"] for r in rows],
        width=0.42,
        color=COLORS["model"],
        linewidth=0,
        label="Share of bouts the model's pick won",
    )
    ax.vlines(
        positions,
        [r["win_rate_low"] for r in rows],
        [r["win_rate_high"] for r in rows],
        color=INK_SECONDARY,
        linewidth=1.0,
    )
    ax.scatter(
        positions,
        [r["mean_confidence"] for r in rows],
        marker="D",
        s=42,
        color=INK,
        edgecolor=SURFACE,
        linewidth=1.5,
        zorder=4,
        label="Mean probability the model gave its pick",
    )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(
        [f"{r['low']:.0%} to {r['high']:.0%}\nn = {r['n']}" for r in rows],
        color=INK_SECONDARY,
    )
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("Probability given to the favoured fighter")
    ax.set_ylabel("Share")
    _style(ax)
    _title(ax, "Accuracy by confidence")
    _legend(ax, loc="upper left")
    return _save(
        fig, path, f"{period}. Whiskers are 95% intervals for the share in a bucket."
    )


# --- walk-forward and ablation --------------------------------------------------


def walk_forward(record: Mapping, path: Path) -> Path:
    """Log loss and accuracy by year, each year scored by a model refitted on
    earlier years only."""
    folds = record["folds"]
    years = [f["year"] for f in folds]
    fig, axes = _figure(2, width=5.6)
    for ax, metric, label in (
        (axes[0, 0], "log_loss", "Log loss (lower is better)"),
        (axes[0, 1], "accuracy", "Accuracy"),
    ):
        inside = [f["year"] for f in folds if f["in_test_period"]]
        if inside:
            ax.axvspan(
                inside[0] - 0.5, inside[-1] + 0.5, color=GRID, alpha=0.6, linewidth=0
            )
        for name in BASELINE_ORDER:
            _points(ax, years, [f["metrics"][name][metric] for f in folds], name)
        ax.axhline(0.6931 if metric == "log_loss" else 0.5, color=AXIS, linewidth=1.0)
        ax.set_xticks(years)
        ax.set_xticklabels([f"'{str(y)[2:]}" for y in years])
        ax.set_xlabel("Year scored")
        ax.set_ylabel(label)
        _style(ax)
        _title(
            ax,
            f"Walk-forward backtest: {'log loss' if metric == 'log_loss' else 'accuracy'}",
        )
        if inside:
            top = ax.get_ylim()[1]
            ax.annotate(
                "test period",
                xy=(inside[0] - 0.4, top),
                xytext=(0, -11),
                textcoords="offset points",
                color=INK_SECONDARY,
                fontsize=8,
            )
    _legend(axes[0, 0], loc="lower left")
    return _save(
        fig,
        path,
        "Each year: fitted through Y-3, trees and calibrator from Y-2 and Y-1. The "
        "recipe was chosen on 2023 to 2024, so only the shaded years are untouched.",
    )


def ablation(record: Mapping, path: Path) -> Path:
    """Change in validation log loss when each family is removed."""
    rows = sorted(record["families"], key=lambda r: r["delta"])
    fig, axes = _figure(1, width=7.4, height=0.36 * len(rows) + 1.6)
    ax = axes[0, 0]
    positions = range(len(rows))
    ax.barh(
        positions,
        [r["delta"] for r in rows],
        height=0.5,
        color=COLORS["model"],
        linewidth=0,
    )
    ax.hlines(
        positions,
        [r["delta_low"] for r in rows],
        [r["delta_high"] for r in rows],
        color=INK_SECONDARY,
        linewidth=1.0,
    )
    ax.axvline(0.0, color=AXIS, linewidth=1.0)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(
        [f"{r['family'].replace('_', ' ')} ({r['n_features']})" for r in rows],
        color=INK_SECONDARY,
    )
    ax.set_xlabel(
        "Validation log loss without the family minus with it\n"
        "(right of zero: the family helps; left: the model was no worse without it)"
    )
    _style(ax, grid="x")
    _title(ax, "Feature-family ablation")
    return _save(
        fig,
        path,
        f"Mean over {record['n_seeds']} seeds. Lines are 95% paired bootstrap "
        "intervals over validation bouts.",
    )


def period_plots(record: Mapping, directory: Path, period: str) -> list[Path]:
    """The standard set for one evaluated period."""
    debut = {
        label: record["by_debut"][label]
        for label in ("no debutant", "any debutant")
        if label in record["by_debut"]
    }
    made = [
        reliability(record, directory / "reliability.png", period),
        by_confidence(record, directory / "by_confidence.png", period),
        by_group(record["by_year"], directory / "by_year.png", "By year", period),
    ]
    if debut:
        made.append(
            by_group(
                debut,
                directory / "debut.png",
                "Debut bouts",
                f"{period}. A debutant has no earlier UFC bout, so his stats are missing.",
            )
        )
    return made
