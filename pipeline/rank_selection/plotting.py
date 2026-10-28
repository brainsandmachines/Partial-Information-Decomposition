"""Plots of rank curves: one result, overlays per dataset, and toy rank recovery.

Colors follow the method family in a fixed order (validated categorical palette);
variants of one family differ by line style. Reference lines (selected rank,
1-SE rank, true rank, permutation threshold) use neutral ink.
"""

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.rank_selection.result import RankResult

# Fixed method-family colors (validated: light surface, 6 slots).
METHOD_COLORS = {
    "variance_threshold": "#2a78d6",
    "eigenvector": "#eb6834",
    "repeat_reconstruction": "#1baf7a",
    "mle": "#eda100",
    "cross_decomposition": "#e87ba4",
    "pls": "#008300",
}
LINE_STYLES = ("-", "--", ":", "-.")
INK = "#3d3d3a"
MUTED_INK = "#8a8985"
# Extra curves drawn on the score axis (same units); boolean curves are not drawn.
SAME_AXIS_CURVES = ("null_threshold",)


def method_family(method: str) -> str:
    """Return the method name without its options.

    Input: method, str label such as ``"eigenvector[basis=pca,folds=5]"`` or
        the file label ``"eigenvector-basis=pca-folds=5"``.
    Output: str, e.g. ``"eigenvector"``.
    """
    return method.split("[", 1)[0].split("-", 1)[0]


def _style_axis(axis: plt.Axes) -> None:
    """Make grid and spines recessive.

    Input: axis, matplotlib Axes.
    Output: None.
    """
    axis.grid(True, color="#e6e5e0", linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED_INK)
    axis.tick_params(colors=INK)


def plot_result(result: RankResult, path: str | Path) -> Path:
    """Plot one method's curve with its selected, 1-SE, and true ranks.

    Inputs: result, RankResult; path, str or Path of the PDF to write.
    Output: Path of the written PDF. Float extra curves in other units (e.g. the
        naive held-out error) get their own panel rather than a second y-axis.
    """
    other_curves = {
        name: np.asarray(curve, dtype=float)
        for name, curve in result.extra_curves.items()
        if name not in SAME_AXIS_CURVES and np.asarray(curve).dtype != bool
        and name not in ("fold_sd", "empirical_p")
    }
    n_panels = 1 + len(other_curves)
    figure, axes = plt.subplots(n_panels, 1, figsize=(7, 3.6 * n_panels), squeeze=False)
    axes = axes[:, 0]
    color = METHOD_COLORS.get(method_family(result.method), INK)

    axis = axes[0]
    axis.plot(result.rank, result.score, color=color, linewidth=2, label=result.method)
    if result.standard_error is not None:
        axis.fill_between(
            result.rank, result.score - result.standard_error, result.score + result.standard_error,
            color=color, alpha=0.2, linewidth=0, label="± 1 SE across folds",
        )
    for name in SAME_AXIS_CURVES:
        if name in result.extra_curves:
            axis.plot(result.rank, result.extra_curves[name], color=MUTED_INK,
                      linewidth=1.5, linestyle="--", label=name.replace("_", " "))
    axis.set(xlabel="rank", ylabel=result.score_name)
    axis.set_title(f"{result.method} — {result.dataset}", color=INK, fontsize=10)

    for panel, (name, curve) in zip(axes[1:], other_curves.items()):
        panel.plot(result.rank, curve, color=color, linewidth=2, label=name.replace("_", " "))
        panel.set(xlabel="rank", ylabel=name.replace("_", " "))

    for panel in axes:
        panel.axvline(result.selected_rank, color=INK, linestyle="--", linewidth=1.2,
                      label=f"selected = {result.selected_rank}")
        if result.one_se_rank is not None and result.one_se_rank != result.selected_rank:
            panel.axvline(result.one_se_rank, color=MUTED_INK, linestyle=":", linewidth=1.2,
                          label=f"1-SE rank = {result.one_se_rank}")
        if result.true_rank is not None:
            panel.axvline(result.true_rank, color=INK, linestyle="-", linewidth=0.8, alpha=0.6,
                          label=f"true rank = {result.true_rank}")
        _style_axis(panel)
        panel.legend(frameon=False, fontsize=8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_dataset_overlay(results: list[RankResult], path: str | Path) -> Path:
    """Overlay every method's curve for one dataset, one panel per score type.

    Inputs: results, list[RankResult] for the same dataset; path, PDF path.
    Output: Path of the written PDF. Each curve is labeled directly at its
        selected rank, so identity never relies on color alone.
    """
    score_names = list(dict.fromkeys(result.score_name for result in results))
    figure, axes = plt.subplots(
        len(score_names), 1, figsize=(8, 3.8 * len(score_names)), squeeze=False,
    )
    variant_index: dict[str, list[str]] = {}
    for axis, score_name in zip(axes[:, 0], score_names):
        for result in (r for r in results if r.score_name == score_name):
            family = method_family(result.method)
            variants = variant_index.setdefault(family, [])
            if result.method not in variants:
                variants.append(result.method)
            style = LINE_STYLES[variants.index(result.method) % len(LINE_STYLES)]
            color = METHOD_COLORS.get(family, INK)
            label = result.method if result.seed is None else f"{result.method} seed {result.seed}"
            axis.plot(result.rank, result.score, color=color, linestyle=style, linewidth=2, label=label)
            at_selected = np.flatnonzero(result.rank == result.selected_rank)
            if at_selected.size:
                y = result.score[at_selected[0]]
                axis.plot(result.selected_rank, y, "o", color=color, markersize=8,
                          markeredgecolor="white", markeredgewidth=2)
                # Variants of one family often select the same rank: stack their labels.
                variant = result.method[len(family):].strip("[]") or family
                axis.annotate(f"{variant}: {result.selected_rank}", (result.selected_rank, y),
                              textcoords="offset points",
                              xytext=(6, 6 + 11 * variants.index(result.method)),
                              fontsize=7, color=INK)
        true_ranks = {r.true_rank for r in results if r.true_rank is not None}
        for true_rank in true_ranks:
            axis.axvline(true_rank, color=INK, linewidth=0.8, alpha=0.6, label=f"true rank = {true_rank}")
        axis.set(xlabel="rank", ylabel=score_name)
        _style_axis(axis)
        axis.legend(frameon=False, fontsize=8)
    axes[0, 0].set_title(results[0].dataset, color=INK, fontsize=10)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_rank_recovery(summary: pd.DataFrame, path: str | Path) -> Path | None:
    """Plot selected minus true rank for every method across toy datasets.

    Inputs: summary, pd.DataFrame with ``method``, ``selected_rank``,
        ``true_rank`` (and ``status``) columns; path, PDF path.
    Output: Path of the written PDF, or None when no result has a true rank.
    """
    done = summary[(summary.get("status", "done") == "done") & summary["true_rank"].notna()]
    if done.empty:
        return None
    methods = sorted(done["method"].unique())
    figure, axis = plt.subplots(figsize=(max(6, 1.2 * len(methods)), 4))
    rng = np.random.default_rng(0)
    for position, method in enumerate(methods):
        rows = done[done["method"] == method]
        error = rows["selected_rank"].to_numpy(float) - rows["true_rank"].to_numpy(float)
        jitter = rng.uniform(-0.15, 0.15, size=error.shape[0])
        axis.plot(position + jitter, error, "o", markersize=8, alpha=0.8,
                  color=METHOD_COLORS.get(method_family(method), INK),
                  markeredgecolor="white", markeredgewidth=1)
    axis.axhline(0, color=INK, linewidth=1)
    axis.set_xticks(range(len(methods)))
    axis.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    axis.set(ylabel="selected rank − true rank")
    axis.set_title("Rank recovery on toy data (0 = correct)", color=INK, fontsize=10)
    _style_axis(axis)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def _mean_curve(curves: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average curves that may cover different ranks.

    Input: curves, list of dicts with ``rank`` and ``score`` arrays.
    Output: ranks (n_ranks,), mean score (n_ranks,), and SD across curves
        (n_ranks,); ranks a curve lacks are ignored for that curve.
    """
    # n_curves Series indexed by rank -> (n_ranks, n_curves)
    table = pd.concat(
        [pd.Series(curve["score"], index=curve["rank"]) for curve in curves], axis=1
    ).sort_index()
    spread = table.std(axis=1, ddof=1) if table.shape[1] > 1 else table.iloc[:, 0] * 0.0
    return table.index.to_numpy(), table.mean(axis=1).to_numpy(), spread.to_numpy()


def _condition_title(condition: str) -> str:
    """Format a dataset name such as ``toy_noise_type=isotropic_true_rank=5`` as lines.

    Input: condition, str dataset name without its seed.
    Output: str with one ``key=value`` per line (the name itself if it has none).
    """
    pairs = re.findall(r"([a-z][a-z_]*?)=([^_]+)", condition.removeprefix("toy_"))
    return "\n".join(f"{key}={value}" for key, value in pairs) or condition


def _ordered_methods(curves: list[dict]) -> list[str]:
    """Order method labels by family (fixed color order), then by label.

    Input: curves, list of dicts with a ``method`` label.
    Output: list[str] of unique method labels.
    """
    family_order = list(METHOD_COLORS)
    return sorted(
        {curve["method"] for curve in curves},
        key=lambda label: (family_order.index(method_family(label))
                           if method_family(label) in family_order else len(family_order), label),
    )


def _draw_mean_cell(axis: plt.Axes, cell: list[dict]) -> None:
    """Draw one method's mean curve over seeds for one condition.

    Inputs: axis, matplotlib Axes; cell, list of curve dicts (one per seed) of
        the same method and condition.
    Output: None. Draws the mean with a ±1 SD band, the mean selected rank
        (dashed), and the true rank (solid), zoomed to ranks up to 1.5x the
        largest selected or true rank (at least 20).
    """
    color = METHOD_COLORS.get(method_family(cell[0]["method"]), INK)
    ranks, mean, spread = _mean_curve(cell)
    selected = np.array([c["selected_rank"] for c in cell], dtype=float)
    true_ranks = {c["true_rank"] for c in cell if c["true_rank"] is not None}
    # Zoom to the ranks that matter, so blow-ups at very high ranks do not flatten the curve.
    x_max = max(20.0, 1.5 * max([selected.max(), *true_ranks]))
    shown = ranks <= x_max  # (n_ranks,)
    ranks, mean, spread = ranks[shown], mean[shown], spread[shown]
    axis.plot(ranks, mean, color=color, linewidth=2)
    axis.fill_between(ranks, mean - spread, mean + spread, color=color, alpha=0.2, linewidth=0)
    finite = np.isfinite(mean)
    if finite.any():
        low = np.nanmin((mean - spread)[finite])
        high = np.nanmax((mean + spread)[finite])
        margin = 0.05 * (high - low) if high > low else 1.0
        axis.set_ylim(low - margin, high + margin)
    axis.set_xlim(left=min(0, ranks.min()) if ranks.size else 0, right=x_max)
    axis.axvline(selected.mean(), color=INK, linestyle="--", linewidth=1.2)
    text = f"selected {selected.mean():.1f}" + (f" ± {selected.std(ddof=1):.1f}" if selected.size > 1 else "")
    for true_rank in true_ranks:
        axis.axvline(true_rank, color=INK, linewidth=0.8, alpha=0.6)
        text += f"\ntrue {true_rank}"
    text += f"\nn = {len(cell)} seeds"
    axis.text(0.97, 0.95, text, transform=axis.transAxes, ha="right", va="top", fontsize=8, color=INK)
    axis.set_title(cell[0]["method_detail"], fontsize=9, color=INK)
    axis.set_xlabel("rank", fontsize=8)
    axis.set_ylabel(cell[0]["score_name"], fontsize=7, color=INK)
    _style_axis(axis)
    axis.tick_params(labelsize=7)


def plot_condition_grids(curves: list[dict], folder: str | Path, n_columns: int = 4) -> list[Path]:
    """Write one PDF per condition with every method's mean curve over seeds.

    Inputs:
        curves: list of dicts, one per finished result, with ``method`` (label),
            ``method_detail``, ``condition`` (dataset name without its seed),
            ``score_name``, ``rank`` and ``score`` arrays, ``selected_rank``, and
            ``true_rank`` (int or None).
        folder: str or Path; files are named ``<condition>.pdf``.
        n_columns: int, panels per row.
    Output: list[Path] of the written PDFs. Each panel is one method, averaged
        over all seeds of the condition; methods keep their own y-axis because
        their scores have different units.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for condition in sorted({curve["condition"] for curve in curves}):
        in_condition = [c for c in curves if c["condition"] == condition]
        methods = _ordered_methods(in_condition)
        n_rows = int(np.ceil(len(methods) / n_columns))
        figure, axes = plt.subplots(n_rows, n_columns, figsize=(4.2 * n_columns, 3.3 * n_rows), squeeze=False)
        for axis in axes.ravel()[len(methods):]:
            axis.set_axis_off()
        for axis, method in zip(axes.ravel(), methods):
            _draw_mean_cell(axis, [c for c in in_condition if c["method"] == method])
        figure.suptitle(
            f"{_condition_title(condition).replace(chr(10), ', ')} — mean over seeds (band ±1 SD); "
            "dashed = mean selected rank, solid = true rank",
            fontsize=11, color=INK,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        path = folder / f"{condition}.pdf"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths
