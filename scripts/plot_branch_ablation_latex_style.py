#!/usr/bin/env python3
"""Generate publication-style branch ablation plots from summary_ablation.csv."""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


GROUP_ORDER = [1, 2, 4, 8, 16]
COLORS = {
    1: "#3B6EA8",
    2: "#D9822B",
    4: "#4E9A51",
    8: "#8E6CA8",
    16: "#C44E52",
}
MARKERS = {1: "o", 2: "s", 4: "^", 8: "D", 16: "P"}
LINESTYLES = {1: "-", 2: "--", 4: "-.", 8: ":", 16: (0, (5, 1.5))}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9.5,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#3A3A3A",
            "grid.color": "#D6D6D6",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.75,
        }
    )


def token_formatter(value: float, _pos: int) -> str:
    if abs(value) >= 1000:
        return rf"{value / 1000:.0f}k"
    return f"{value:g}"


def load_rows(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            rows.append(
                {
                    "method": row["method"],
                    "condition": row["condition"],
                    "group": int(row["branch_group_size"]),
                    "k": int(row["k"]),
                    "pass_at_k": float(row["pass_at_k"]),
                    "avg_cost_tokens": float(row["avg_cost_tokens"]),
                    "num_prompts": int(row["num_prompts"]),
                    "adaptive_shared_count": int(row["adaptive_shared_count"])
                    if row.get("adaptive_shared_count")
                    else None,
                    "adaptive_fixed_count": int(row["adaptive_fixed_count"])
                    if row.get("adaptive_fixed_count")
                    else None,
                    "confidence_threshold": float(row["confidence_threshold"])
                    if row.get("confidence_threshold")
                    else None,
                }
            )
    return rows


def group_label(group: int, adaptive: bool = False, row: Dict[str, object] = None) -> str:
    if adaptive and row is not None:
        shared = int(row.get("adaptive_shared_count") or group)
        fixed = int(row.get("adaptive_fixed_count") or 0)
        threshold = row.get("confidence_threshold")
        if threshold is not None:
            return rf"Shared every {shared} + fixed {fixed} if conf$<{float(threshold):g}$"
        return f"Shared every {shared} + fixed {fixed}"
    if group == 1:
        return "Every 1 / fixed trace"
    return f"Shared every {group}"


def short_group_label(group: int, adaptive: bool = False, row: Dict[str, object] = None) -> str:
    if adaptive and row is not None:
        shared = int(row.get("adaptive_shared_count") or group)
        fixed = int(row.get("adaptive_fixed_count") or 0)
        return f"Shared {shared} + fixed {fixed}"
    if group == 1:
        return "Every 1"
    return f"Every {group}"


def style_for(group: int) -> Dict[str, object]:
    return {
        "color": COLORS[group],
        "marker": MARKERS[group],
        "linestyle": LINESTYLES[group],
        "linewidth": 2.0,
        "markersize": 5.5,
        "markeredgewidth": 0.8,
        "markeredgecolor": "white",
    }


def rows_for(rows: Iterable[Dict[str, object]], group: int) -> List[Dict[str, object]]:
    return sorted((row for row in rows if row["group"] == group), key=lambda row: int(row["k"]))


def tight_ylim(values: Iterable[float], pad: float = 0.035) -> Tuple[float, float]:
    vals = list(values)
    lo = max(0.0, min(vals) - pad)
    hi = min(1.0, max(vals) + pad)
    if hi - lo < 0.08:
        mid = (hi + lo) / 2
        lo = max(0.0, mid - 0.04)
        hi = min(1.0, mid + 0.04)
    return lo, hi


def finish_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y")
    ax.grid(True, axis="x", alpha=0.32)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def ordered_groups(rows: List[Dict[str, object]]) -> List[int]:
    present = sorted({int(row["group"]) for row in rows})
    ordered = [group for group in GROUP_ORDER if group in present]
    return ordered + [group for group in present if group not in GROUP_ORDER]


def plot_passk_superimposed(rows: List[Dict[str, object]], out_dir: Path, adaptive: bool = False) -> None:
    y_values = [float(row["pass_at_k"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for group in ordered_groups(rows):
        series = rows_for(rows, group)
        ax.plot(
            [int(row["k"]) for row in series],
            [float(row["pass_at_k"]) for row in series],
            label=group_label(group, adaptive, series[0]),
            **style_for(group),
        )

    ax.set_xscale("log", basex=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"Pass@$k$")
    ax.set_ylim(*tight_ylim(y_values, pad=0.025))
    ax.set_title(r"Branch Ablation: Pass@$k$ Scaling")
    finish_axes(ax)
    ax.legend(title="Technique", frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_passk_scaling_curves_latex.png")
    fig.savefig(out_dir / "ablation_passk_scaling_curves_zoomed.png")
    plt.close(fig)


def plot_passk_subplots(rows: List[Dict[str, object]], out_dir: Path, adaptive: bool = False) -> None:
    y_values = [float(row["pass_at_k"]) for row in rows]
    groups = ordered_groups(rows)
    cols = min(3, max(1, len(groups)))
    subplot_count = len(groups)
    rows_count = int(math.ceil(subplot_count / float(cols)))
    fig, axes = plt.subplots(rows_count, cols, figsize=(3.45 * cols, 3.05 * rows_count), sharex=True, sharey=True)
    if not isinstance(axes, (list, tuple)):
        try:
            flat_axes = axes.ravel()
        except AttributeError:
            flat_axes = [axes]
    else:
        flat_axes = axes
    ylim = tight_ylim(y_values, pad=0.025)
    for ax, group in zip(flat_axes, groups):
        series = rows_for(rows, group)
        ax.plot(
            [int(row["k"]) for row in series],
            [float(row["pass_at_k"]) for row in series],
            **style_for(group),
        )
        ax.set_title(short_group_label(group, adaptive, series[0]), color=COLORS[group], pad=6)
        ax.set_xscale("log", basex=2)
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_xticklabels(["1", "2", "4", "8", "16"])
        ax.set_ylim(*ylim)
        finish_axes(ax)
    for ax in flat_axes[len(groups) :]:
        ax.axis("off")
    fig.text(0.52, 0.035, r"$k$", ha="center", va="center", fontsize=11)
    fig.text(0.025, 0.52, r"Pass@$k$", ha="center", va="center", rotation="vertical", fontsize=11)
    fig.suptitle(r"Branch Ablation: Pass@$k$ by Sharing Frequency", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0.04, 0.04, 1.0, 0.96))
    fig.savefig(out_dir / "ablation_passk_scaling_subplots_latex.png")
    plt.close(fig)


def plot_memory_superimposed(rows: List[Dict[str, object]], out_dir: Path, adaptive: bool = False) -> None:
    y_values = [float(row["pass_at_k"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for group in ordered_groups(rows):
        series = rows_for(rows, group)
        ax.plot(
            [float(row["avg_cost_tokens"]) for row in series],
            [float(row["pass_at_k"]) for row in series],
            label=group_label(group, adaptive, series[0]),
            **style_for(group),
        )

    ax.set_xlabel("Memory Usage (tokens)")
    ax.set_ylabel(r"Pass@$k$")
    ax.set_ylim(*tight_ylim(y_values, pad=0.025))
    ax.xaxis.set_major_formatter(FuncFormatter(token_formatter))
    ax.set_title(r"Branch Ablation: Memory Usage vs. Pass@$k$")
    finish_axes(ax)
    ax.legend(title="Technique", frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_memory_vs_passk_latex.png")
    fig.savefig(out_dir / "ablation_memory_vs_passk.png")
    plt.close(fig)


def plot_memory_at_k16(rows: List[Dict[str, object]], out_dir: Path, adaptive: bool = False) -> None:
    max_k = max(int(row["k"]) for row in rows)
    max_rows = [row for row in rows if int(row["k"]) == max_k]
    y_values = [float(row["pass_at_k"]) for row in max_rows]
    x_values = [float(row["avg_cost_tokens"]) for row in max_rows]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for row in sorted(max_rows, key=lambda item: int(item["group"])):
        group = int(row["group"])
        ax.scatter(
            [float(row["avg_cost_tokens"])],
            [float(row["pass_at_k"])],
            s=72,
            color=COLORS[group],
            marker=MARKERS[group],
            edgecolor="white",
            linewidth=0.8,
            label=group_label(group, adaptive, row),
            zorder=3,
        )
        ax.annotate(
            short_group_label(group, adaptive, row),
            (float(row["avg_cost_tokens"]), float(row["pass_at_k"])),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
            color="#222222",
        )
    ax.set_xlabel("Memory Usage (tokens)")
    ax.set_ylabel(rf"Pass@{max_k}")
    ax.set_ylim(*tight_ylim(y_values, pad=0.035))
    x_pad = (max(x_values) - min(x_values)) * 0.08
    ax.set_xlim(min(x_values) - x_pad, max(x_values) + x_pad)
    ax.xaxis.set_major_formatter(FuncFormatter(token_formatter))
    ax.set_title(rf"Branch Ablation: Memory Usage vs. Pass@{max_k}")
    finish_axes(ax)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0)
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    fig.savefig(out_dir / f"ablation_memory_vs_pass{max_k}_latex.png")
    fig.savefig(out_dir / f"ablation_memory_vs_pass{max_k}.png")
    plt.close(fig)


def plot_memory_subplots(rows: List[Dict[str, object]], out_dir: Path, adaptive: bool = False) -> None:
    y_values = [float(row["pass_at_k"]) for row in rows]
    groups = ordered_groups(rows)
    cols = min(3, max(1, len(groups)))
    rows_count = int(math.ceil(len(groups) / float(cols)))
    fig, axes = plt.subplots(rows_count, cols, figsize=(3.55 * cols, 3.1 * rows_count), sharey=True)
    try:
        flat_axes = axes.ravel()
    except AttributeError:
        flat_axes = [axes]
    ylim = tight_ylim(y_values, pad=0.025)
    for ax, group in zip(flat_axes, groups):
        series = rows_for(rows, group)
        ax.plot(
            [float(row["avg_cost_tokens"]) for row in series],
            [float(row["pass_at_k"]) for row in series],
            **style_for(group),
        )
        ax.set_title(short_group_label(group, adaptive, series[0]), color=COLORS[group], pad=6)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_formatter(FuncFormatter(token_formatter))
        finish_axes(ax)
    for ax in flat_axes[len(groups) :]:
        ax.axis("off")
    fig.text(0.52, 0.035, "Memory Usage (tokens)", ha="center", va="center", fontsize=11)
    fig.text(0.025, 0.52, r"Pass@$k$", ha="center", va="center", rotation="vertical", fontsize=11)
    fig.suptitle(r"Branch Ablation: Memory/Performance Tradeoff by Sharing Frequency", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0.04, 0.04, 1.0, 0.96))
    fig.savefig(out_dir / "ablation_memory_vs_passk_subplots_latex.png")
    plt.close(fig)


def plot_tokens_superimposed(rows: List[Dict[str, object]], out_dir: Path, adaptive: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for group in ordered_groups(rows):
        series = rows_for(rows, group)
        ax.plot(
            [int(row["k"]) for row in series],
            [float(row["avg_cost_tokens"]) for row in series],
            label=group_label(group, adaptive, series[0]),
            **style_for(group),
        )
    ax.set_xscale("log", basex=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.yaxis.set_major_formatter(FuncFormatter(token_formatter))
    ax.set_xlabel(r"$k$")
    ax.set_ylabel("Memory Usage (tokens)")
    ax.set_title(r"Branch Ablation: Memory Usage vs. $k$")
    finish_axes(ax)
    ax.legend(title="Technique", frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_k_vs_generated_tokens_latex.png")
    fig.savefig(out_dir / "ablation_k_vs_generated_tokens.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=Path("final_eval_outputs/passk-ablation-raivn-20260424-101750"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--allow-missing-groups", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir
    out_dir = args.out_dir or run_dir
    summary_path = run_dir / "summary_ablation.csv"
    if not summary_path.exists():
        raise SystemExit(f"missing summary CSV: {summary_path}")

    configure_style()
    rows = load_rows(summary_path)
    present_groups = sorted({int(row["group"]) for row in rows})
    missing_groups = sorted(set(GROUP_ORDER) - set(present_groups))
    if missing_groups and not args.allow_missing_groups:
        raise SystemExit(f"missing branch groups in summary: {missing_groups}")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_passk_superimposed(rows, out_dir, adaptive=args.adaptive)
    plot_passk_subplots(rows, out_dir, adaptive=args.adaptive)
    plot_memory_superimposed(rows, out_dir, adaptive=args.adaptive)
    plot_memory_at_k16(rows, out_dir, adaptive=args.adaptive)
    plot_memory_subplots(rows, out_dir, adaptive=args.adaptive)
    plot_tokens_superimposed(rows, out_dir, adaptive=args.adaptive)
    print(f"wrote LaTeX-style branch ablation plots to {out_dir}")


if __name__ == "__main__":
    main()
