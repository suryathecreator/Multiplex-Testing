#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


FIXED_METHOD = "baseline_independent"
SHARED_METHOD = "shared_trace_branch_after_prefix"
ANNOTATE_K = {1, 8, 64}
BUDGET_MARKERS = {256: "o", 512: "s", 1024: "^", 2048: "D", 4096: "P", 6144: "X"}
BUDGET_COLORS = {
    256: "#4C78A8",
    512: "#72B7B2",
    1024: "#54A24B",
    2048: "#F58518",
    4096: "#B279A2",
    6144: "#E45756",
}
BUDGET_LINESTYLES = {
    256: "-",
    512: "--",
    1024: "-.",
    2048: ":",
    4096: (0, (4, 1.4)),
    6144: (0, (2.2, 1.2)),
}
METHOD_STYLES = {
    FIXED_METHOD: {
        "color": "#2B5C8A",
        "marker": "o",
        "linestyle": "-",
        "label": "Fixed trace",
    },
    SHARED_METHOD: {
        "color": "#B86E2B",
        "marker": "s",
        "linestyle": "--",
        "label": "Shared trace",
    },
}
EQUATION_TEXT = (
    r"$\mathrm{Token\ Savings\ (\%)} = 100 \times "
    r"\left(1 - \frac{\mathrm{shared\_tokens}}{\mathrm{fixed\_tokens}}\right)$"
    "\n"
    r"$\mathrm{Accuracy\ Loss} = \mathrm{fixed\_acc} - \mathrm{shared\_acc}$"
)
LABEL_BBOX = {
    "boxstyle": "round,pad=0.12",
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.82,
}


def method_label(method: str) -> str:
    if method == FIXED_METHOD:
        return "Fixed trace"
    if method == SHARED_METHOD:
        return "Shared trace"
    return method


def load_rows(path: Path) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    with path.open(newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            rows.append(
                {
                    "method": row["method"],
                    "k": int(row["k"]),
                    "reasoning_prefix_tokens": int(row["reasoning_prefix_tokens"]),
                    "avg_cost_tokens": float(row["avg_cost_tokens"]),
                    "pass_at_k": float(row["pass_at_k"]),
                    "num_prompts": int(float(row["num_prompts"])),
                }
            )
    return rows


def average_by_method_and_k(rows: Iterable[Dict[str, float | int | str]]) -> List[Dict[str, float | int | str]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["k"]))].append(row)
    averaged: List[Dict[str, float | int | str]] = []
    for (method, k), bucket in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        averaged.append(
            {
                "method": method,
                "k": k,
                "avg_cost_tokens": mean(float(row["avg_cost_tokens"]) for row in bucket),
                "pass_at_k": mean(float(row["pass_at_k"]) for row in bucket),
            }
        )
    return averaged


def paired_by_budget_and_k(rows: Iterable[Dict[str, float | int | str]]) -> List[Dict[str, float | int]]:
    lookup = {
        (str(row["method"]), int(row["reasoning_prefix_tokens"]), int(row["k"])): row
        for row in rows
    }
    pairs: List[Dict[str, float | int]] = []
    keys = sorted({(int(row["reasoning_prefix_tokens"]), int(row["k"])) for row in rows})
    for budget, k in keys:
        fixed = lookup.get((FIXED_METHOD, budget, k))
        shared = lookup.get((SHARED_METHOD, budget, k))
        if fixed is None or shared is None:
            continue
        fixed_tokens = float(fixed["avg_cost_tokens"])
        shared_tokens = float(shared["avg_cost_tokens"])
        fixed_acc = float(fixed["pass_at_k"])
        shared_acc = float(shared["pass_at_k"])
        if fixed_tokens <= 0:
            continue
        pairs.append(
            {
                "reasoning_prefix_tokens": budget,
                "k": k,
                "token_savings": 1.0 - (shared_tokens / fixed_tokens),
                "accuracy_loss": fixed_acc - shared_acc,
                "tokens_fixed": fixed_tokens,
                "tokens_shared": shared_tokens,
                "acc_fixed": fixed_acc,
                "acc_shared": shared_acc,
            }
        )
    return pairs


def average_tradeoff_by_k(pairs: Iterable[Dict[str, float | int]]) -> List[Dict[str, float | int]]:
    grouped: Dict[int, List[Dict[str, float | int]]] = defaultdict(list)
    for pair in pairs:
        grouped[int(pair["k"])].append(pair)
    return [
        {
            "k": k,
            "token_savings": mean(float(row["token_savings"]) for row in bucket),
            "accuracy_loss": mean(float(row["accuracy_loss"]) for row in bucket),
        }
        for k, bucket in sorted(grouped.items())
    ]


def pareto_frontier(points: List[Dict[str, float | int]]) -> List[Dict[str, float | int]]:
    frontier: List[Dict[str, float | int]] = []
    for point in points:
        savings = float(point["token_savings"])
        loss = float(point["accuracy_loss"])
        dominated = False
        for other in points:
            if other is point:
                continue
            other_savings = float(other["token_savings"])
            other_loss = float(other["accuracy_loss"])
            if (
                other_savings >= savings
                and other_loss <= loss
                and (other_savings > savings or other_loss < loss)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda row: float(row["token_savings"]))


def pct_formatter(value: float, _pos: int) -> str:
    return f"{value * 100:.0f}%"


def token_formatter(value: float, _pos: int) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def add_equation_note(ax: plt.Axes, loc: str = "upper left") -> None:
    x = 0.025 if "left" in loc else 0.975
    y = 0.965 if "upper" in loc else 0.035
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    ax.text(
        x,
        y,
        EQUATION_TEXT,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=7.5,
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#D0D0D0",
            "linewidth": 0.4,
            "alpha": 0.78,
        },
        zorder=20,
    )


def set_k_axis(ax: plt.Axes, k_values: List[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])


def grouped_tradeoff_by_budget(
    pairs: Iterable[Dict[str, float | int]],
) -> Dict[int, List[Dict[str, float | int]]]:
    grouped: Dict[int, List[Dict[str, float | int]]] = defaultdict(list)
    for pair in pairs:
        row = dict(pair)
        row["token_savings_pct"] = 100.0 * float(row["token_savings"])
        grouped[int(row["reasoning_prefix_tokens"])].append(row)
    return {
        budget: sorted(bucket, key=lambda row: int(row["k"]))
        for budget, bucket in sorted(grouped.items())
    }


def average_tradeoff_percent_by_k(
    pairs: Iterable[Dict[str, float | int]],
) -> List[Dict[str, float | int]]:
    grouped: Dict[int, List[Dict[str, float | int]]] = defaultdict(list)
    for pair in pairs:
        grouped[int(pair["k"])].append(pair)
    return [
        {
            "k": k,
            "token_savings_pct": 100.0
            * mean(float(row["token_savings"]) for row in bucket),
            "accuracy_loss": mean(float(row["accuracy_loss"]) for row in bucket),
        }
        for k, bucket in sorted(grouped.items())
    ]


def load_discrete_token_summary(
    root: Path, pass_at_ks: List[int]
) -> List[Dict[str, float | int | str]]:
    """Average post-prefix continuation tokens for each budget/method/k."""
    rows: List[Dict[str, float | int | str]] = []
    method_labels = {
        FIXED_METHOD: "Fixed trace",
        SHARED_METHOD: "Shared trace",
    }
    max_k = max(pass_at_ks)
    prefix_dirs = sorted(root.glob("prefix_*"))
    for prefix_dir in prefix_dirs:
        try:
            budget = int(prefix_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        samples_path = prefix_dir / "samples.jsonl"
        if not samples_path.exists():
            continue
        grouped: Dict[Tuple[str, int], List[Dict[str, float | int | str]]] = defaultdict(list)
        with samples_path.open(encoding="utf-8") as fp:
            for line in fp:
                sample = json.loads(line)
                method = str(sample.get("method", ""))
                if method not in method_labels:
                    continue
                if not bool(sample.get("usable_for_eval", True)):
                    continue
                grouped[(method, int(sample["prompt_index"]))].append(sample)

        for method, label in method_labels.items():
            prompt_groups: List[List[Dict[str, float | int | str]]] = []
            for (group_method, _prompt_index), samples in grouped.items():
                if group_method != method:
                    continue
                ordered = sorted(samples, key=lambda row: int(row["sample_index"]))
                if len(ordered) >= max_k:
                    prompt_groups.append(ordered)
            if not prompt_groups:
                continue
            for k in pass_at_ks:
                per_prompt_totals = [
                    sum(int(sample["completion_tokens"]) for sample in samples[:k])
                    for samples in prompt_groups
                ]
                rows.append(
                    {
                        "method": method,
                        "method_label": label,
                        "reasoning_prefix_tokens": budget,
                        "k": k,
                        "avg_discrete_tokens": mean(per_prompt_totals),
                        "num_prompts": len(prompt_groups),
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            str(row["method"]),
            int(row["k"]),
            int(row["reasoning_prefix_tokens"]),
        ),
    )


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman"],
            "mathtext.fontset": "stix",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.9,
            "axes.labelsize": 11,
            "axes.titlesize": 12.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def tradeoff_axis_limits(pairs: List[Dict[str, float | int]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xs = [float(row["token_savings"]) for row in pairs]
    ys = [float(row["accuracy_loss"]) for row in pairs]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    x_pad = max(0.015, x_span * 0.08)
    y_pad = max(0.015, y_span * 0.10)
    return (min(xs) - x_pad, max(xs) + x_pad), (min(ys) - y_pad, max(ys) + y_pad)


def group_memory_rows_by_budget(
    rows: Iterable[Dict[str, float | int | str]]
) -> Dict[int, Dict[str, List[Dict[str, float | int | str]]]]:
    grouped: Dict[int, Dict[str, List[Dict[str, float | int | str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[int(row["reasoning_prefix_tokens"])][str(row["method"])].append(row)
    return {
        budget: {
            method: sorted(bucket, key=lambda item: int(item["k"]))
            for method, bucket in methods.items()
        }
        for budget, methods in sorted(grouped.items())
    }


def annotate_memory_curve(
    ax: plt.Axes,
    fixed_series: List[Dict[str, float | int | str]],
    shared_series: List[Dict[str, float | int | str]],
    fontsize: float = 7.8,
) -> None:
    fixed_by_k = {int(row["k"]): row for row in fixed_series}
    shared_by_k = {int(row["k"]): row for row in shared_series}

    if 1 in fixed_by_k and 1 in shared_by_k:
        fixed = fixed_by_k[1]
        shared = shared_by_k[1]
        x = (float(fixed["avg_cost_tokens"]) + float(shared["avg_cost_tokens"])) / 2
        y = (float(fixed["pass_at_k"]) + float(shared["pass_at_k"])) / 2
        ax.annotate(
            r"$k=1$",
            (x, y),
            textcoords="offset points",
            xytext=(8, -14),
            fontsize=fontsize,
            color="#222222",
            bbox=LABEL_BBOX,
            zorder=8,
        )

    label_offsets = {
        (FIXED_METHOD, 64): (8, 12),
        (SHARED_METHOD, 64): (8, -20),
    }
    for method, series in ((FIXED_METHOD, fixed_series), (SHARED_METHOD, shared_series)):
        color = METHOD_STYLES[method]["color"]
        for row in series:
            k = int(row["k"])
            if k != 64:
                continue
            ax.annotate(
                rf"$k={k}$",
                (float(row["avg_cost_tokens"]), float(row["pass_at_k"])),
                textcoords="offset points",
                xytext=label_offsets[(method, k)],
                fontsize=fontsize,
                color=color,
                bbox=LABEL_BBOX,
                zorder=8,
            )


def plot_average_clean_linear(out_dir: Path, averaged_rows: List[Dict[str, float | int | str]]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    styles = {
        FIXED_METHOD: {
            "color": "#2B5C8A",
            "marker": "o",
            "linestyle": "-",
            "label": "Fixed trace",
        },
        SHARED_METHOD: {
            "color": "#B86E2B",
            "marker": "s",
            "linestyle": "--",
            "label": "Shared trace",
        },
    }
    plotted: Dict[str, List[Dict[str, float | int | str]]] = {}
    for method in (FIXED_METHOD, SHARED_METHOD):
        series = sorted(
            [row for row in averaged_rows if str(row["method"]) == method],
            key=lambda row: int(row["k"]),
        )
        plotted[method] = series
        style = styles[method]
        ax.plot(
            [float(row["avg_cost_tokens"]) for row in series],
            [float(row["pass_at_k"]) for row in series],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            markersize=5.4,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=style["label"],
        )

    # Label k=1 once between the two nearly overlapping starting points. Label
    # k=8 and k=64 on both curves; this keeps orientation without crowding.
    fixed_k1 = next((row for row in plotted[FIXED_METHOD] if int(row["k"]) == 1), None)
    shared_k1 = next((row for row in plotted[SHARED_METHOD] if int(row["k"]) == 1), None)
    if fixed_k1 and shared_k1:
        x = (float(fixed_k1["avg_cost_tokens"]) + float(shared_k1["avg_cost_tokens"])) / 2
        y = (float(fixed_k1["pass_at_k"]) + float(shared_k1["pass_at_k"])) / 2
        ax.annotate(
            r"$k=1$",
            (x, y),
            textcoords="offset points",
            xytext=(12, -18),
            fontsize=9,
            color="#222222",
            bbox=LABEL_BBOX,
        )

    offsets = {
        (FIXED_METHOD, 8): (-34, 20),
        (FIXED_METHOD, 64): (10, 14),
        (SHARED_METHOD, 8): (16, -22),
        (SHARED_METHOD, 64): (10, -22),
    }
    for method in (FIXED_METHOD, SHARED_METHOD):
        color = styles[method]["color"]
        for row in plotted[method]:
            k = int(row["k"])
            if k not in {8, 64}:
                continue
            ax.annotate(
                rf"$k={k}$",
                (float(row["avg_cost_tokens"]), float(row["pass_at_k"])),
                textcoords="offset points",
                xytext=offsets.get((method, k), (5, 5)),
                fontsize=9,
                color=color,
                bbox=LABEL_BBOX,
            )

    ax.set_xlabel("Memory Usage (tokens)")
    ax.set_ylabel(r"Pass@$k$ Performance")
    ax.set_ylim(0.28, 0.76)
    ax.xaxis.set_major_formatter(FuncFormatter(token_formatter))
    ax.set_title("Average Across Reasoning Budgets")
    ax.grid(True, axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.6)
    ax.grid(True, axis="x", color="#E6E6E6", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "average_across_budgets_memory_vs_passk_clean_linear.png", dpi=300)
    plt.close(fig)


def plot_average_clean_logx(out_dir: Path, averaged_rows: List[Dict[str, float | int | str]]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    styles = {
        FIXED_METHOD: {"color": "#3366AA", "marker": "o", "linestyle": "-", "label": "Fixed trace"},
        SHARED_METHOD: {"color": "#DD7E2A", "marker": "s", "linestyle": "--", "label": "Shared trace"},
    }
    for method in (FIXED_METHOD, SHARED_METHOD):
        series = sorted(
            [row for row in averaged_rows if str(row["method"]) == method],
            key=lambda row: int(row["k"]),
        )
        style = styles[method]
        ax.plot(
            [float(row["avg_cost_tokens"]) for row in series],
            [float(row["pass_at_k"]) for row in series],
            linewidth=2.4,
            markersize=5.8,
            **style,
        )
        for row in series:
            k = int(row["k"])
            if k not in ANNOTATE_K:
                continue
            ax.annotate(
                f"k={k}",
                (float(row["avg_cost_tokens"]), float(row["pass_at_k"])),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color=style["color"],
                bbox=LABEL_BBOX,
            )
    ax.set_xscale("log")
    ax.set_xlabel("Memory Usage (tokens, log scale)")
    ax.set_ylabel("Pass@k Performance")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Average Across Reasoning Budgets")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "average_across_budgets_memory_vs_passk_clean_logx.png", dpi=300)
    plt.close(fig)


def plot_dual_axis_tradeoff(out_dir: Path, avg_tradeoff_rows: List[Dict[str, float | int]]) -> None:
    rows = sorted(avg_tradeoff_rows, key=lambda row: int(row["k"]))
    fig, ax1 = plt.subplots(figsize=(8.8, 5.4))
    ax2 = ax1.twinx()

    k_values = [int(row["k"]) for row in rows]
    savings = [float(row["token_savings"]) for row in rows]
    losses = [float(row["accuracy_loss"]) for row in rows]

    line1 = ax1.plot(
        k_values,
        savings,
        color="#228B7E",
        marker="o",
        linewidth=2.4,
        markersize=5.8,
        label="Token savings",
    )
    line2 = ax2.plot(
        k_values,
        losses,
        color="#B44E55",
        marker="s",
        linewidth=2.4,
        markersize=5.8,
        linestyle="--",
        label="Accuracy loss",
    )

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(k_values)
    ax1.set_xticklabels([str(k) for k in k_values])
    ax1.set_xlabel("k")
    ax1.set_ylabel("Token Savings")
    ax1.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax2.set_ylabel("Pass@k Gap (Fixed - Shared)")
    ax1.set_title("Shared Trace Token Savings vs Accuracy Loss")
    ax1.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.6)
    ax2.axhline(0.0, color="#B44E55", linewidth=0.8, alpha=0.45)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "token_savings_accuracy_loss_dual_axis.png", dpi=300)
    plt.close(fig)


def plot_pareto(out_dir: Path, pairs: List[Dict[str, float | int]]) -> None:
    frontier = pareto_frontier(pairs)
    xlim, ylim = tradeoff_axis_limits(pairs)
    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    budgets = sorted({int(row["reasoning_prefix_tokens"]) for row in pairs})
    for budget in budgets:
        subset = [row for row in pairs if int(row["reasoning_prefix_tokens"]) == budget]
        ax.scatter(
            [float(row["token_savings"]) for row in subset],
            [float(row["accuracy_loss"]) for row in subset],
            s=46,
            marker=BUDGET_MARKERS.get(budget, "o"),
            color=BUDGET_COLORS.get(budget, "#777777"),
            alpha=0.66,
            edgecolor="white",
            linewidth=0.55,
            label=f"{budget} tokens",
        )
    if frontier:
        ax.plot(
            [float(row["token_savings"]) for row in frontier],
            [float(row["accuracy_loss"]) for row in frontier],
            color="#666666",
            linestyle="--",
            linewidth=2.0,
            marker="o",
            markersize=4.0,
            alpha=0.7,
            label="Pareto frontier",
        )

        for row in frontier:
            k = int(row["k"])
            if k not in {1, 16, 64}:
                continue
            ax.annotate(
                rf"$k={k}$",
                (float(row["token_savings"]), float(row["accuracy_loss"])),
                textcoords="offset points",
                xytext=(7, -13 if k == 1 else 7),
                fontsize=9,
                color="#333333",
                bbox=LABEL_BBOX,
            )

    ax.axhline(0.0, color="#777777", linewidth=0.9, alpha=0.7, zorder=0)
    ax.axvline(0.0, color="#777777", linewidth=0.9, alpha=0.7, zorder=0)

    x_min, x_max = xlim
    y_min, y_max = ylim
    x_left = x_min + 0.045 * (x_max - x_min)
    x_right = x_max - 0.045 * (x_max - x_min)
    y_bottom = y_min + 0.085 * (y_max - y_min)
    y_top = y_max - 0.085 * (y_max - y_min)
    quadrant_style = {
        "fontsize": 9.5,
        "color": "#555555",
        "alpha": 0.88,
        "bbox": {
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.72,
        },
    }
    ax.text(x_right, y_bottom, "Best tradeoff", ha="right", va="bottom", **quadrant_style)
    ax.text(x_right, y_top, "High savings, high loss", ha="right", va="top", **quadrant_style)
    ax.text(x_left, y_bottom, "Accuracy gain, low savings", ha="left", va="bottom", **quadrant_style)
    ax.text(x_left, y_top, "Bad tradeoff", ha="left", va="top", **quadrant_style)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax.set_xlabel("Token savings from sharing")
    ax.set_ylabel(r"Accuracy loss from sharing (Fixed $-$ Shared)")
    ax.set_title("Token Savings vs Accuracy Tradeoff")
    ax.grid(True, color="#D8D8D8", linestyle="--", linewidth=0.65, alpha=0.55)
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        title="Reasoning budget",
    )
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(out_dir / "token_savings_accuracy_loss_pareto_frontier.png", dpi=300)
    fig.savefig(out_dir / "token_savings_accuracy_tradeoff_clean.png", dpi=300)
    plt.close(fig)


def plot_pareto_by_budget_subplots(out_dir: Path, pairs: List[Dict[str, float | int]]) -> None:
    budgets = sorted({int(row["reasoning_prefix_tokens"]) for row in pairs})
    xlim, ylim = tradeoff_axis_limits(pairs)
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), sharex=True, sharey=True)
    axes_flat = list(axes.ravel())
    for ax, budget in zip(axes_flat, budgets):
        subset = sorted(
            [row for row in pairs if int(row["reasoning_prefix_tokens"]) == budget],
            key=lambda row: int(row["k"]),
        )
        frontier = pareto_frontier(subset)
        color = BUDGET_COLORS.get(budget, "#777777")
        marker = BUDGET_MARKERS.get(budget, "o")
        ax.plot(
            [float(row["token_savings"]) for row in subset],
            [float(row["accuracy_loss"]) for row in subset],
            color=color,
            linewidth=1.45,
            alpha=0.42,
            zorder=1,
        )
        ax.scatter(
            [float(row["token_savings"]) for row in subset],
            [float(row["accuracy_loss"]) for row in subset],
            s=48,
            marker=marker,
            color=color,
            alpha=0.76,
            edgecolor="white",
            linewidth=0.6,
            label=f"{budget} tokens",
            zorder=3,
        )
        if frontier:
            ax.plot(
                [float(row["token_savings"]) for row in frontier],
                [float(row["accuracy_loss"]) for row in frontier],
                color="#666666",
                linestyle="--",
                linewidth=2.0,
                marker="o",
                markersize=3.4,
                alpha=0.68,
                label="Within-budget frontier",
                zorder=4,
            )
        for row in subset:
            k = int(row["k"])
            if k not in {1, 16, 64}:
                continue
            offset = {
                1: (9, -17),
                16: (10, 15),
                64: (10, 15),
            }[k]
            ax.annotate(
                rf"$k={k}$",
                (float(row["token_savings"]), float(row["accuracy_loss"])),
                textcoords="offset points",
                xytext=offset,
                fontsize=8.5,
                color="#222222",
                bbox=LABEL_BBOX,
                zorder=6,
            )
        ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.55, zorder=0)
        ax.axvline(0.0, color="#777777", linewidth=0.8, alpha=0.55, zorder=0)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_formatter(FuncFormatter(pct_formatter))
        ax.grid(True, color="#DCDCDC", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_title(f"{budget} reasoning tokens", fontsize=11.5)
    for ax in axes_flat[len(budgets) :]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("Token savings from sharing")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Accuracy loss from sharing (Fixed $-$ Shared)")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles[-1:],
            labels[-1:],
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.975),
        )
    fig.suptitle("Token Savings vs Accuracy Tradeoff by Reasoning Budget", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "token_savings_accuracy_loss_by_budget_subplots.png", dpi=300)
    fig.savefig(out_dir / "token_savings_accuracy_tradeoff_by_budget_subplots.png", dpi=300)
    plt.close(fig)


def plot_pareto_budget_overlay(out_dir: Path, pairs: List[Dict[str, float | int]]) -> None:
    budgets = sorted({int(row["reasoning_prefix_tokens"]) for row in pairs})
    xlim, ylim = tradeoff_axis_limits(pairs)
    global_frontier = pareto_frontier(pairs)
    fig, ax = plt.subplots(figsize=(8.8, 5.9))
    for budget in budgets:
        subset = sorted(
            [row for row in pairs if int(row["reasoning_prefix_tokens"]) == budget],
            key=lambda row: int(row["k"]),
        )
        frontier = pareto_frontier(subset)
        color = BUDGET_COLORS.get(budget, "#777777")
        marker = BUDGET_MARKERS.get(budget, "o")
        linestyle = BUDGET_LINESTYLES.get(budget, "-")
        ax.scatter(
            [float(row["token_savings"]) for row in subset],
            [float(row["accuracy_loss"]) for row in subset],
            s=38,
            marker=marker,
            color=color,
            alpha=0.42,
            edgecolor="white",
            linewidth=0.45,
        )
        if frontier:
            ax.plot(
                [float(row["token_savings"]) for row in frontier],
                [float(row["accuracy_loss"]) for row in frontier],
                color=color,
                linestyle=linestyle,
                linewidth=1.9,
                marker=marker,
                markersize=4.4,
                markeredgecolor="white",
                markeredgewidth=0.5,
                alpha=0.88,
                label=f"{budget} tokens",
            )
    if global_frontier:
        ax.plot(
            [float(row["token_savings"]) for row in global_frontier],
            [float(row["accuracy_loss"]) for row in global_frontier],
            color="#111111",
            linewidth=2.6,
            marker="o",
            markersize=4.2,
            alpha=0.92,
            label="Global frontier",
            zorder=10,
        )
    ax.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.55)
    ax.axvline(0.0, color="#555555", linewidth=0.8, alpha=0.55)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax.set_xlabel("Token Savings")
    ax.set_ylabel("Accuracy Loss (Fixed - Shared)")
    ax.set_title("Token Savings vs Accuracy Loss Pareto Frontier")
    ax.grid(True, color="#DDDDDD", linestyle="--", linewidth=0.65, alpha=0.55)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out_dir / "token_savings_accuracy_loss_budget_overlay.png", dpi=300)
    plt.close(fig)


def plot_avg_tradeoff_independent(
    out_dir: Path, pairs: List[Dict[str, float | int]]
) -> None:
    rows = average_tradeoff_percent_by_k(pairs)
    k_values = [int(row["k"]) for row in rows]
    savings = [float(row["token_savings_pct"]) for row in rows]
    losses = [float(row["accuracy_loss"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.9, 4.8))
    ax.plot(
        k_values,
        savings,
        color="#287C75",
        marker="o",
        markersize=5.4,
        markeredgecolor="white",
        markeredgewidth=0.7,
        linewidth=2.2,
    )
    set_k_axis(ax, k_values)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
    ax.set_xlabel(r"$k$")
    ax.set_ylabel("Token savings from sharing")
    ax.set_title("Average Token Savings Across Reasoning Budgets")
    ax.grid(True, color="#D8D8D8", linestyle="--", linewidth=0.65, alpha=0.55)
    add_equation_note(ax, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "shared_trace_avg_token_savings_vs_k.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.9, 4.8))
    ax.plot(
        k_values,
        losses,
        color="#A94E5A",
        marker="s",
        markersize=5.4,
        markeredgecolor="white",
        markeredgewidth=0.7,
        linewidth=2.2,
    )
    ax.axhline(0.0, color="#666666", linewidth=0.9, alpha=0.65)
    set_k_axis(ax, k_values)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"Accuracy loss from sharing (Fixed $-$ Shared)")
    ax.set_title("Average Accuracy Loss Across Reasoning Budgets")
    ax.grid(True, color="#D8D8D8", linestyle="--", linewidth=0.65, alpha=0.55)
    add_equation_note(ax, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "shared_trace_avg_accuracy_loss_vs_k.png", dpi=300)
    plt.close(fig)


def plot_avg_tradeoff_stacked(
    out_dir: Path, pairs: List[Dict[str, float | int]]
) -> None:
    rows = average_tradeoff_percent_by_k(pairs)
    k_values = [int(row["k"]) for row in rows]
    savings = [float(row["token_savings_pct"]) for row in rows]
    losses = [float(row["accuracy_loss"]) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.3), sharex=True)
    top, bottom = axes
    top.plot(
        k_values,
        savings,
        color="#287C75",
        marker="o",
        markersize=5.2,
        markeredgecolor="white",
        markeredgewidth=0.7,
        linewidth=2.15,
    )
    top.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
    top.set_ylabel("Token savings")
    top.set_title("Average Shared Trace Tradeoff Across Reasoning Budgets")
    top.grid(True, color="#D8D8D8", linestyle="--", linewidth=0.65, alpha=0.55)
    add_equation_note(top, loc="upper left")

    bottom.plot(
        k_values,
        losses,
        color="#A94E5A",
        marker="s",
        markersize=5.2,
        markeredgecolor="white",
        markeredgewidth=0.7,
        linewidth=2.15,
    )
    bottom.axhline(0.0, color="#666666", linewidth=0.9, alpha=0.65)
    bottom.set_ylabel("Accuracy loss")
    bottom.set_xlabel(r"$k$")
    bottom.grid(True, color="#D8D8D8", linestyle="--", linewidth=0.65, alpha=0.55)
    set_k_axis(bottom, k_values)
    fig.tight_layout()
    fig.savefig(out_dir / "shared_trace_avg_tradeoff_stacked_vs_k.png", dpi=300)
    plt.close(fig)


def plot_budget_tradeoff_stacked_grid(
    out_dir: Path, pairs: List[Dict[str, float | int]]
) -> None:
    grouped = grouped_tradeoff_by_budget(pairs)
    budgets = list(grouped)
    k_values = sorted({int(row["k"]) for row in pairs})
    fig = plt.figure(figsize=(14.6, 8.5), constrained_layout=True)
    outer = fig.add_gridspec(2, 3, wspace=0.24, hspace=0.34)

    for idx, budget in enumerate(budgets):
        inner = outer[idx].subgridspec(2, 1, hspace=0.08)
        top = fig.add_subplot(inner[0])
        bottom = fig.add_subplot(inner[1], sharex=top)

        rows = grouped[budget]
        xs = [int(row["k"]) for row in rows]
        savings = [float(row["token_savings_pct"]) for row in rows]
        losses = [float(row["accuracy_loss"]) for row in rows]
        color = BUDGET_COLORS.get(budget, "#777777")
        marker = BUDGET_MARKERS.get(budget, "o")

        top.plot(
            xs,
            savings,
            color=color,
            marker=marker,
            markersize=4.8,
            markeredgecolor="white",
            markeredgewidth=0.65,
            linewidth=1.9,
        )
        top.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
        top.set_title(f"{budget} reasoning tokens", fontsize=11.2)
        top.set_ylabel("Savings")
        top.grid(True, color="#DEDEDE", linestyle="--", linewidth=0.55, alpha=0.5)
        set_k_axis(top, k_values)
        top.tick_params(axis="x", labelbottom=False)

        bottom.plot(
            xs,
            losses,
            color=color,
            marker=marker,
            markersize=4.8,
            markeredgecolor="white",
            markeredgewidth=0.65,
            linewidth=1.9,
        )
        bottom.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.6)
        bottom.set_ylabel("Loss")
        bottom.grid(True, color="#DEDEDE", linestyle="--", linewidth=0.55, alpha=0.5)
        set_k_axis(bottom, k_values)
        if idx >= 3:
            bottom.set_xlabel(r"$k$")

    fig.suptitle("Shared Trace Tradeoff by Reasoning Budget")
    fig.savefig(out_dir / "shared_trace_tradeoff_by_budget_stacked_grid.png", dpi=300)
    plt.close(fig)


def plot_all_budgets_tradeoff_stacked(
    out_dir: Path, pairs: List[Dict[str, float | int]]
) -> None:
    grouped = grouped_tradeoff_by_budget(pairs)
    k_values = sorted({int(row["k"]) for row in pairs})
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 6.8), sharex=True)
    top, bottom = axes

    for budget, rows in grouped.items():
        xs = [int(row["k"]) for row in rows]
        savings = [float(row["token_savings_pct"]) for row in rows]
        losses = [float(row["accuracy_loss"]) for row in rows]
        color = BUDGET_COLORS.get(budget, "#777777")
        marker = BUDGET_MARKERS.get(budget, "o")
        linestyle = BUDGET_LINESTYLES.get(budget, "-")
        line_kwargs = {
            "color": color,
            "linestyle": linestyle,
            "marker": marker,
            "markersize": 4.9,
            "markeredgecolor": "white",
            "markeredgewidth": 0.6,
            "linewidth": 1.9,
            "alpha": 0.94,
            "label": f"{budget} tokens",
        }
        top.plot(xs, savings, **line_kwargs)
        bottom.plot(xs, losses, **line_kwargs)

    top.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
    top.set_ylabel("Token savings")
    top.set_title("Shared Trace Tradeoff Across Reasoning Budgets")
    top.grid(True, color="#D8D8D8", linestyle="--", linewidth=0.65, alpha=0.55)
    add_equation_note(top, loc="upper left")

    bottom.axhline(0.0, color="#666666", linewidth=0.9, alpha=0.65)
    bottom.set_ylabel("Accuracy loss")
    bottom.set_xlabel(r"$k$")
    bottom.grid(True, color="#D8D8D8", linestyle="--", linewidth=0.65, alpha=0.55)
    set_k_axis(bottom, k_values)

    handles, labels = top.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.84, 0.5),
        title="Reasoning budget",
    )
    fig.tight_layout(rect=(0, 0, 0.80, 1))
    fig.savefig(
        out_dir / "shared_trace_tradeoff_all_budgets_stacked_vs_k.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_memory_passk_by_budget_subplots(
    out_dir: Path, rows: List[Dict[str, float | int | str]]
) -> None:
    grouped = group_memory_rows_by_budget(rows)
    budgets = list(grouped)
    all_acc = [float(row["pass_at_k"]) for row in rows]
    y_pad = max(0.02, (max(all_acc) - min(all_acc)) * 0.08)
    ylim = (max(0.0, min(all_acc) - y_pad), min(1.0, max(all_acc) + y_pad))

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.4), sharey=True)
    axes_flat = list(axes.ravel())
    for idx, (ax, budget) in enumerate(zip(axes_flat, budgets)):
        methods = grouped[budget]
        for method in (FIXED_METHOD, SHARED_METHOD):
            series = methods.get(method, [])
            if not series:
                continue
            style = METHOD_STYLES[method]
            ax.plot(
                [float(row["avg_cost_tokens"]) for row in series],
                [float(row["pass_at_k"]) for row in series],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                markersize=4.7,
                markeredgecolor="white",
                markeredgewidth=0.65,
                label=style["label"],
            )
        fixed_series = methods.get(FIXED_METHOD, [])
        shared_series = methods.get(SHARED_METHOD, [])
        if fixed_series and shared_series:
            annotate_memory_curve(ax, fixed_series, shared_series, fontsize=7.2)

        ax.set_title(f"{budget} reasoning tokens", fontsize=11.2)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_formatter(FuncFormatter(token_formatter))
        ax.grid(True, axis="y", color="#D8D8D8", linewidth=0.65, alpha=0.58)
        ax.grid(True, axis="x", color="#E6E6E6", linewidth=0.5, alpha=0.35)
        ax.tick_params(axis="x", labelsize=8.6)
        if idx % 3 == 0:
            ax.set_ylabel(r"Pass@$k$ Performance")
        if idx >= 3:
            ax.set_xlabel("Memory Usage (tokens)")

    for ax in axes_flat[len(budgets) :]:
        ax.axis("off")

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLES[method]["color"],
            linestyle=METHOD_STYLES[method]["linestyle"],
            marker=METHOD_STYLES[method]["marker"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=2.1,
            label=METHOD_STYLES[method]["label"],
        )
        for method in (FIXED_METHOD, SHARED_METHOD)
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Memory Usage vs Pass@k by Reasoning Budget", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "memory_usage_vs_passk_by_budget_subplots.png", dpi=300)
    plt.close(fig)


def plot_memory_passk_all_budgets_overlay(
    out_dir: Path, rows: List[Dict[str, float | int | str]]
) -> None:
    grouped = group_memory_rows_by_budget(rows)
    all_acc = [float(row["pass_at_k"]) for row in rows]
    y_pad = max(0.02, (max(all_acc) - min(all_acc)) * 0.08)

    fig, ax = plt.subplots(figsize=(9.7, 5.9))
    for budget, methods in grouped.items():
        marker = BUDGET_MARKERS.get(budget, "o")
        for method in (FIXED_METHOD, SHARED_METHOD):
            series = methods.get(method, [])
            if not series:
                continue
            style = METHOD_STYLES[method]
            ax.plot(
                [float(row["avg_cost_tokens"]) for row in series],
                [float(row["pass_at_k"]) for row in series],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=marker,
                linewidth=1.65,
                markersize=4.6,
                markeredgecolor="white",
                markeredgewidth=0.55,
                alpha=0.74,
            )

    ax.set_xlabel("Memory Usage (tokens)")
    ax.set_ylabel(r"Pass@$k$ Performance")
    ax.set_ylim(max(0.0, min(all_acc) - y_pad), min(1.0, max(all_acc) + y_pad))
    ax.xaxis.set_major_formatter(FuncFormatter(token_formatter))
    ax.set_title("Memory Usage vs Pass@k: All Reasoning Budgets")
    ax.grid(True, axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.6)
    ax.grid(True, axis="x", color="#E6E6E6", linewidth=0.5, alpha=0.35)

    technique_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLES[method]["color"],
            linestyle=METHOD_STYLES[method]["linestyle"],
            linewidth=2.2,
            label=METHOD_STYLES[method]["label"],
        )
        for method in (FIXED_METHOD, SHARED_METHOD)
    ]
    budget_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker=BUDGET_MARKERS.get(budget, "o"),
            linestyle="None",
            markersize=6.0,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=f"{budget} tokens",
        )
        for budget in grouped
    ]
    technique_legend = ax.legend(
        handles=technique_handles,
        frameon=False,
        loc="upper left",
        title="Technique",
    )
    ax.add_artist(technique_legend)
    ax.legend(
        handles=budget_handles,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        title="Reasoning budget",
    )
    fig.tight_layout(rect=(0, 0, 0.83, 1))
    fig.savefig(out_dir / "memory_usage_vs_passk_all_budgets_superimposed.png", dpi=300)
    plt.close(fig)


def write_discrete_token_summary_csv(
    out_dir: Path, rows: List[Dict[str, float | int | str]]
) -> None:
    path = out_dir / "discrete_tokens_by_budget_k.csv"
    fieldnames = [
        "method",
        "method_label",
        "reasoning_prefix_tokens",
        "k",
        "avg_discrete_tokens",
        "num_prompts",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def plot_discrete_tokens_by_k_subplots(
    out_dir: Path, rows: List[Dict[str, float | int | str]]
) -> None:
    budgets = sorted({int(row["reasoning_prefix_tokens"]) for row in rows})
    pass_at_ks = sorted({int(row["k"]) for row in rows})
    positions = list(range(len(budgets)))
    budget_labels = [str(budget) for budget in budgets]
    lookup = {
        (str(row["method"]), int(row["k"]), int(row["reasoning_prefix_tokens"])): float(
            row["avg_discrete_tokens"]
        )
        for row in rows
    }

    fig, axes = plt.subplots(1, len(pass_at_ks), figsize=(18.8, 3.9), sharex=True)
    if len(pass_at_ks) == 1:
        axes = [axes]
    for ax, k in zip(axes, pass_at_ks):
        for method in (FIXED_METHOD, SHARED_METHOD):
            style = METHOD_STYLES[method]
            ys = [lookup.get((method, k, budget), float("nan")) for budget in budgets]
            ax.plot(
                positions,
                ys,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.9,
                markersize=4.5,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=style["label"],
            )
        ax.set_title(rf"$k={k}$", fontsize=11.2)
        ax.set_xticks(positions)
        ax.set_xticklabels(budget_labels, rotation=35, ha="right")
        ax.yaxis.set_major_formatter(FuncFormatter(token_formatter))
        ax.grid(True, axis="y", color="#D8D8D8", linewidth=0.65, alpha=0.58)
        ax.grid(True, axis="x", color="#E6E6E6", linewidth=0.5, alpha=0.35)
    axes[0].set_ylabel("Discrete tokens")
    for ax in axes:
        ax.set_xlabel("Reasoning budget")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Post-Prefix Discrete Tokens vs Reasoning Budget", y=1.13, fontsize=14.5)
    fig.tight_layout()
    fig.savefig(
        out_dir / "discrete_tokens_vs_reasoning_budget_by_k_subplots.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_discrete_tokens_avg_across_k(
    out_dir: Path, rows: List[Dict[str, float | int | str]]
) -> None:
    budgets = sorted({int(row["reasoning_prefix_tokens"]) for row in rows})
    positions = list(range(len(budgets)))
    budget_labels = [str(budget) for budget in budgets]
    grouped: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["reasoning_prefix_tokens"]))].append(
            float(row["avg_discrete_tokens"])
        )

    fig, ax = plt.subplots(figsize=(8.7, 5.2))
    for method in (FIXED_METHOD, SHARED_METHOD):
        style = METHOD_STYLES[method]
        ys = [mean(grouped[(method, budget)]) for budget in budgets]
        ax.plot(
            positions,
            ys,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            markersize=5.4,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=style["label"],
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(budget_labels)
    ax.yaxis.set_major_formatter(FuncFormatter(token_formatter))
    ax.set_xlabel("Reasoning budget")
    ax.set_ylabel("Average discrete tokens")
    ax.set_title("Average Discrete Tokens Across k Values")
    ax.grid(True, axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.6)
    ax.grid(True, axis="x", color="#E6E6E6", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "average_discrete_tokens_vs_reasoning_budget.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot memory/pass@k token-savings tradeoffs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/gscratch/scrubbed/suryadv/repos/Multiplex-Testing/final_eval_outputs/"
            "passk-memory-raivn-20260424-101750"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    rows = load_rows(root / "summary_overall.csv")
    rows = [
        row
        for row in rows
        if str(row["method"]) in {FIXED_METHOD, SHARED_METHOD}
        and int(row["num_prompts"]) > 0
    ]
    out_dir = root / "requested_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    averaged_rows = average_by_method_and_k(rows)
    pairs = paired_by_budget_and_k(rows)
    avg_tradeoffs = average_tradeoff_by_k(pairs)
    pass_at_ks = sorted({int(row["k"]) for row in rows})
    discrete_token_rows = load_discrete_token_summary(root, pass_at_ks)
    plot_average_clean_linear(out_dir, averaged_rows)
    plot_average_clean_logx(out_dir, averaged_rows)
    plot_dual_axis_tradeoff(out_dir, avg_tradeoffs)
    plot_pareto(out_dir, pairs)
    plot_pareto_by_budget_subplots(out_dir, pairs)
    plot_pareto_budget_overlay(out_dir, pairs)
    plot_avg_tradeoff_independent(out_dir, pairs)
    plot_avg_tradeoff_stacked(out_dir, pairs)
    plot_budget_tradeoff_stacked_grid(out_dir, pairs)
    plot_all_budgets_tradeoff_stacked(out_dir, pairs)
    plot_memory_passk_by_budget_subplots(out_dir, rows)
    plot_memory_passk_all_budgets_overlay(out_dir, rows)
    write_discrete_token_summary_csv(out_dir, discrete_token_rows)
    plot_discrete_tokens_by_k_subplots(out_dir, discrete_token_rows)
    plot_discrete_tokens_avg_across_k(out_dir, discrete_token_rows)

    print(out_dir)
    print("wrote average_across_budgets_memory_vs_passk_clean_linear.png")
    print("wrote average_across_budgets_memory_vs_passk_clean_logx.png")
    print("wrote token_savings_accuracy_loss_dual_axis.png")
    print("wrote token_savings_accuracy_loss_pareto_frontier.png")
    print("wrote token_savings_accuracy_loss_by_budget_subplots.png")
    print("wrote token_savings_accuracy_loss_budget_overlay.png")
    print("wrote shared_trace_avg_token_savings_vs_k.png")
    print("wrote shared_trace_avg_accuracy_loss_vs_k.png")
    print("wrote shared_trace_avg_tradeoff_stacked_vs_k.png")
    print("wrote shared_trace_tradeoff_by_budget_stacked_grid.png")
    print("wrote shared_trace_tradeoff_all_budgets_stacked_vs_k.png")
    print("wrote memory_usage_vs_passk_by_budget_subplots.png")
    print("wrote memory_usage_vs_passk_all_budgets_superimposed.png")
    print("wrote discrete_tokens_by_budget_k.csv")
    print("wrote discrete_tokens_vs_reasoning_budget_by_k_subplots.png")
    print("wrote average_discrete_tokens_vs_reasoning_budget.png")


if __name__ == "__main__":
    main()
