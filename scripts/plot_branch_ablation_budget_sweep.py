#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def group_label(group_size: int) -> str:
    if group_size == 1:
        return "Shared every 1 / fixed trace"
    return f"Shared every {group_size}"


def group_slug(group_size: int) -> str:
    return "shared_every_01" if group_size == 1 else f"shared_every_{group_size:02d}"


def group_style(group_size: int) -> Dict[str, object]:
    colors = {
        1: "#4C78A8",
        2: "#F58518",
        4: "#54A24B",
        8: "#B279A2",
        16: "#E45756",
        32: "#72B7B2",
        64: "#EECA3B",
    }
    markers = {1: "o", 2: "s", 4: "^", 8: "D", 16: "P", 32: "X", 64: "v"}
    return {
        "color": colors.get(group_size, "#7f7f7f"),
        "marker": markers.get(group_size, "o"),
        "markersize": 5.2,
        "linewidth": 1.8,
    }


def budget_style(budget: int) -> Dict[str, object]:
    linestyles = {1024: "-", 2048: "--", 4096: ":"}
    markers = {1024: "o", 2048: "s", 4096: "^"}
    return {
        "linestyle": linestyles.get(budget, "-."),
        "marker": markers.get(budget, "o"),
        "markersize": 5.0,
        "linewidth": 2.0,
    }


def load_rows(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for budget_dir in sorted(root.glob("budget_*")):
        if not budget_dir.is_dir():
            continue
        try:
            budget = int(budget_dir.name.rsplit("_", 1)[1])
        except ValueError:
            continue
        summary_path = budget_dir / "summary_ablation.csv"
        if not summary_path.exists():
            continue
        with summary_path.open(newline="", encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                rows.append(
                    {
                        "reasoning_prefix_tokens": budget,
                        "branch_group_size": int(row["branch_group_size"]),
                        "condition": row["condition"],
                        "method": row["method"],
                        "k": int(row["k"]),
                        "num_prompts": int(row["num_prompts"]),
                        "pass_at_k": float(row["pass_at_k"]),
                        "avg_cost_tokens": float(row["avg_cost_tokens"]),
                        "num_failures": int(row["num_failures"]),
                    }
                )
    return rows


def write_combined_csv(root: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "reasoning_prefix_tokens",
        "branch_group_size",
        "condition",
        "method",
        "k",
        "num_prompts",
        "pass_at_k",
        "avg_cost_tokens",
        "num_failures",
    ]
    with (root / "summary_ablation_budget_sweep.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                int(item["reasoning_prefix_tokens"]),
                int(item["branch_group_size"]),
                int(item["k"]),
            ),
        ):
            writer.writerow({name: row[name] for name in fieldnames})


def iter_series(
    rows: Iterable[Dict[str, object]],
    *,
    group_size: int | None = None,
    budget: int | None = None,
) -> Iterable[Tuple[int, int, List[Dict[str, object]]]]:
    by_key: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        row_group = int(row["branch_group_size"])
        row_budget = int(row["reasoning_prefix_tokens"])
        if group_size is not None and row_group != group_size:
            continue
        if budget is not None and row_budget != budget:
            continue
        by_key[(row_group, row_budget)].append(row)
    for (row_group, row_budget), series_rows in sorted(by_key.items()):
        yield row_group, row_budget, sorted(series_rows, key=lambda item: int(item["k"]))


def plot_overlay(
    path: Path,
    rows: List[Dict[str, object]],
    *,
    y_full: bool,
    title: str,
) -> None:
    plt.figure(figsize=(11.2, 6.8))
    ax = plt.gca()
    for group_size, budget, series_rows in iter_series(rows):
        style = group_style(group_size)
        bstyle = budget_style(budget)
        ax.plot(
            [int(row["k"]) for row in series_rows],
            [float(row["pass_at_k"]) for row in series_rows],
            color=style["color"],
            marker=bstyle["marker"],
            linestyle=bstyle["linestyle"],
            linewidth=bstyle["linewidth"],
            markersize=bstyle["markersize"],
            label=group_label(group_size) if budget == 0 else f"{group_label(group_size)} / T={budget}",
        )
    observed = [float(row["pass_at_k"]) for row in rows]
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({int(row["k"]) for row in rows}))
    ax.set_xticklabels([str(value) for value in sorted({int(row["k"]) for row in rows})])
    ax.set_xlabel("k")
    ax.set_ylabel("Pass@k Performance")
    if y_full:
        ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylim(max(0.0, min(observed) - 0.04), min(1.0, max(observed) + 0.04))
    ax.set_title(title)
    ax.grid(True, which="major", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def plot_memory_and_tokens(root: Path, rows: List[Dict[str, object]], *, prefix: str) -> None:
    for filename, x_key, y_key, xlabel, ylabel, title, log_x in [
        (
            f"{prefix}_memory_vs_passk.png",
            "avg_cost_tokens",
            "pass_at_k",
            "Memory Usage (tokens)",
            "Pass@k Performance",
            "Memory Usage vs Pass@k",
            False,
        ),
        (
            f"{prefix}_k_vs_generated_tokens.png",
            "k",
            "avg_cost_tokens",
            "k",
            "Generated Tokens",
            "k vs Generated Tokens",
            True,
        ),
    ]:
        plt.figure(figsize=(11.2, 6.8))
        ax = plt.gca()
        for group_size, budget, series_rows in iter_series(rows):
            style = group_style(group_size)
            bstyle = budget_style(budget)
            ax.plot(
                [float(row[x_key]) for row in series_rows],
                [float(row[y_key]) for row in series_rows],
                color=style["color"],
                marker=bstyle["marker"],
                linestyle=bstyle["linestyle"],
                linewidth=bstyle["linewidth"],
                markersize=bstyle["markersize"],
                label=group_label(group_size) if budget == 0 else f"{group_label(group_size)} / T={budget}",
            )
        if log_x:
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted({int(row["k"]) for row in rows}))
            ax.set_xticklabels([str(value) for value in sorted({int(row["k"]) for row in rows})])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if y_key == "pass_at_k":
            ax.set_ylim(0.0, 1.0)
        ax.set_title(title)
        ax.grid(True, which="major", linestyle="--", alpha=0.3)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(root / filename, dpi=240)
        plt.close()


def plot_by_technique(root: Path, rows: List[Dict[str, object]]) -> None:
    out_dir = root / "technique_budget_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    for group_size in sorted({int(row["branch_group_size"]) for row in rows}):
        group_rows = [row for row in rows if int(row["branch_group_size"]) == group_size]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
        for budget in sorted({int(row["reasoning_prefix_tokens"]) for row in group_rows}):
            series_rows = sorted(
                [row for row in group_rows if int(row["reasoning_prefix_tokens"]) == budget],
                key=lambda item: int(item["k"]),
            )
            bstyle = budget_style(budget)
            axes[0].plot(
                [int(row["k"]) for row in series_rows],
                [float(row["pass_at_k"]) for row in series_rows],
                label=f"T={budget}",
                **bstyle,
            )
            axes[1].plot(
                [float(row["avg_cost_tokens"]) for row in series_rows],
                [float(row["pass_at_k"]) for row in series_rows],
                label=f"T={budget}",
                **bstyle,
            )
            axes[2].plot(
                [int(row["k"]) for row in series_rows],
                [float(row["avg_cost_tokens"]) for row in series_rows],
                label=f"T={budget}",
                **bstyle,
            )
        axes[0].set_xscale("log", base=2)
        axes[0].set_xticks(sorted({int(row["k"]) for row in group_rows}))
        axes[0].set_xticklabels([str(value) for value in sorted({int(row["k"]) for row in group_rows})])
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("Pass@k Performance")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_title("k vs Pass@k")
        axes[1].set_xlabel("Memory Usage (tokens)")
        axes[1].set_ylabel("Pass@k Performance")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("Memory Usage vs Pass@k")
        axes[2].set_xscale("log", base=2)
        axes[2].set_xticks(sorted({int(row["k"]) for row in group_rows}))
        axes[2].set_xticklabels([str(value) for value in sorted({int(row["k"]) for row in group_rows})])
        axes[2].set_xlabel("k")
        axes[2].set_ylabel("Generated Tokens")
        axes[2].set_title("k vs Generated Tokens")
        for ax in axes:
            ax.grid(True, which="major", linestyle="--", alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
        fig.suptitle(group_label(group_size), y=1.03)
        plt.tight_layout()
        plt.savefig(out_dir / f"{group_slug(group_size)}_budget_overlay.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def averaged_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_key: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_key[(int(row["branch_group_size"]), int(row["k"]))].append(row)
    averaged: List[Dict[str, object]] = []
    for (group_size, k), key_rows in sorted(by_key.items()):
        averaged.append(
            {
                "reasoning_prefix_tokens": 0,
                "branch_group_size": group_size,
                "condition": f"average_shared_every_{group_size}",
                "method": str(key_rows[0]["method"]),
                "k": k,
                "num_prompts": int(mean(float(row["num_prompts"]) for row in key_rows)),
                "pass_at_k": mean(float(row["pass_at_k"]) for row in key_rows),
                "avg_cost_tokens": mean(float(row["avg_cost_tokens"]) for row in key_rows),
                "num_failures": int(mean(float(row["num_failures"]) for row in key_rows)),
            }
        )
    return averaged


def write_average_plots(root: Path, rows: List[Dict[str, object]]) -> None:
    avg_dir = root / "average_across_budgets"
    avg_dir.mkdir(parents=True, exist_ok=True)
    avg_rows = averaged_rows(rows)
    with (avg_dir / "summary_ablation_average_across_budgets.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fp:
        fieldnames = [
            "branch_group_size",
            "condition",
            "method",
            "k",
            "num_prompts",
            "pass_at_k",
            "avg_cost_tokens",
            "num_failures",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in avg_rows:
            writer.writerow({name: row[name] for name in fieldnames})
    plot_overlay(
        avg_dir / "average_k_vs_passk_zoomed.png",
        avg_rows,
        y_full=False,
        title="Average Across Reasoning Budgets: k vs Pass@k",
    )
    plot_overlay(
        avg_dir / "average_k_vs_passk_zoomout.png",
        avg_rows,
        y_full=True,
        title="Average Across Reasoning Budgets: k vs Pass@k",
    )
    plot_memory_and_tokens(avg_dir, avg_rows, prefix="average")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot branch ablation budget sweep summaries.")
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = load_rows(root)
    if not rows:
        raise SystemExit(f"No budget summary rows found under {root}")
    max_k = max(int(row["k"]) for row in rows)
    write_combined_csv(root, rows)
    combined_dir = root / "combined_plots"
    combined_dir.mkdir(parents=True, exist_ok=True)
    plot_overlay(
        combined_dir / "combined_k_vs_passk_compact.png",
        rows,
        y_full=False,
        title=f"Pass@{max_k} Branch Ablation: k vs Pass@k (Compact)",
    )
    plot_overlay(
        combined_dir / "combined_k_vs_passk_zoomout.png",
        rows,
        y_full=True,
        title=f"Pass@{max_k} Branch Ablation: k vs Pass@k (Full Scale)",
    )
    plot_memory_and_tokens(combined_dir, rows, prefix="combined")
    plot_by_technique(root, rows)
    write_average_plots(root, rows)
    print(root)


if __name__ == "__main__":
    main()
