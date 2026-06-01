#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker


TASK_LABELS = {
    "discrete_untrained": "Discrete untrained",
    "discrete_trained": "Discrete trained",
    "multiplex_trained_discrete_eval": "Multiplex trained, discrete eval",
    "multiplex_untrained": "Multiplex untrained",
    "multiplex_trained": "Multiplex trained",
    "discrete_trained_multiplex_eval": "Discrete trained, multiplex eval",
    "shared4_untrained": "Shared4 untrained",
    "shared4_multiplex_trained": "Shared4 from multiplex trained",
    "shared4_joint_trained": "Shared4 joint trained",
    "shared4_thinking_trained": "Shared4 thinking-only trained",
    "shared4_answer_trained": "Shared4 answer-only trained",
}

COMPARISON_GROUPS = {
    "discrete_eval": [
        "discrete_untrained",
        "discrete_trained",
        "multiplex_trained_discrete_eval",
    ],
    "multiplex_eval": [
        "multiplex_untrained",
        "multiplex_trained",
        "discrete_trained_multiplex_eval",
    ],
    "shared4_eval": [
        "shared4_untrained",
        "shared4_multiplex_trained",
        "shared4_joint_trained",
        "shared4_thinking_trained",
        "shared4_answer_trained",
    ],
}

PAIR_COMPARISON_GROUPS = {
    "discrete_untrained_vs_trained": [
        "discrete_untrained",
        "discrete_trained",
    ],
    "multiplex_untrained_vs_trained": [
        "multiplex_untrained",
        "multiplex_trained",
    ],
    "shared4_untrained_vs_joint_trained": [
        "shared4_untrained",
        "shared4_joint_trained",
    ],
    "discrete_eval_with_cross_model": [
        "discrete_untrained",
        "discrete_trained",
        "multiplex_trained_discrete_eval",
    ],
    "multiplex_eval_with_cross_model": [
        "multiplex_untrained",
        "multiplex_trained",
        "discrete_trained_multiplex_eval",
    ],
}


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def maybe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"true", "false"}:
        return 1.0 if truthy(text) else 0.0
    try:
        return float(text)
    except ValueError:
        return None


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_mean(values: List[float], runs: int, rng: random.Random) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1 or runs <= 0:
        return mean, mean, mean
    boot = []
    for _ in range(runs):
        sample_total = sum(values[rng.randrange(len(values))] for _ in values)
        boot.append(sample_total / len(values))
    boot.sort()
    return mean, percentile(boot, 0.025), percentile(boot, 0.975)


def task_dirs(eval_root: Path, task: str) -> List[Path]:
    return sorted(path / task for path in eval_root.glob("repeat_*") if (path / task).is_dir())


def per_prompt_values(task_dir: Path) -> Dict[int, List[float]]:
    values: Dict[int, List[float]] = {}
    candidates = [
        task_dir / "standard_generation" / "per_prompt_correctness.csv",
        task_dir / "baseline" / "per_prompt_correctness.csv",
        task_dir / "passk_sweep" / "per_prompt_correctness.csv",
    ]
    candidates.extend(sorted(task_dir.glob("*/per_prompt_correctness.csv")))
    seen: set[Path] = set()
    for path in candidates:
        if not path.exists() or path in seen:
            continue
        seen.add(path)
        for row in read_csv(path):
            for key, value in row.items():
                if not key.startswith("pass_at_"):
                    continue
                k_text = key.removeprefix("pass_at_")
                if not k_text.isdigit():
                    continue
                parsed = maybe_float(value)
                if parsed is None:
                    continue
                values.setdefault(int(k_text), []).append(parsed)
    return values


def summary_values(task_dir: Path) -> Dict[int, List[float]]:
    values: Dict[int, List[float]] = {}
    for filename in ("summary_overall.csv", "summary_ablation.csv"):
        path = task_dir / filename
        if not path.exists():
            continue
        for row in read_csv(path):
            k_value = row.get("k") or row.get("group_size") or row.get("shared_count")
            score = (
                row.get("pass_at_k")
                or row.get("accuracy")
                or row.get("mean_pass")
                or row.get("pass_rate")
            )
            k_float = maybe_float(k_value)
            score_float = maybe_float(score)
            if k_float is None or score_float is None:
                continue
            values.setdefault(int(k_float), []).append(score_float)
    return values


def collect_task_values(eval_root: Path, task: str) -> Tuple[Dict[int, List[float]], str]:
    merged: Dict[int, List[float]] = {}
    source = "per_prompt"
    for directory in task_dirs(eval_root, task):
        directory_values = per_prompt_values(directory)
        if not directory_values:
            source = "summary"
            directory_values = summary_values(directory)
        for k, values in directory_values.items():
            merged.setdefault(k, []).extend(values)
    return merged, source


def plot_group(group_name: str, rows: List[Dict[str, object]], output_dir: Path) -> Optional[str]:
    group_rows = [row for row in rows if row["group"] == group_name]
    if not group_rows:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for task in COMPARISON_GROUPS[group_name]:
        task_rows = [row for row in group_rows if row["task"] == task]
        if not task_rows:
            continue
        task_rows.sort(key=lambda row: int(row["k"]))
        xs = [int(row["k"]) for row in task_rows]
        ys = [float(row["pass_at_k_mean"]) for row in task_rows]
        lows = [max(0.0, y - float(row["pass_at_k_ci_low"])) for y, row in zip(ys, task_rows)]
        highs = [max(0.0, float(row["pass_at_k_ci_high"]) - y) for y, row in zip(ys, task_rows)]
        ax.errorbar(xs, ys, yerr=[lows, highs], marker="o", capsize=3, label=TASK_LABELS.get(task, task))
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({int(row["k"]) for row in group_rows}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("k")
    ax.set_ylabel("Pass@k")
    ax.set_title(group_name.replace("_", " ").title())
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    output_path = output_dir / f"{group_name}_bootstrap_passk.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return str(output_path)


def plot_task(task: str, rows: List[Dict[str, object]], output_dir: Path) -> Optional[str]:
    task_rows = [row for row in rows if row["task"] == task]
    if not task_rows:
        return None
    task_rows.sort(key=lambda row: int(row["k"]))
    xs = [int(row["k"]) for row in task_rows]
    ys = [float(row["pass_at_k_mean"]) for row in task_rows]
    lows = [max(0.0, y - float(row["pass_at_k_ci_low"])) for y, row in zip(ys, task_rows)]
    highs = [max(0.0, float(row["pass_at_k_ci_high"]) - y) for y, row in zip(ys, task_rows)]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.errorbar(xs, ys, yerr=[lows, highs], marker="o", capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("k")
    ax.set_ylabel("Pass@k")
    ax.set_title(TASK_LABELS.get(task, task))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output_path = output_dir / f"{task}_passk.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return str(output_path)


def plot_pair_group(group_name: str, tasks: List[str], rows: List[Dict[str, object]], output_dir: Path) -> Optional[str]:
    group_rows = [row for row in rows if row["task"] in tasks]
    if not group_rows:
        return None
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    plotted = False
    for task in tasks:
        task_rows = [row for row in group_rows if row["task"] == task]
        if not task_rows:
            continue
        task_rows.sort(key=lambda row: int(row["k"]))
        xs = [int(row["k"]) for row in task_rows]
        ys = [float(row["pass_at_k_mean"]) for row in task_rows]
        lows = [max(0.0, y - float(row["pass_at_k_ci_low"])) for y, row in zip(ys, task_rows)]
        highs = [max(0.0, float(row["pass_at_k_ci_high"]) - y) for y, row in zip(ys, task_rows)]
        ax.errorbar(xs, ys, yerr=[lows, highs], marker="o", capsize=3, label=TASK_LABELS.get(task, task))
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({int(row["k"]) for row in group_rows}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("k")
    ax.set_ylabel("Pass@k")
    ax.set_title(group_name.replace("_", " ").title())
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    output_path = output_dir / f"{group_name}_paired_passk.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return str(output_path)


def plot_all_tasks(rows: List[Dict[str, object]], output_dir: Path) -> Optional[str]:
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    plotted = False
    for task in TASK_LABELS:
        task_rows = [row for row in rows if row["task"] == task]
        if not task_rows:
            continue
        task_rows.sort(key=lambda row: int(row["k"]))
        xs = [int(row["k"]) for row in task_rows]
        ys = [float(row["pass_at_k_mean"]) for row in task_rows]
        ax.plot(xs, ys, marker="o", linewidth=1.6, label=TASK_LABELS.get(task, task))
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({int(row["k"]) for row in rows}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("k")
    ax.set_ylabel("Pass@k")
    ax.set_title("All Pass@k Comparisons")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    output_path = output_dir / "all_tasks_passk_overlay.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot bootstrapped eval comparisons.")
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=26010808)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows: List[Dict[str, object]] = []
    for group_name, tasks in COMPARISON_GROUPS.items():
        for task in tasks:
            task_values, source = collect_task_values(args.eval_root, task)
            for k, values in sorted(task_values.items()):
                mean, low, high = bootstrap_mean(values, args.bootstrap_runs, rng)
                rows.append(
                    {
                        "group": group_name,
                        "task": task,
                        "label": TASK_LABELS.get(task, task),
                        "k": k,
                        "pass_at_k_mean": mean,
                        "pass_at_k_ci_low": low,
                        "pass_at_k_ci_high": high,
                        "n_units": len(values),
                        "source": source,
                        "bootstrap_runs": args.bootstrap_runs,
                    }
                )

    write_csv(args.output_dir / "bootstrap_comparison_summary.csv", rows)
    group_plots = [plot_group(group_name, rows, args.output_dir) for group_name in COMPARISON_GROUPS]
    task_plots = [plot_task(task, rows, args.output_dir) for task in TASK_LABELS]
    pair_plots = [
        plot_pair_group(group_name, tasks, rows, args.output_dir)
        for group_name, tasks in PAIR_COMPARISON_GROUPS.items()
    ]
    overlay_plot = plot_all_tasks(rows, args.output_dir)
    summary = {
        "eval_root": str(args.eval_root),
        "bootstrap_runs": args.bootstrap_runs,
        "seed": args.seed,
        "rows": rows,
        "group_plots": [plot for plot in group_plots if plot],
        "task_plots": [plot for plot in task_plots if plot],
        "pair_plots": [plot for plot in pair_plots if plot],
        "overlay_plot": overlay_plot,
        "plots": [plot for plot in [*group_plots, *task_plots, *pair_plots, overlay_plot] if plot],
    }
    (args.output_dir / "bootstrap_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[aggregate] wrote {args.output_dir / 'bootstrap_comparison_summary.csv'}")


if __name__ == "__main__":
    main()
