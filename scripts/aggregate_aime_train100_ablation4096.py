#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PASS_AT_KS = [1, 2, 4, 8, 16, 32]
GROUP_LABELS = {
    1: "Every 1 / fixed trace",
    2: "Shared every 2",
    4: "Shared every 4",
    8: "Shared every 8",
    16: "Shared every 16",
    32: "Shared every 32",
}
GROUP_COLORS = {
    1: "#4C78A8",
    2: "#F58518",
    4: "#54A24B",
    8: "#B279A2",
    16: "#E45756",
    32: "#72B7B2",
}
GROUP_MARKERS = {
    1: "o",
    2: "s",
    4: "^",
    8: "D",
    16: "P",
    32: "X",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate balanced AIME-train branch ablation repeats and make LaTeX-style plots."
    )
    parser.add_argument("--root", required=True, help="Root output directory containing repeat_00, repeat_01, ...")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seeds", default="409600,409601,409602,409603,409604")
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def mean_std(values: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean:
        return None, None, 0
    if len(clean) == 1:
        return clean[0], 0.0, 1
    return statistics.mean(clean), statistics.stdev(clean), len(clean)


def configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.7,
            "figure.dpi": 180,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )
    return plt


def collect_ablation_rows(root: Path, repeats: int, seeds: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for repeat_index in range(repeats):
        repeat_dir = root / f"repeat_{repeat_index:02d}"
        seed = seeds[repeat_index] if repeat_index < len(seeds) else ""
        for bundle_dir in sorted(repeat_dir.glob("bundle_*")):
            for row in read_csv_rows(bundle_dir / "summary_ablation.csv"):
                out = dict(row)
                out["repeat_index"] = repeat_index
                out["seed"] = seed
                out["bundle"] = bundle_dir.name
                rows.append(out)
    return rows


def aggregate_ablation(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    repeat_buckets: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
    for row in rows:
        group_size = int(float(row.get("branch_group_size") or 0))
        k = int(float(row.get("k") or 0))
        if group_size <= 0 or k <= 0:
            continue
        repeat_index = int(float(row.get("repeat_index") or 0))
        repeat_buckets.setdefault((repeat_index, group_size, k), []).append(row)

    repeat_rows: List[Dict[str, Any]] = []
    for (repeat_index, group_size, k), bucket in sorted(repeat_buckets.items()):
        weights = [to_float(row.get("num_prompts")) or 0.0 for row in bucket]
        total_weight = sum(weight for weight in weights if weight > 0)

        def weighted_metric(field: str) -> Optional[float]:
            values = [to_float(row.get(field)) for row in bucket]
            if total_weight > 0 and any(value is not None for value in values):
                numerator = sum(
                    float(value) * weight
                    for value, weight in zip(values, weights)
                    if value is not None and weight > 0
                )
                return numerator / total_weight
            mean_value, _, _ = mean_std(values)
            return mean_value

        repeat_rows.append(
            {
                "repeat_index": repeat_index,
                "group_size": group_size,
                "k": k,
                "pass_at_k": weighted_metric("pass_at_k"),
                "avg_cost_tokens": weighted_metric("avg_cost_tokens"),
                "num_prompts": total_weight if total_weight > 0 else None,
            }
        )

    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for row in repeat_rows:
        buckets.setdefault((int(row["group_size"]), int(row["k"])), []).append(row)

    summary: List[Dict[str, Any]] = []
    for (group_size, k), bucket in sorted(buckets.items()):
        pass_mean, pass_std, n = mean_std(to_float(row.get("pass_at_k")) for row in bucket)
        cost_mean, cost_std, _ = mean_std(to_float(row.get("avg_cost_tokens")) for row in bucket)
        prompts_mean, _, _ = mean_std(to_float(row.get("num_prompts")) for row in bucket)
        summary.append(
            {
                "group_size": group_size,
                "condition": GROUP_LABELS.get(group_size, f"Shared every {group_size}"),
                "k": k,
                "num_repeats": n,
                "mean_pass_at_k": pass_mean,
                "std_pass_at_k": pass_std,
                "mean_cost_tokens": cost_mean,
                "std_cost_tokens": cost_std,
                "mean_num_prompts": prompts_mean,
            }
        )
    return summary


def collect_memory_match_rows(root: Path, repeats: int, seeds: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for repeat_index in range(repeats):
        repeat_dir = root / f"repeat_{repeat_index:02d}"
        seed = seeds[repeat_index] if repeat_index < len(seeds) else ""
        topup_dirs = sorted(repeat_dir.glob("memory_match_topup*"))
        for topup_dir in topup_dirs:
            for row in read_csv_rows(topup_dir / "summary_memory_match.csv"):
                out = dict(row)
                out["repeat_index"] = repeat_index
                out["seed"] = seed
                out["topup_bundle"] = topup_dir.name
                rows.append(out)
    return rows


def aggregate_memory_match(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        group_size = int(float(row.get("group_size") or 0))
        if group_size <= 0:
            continue
        buckets.setdefault(group_size, []).append(row)
    summary: List[Dict[str, Any]] = []
    metrics = [
        "before_accuracy",
        "after_accuracy",
        "target_cost_tokens",
        "before_cost_tokens",
        "after_cost_tokens",
        "effective_k",
        "topup_count",
        "reached_target_rate",
    ]
    for group_size, bucket in sorted(buckets.items()):
        row: Dict[str, Any] = {
            "group_size": group_size,
            "condition": GROUP_LABELS.get(group_size, f"Shared every {group_size}"),
            "num_repeats": len(bucket),
        }
        for metric in metrics:
            mean_value, std_value, n = mean_std(to_float(item.get(metric)) for item in bucket)
            row[f"mean_{metric}"] = mean_value
            row[f"std_{metric}"] = std_value
            row[f"n_{metric}"] = n
        summary.append(row)
    return summary


def row_for(summary: List[Dict[str, Any]], group_size: int, k: int) -> Optional[Dict[str, Any]]:
    for row in summary:
        if int(row["group_size"]) == group_size and int(row["k"]) == k:
            return row
    return None


def plot_passk_scaling(plt: Any, out_dir: Path, summary: List[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    group_sizes = sorted({int(row["group_size"]) for row in summary})
    for group_size in group_sizes:
        rows = [row for row in summary if int(row["group_size"]) == group_size]
        rows.sort(key=lambda row: int(row["k"]))
        xs = [int(row["k"]) for row in rows]
        ys = [float(row["mean_pass_at_k"]) for row in rows if row["mean_pass_at_k"] is not None]
        if not ys:
            continue
        yvals = [float(row["mean_pass_at_k"]) for row in rows]
        yerr = [float(row["std_pass_at_k"] or 0.0) for row in rows]
        ax.errorbar(
            xs,
            yvals,
            yerr=yerr,
            marker=GROUP_MARKERS.get(group_size, "o"),
            color=GROUP_COLORS.get(group_size),
            linewidth=1.8,
            capsize=3,
            label=GROUP_LABELS.get(group_size, str(group_size)),
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(PASS_AT_KS)
    ax.set_xticklabels([str(k) for k in PASS_AT_KS])
    ax.set_xlabel("k")
    ax.set_ylabel(r"Pass@$k$")
    ax.set_title(r"AIME Train: Pass@$k$ Scaling")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "passk_scaling_mean_std.png")
    plt.close(fig)


def plot_memory_vs_passk(plt: Any, out_dir: Path, summary: List[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for group_size in sorted({int(row["group_size"]) for row in summary}):
        rows = [
            row
            for row in summary
            if int(row["group_size"]) == group_size
            and row.get("mean_cost_tokens") is not None
            and row.get("mean_pass_at_k") is not None
        ]
        rows.sort(key=lambda row: int(row["k"]))
        if not rows:
            continue
        ax.errorbar(
            [float(row["mean_cost_tokens"]) for row in rows],
            [float(row["mean_pass_at_k"]) for row in rows],
            xerr=[float(row["std_cost_tokens"] or 0.0) for row in rows],
            yerr=[float(row["std_pass_at_k"] or 0.0) for row in rows],
            marker=GROUP_MARKERS.get(group_size, "o"),
            color=GROUP_COLORS.get(group_size),
            linewidth=1.7,
            capsize=3,
            label=GROUP_LABELS.get(group_size, str(group_size)),
        )
    ax.set_xlabel("Memory Usage (tokens)")
    ax.set_ylabel(r"Pass@$k$")
    ax.set_title(r"AIME Train: Memory Usage vs. Pass@$k$")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "memory_vs_passk_mean_std.png")
    plt.close(fig)


def plot_focus_k(plt: Any, out_dir: Path, summary: List[Dict[str, Any]], k: int) -> None:
    rows = [
        row
        for row in summary
        if int(row["k"]) == k and row.get("mean_cost_tokens") is not None and row.get("mean_pass_at_k") is not None
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for row in rows:
        group_size = int(row["group_size"])
        ax.errorbar(
            [float(row["mean_cost_tokens"])],
            [float(row["mean_pass_at_k"])],
            xerr=[float(row["std_cost_tokens"] or 0.0)],
            yerr=[float(row["std_pass_at_k"] or 0.0)],
            marker=GROUP_MARKERS.get(group_size, "o"),
            color=GROUP_COLORS.get(group_size),
            linestyle="none",
            markersize=7,
            capsize=3,
            label=GROUP_LABELS.get(group_size, str(group_size)),
        )
    ax.set_xlabel("Memory Usage (tokens)")
    ax.set_ylabel(rf"Pass@{k}")
    ax.set_title(rf"AIME Train: Memory Usage vs. Pass@{k}")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out_dir / f"memory_vs_pass{k}_mean_std.png")
    plt.close(fig)


def plot_memory_match(
    plt: Any,
    out_dir: Path,
    ablation_summary: List[Dict[str, Any]],
    memory_summary: List[Dict[str, Any]],
) -> None:
    fixed = row_for(ablation_summary, 1, 32)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    if fixed and fixed.get("mean_cost_tokens") is not None and fixed.get("mean_pass_at_k") is not None:
        ax.errorbar(
            [float(fixed["mean_cost_tokens"])],
            [float(fixed["mean_pass_at_k"])],
            xerr=[float(fixed["std_cost_tokens"] or 0.0)],
            yerr=[float(fixed["std_pass_at_k"] or 0.0)],
            marker=GROUP_MARKERS[1],
            color=GROUP_COLORS[1],
            linestyle="none",
            markersize=8,
            capsize=3,
            label="Fixed@32",
        )
    for row in memory_summary:
        group_size = int(row["group_size"])
        before_x = row.get("mean_before_cost_tokens")
        before_y = row.get("mean_before_accuracy")
        after_x = row.get("mean_after_cost_tokens")
        after_y = row.get("mean_after_accuracy")
        if None in (before_x, before_y, after_x, after_y):
            continue
        color = GROUP_COLORS.get(group_size)
        ax.errorbar(
            [float(before_x)],
            [float(before_y)],
            xerr=[float(row.get("std_before_cost_tokens") or 0.0)],
            yerr=[float(row.get("std_before_accuracy") or 0.0)],
            marker=GROUP_MARKERS.get(group_size, "o"),
            color=color,
            linestyle="none",
            capsize=3,
            label=f"{GROUP_LABELS.get(group_size, group_size)} before",
        )
        ax.errorbar(
            [float(after_x)],
            [float(after_y)],
            xerr=[float(row.get("std_after_cost_tokens") or 0.0)],
            yerr=[float(row.get("std_after_accuracy") or 0.0)],
            marker=GROUP_MARKERS.get(group_size, "o"),
            markerfacecolor="white",
            color=color,
            linestyle="none",
            capsize=3,
            label=f"{GROUP_LABELS.get(group_size, group_size)} matched",
        )
        ax.annotate(
            "",
            xy=(float(after_x), float(after_y)),
            xytext=(float(before_x), float(before_y)),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2, "alpha": 0.75},
        )
    ax.set_xlabel("Memory Usage (tokens)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Memory-Matched Top-Up vs. Fixed@32")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out_dir / "memory_matched_topup_mean_std.png")
    plt.close(fig)


def plot_effective_k(plt: Any, out_dir: Path, memory_summary: List[Dict[str, Any]]) -> None:
    rows = [row for row in memory_summary if row.get("mean_effective_k") is not None]
    rows.sort(key=lambda row: int(row["group_size"]))
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    xs = list(range(len(rows)))
    ax.bar(
        xs,
        [float(row["mean_effective_k"]) for row in rows],
        yerr=[float(row.get("std_effective_k") or 0.0) for row in rows],
        color=[GROUP_COLORS.get(int(row["group_size"]), "#999999") for row in rows],
        capsize=3,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(row["group_size"])) for row in rows])
    ax.set_xlabel("Shared every X")
    ax.set_ylabel("Effective k After Matching")
    ax.set_title("Effective Attempts Under Fixed@32 Memory")
    fig.tight_layout()
    fig.savefig(out_dir / "effective_k_after_topup_mean_std.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [item.strip() for item in args.seeds.split(",") if item.strip()]

    ablation_rows = collect_ablation_rows(root, args.repeats, seeds)
    ablation_summary = aggregate_ablation(ablation_rows)
    memory_rows = collect_memory_match_rows(root, args.repeats, seeds)
    memory_summary = aggregate_memory_match(memory_rows)

    write_csv(out_dir / "per_repeat_ablation.csv", ablation_rows)
    write_csv(out_dir / "summary_ablation_mean_std.csv", ablation_summary)
    write_csv(out_dir / "per_repeat_memory_match.csv", memory_rows)
    write_csv(out_dir / "summary_memory_match_mean_std.csv", memory_summary)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "ablation": ablation_summary,
                "memory_match": memory_summary,
                "root": str(root),
                "repeats": args.repeats,
                "seeds": seeds,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    plt = configure_matplotlib()
    if ablation_summary:
        plot_passk_scaling(plt, out_dir, ablation_summary)
        plot_memory_vs_passk(plt, out_dir, ablation_summary)
        for k in (8, 16, 32):
            plot_focus_k(plt, out_dir, ablation_summary, k)
    if memory_summary:
        plot_memory_match(plt, out_dir, ablation_summary, memory_summary)
        plot_effective_k(plt, out_dir, memory_summary)

    print(f"[aggregate] wrote {out_dir}")


if __name__ == "__main__":
    main()
