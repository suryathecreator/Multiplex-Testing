#!/usr/bin/env python3

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_PASS_KS = [1, 2, 4, 8, 16, 32]
FIXED_CONDITION = "Fixed_32"
SHARED_CONDITION = "Shared_32"
MIXED_CONDITION = "Mixed Fixed+Shared"
SANITY_TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate AIME-train Fixed_32/Shared_32 samples into pure and mixed Pass@k plots."
    )
    parser.add_argument("--root", required=True, help="Root output directory containing repeat_XX shard dirs.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seeds", default="409600,409601,409602")
    parser.add_argument("--max-k", type=int, default=32)
    parser.add_argument("--sanity-tolerance", type=float, default=SANITY_TOLERANCE)
    return parser.parse_args()


def pass_ks_for(max_k: int) -> List[int]:
    return [value for value in DEFAULT_PASS_KS if value <= max_k]


def mixed_ks_for(max_k: int) -> List[int]:
    return [value for value in pass_ks_for(max_k) if value >= 2 and value % 2 == 0]


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
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
        value_float = float(value)
    except Exception:
        return None
    if math.isnan(value_float):
        return None
    return value_float


def to_bool_or_none(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def mean_std(values: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None, 0
    if len(clean) == 1:
        return clean[0], 0.0, 1
    return statistics.mean(clean), statistics.stdev(clean), len(clean)


def row_has_k(row: Dict[str, Any], k: int) -> bool:
    return to_bool_or_none(row.get(f"pass_at_{k}")) is not None and to_float(row.get(f"cost_at_{k}")) is not None


def row_has_full_k(row: Dict[str, Any], max_k: int) -> bool:
    usable = int(row.get("usable_sample_count") or 0)
    return usable >= max_k and row_has_k(row, max_k)


def extract_condition_summary(shard_dir: Path, condition: str) -> Optional[Dict[str, Any]]:
    summary_ablation = read_json(shard_dir / "summary_ablation.json")
    if summary_ablation:
        if condition == "fixed":
            baseline = summary_ablation.get("baseline")
            if isinstance(baseline, dict):
                return baseline
        elif condition == "shared32":
            groups = summary_ablation.get("groups")
            if isinstance(groups, dict):
                group_summary = groups.get("32") or groups.get(32)
                if isinstance(group_summary, dict):
                    return group_summary

    if condition == "fixed":
        return read_json(shard_dir / "fixed_trace_independent" / "summary.json")
    if condition == "shared32":
        return read_json(shard_dir / "shared_every_32" / "summary.json")
    raise ValueError(f"unknown condition: {condition}")


def merge_prompt_rows(
    existing: Dict[int, Dict[str, Any]],
    rows: Iterable[Dict[str, Any]],
    *,
    source: Path,
) -> None:
    for row in rows:
        if "prompt_index" not in row:
            continue
        prompt_index = int(row["prompt_index"])
        if prompt_index in existing:
            raise ValueError(
                f"duplicate prompt_index={prompt_index} while merging {source}; "
                "prompt shards must be disjoint"
            )
        existing[prompt_index] = dict(row)


def load_condition_rows(repeat_dir: Path, condition: str) -> Dict[int, Dict[str, Any]]:
    pattern = "fixed32_shard*" if condition == "fixed" else "shared32_shard*"
    merged: Dict[int, Dict[str, Any]] = {}
    for shard_dir in sorted(path for path in repeat_dir.glob(pattern) if path.is_dir()):
        summary = extract_condition_summary(shard_dir, condition)
        if not summary:
            continue
        rows = summary.get("all_per_prompt_rows") or summary.get("per_prompt_rows") or []
        merge_prompt_rows(merged, rows, source=shard_dir)
    return merged


def aggregate_prompt_rows(rows: Sequence[Dict[str, Any]], group_fields: Sequence[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(tuple(row[field] for field in group_fields), []).append(row)

    out: List[Dict[str, Any]] = []
    for key, bucket in sorted(buckets.items()):
        row = {field: value for field, value in zip(group_fields, key)}
        accuracy, _, _ = mean_std(1.0 if item["correct"] else 0.0 for item in bucket)
        cost, _, _ = mean_std(to_float(item.get("cost_tokens")) for item in bucket)
        row.update(
            {
                "num_prompts": len(bucket),
                "accuracy": accuracy,
                "cost_tokens": cost,
            }
        )
        component_fields = [
            "fixed_component_correct",
            "shared_component_correct",
            "fixed_component_cost_tokens",
            "shared_component_cost_tokens",
        ]
        for field in component_fields:
            if field not in bucket[0]:
                continue
            if field.endswith("_correct"):
                value, _, _ = mean_std(1.0 if item[field] else 0.0 for item in bucket)
            else:
                value, _, _ = mean_std(to_float(item.get(field)) for item in bucket)
            row[field.replace("_correct", "_accuracy")] = value
        out.append(row)
    return out


def compute_repeat_metrics(
    *,
    repeat_index: int,
    seed: str,
    fixed_rows: Dict[int, Dict[str, Any]],
    shared_rows: Dict[int, Dict[str, Any]],
    max_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    pure_prompt_rows: List[Dict[str, Any]] = []
    mixed_prompt_rows: List[Dict[str, Any]] = []
    pass_ks = pass_ks_for(max_k)
    mixed_ks = mixed_ks_for(max_k)
    common_prompts = sorted(
        prompt_index
        for prompt_index in set(fixed_rows) & set(shared_rows)
        if row_has_full_k(fixed_rows[prompt_index], max_k)
        and row_has_full_k(shared_rows[prompt_index], max_k)
    )

    for prompt_index in common_prompts:
        fixed = fixed_rows[prompt_index]
        shared = shared_rows[prompt_index]
        for condition, source_row in ((FIXED_CONDITION, fixed), (SHARED_CONDITION, shared)):
            for k in pass_ks:
                pure_prompt_rows.append(
                    {
                        "repeat_index": repeat_index,
                        "seed": seed,
                        "prompt_index": prompt_index,
                        "condition": condition,
                        "k": k,
                        "correct": bool(to_bool_or_none(source_row.get(f"pass_at_{k}"))),
                        "cost_tokens": to_float(source_row.get(f"cost_at_{k}")),
                    }
                )
        for k in mixed_ks:
            component_k = k // 2
            fixed_correct = bool(to_bool_or_none(fixed.get(f"pass_at_{component_k}")))
            shared_correct = bool(to_bool_or_none(shared.get(f"pass_at_{component_k}")))
            fixed_cost = to_float(fixed.get(f"cost_at_{component_k}"))
            shared_cost = to_float(shared.get(f"cost_at_{component_k}"))
            mixed_prompt_rows.append(
                {
                    "repeat_index": repeat_index,
                    "seed": seed,
                    "prompt_index": prompt_index,
                    "condition": MIXED_CONDITION,
                    "k": k,
                    "component_k": component_k,
                    "correct": fixed_correct or shared_correct,
                    "cost_tokens": (
                        None if fixed_cost is None or shared_cost is None else fixed_cost + shared_cost
                    ),
                    "fixed_component_correct": fixed_correct,
                    "shared_component_correct": shared_correct,
                    "fixed_component_cost_tokens": fixed_cost,
                    "shared_component_cost_tokens": shared_cost,
                }
            )

    per_repeat_pure = aggregate_prompt_rows(
        pure_prompt_rows,
        group_fields=["repeat_index", "seed", "condition", "k"],
    )
    per_repeat_mixed = aggregate_prompt_rows(
        mixed_prompt_rows,
        group_fields=["repeat_index", "seed", "condition", "k", "component_k"],
    )
    return pure_prompt_rows, mixed_prompt_rows, per_repeat_pure, per_repeat_mixed


def summarize_repeats(rows: Sequence[Dict[str, Any]], group_fields: Sequence[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(tuple(row[field] for field in group_fields), []).append(row)

    summary: List[Dict[str, Any]] = []
    metric_fields = [
        "accuracy",
        "cost_tokens",
        "num_prompts",
        "fixed_component_accuracy",
        "shared_component_accuracy",
        "fixed_component_cost_tokens",
        "shared_component_cost_tokens",
    ]
    for key, bucket in sorted(buckets.items()):
        row = {field: value for field, value in zip(group_fields, key)}
        row["num_repeats"] = len(bucket)
        for metric in metric_fields:
            if not any(metric in item for item in bucket):
                continue
            mean_value, std_value, n = mean_std(to_float(item.get(metric)) for item in bucket)
            row[f"mean_{metric}"] = mean_value
            row[f"std_{metric}"] = std_value
            row[f"n_{metric}"] = n
        summary.append(row)
    return summary


def check_mixed_sanity(
    mixed_prompt_rows: Sequence[Dict[str, Any]],
    per_repeat_mixed: Sequence[Dict[str, Any]],
    *,
    tolerance: float = SANITY_TOLERANCE,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    checks: List[Dict[str, Any]] = []

    for row in mixed_prompt_rows:
        fixed_correct = bool(row["fixed_component_correct"])
        shared_correct = bool(row["shared_component_correct"])
        expected = fixed_correct or shared_correct
        actual = bool(row["correct"])
        passed = actual == expected and (actual or not fixed_correct) and (actual or not shared_correct)
        checks.append(
            {
                "level": "prompt",
                "status": "passed" if passed else "violation",
                "repeat_index": row["repeat_index"],
                "seed": row["seed"],
                "prompt_index": row["prompt_index"],
                "k": row["k"],
                "component_k": row["component_k"],
                "fixed_component_correct": fixed_correct,
                "shared_component_correct": shared_correct,
                "mixed_correct": actual,
                "message": "" if passed else "mixed_correct does not equal fixed OR shared",
            }
        )

    for row in per_repeat_mixed:
        mixed_accuracy = to_float(row.get("accuracy"))
        fixed_accuracy = to_float(row.get("fixed_component_accuracy"))
        shared_accuracy = to_float(row.get("shared_component_accuracy"))
        passed = (
            mixed_accuracy is not None
            and fixed_accuracy is not None
            and shared_accuracy is not None
            and mixed_accuracy + tolerance >= fixed_accuracy
            and mixed_accuracy + tolerance >= shared_accuracy
        )
        checks.append(
            {
                "level": "aggregate",
                "status": "passed" if passed else "violation",
                "repeat_index": row["repeat_index"],
                "seed": row["seed"],
                "prompt_index": "",
                "k": row["k"],
                "component_k": row["component_k"],
                "fixed_component_correct": fixed_accuracy,
                "shared_component_correct": shared_accuracy,
                "mixed_correct": mixed_accuracy,
                "message": "" if passed else "mixed aggregate accuracy is below a component accuracy",
            }
        )

    violations = [row for row in checks if row["status"] != "passed"]
    return checks, violations


def configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "figure.dpi": 180,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
        }
    )
    return plt


PLOT_STYLE = {
    FIXED_CONDITION: {"label": "Fixed", "color": "#4C78A8", "marker": "o", "linestyle": "-"},
    MIXED_CONDITION: {"label": "Fixed + Shared", "color": "#E45756", "marker": "D", "linestyle": "-"},
    SHARED_CONDITION: {"label": "Shared", "color": "#54A24B", "marker": "^", "linestyle": "-"},
}


def combined_summary_rows(
    pure_summary: Sequence[Dict[str, Any]],
    mixed_summary: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in pure_summary:
        rows.append(
            {
                "condition": row["condition"],
                "k": int(row["k"]),
                "mean_accuracy": row.get("mean_accuracy"),
                "std_accuracy": row.get("std_accuracy"),
                "mean_cost_tokens": row.get("mean_cost_tokens"),
                "std_cost_tokens": row.get("std_cost_tokens"),
                "mean_num_prompts": row.get("mean_num_prompts"),
            }
        )
    for row in mixed_summary:
        rows.append(
            {
                "condition": row["condition"],
                "k": int(row["k"]),
                "component_k": int(row["component_k"]),
                "mean_accuracy": row.get("mean_accuracy"),
                "std_accuracy": row.get("std_accuracy"),
                "mean_cost_tokens": row.get("mean_cost_tokens"),
                "std_cost_tokens": row.get("std_cost_tokens"),
                "mean_num_prompts": row.get("mean_num_prompts"),
                "mean_fixed_component_accuracy": row.get("mean_fixed_component_accuracy"),
                "mean_shared_component_accuracy": row.get("mean_shared_component_accuracy"),
            }
        )
    return rows


def plot_passk_scaling(plt: Any, out_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for condition in (FIXED_CONDITION, MIXED_CONDITION, SHARED_CONDITION):
        condition_rows = sorted(
            [row for row in rows if row["condition"] == condition and row.get("mean_accuracy") is not None],
            key=lambda row: int(row["k"]),
        )
        if not condition_rows:
            continue
        style = PLOT_STYLE[condition]
        ax.errorbar(
            [int(row["k"]) for row in condition_rows],
            [float(row["mean_accuracy"]) for row in condition_rows],
            yerr=[float(row.get("std_accuracy") or 0.0) for row in condition_rows],
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            capsize=3,
            label=style["label"],
        )
    ax.set_xscale("log", base=2)
    observed_ks = sorted({int(row["k"]) for row in rows})
    ax.set_xticks(observed_ks)
    ax.set_xticklabels([str(k) for k in observed_ks])
    ax.set_xlabel("k")
    ax.set_ylabel(r"Pass@$k$")
    ax.set_title(r"AIME Train: Fixed/Shared/Mixed Pass@$k$")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "passk_scaling_mean_std.png")
    plt.close(fig)


def plot_memory_vs_passk(plt: Any, out_dir: Path, rows: Sequence[Dict[str, Any]], filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.7, 5.0))
    for condition in (FIXED_CONDITION, MIXED_CONDITION, SHARED_CONDITION):
        condition_rows = sorted(
            [
                row
                for row in rows
                if row["condition"] == condition
                and row.get("mean_accuracy") is not None
                and row.get("mean_cost_tokens") is not None
            ],
            key=lambda row: int(row["k"]),
        )
        if not condition_rows:
            continue
        style = PLOT_STYLE[condition]
        ax.errorbar(
            [float(row["mean_cost_tokens"]) for row in condition_rows],
            [float(row["mean_accuracy"]) for row in condition_rows],
            xerr=[float(row.get("std_cost_tokens") or 0.0) for row in condition_rows],
            yerr=[float(row.get("std_accuracy") or 0.0) for row in condition_rows],
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            capsize=3,
            label=style["label"],
        )
    ax.set_xlabel("Memory Usage (tokens)")
    ax.set_ylabel(r"Pass@$k$")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)


def plot_k_vs_tokens(plt: Any, out_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for condition in (FIXED_CONDITION, MIXED_CONDITION, SHARED_CONDITION):
        condition_rows = sorted(
            [row for row in rows if row["condition"] == condition and row.get("mean_cost_tokens") is not None],
            key=lambda row: int(row["k"]),
        )
        if not condition_rows:
            continue
        style = PLOT_STYLE[condition]
        ax.errorbar(
            [int(row["k"]) for row in condition_rows],
            [float(row["mean_cost_tokens"]) for row in condition_rows],
            yerr=[float(row.get("std_cost_tokens") or 0.0) for row in condition_rows],
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            capsize=3,
            label=style["label"],
        )
    observed_ks = sorted({int(row["k"]) for row in rows})
    ax.set_xscale("log", base=2)
    ax.set_xticks(observed_ks)
    ax.set_xticklabels([str(k) for k in observed_ks])
    ax.set_xlabel("k")
    ax.set_ylabel("Generated Tokens")
    ax.set_title("AIME Train: k vs. Generated Tokens")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "k_vs_generated_tokens_mean_std.png")
    plt.close(fig)


def plot_focus_k(plt: Any, out_dir: Path, rows: Sequence[Dict[str, Any]], k: int) -> None:
    focus_rows = [
        row
        for row in rows
        if int(row["k"]) == k
        and row.get("mean_accuracy") is not None
        and row.get("mean_cost_tokens") is not None
    ]
    if not focus_rows:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    for row in focus_rows:
        style = PLOT_STYLE[row["condition"]]
        ax.errorbar(
            [float(row["mean_cost_tokens"])],
            [float(row["mean_accuracy"])],
            xerr=[float(row.get("std_cost_tokens") or 0.0)],
            yerr=[float(row.get("std_accuracy") or 0.0)],
            marker=style["marker"],
            color=style["color"],
            linestyle="none",
            markersize=7,
            capsize=3,
            label=style["label"],
        )
    ax.set_xlabel("Memory Usage (tokens)")
    ax.set_ylabel(rf"Pass@{k}")
    ax.set_title(rf"AIME Train: Memory Usage vs. Pass@{k}")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"memory_vs_pass{k}_mean_std.png")
    plt.close(fig)


def plot_component_comparison(plt: Any, out_dir: Path, mixed_summary: Sequence[Dict[str, Any]]) -> None:
    rows = sorted(
        [row for row in mixed_summary if row.get("mean_accuracy") is not None],
        key=lambda row: int(row["k"]),
    )
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    xs = list(range(len(rows)))
    width = 0.24
    ax.bar(
        [x - width for x in xs],
        [float(row.get("mean_fixed_component_accuracy") or 0.0) for row in rows],
        width=width,
        color=PLOT_STYLE[FIXED_CONDITION]["color"],
        label="Fixed half",
    )
    ax.bar(
        xs,
        [float(row.get("mean_shared_component_accuracy") or 0.0) for row in rows],
        width=width,
        color=PLOT_STYLE[SHARED_CONDITION]["color"],
        label="Shared half",
    )
    ax.bar(
        [x + width for x in xs],
        [float(row.get("mean_accuracy") or 0.0) for row in rows],
        width=width,
        color=PLOT_STYLE[MIXED_CONDITION]["color"],
        label="Mixed union",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(row['k'])}\n({int(row['component_k'])}+{int(row['component_k'])})" for row in rows])
    ax.set_xlabel("Mixed Pass@k")
    ax.set_ylabel("Accuracy")
    ax.set_title("Mixed Pass@k vs. Component Half-k Accuracy")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "component_comparison_mean_std.png")
    plt.close(fig)


def write_plots(
    out_dir: Path,
    pure_summary: Sequence[Dict[str, Any]],
    mixed_summary: Sequence[Dict[str, Any]],
    max_k: int,
) -> None:
    plt = configure_matplotlib()
    rows = combined_summary_rows(pure_summary, mixed_summary)
    plot_passk_scaling(plt, out_dir, rows)
    plot_memory_vs_passk(
        plt,
        out_dir,
        rows,
        filename="primary_memory_vs_passk.png",
        title=r"Fixed@32, Mixed Fixed@16+Shared@16, and Shared@32",
    )
    plot_memory_vs_passk(
        plt,
        out_dir,
        rows,
        filename="memory_vs_passk_mean_std.png",
        title=r"AIME Train: Memory Usage vs. Pass@$k$",
    )
    plot_k_vs_tokens(plt, out_dir, rows)
    for k in mixed_ks_for(max_k):
        plot_focus_k(plt, out_dir, rows, k)
    plot_component_comparison(plt, out_dir, mixed_summary)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [item.strip() for item in args.seeds.split(",") if item.strip()]

    all_pure_prompt_rows: List[Dict[str, Any]] = []
    all_mixed_prompt_rows: List[Dict[str, Any]] = []
    all_per_repeat_pure: List[Dict[str, Any]] = []
    all_per_repeat_mixed: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []

    for repeat_index in range(args.repeats):
        repeat_dir = root / f"repeat_{repeat_index:02d}"
        seed = seeds[repeat_index] if repeat_index < len(seeds) else ""
        fixed_rows = load_condition_rows(repeat_dir, "fixed")
        shared_rows = load_condition_rows(repeat_dir, "shared32")
        pure_prompt, mixed_prompt, per_repeat_pure, per_repeat_mixed = compute_repeat_metrics(
            repeat_index=repeat_index,
            seed=seed,
            fixed_rows=fixed_rows,
            shared_rows=shared_rows,
            max_k=args.max_k,
        )
        all_pure_prompt_rows.extend(pure_prompt)
        all_mixed_prompt_rows.extend(mixed_prompt)
        all_per_repeat_pure.extend(per_repeat_pure)
        all_per_repeat_mixed.extend(per_repeat_mixed)
        coverage_rows.append(
            {
                "repeat_index": repeat_index,
                "seed": seed,
                "fixed_prompts_loaded": len(fixed_rows),
                "shared_prompts_loaded": len(shared_rows),
                "common_full_k_prompts": len({row["prompt_index"] for row in pure_prompt}),
            }
        )

    sanity_rows, violations = check_mixed_sanity(
        all_mixed_prompt_rows,
        all_per_repeat_mixed,
        tolerance=float(args.sanity_tolerance),
    )
    pure_summary = summarize_repeats(all_per_repeat_pure, group_fields=["condition", "k"])
    mixed_summary = summarize_repeats(
        all_per_repeat_mixed,
        group_fields=["condition", "k", "component_k"],
    )

    write_csv(out_dir / "per_prompt_pure.csv", all_pure_prompt_rows)
    write_csv(out_dir / "per_prompt_mixed.csv", all_mixed_prompt_rows)
    write_csv(out_dir / "per_repeat_pure.csv", all_per_repeat_pure)
    write_csv(out_dir / "per_repeat_mixed.csv", all_per_repeat_mixed)
    write_csv(out_dir / "summary_pure_mean_std.csv", pure_summary)
    write_csv(out_dir / "summary_mixed_mean_std.csv", mixed_summary)
    write_csv(out_dir / "mixed_sanity_checks.csv", sanity_rows)
    write_csv(out_dir / "coverage.csv", coverage_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "root": str(root),
                "repeats": args.repeats,
                "seeds": seeds,
                "max_k": args.max_k,
                "pure": pure_summary,
                "mixed": mixed_summary,
                "coverage": coverage_rows,
                "sanity": {
                    "num_checks": len(sanity_rows),
                    "num_violations": len(violations),
                    "tolerance": float(args.sanity_tolerance),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_plots(out_dir, pure_summary, mixed_summary, max_k=args.max_k)

    if violations:
        raise SystemExit(
            f"mixed sanity checks failed with {len(violations)} violation(s); "
            f"see {out_dir / 'mixed_sanity_checks.csv'}"
        )
    print(f"[aggregate] wrote {out_dir}")


if __name__ == "__main__":
    main()
