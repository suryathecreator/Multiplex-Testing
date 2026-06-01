#!/usr/bin/env python3
"""Aggregate the 4-hour Multiplex scaling experiment and write final plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


SCALING_SERIES = {
    "discrete_cot_untrained": {
        "label": "Discrete CoT, untrained",
        "summary": ("discrete_cot_untrained", "summary_overall.csv"),
        "method": "standard_generation_independent",
        "color": "#1f77b4",
        "marker": "D",
    },
    "discrete_rl_trained": {
        "label": "Discrete RL, trained",
        "summary": ("discrete_rl", "summary_overall.csv"),
        "method": "standard_generation_independent",
        "color": "#ff7f0e",
        "marker": "^",
    },
    "multiplex_thinking_trained": {
        "label": "Multiplex Thinking, trained",
        "summary": ("multiplex_thinking_fixed", "summary_overall.csv"),
        "method": "baseline_independent",
        "color": "#2ca02c",
        "marker": "o",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-k", type=int, default=8)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mu = sum(values) / len(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def repeat_dirs(eval_root: Path) -> list[Path]:
    return sorted(path for path in eval_root.glob("repeat_*") if path.is_dir())


def collect_scaling(eval_root: Path) -> list[dict[str, object]]:
    buckets: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    for repeat_dir in repeat_dirs(eval_root):
        repeat_name = repeat_dir.name
        for key, spec in SCALING_SERIES.items():
            condition_dir, summary_name = spec["summary"]
            for row in read_csv(repeat_dir / condition_dir / summary_name):
                if row.get("method") != spec["method"]:
                    continue
                k_value = int(float(row["k"]))
                pass_at_k = as_float(row.get("pass_at_k"))
                cost = as_float(row.get("avg_cost_tokens"))
                if pass_at_k is None:
                    continue
                buckets[(key, k_value)].append(
                    {
                        "repeat": repeat_name,
                        "pass_at_k": pass_at_k,
                        "avg_cost_tokens": cost if cost is not None else float("nan"),
                    }
                )
    rows: list[dict[str, object]] = []
    for (key, k_value), values in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1])):
        passes = [item["pass_at_k"] for item in values]
        costs = [item["avg_cost_tokens"] for item in values if math.isfinite(item["avg_cost_tokens"])]
        rows.append(
            {
                "series": key,
                "label": SCALING_SERIES[key]["label"],
                "k": k_value,
                "repeats": len(values),
                "mean_pass_at_k": mean(passes),
                "std_pass_at_k": std(passes),
                "mean_cost_tokens": mean(costs),
                "std_cost_tokens": std(costs),
            }
        )
    return rows


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def collect_compute_normalized_shared4(eval_root: Path) -> list[dict[str, object]]:
    """Collect shared4 base and shared4+shared2 top-up points.

    The top-up points are normalized per prompt: start from shared4 base at k,
    then add shared2 top-up samples in generation order until that prompt's
    fixed-trace independent cost_at_k is reached or exceeded.
    """

    raw_rows: list[dict[str, object]] = []
    for repeat_dir in repeat_dirs(eval_root):
        repeat_name = repeat_dir.name
        branch_dir = repeat_dir / "multiplex_branch_shared4"
        fixed_rows = read_csv(branch_dir / "fixed_trace_independent" / "per_prompt_correctness.csv")
        shared_rows = read_csv(branch_dir / "shared_every_04" / "per_prompt_correctness.csv")
        fixed_by_prompt = {int(float(row["prompt_index"])): row for row in fixed_rows if row.get("prompt_index")}
        shared_by_prompt = {int(float(row["prompt_index"])): row for row in shared_rows if row.get("prompt_index")}

        topup_by_prompt: dict[int, list[dict[str, object]]] = defaultdict(list)
        for sample in read_jsonl(repeat_dir / "multiplex_shared4_shared2_topup" / "topup_samples.jsonl"):
            if not sample.get("usable_for_eval", True):
                continue
            prompt_index = int(sample["prompt_index"])
            topup_by_prompt[prompt_index].append(sample)
        for prompt_samples in topup_by_prompt.values():
            prompt_samples.sort(key=lambda item: int(item.get("sample_index", 0)))

        prompt_indices = sorted(set(fixed_by_prompt) & set(shared_by_prompt))
        for k_value in [1, 2, 4, 8]:
            pass_values: list[float] = []
            costs: list[float] = []
            for prompt_index in prompt_indices:
                shared = shared_by_prompt[prompt_index]
                pass_values.append(1.0 if as_bool(shared.get(f"pass_at_{k_value}")) else 0.0)
                cost = as_float(shared.get(f"cost_at_{k_value}"))
                if cost is not None:
                    costs.append(cost)
            if pass_values:
                raw_rows.append(
                    {
                        "repeat": repeat_name,
                        "series": "shared4_base",
                        "label": "Shared4 base",
                        "k": k_value,
                        "pass_at_k": mean(pass_values),
                        "cost_tokens": mean(costs),
                        "effective_k": k_value,
                        "normalization": "observed_shared4_cost",
                    }
                )

        for k_value in [4, 8]:
            pass_values = []
            costs = []
            effective_ks = []
            reached = []
            for prompt_index in prompt_indices:
                fixed = fixed_by_prompt[prompt_index]
                shared = shared_by_prompt[prompt_index]
                target_cost = as_float(fixed.get(f"cost_at_{k_value}"))
                current_cost = as_float(shared.get(f"cost_at_{k_value}"))
                if target_cost is None or current_cost is None:
                    continue
                correct = as_bool(shared.get(f"pass_at_{k_value}"))
                topup_count = 0
                for sample in topup_by_prompt.get(prompt_index, []):
                    if current_cost >= target_cost:
                        break
                    current_cost += float(sample.get("completion_tokens") or 0.0)
                    current_cost += float(sample.get("prefix_completion_tokens") or 0.0)
                    topup_count += 1
                    correct = correct or bool(sample.get("correct"))
                pass_values.append(1.0 if correct else 0.0)
                costs.append(current_cost)
                effective_ks.append(k_value + topup_count)
                reached.append(1.0 if current_cost >= target_cost else 0.0)
            if pass_values:
                raw_rows.append(
                    {
                        "repeat": repeat_name,
                        "series": "shared4_shared2_topup_compute_normalized",
                        "label": "Shared4 + Shared2 top-up, compute-normalized",
                        "k": k_value,
                        "pass_at_k": mean(pass_values),
                        "cost_tokens": mean(costs),
                        "effective_k": mean(effective_ks),
                        "reached_target_rate": mean(reached),
                        "normalization": f"per_prompt_fixed_trace_cost_at_{k_value}",
                    }
                )

    summary: list[dict[str, object]] = []
    buckets: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        buckets[(str(row["series"]), int(row["k"]))].append(row)
    for (series, k_value), values in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1])):
        pass_values = [float(row["pass_at_k"]) for row in values]
        costs = [float(row["cost_tokens"]) for row in values if row.get("cost_tokens") not in (None, "")]
        effective_ks = [float(row["effective_k"]) for row in values if row.get("effective_k") not in (None, "")]
        reached_values = [
            float(row["reached_target_rate"])
            for row in values
            if row.get("reached_target_rate") not in (None, "")
        ]
        summary.append(
            {
                "series": series,
                "label": str(values[0]["label"]),
                "k": k_value,
                "repeats": len(values),
                "mean_pass_at_k": mean(pass_values),
                "std_pass_at_k": std(pass_values),
                "mean_cost_tokens": mean(costs),
                "std_cost_tokens": std(costs),
                "mean_effective_k": mean(effective_ks),
                "mean_reached_target_rate": mean(reached_values),
                "normalization": str(values[0].get("normalization", "")),
            }
        )
    return summary


def collect_shared_budget(eval_root: Path, max_k: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for repeat_dir in repeat_dirs(eval_root):
        repeat_name = repeat_dir.name
        branch_rows = read_csv(repeat_dir / "multiplex_branch_shared4" / "summary_ablation.csv")
        memory_rows = read_csv(repeat_dir / "multiplex_shared4_shared2_topup" / "summary_memory_match.csv")
        for row in branch_rows:
            if int(float(row.get("k") or 0)) != max_k:
                continue
            condition = row.get("condition", "")
            if condition == "fixed_trace_independent":
                label = "Fixed independent pass@8"
            elif condition == "shared_every_4":
                label = "Shared4 base pass@8"
            else:
                continue
            pass_value = as_float(row.get("pass_at_k"))
            cost = as_float(row.get("avg_cost_tokens"))
            if pass_value is not None:
                rows.append(
                    {
                        "repeat": repeat_name,
                        "series": condition,
                        "label": label,
                        "pass_value": pass_value,
                        "cost_tokens": cost,
                        "effective_k": max_k,
                    }
                )
        for row in memory_rows:
            if int(float(row.get("group_size") or 0)) != 4:
                continue
            pass_value = as_float(row.get("after_accuracy"))
            cost = as_float(row.get("after_cost_tokens"))
            effective_k = as_float(row.get("effective_k"))
            if pass_value is not None:
                rows.append(
                    {
                        "repeat": repeat_name,
                        "series": "shared4_shared2_topup",
                        "label": "Shared4 + Shared2 top-up to fixed budget",
                        "pass_value": pass_value,
                        "cost_tokens": cost,
                        "effective_k": effective_k,
                    }
                )
    summary_rows: list[dict[str, object]] = []
    by_series: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_series[str(row["series"])].append(row)
    order = ["fixed_trace_independent", "shared_every_4", "shared4_shared2_topup"]
    for series in order:
        values = by_series.get(series, [])
        if not values:
            continue
        pass_values = [float(row["pass_value"]) for row in values]
        costs = [float(row["cost_tokens"]) for row in values if row.get("cost_tokens") not in (None, "")]
        effective_ks = [float(row["effective_k"]) for row in values if row.get("effective_k") not in (None, "")]
        summary_rows.append(
            {
                "series": series,
                "label": values[0]["label"],
                "repeats": len(values),
                "mean_pass": mean(pass_values),
                "std_pass": std(pass_values),
                "mean_cost_tokens": mean(costs),
                "std_cost_tokens": std(costs),
                "mean_effective_k": mean(effective_ks),
            }
        )
    return rows + [{"_summary": True, **row} for row in summary_rows]


def plot_scaling(
    rows: list[dict[str, object]],
    output_dir: Path,
    *,
    filename: str = "final_passk_scaling_curve.png",
    include_error_bars: bool = True,
    include_shared_series: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    plot_specs = {
        **SCALING_SERIES,
        "shared4_base": {
            "label": "Shared4 base",
            "color": "#9467bd",
            "marker": "s",
            "linestyle": "--",
        },
        "shared4_shared2_topup_compute_normalized": {
            "label": "Shared4 + Shared2 top-up, compute-normalized",
            "color": "#d62728",
            "marker": "P",
            "linestyle": "-.",
        },
    }
    if not include_shared_series:
        plot_specs = {key: spec for key, spec in plot_specs.items() if key in SCALING_SERIES}

    for key, spec in plot_specs.items():
        series_rows = [row for row in rows if row["series"] == key]
        if not series_rows:
            continue
        xs = [int(row["k"]) for row in series_rows]
        ys = [float(row["mean_pass_at_k"]) for row in series_rows]
        yerr = [float(row["std_pass_at_k"] or 0.0) for row in series_rows]
        if include_error_bars:
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                label=spec["label"],
                color=spec["color"],
                marker=spec["marker"],
                linestyle=spec.get("linestyle", "-"),
                linewidth=2.0,
                capsize=3,
            )
        else:
            ax.plot(
                xs,
                ys,
                label=spec["label"],
                color=spec["color"],
                marker=spec["marker"],
                linestyle=spec.get("linestyle", "-"),
                linewidth=2.0,
            )
    ax.set_title("AIME 2024 Pass@k Scaling, 1.5B, 4096 Reasoning Budget")
    ax.set_xlabel("k")
    ax.set_ylabel("Pass@k")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=220)
    plt.close(fig)


def plot_shared_budget(rows: list[dict[str, object]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_rows = [row for row in rows if row.get("_summary")]
    if not summary_rows:
        return
    labels = [str(row["label"]) for row in summary_rows]
    ys = [float(row["mean_pass"]) for row in summary_rows]
    yerr = [float(row["std_pass"] or 0.0) for row in summary_rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(range(len(labels)), ys, yerr=yerr, capsize=4, color=["#4c78a8", "#59a14f", "#f28e2b"])
    ax.set_title("Shared Branching Budget Match, AIME 2024, 4096 Prefix")
    ax.set_ylabel("Accuracy / Pass@8-equivalent")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "shared4_shared2_budget_match.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scaling_rows = collect_scaling(args.eval_root)
    compute_normalized_rows = collect_compute_normalized_shared4(args.eval_root)
    scaling_rows = scaling_rows + compute_normalized_rows
    shared_rows = collect_shared_budget(args.eval_root, args.max_k)
    write_csv(args.output_dir / "final_passk_scaling_curve.csv", scaling_rows)
    write_csv(args.output_dir / "shared4_shared2_compute_normalized_curve.csv", compute_normalized_rows)
    write_csv(args.output_dir / "shared4_shared2_budget_match.csv", shared_rows)
    plot_scaling(scaling_rows, args.output_dir)
    plot_scaling(
        scaling_rows,
        args.output_dir,
        filename="final_passk_scaling_curve_no_error_bars.png",
        include_error_bars=False,
    )
    plot_scaling(
        scaling_rows,
        args.output_dir,
        filename="final_passk_scaling_curve_main_methods_no_error_bars.png",
        include_error_bars=False,
        include_shared_series=False,
    )
    plot_shared_budget(shared_rows, args.output_dir)
    summary = {
        "eval_root": str(args.eval_root),
        "scaling_rows": scaling_rows,
        "shared4_shared2_compute_normalized_rows": compute_normalized_rows,
        "shared_budget_rows": shared_rows,
        "plots": [
            str(args.output_dir / "final_passk_scaling_curve.png"),
            str(args.output_dir / "final_passk_scaling_curve_no_error_bars.png"),
            str(args.output_dir / "final_passk_scaling_curve_main_methods_no_error_bars.png"),
            str(args.output_dir / "shared4_shared2_budget_match.png"),
        ],
    }
    (args.output_dir / "final_scaling_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[aggregate] wrote {args.output_dir / 'final_passk_scaling_curve.png'}")
    print(f"[aggregate] wrote {args.output_dir / 'shared4_shared2_budget_match.png'}")


if __name__ == "__main__":
    main()
