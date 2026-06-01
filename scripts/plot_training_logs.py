#!/usr/bin/env python3
"""Parse VERL console logs and write CPU-only training metric plots."""

import argparse
import csv
import json
import math
import re
from pathlib import Path


STEP_RE = re.compile(r"(?:^|\s)step:(?P<step>-?\d+)")
NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
NP_SCALAR_RE = re.compile(
    r"^np\.(?:float(?:16|32|64)?|int(?:8|16|32|64)?|uint(?:8|16|32|64)?)\((?P<value>.+)\)$"
)

PLOT_SPECS = {
    "training_reward_by_step.png": {
        "title": "Training Reward",
        "ylabel": "Reward",
        "metrics": ("critic/rewards/mean", "critic/score/mean"),
    },
    "training_generated_tokens_by_step.png": {
        "title": "Generated Tokens Per Step",
        "ylabel": "Generated tokens",
        "metrics": ("derived/generated_tokens_per_step",),
    },
    "training_response_length_by_step.png": {
        "title": "Response Length",
        "ylabel": "Tokens",
        "metrics": ("response_length/mean", "response_length_non_aborted/mean"),
    },
    "training_avg_generated_tokens_by_step.png": {
        "title": "Avg Generated Tokens",
        "ylabel": "Tokens",
        "metrics": ("derived/generated_tokens_per_sample", "response_length/mean", "response_length_non_aborted/mean"),
    },
    "training_total_tokens_by_step.png": {
        "title": "Total Tokens Per Step",
        "ylabel": "Tokens",
        "metrics": ("perf/total_num_tokens",),
    },
    "training_generated_tokens_per_second_by_step.png": {
        "title": "Generated Tokens Per Second",
        "ylabel": "Generated tokens / second",
        "metrics": ("derived/generated_tokens_per_second",),
    },
    "training_loss_by_step.png": {
        "title": "Policy Loss",
        "ylabel": "Loss",
        "metrics": ("actor/pg_loss",),
    },
    "training_grad_norm_by_step.png": {
        "title": "Gradient Norm",
        "ylabel": "Norm",
        "metrics": ("actor/grad_norm",),
    },
    "training_step_time_by_step.png": {
        "title": "Step Time",
        "ylabel": "Seconds",
        "metrics": ("perf/time_per_step", "timing_s/step"),
    },
}

PAIR_RUN_GROUPS = {
    "discrete_vs_multiplex": ("discrete_rl", "multiplex_thinking"),
    "shared4_modes": ("shared4_joint", "shared4_thinking_only", "shared4_answer_only"),
    "all_trained_conditions": (
        "discrete_rl",
        "multiplex_thinking",
        "shared4_joint",
        "shared4_thinking_only",
        "shared4_answer_only",
    ),
}

EXTRACTED_METRIC_CANDIDATES = {
    "reward_mean": ("critic/rewards/mean", "critic/score/mean"),
    "reward_std": ("critic/rewards/std", "critic/score/std"),
    "response_length_mean": ("response_length/mean", "response_length_non_aborted/mean"),
    "avg_generated_tokens": ("derived/generated_tokens_per_sample", "response_length/mean", "response_length_non_aborted/mean"),
    "generated_tokens_per_step": ("derived/generated_tokens_per_step",),
    "generated_tokens_per_second": ("derived/generated_tokens_per_second",),
    "total_tokens_per_step": ("perf/total_num_tokens",),
    "prompt_length_mean": ("prompt_length/mean",),
    "policy_loss": ("actor/pg_loss",),
    "entropy": ("actor/entropy", "actor/entropy_loss"),
    "kl": ("actor/ppo_kl", "actor/reward_kl_penalty"),
    "grad_norm": ("actor/grad_norm",),
    "learning_rate": ("actor/lr", "critic/lr"),
    "step_time": ("perf/time_per_step", "timing_s/step"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=LOG",
        help="Condition name and log file path. May be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-json", default="")
    return parser.parse_args(argv)


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "run"


def parse_number(raw):
    text = raw.strip().strip(",")
    np_match = NP_SCALAR_RE.match(text)
    if np_match:
        text = np_match.group("value").strip().strip(",")
    if text in {"nan", "NaN", "inf", "-inf", "Infinity", "-Infinity"}:
        return float(text.replace("Infinity", "inf"))
    if NUMERIC_RE.match(text):
        return float(text)
    return None


def parse_console_metric_line(line):
    match = STEP_RE.search(line)
    if not match:
        return None
    row = {"step": int(match.group("step"))}
    metric_text = line[match.end():].strip()
    if metric_text.startswith("- "):
        metric_text = metric_text[2:]
    for part in metric_text.split(" - "):
        if ":" not in part:
            continue
        key, raw_value = part.split(":", 1)
        key = key.strip()
        value = parse_number(raw_value)
        if key and value is not None and math.isfinite(value):
            row[key] = value
    return row if len(row) > 1 else None


def parse_json_metric_line(line):
    try:
        payload = json.loads(line)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    step = payload.get("step")
    if not isinstance(data, dict) or step is None:
        return None
    row = {"step": int(step)}
    for key, value in data.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            row[str(key)] = float(value)
    return row if len(row) > 1 else None


def parse_log(path):
    by_step = {}
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            row = parse_json_metric_line(line)
            if row is None:
                row = parse_console_metric_line(line)
            if row is None:
                continue
            step = row.pop("step")
            existing = by_step.setdefault(step, {"step": step})
            existing.update(row)
    return [by_step[step] for step in sorted(by_step)]


def write_csv(path, rows):
    fieldnames = ["step"]
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    for key in sorted(all_keys):
        if key != "step":
            fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def first_present_metric(rows, candidates):
    for metric in candidates:
        if any(metric in row for row in rows):
            return metric
    return None


def add_derived_metrics(rows):
    for row in rows:
        response_mean = row.get("response_length/mean")
        if response_mean is None:
            response_mean = row.get("response_length_non_aborted/mean")
        prompt_mean = row.get("prompt_length/mean")
        total_tokens = row.get("perf/total_num_tokens")
        step_time = row.get("perf/time_per_step", row.get("timing_s/step"))

        if response_mean is not None:
            row["derived/generated_tokens_per_sample"] = response_mean
        if response_mean is not None and prompt_mean is not None and total_tokens is not None:
            denom = response_mean + prompt_mean
            if denom > 0:
                generated_tokens = total_tokens * response_mean / denom
                row["derived/generated_tokens_per_step"] = generated_tokens
                if step_time is not None and step_time > 0:
                    row["derived/generated_tokens_per_second"] = generated_tokens / step_time


def plot_metric(output_path, runs, spec, plt):
    plotted = False
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    used_metrics = {}
    for name, rows in runs.items():
        metric = first_present_metric(rows, spec["metrics"])
        if metric is None:
            continue
        xs = [row["step"] for row in rows if metric in row]
        ys = [row[metric] for row in rows if metric in row]
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=name)
        used_metrics[name] = metric
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_title(spec["title"])
    ax.set_xlabel("Training step")
    ax.set_ylabel(spec["ylabel"])
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"path": str(output_path), "metrics": used_metrics}


def plot_metric_for_selected_runs(output_path, runs, spec, plt, selected_names):
    selected_runs = {name: runs[name] for name in selected_names if name in runs}
    if not selected_runs:
        return None
    return plot_metric(output_path, selected_runs, spec, plt)


def plot_single_run_metrics(output_dir, runs, plt):
    results = {}
    for run_name, rows in runs.items():
        for filename, spec in PLOT_SPECS.items():
            metric = first_present_metric(rows, spec["metrics"])
            if metric is None:
                continue
            output_path = output_dir / f"{Path(filename).stem}_{safe_name(run_name)}.png"
            result = plot_metric(output_path, {run_name: rows}, spec, plt)
            if result is not None:
                results[f"{Path(filename).stem}_{safe_name(run_name)}.png"] = result
    return results


def plot_paired_metric_groups(output_dir, runs, plt):
    results = {}
    focus_specs = {
        "reward": PLOT_SPECS["training_reward_by_step.png"],
        "generated_tokens": PLOT_SPECS["training_generated_tokens_by_step.png"],
        "step_time": PLOT_SPECS["training_step_time_by_step.png"],
    }
    for group_name, selected_names in PAIR_RUN_GROUPS.items():
        for metric_name, spec in focus_specs.items():
            output_path = output_dir / f"training_{metric_name}_{group_name}.png"
            result = plot_metric_for_selected_runs(output_path, runs, spec, plt, selected_names)
            if result is not None:
                results[output_path.name] = result
    return results


def plot_combined(output_path, runs, plt):
    combined_specs = [
        ("Reward", ("critic/rewards/mean", "critic/score/mean")),
        ("Generated tokens", ("derived/generated_tokens_per_step",)),
        ("Avg generated tokens", ("derived/generated_tokens_per_sample", "response_length/mean", "response_length_non_aborted/mean")),
        ("Step time", ("perf/time_per_step", "timing_s/step")),
        ("Generated tokens/sec", ("derived/generated_tokens_per_second",)),
        ("Policy loss", ("actor/pg_loss",)),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 11.0))
    plotted_any = False
    used_metrics = {}
    for ax, (title, metrics) in zip(axes.flat, combined_specs):
        plotted = False
        for name, rows in runs.items():
            metric = first_present_metric(rows, metrics)
            if metric is None:
                continue
            xs = [row["step"] for row in rows if metric in row]
            ys = [row[metric] for row in rows if metric in row]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", linewidth=1.6, label=name)
            used_metrics.setdefault(title, {})[name] = metric
            plotted = True
            plotted_any = True
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.grid(True, alpha=0.25)
        if plotted:
            ax.legend(fontsize=8)
    if not plotted_any:
        plt.close(fig)
        return None
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"path": str(output_path), "metrics": used_metrics}


def main_from_args(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    csv_paths = {}
    missing_logs = []
    for item in args.run:
        if "=" not in item:
            raise ValueError(f"--run must be NAME=LOG, got {item!r}")
        name, raw_path = item.split("=", 1)
        log_path = Path(raw_path)
        if not log_path.exists():
            missing_logs.append(str(log_path))
        rows = parse_log(log_path)
        add_derived_metrics(rows)
        runs[name] = rows
        csv_path = output_dir / f"training_metrics_{safe_name(name)}.csv"
        write_csv(csv_path, rows)
        csv_paths[name] = str(csv_path)

    summary = {
        "csv_paths": csv_paths,
        "num_rows": {name: len(rows) for name, rows in runs.items()},
        "missing_logs": missing_logs,
        "plots": {},
        "missing_metric_groups": {},
    }
    for name, rows in runs.items():
        summary["available_extracted_metrics"] = summary.get("available_extracted_metrics", {})
        summary["available_extracted_metrics"][name] = {
            label: first_present_metric(rows, candidates)
            for label, candidates in EXTRACTED_METRIC_CANDIDATES.items()
            if first_present_metric(rows, candidates) is not None
        }

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        summary["plot_error"] = f"{type(exc).__name__}: {exc}"
    else:
        for filename, spec in PLOT_SPECS.items():
            result = plot_metric(output_dir / filename, runs, spec, plt)
            if result is None:
                summary["missing_metric_groups"][filename] = list(spec["metrics"])
            else:
                summary["plots"][filename] = result
        summary["single_run_plots"] = plot_single_run_metrics(output_dir, runs, plt)
        summary["paired_metric_plots"] = plot_paired_metric_groups(output_dir, runs, plt)
        combined = plot_combined(output_dir / "training_combined_comparison.png", runs, plt)
        if combined is None:
            summary["missing_metric_groups"]["training_combined_comparison.png"] = [
                "critic/rewards/mean",
                "response_length/mean",
                "actor/pg_loss",
                "perf/time_per_step",
            ]
        else:
            summary["plots"]["training_combined_comparison.png"] = combined

    summary_path = Path(args.summary_json) if args.summary_json else output_dir / "training_plot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[plots] wrote summary={summary_path}")
    for name, path in csv_paths.items():
        print(f"[plots] {name} csv={path} rows={len(runs[name])}")


if __name__ == "__main__":
    main_from_args()
