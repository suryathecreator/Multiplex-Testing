#!/usr/bin/env python3
"""Backfill shared_every_2 budget-matched Pass@k plots from existing samples."""

from __future__ import annotations

import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


PASS_KS = (4, 8, 16, 32)
TABLE_PASS_KS = PASS_KS
LATEX_TABLE_KS = (32, 16, 8, 4)
REPEATS = (0, 1, 2)

ROOT = Path(
    "final_eval_outputs/"
    "aime-train100-ablation4096-r3-stable3b-20260501-103533"
)
OUT = ROOT / "shared2_budget_backfill_graphs"
ALL_POINTS_NO_ARROWS_OUT = OUT / "all_points_no_arrows"
NO_ARROW_ACCURACY_OUT = ALL_POINTS_NO_ARROWS_OUT / "accuracy_matching_points"
NO_ARROW_BUDGET_OUT = ALL_POINTS_NO_ARROWS_OUT / "budget_matching_points"
NO_ARROW_COMBINED_OUT = ALL_POINTS_NO_ARROWS_OUT / "all_points_together"
SHARED2_TOPUP_COMPARISON_OUT = OUT / "shared2_topup_comparison_arrows"
SHARED2_SHARED4_COMPARISON_OUT = OUT / "shared2_shared4_topup_arrows"
GRID_SEARCH_COMPARISON_OUT = OUT / "topup_grid_search_arrows"
LATEST_OUT = OUT / "Latest tables and graphs"
LATEST_SHARED2_TOPUP_COMPARISON_OUT = LATEST_OUT / "shared2_topup_comparison"
LATEST_SHARED2_SHARED4_COMPARISON_OUT = LATEST_OUT / "shared2_shared4_topup_comparison"
LATEST_GRID_SEARCH_COMPARISON_OUT = LATEST_OUT / "topup_grid_search_comparison"
PASS32_CHECKPOINT = (
    ROOT
    / "checkpoint_plots"
    / "memory_match_topup_shared2_sharedgen_fast"
    / "checkpoint_0300_memory_vs_pass32.csv"
)

CONDITIONS = [
    {
        "key": "fixed",
        "label": "Fixed",
        "color": "#4C78A8",
        "marker": "circle",
        "path": ("bundle_fixed", "fixed_trace_independent"),
        "method": "baseline_independent",
    },
    {
        "key": "shared2",
        "label": "Shared every 2",
        "color": "#F58518",
        "marker": "square",
        "path": ("bundle_shared_low", "shared_every_02"),
        "method": "shared_trace_group_2",
    },
    {
        "key": "shared2_topup",
        "label": "Shared every 2 + shared2 top-up",
        "color": "#D55E00",
        "marker": "diamond",
        "computed": True,
        "base_key": "shared2",
        "topup_dir": "memory_match_topup_shared2_sharedgen_fast",
    },
    {
        "key": "shared2_accuracy_match",
        "label": "Shared every 2 + shared2 top-up accuracy match",
        "color": "#111111",
        "marker": "smallcircle",
        "computed": True,
        "accuracy_match_for": "shared2_topup",
    },
    {
        "key": "shared2_fixed_topup",
        "label": "Shared every 2 + fixed top-up",
        "color": "#CC6677",
        "marker": "diamond",
        "computed": True,
        "base_key": "shared2",
        "topup_dir": "memory_match_topup_shared2_fixedgen_fast",
    },
    {
        "key": "shared2_fixed_accuracy_match",
        "label": "Shared every 2 + fixed top-up accuracy match",
        "color": "#7B3144",
        "marker": "smallcircle",
        "computed": True,
        "accuracy_match_for": "shared2_fixed_topup",
    },
    {
        "key": "shared2_shared4_topup",
        "label": "Shared every 2 + shared4 top-up",
        "color": "#0072B2",
        "marker": "diamond",
        "computed": True,
        "base_key": "shared2",
        "topup_dir": "memory_match_topup_shared2_shared4_fast",
    },
    {
        "key": "shared2_shared4_accuracy_match",
        "label": "Shared every 2 + shared4 top-up accuracy match",
        "color": "#004F7C",
        "marker": "smallcircle",
        "computed": True,
        "accuracy_match_for": "shared2_shared4_topup",
    },
    {
        "key": "shared2_shared8_topup",
        "label": "Shared every 2 + shared8 top-up",
        "color": "#8E6BBE",
        "marker": "diamond",
        "computed": True,
        "base_key": "shared2",
        "topup_dir": "memory_match_topup_shared2_shared8_fast",
    },
    {
        "key": "shared2_shared8_accuracy_match",
        "label": "Shared every 2 + shared8 top-up accuracy match",
        "color": "#5E4A86",
        "marker": "smallcircle",
        "computed": True,
        "accuracy_match_for": "shared2_shared8_topup",
    },
    {
        "key": "shared4",
        "label": "Shared every 4",
        "color": "#54A24B",
        "marker": "triangle",
        "path": ("bundle_shared_low", "shared_every_04"),
        "method": "shared_trace_group_4",
    },
    {
        "key": "shared4_fixed_topup",
        "label": "Shared every 4 + fixed top-up",
        "color": "#CC79A7",
        "marker": "diamond",
        "computed": True,
        "base_key": "shared4",
        "topup_dir": "memory_match_topup_shared4_fixedgen_fast",
    },
    {
        "key": "shared4_fixed_accuracy_match",
        "label": "Shared every 4 + fixed top-up accuracy match",
        "color": "#8F4E76",
        "marker": "smallcircle",
        "computed": True,
        "accuracy_match_for": "shared4_fixed_topup",
    },
    {
        "key": "shared4_shared2_topup",
        "label": "Shared every 4 + shared2 top-up",
        "color": "#228833",
        "marker": "diamond",
        "computed": True,
        "base_key": "shared4",
        "topup_dir": "memory_match_topup_shared4_shared2_fast",
    },
    {
        "key": "shared4_shared2_accuracy_match",
        "label": "Shared every 4 + shared2 top-up accuracy match",
        "color": "#176124",
        "marker": "smallcircle",
        "computed": True,
        "accuracy_match_for": "shared4_shared2_topup",
    },
    {
        "key": "shared4_shared4_topup",
        "label": "Shared every 4 + shared4 top-up",
        "color": "#009E73",
        "marker": "diamond",
        "computed": True,
        "base_key": "shared4",
        "topup_dir": "memory_match_topup_shared4_shared4_fast",
    },
    {
        "key": "shared4_shared4_accuracy_match",
        "label": "Shared every 4 + shared4 top-up accuracy match",
        "color": "#006B57",
        "marker": "smallcircle",
        "computed": True,
        "accuracy_match_for": "shared4_shared4_topup",
    },
    {
        "key": "shared4_shared8_topup",
        "label": "Shared every 4 + shared8 top-up",
        "color": "#44AA99",
        "marker": "diamond",
        "computed": True,
        "base_key": "shared4",
        "topup_dir": "memory_match_topup_shared4_shared8_fast",
    },
    {
        "key": "shared4_shared8_accuracy_match",
        "label": "Shared every 4 + shared8 top-up accuracy match",
        "color": "#2A776B",
        "marker": "smallcircle",
        "computed": True,
        "accuracy_match_for": "shared4_shared8_topup",
    },
    {
        "key": "shared8",
        "label": "Shared every 8",
        "color": "#B279A2",
        "marker": "star",
        "path": ("bundle_shared_low", "shared_every_08"),
        "method": "shared_trace_group_8",
    },
    {
        "key": "shared8_fixed_topup",
        "label": "Shared every 8 + fixed top-up",
        "color": "#AA3377",
        "marker": "diamond",
        "computed": True,
        "base_key": "shared8",
        "topup_dir": "memory_match_topup_shared8_fixedgen_fast",
    },
    {
        "key": "shared8_fixed_accuracy_match",
        "label": "Shared every 8 + fixed top-up accuracy match",
        "color": "#742350",
        "marker": "smallcircle",
        "computed": True,
        "accuracy_match_for": "shared8_fixed_topup",
    },
    {
        "key": "shared8_shared2_topup",
        "label": "Shared every 8 + shared2 top-up",
        "color": "#882255",
        "marker": "diamond",
        "computed": True,
        "base_key": "shared8",
        "topup_dir": "memory_match_topup_shared8_shared2_fast",
    },
    {
        "key": "shared8_shared2_accuracy_match",
        "label": "Shared every 8 + shared2 top-up accuracy match",
        "color": "#5F183B",
        "marker": "smallcircle",
        "computed": True,
        "accuracy_match_for": "shared8_shared2_topup",
    },
    {
        "key": "shared8_shared4_topup",
        "label": "Shared every 8 + shared4 top-up",
        "color": "#7A5195",
        "marker": "diamond",
        "computed": True,
        "base_key": "shared8",
        "topup_dir": "memory_match_topup_shared8_shared4_fast",
    },
    {
        "key": "shared8_shared4_accuracy_match",
        "label": "Shared every 8 + shared4 top-up accuracy match",
        "color": "#5B3C79",
        "marker": "smallcircle",
        "computed": True,
        "accuracy_match_for": "shared8_shared4_topup",
    },
    {
        "key": "shared8_shared8_topup",
        "label": "Shared every 8 + shared8 top-up",
        "color": "#A66C9A",
        "marker": "diamond",
        "computed": True,
        "base_key": "shared8",
        "topup_dir": "memory_match_topup_shared8_shared8_fast",
    },
    {
        "key": "shared8_shared8_accuracy_match",
        "label": "Shared every 8 + shared8 top-up accuracy match",
        "color": "#73476C",
        "marker": "smallcircle",
        "computed": True,
        "accuracy_match_for": "shared8_shared8_topup",
    },
    {
        "key": "shared16",
        "label": "Shared every 16",
        "color": "#E45756",
        "marker": "plus",
        "path": ("bundle_shared_16", "shared_every_16"),
        "method": "shared_trace_group_16",
    },
    {
        "key": "shared16_shared16_topup",
        "label": "Shared every 16 + shared16 top-up",
        "color": "#B33A3A",
        "marker": "diamond",
        "computed": True,
        "base_key": "shared16",
        "topup_dir": "memory_match_topup_shared16_shared16_fast",
    },
    {
        "key": "shared16_shared16_accuracy_match",
        "label": "Shared every 16 + shared16 top-up accuracy match",
        "color": "#7C2525",
        "marker": "smallcircle",
        "computed": True,
        "accuracy_match_for": "shared16_shared16_topup",
    },
    {
        "key": "shared32",
        "label": "Shared every 32",
        "color": "#72B7B2",
        "marker": "x",
        "path": ("bundle_shared_32", "shared_every_32"),
        "method": "shared_trace_group_32",
    },
]

SHARED2_TOPUP_SPECS = (
    ("shared2", "shared2_fixed_topup"),
    ("shared2", "shared2_topup"),
    ("shared2", "shared2_shared4_topup"),
    ("shared2", "shared2_shared8_topup"),
)
SHARED2_SHARED4_TOPUP_SPECS = (
    ("shared2", "shared2_topup"),
    ("shared4", "shared4_shared4_topup"),
)
GRID_SEARCH_TOPUP_SPECS = (
    ("shared2", "shared2_fixed_topup"),
    ("shared2", "shared2_topup"),
    ("shared2", "shared2_shared4_topup"),
    ("shared2", "shared2_shared8_topup"),
    ("shared4", "shared4_fixed_topup"),
    ("shared4", "shared4_shared2_topup"),
    ("shared4", "shared4_shared4_topup"),
    ("shared4", "shared4_shared8_topup"),
    ("shared8", "shared8_fixed_topup"),
    ("shared8", "shared8_shared2_topup"),
    ("shared8", "shared8_shared4_topup"),
    ("shared8", "shared8_shared8_topup"),
    ("shared16", "shared16_shared16_topup"),
)
STATIC_CONDITION_KEYS = (
    "fixed",
    "shared2",
    "shared4",
    "shared8",
    "shared16",
    "shared32",
)
BUDGET_MATCH_CONDITION_KEYS = STATIC_CONDITION_KEYS + (
    "shared2_topup",
    "shared2_fixed_topup",
    "shared2_shared4_topup",
    "shared2_shared8_topup",
    "shared4_fixed_topup",
    "shared4_shared2_topup",
    "shared4_shared4_topup",
    "shared4_shared8_topup",
    "shared8_fixed_topup",
    "shared8_shared2_topup",
    "shared8_shared4_topup",
    "shared8_shared8_topup",
    "shared16_shared16_topup",
)
ACCURACY_MATCH_CONDITION_KEYS = STATIC_CONDITION_KEYS + (
    "shared2_accuracy_match",
    "shared2_fixed_accuracy_match",
    "shared2_shared4_accuracy_match",
    "shared2_shared8_accuracy_match",
    "shared4_fixed_accuracy_match",
    "shared4_shared2_accuracy_match",
    "shared4_shared4_accuracy_match",
    "shared4_shared8_accuracy_match",
    "shared8_fixed_accuracy_match",
    "shared8_shared2_accuracy_match",
    "shared8_shared4_accuracy_match",
    "shared8_shared8_accuracy_match",
    "shared16_shared16_accuracy_match",
)


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summary_rows(condition_dir: Path, method: str) -> dict[int, dict]:
    data = read_json(condition_dir / "summary.json")
    rows = data.get("per_prompt_rows") or data.get("all_per_prompt_rows") or []
    out = {}
    for row in rows:
        if row.get("method") == method:
            out[int(row["prompt_index"])] = row
    return out


def samples_by_prompt(path: Path) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        if row.get("usable_for_eval", True) and not row.get("excluded_reason"):
            grouped[int(row["prompt_index"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["sample_index"]))
    return grouped


def topup_samples_by_prompt(topup_dir: Path) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(topup_dir.glob("shard*/topup_samples.jsonl")):
        for row in read_jsonl(path):
            if row.get("usable_for_eval", True) and not row.get("excluded_reason"):
                grouped[int(row["prompt_index"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["sample_index"]))
    return grouped


def completed_memory_match_prompt_rows(topup_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(topup_dir.glob("shard*/memory_match_prompts.jsonl")):
        for row in read_jsonl(path):
            if row.get("status") == "completed":
                rows.append(row)
    return rows


def sample_cost(row: dict) -> float:
    return float(row.get("prefix_completion_tokens") or 0) + float(
        row.get("completion_tokens") or 0
    )


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def compute_budget_topup_rows(
    repeat: int,
    k: int,
    prompts: list[int],
    fixed: dict[int, dict],
    base_rows: dict[int, dict],
    base_samples: dict[int, list[dict]],
    topup_samples: dict[int, list[dict]],
    condition: dict,
) -> dict[int, dict]:
    out = {}
    for prompt_index in prompts:
        fixed_row = fixed.get(prompt_index)
        base = base_rows.get(prompt_index)
        if not fixed_row or not base:
            continue
        target_cost = float(fixed_row[f"cost_at_{k}"])
        total_cost = float(base[f"cost_at_{k}"])
        correct = bool(base[f"pass_at_{k}"])
        extra_count = 0
        extra_sources = (
            [
                row
                for row in base_samples.get(prompt_index, [])
                if int(row["sample_index"]) >= k
            ]
            + topup_samples.get(prompt_index, [])
        )
        for sample in extra_sources:
            if total_cost >= target_cost:
                break
            total_cost += sample_cost(sample)
            correct = correct or bool(sample.get("correct"))
            extra_count += 1
        out[prompt_index] = {
            "repeat": repeat,
            "prompt_index": prompt_index,
            "pass_k": k,
            "condition": condition["key"],
            "label": condition["label"],
            "pass": float(correct),
            "cost": total_cost,
            "target_fixed_cost": target_cost,
            "extra_generations": extra_count,
            "effective_k": k + extra_count,
        }
    return out


def compute_rows(root: Path) -> tuple[list[dict], list[dict]]:
    prompt_rows: list[dict] = []
    accuracy_entries: dict[tuple[int, str], list[dict]] = defaultdict(list)
    static_conditions = [
        condition for condition in CONDITIONS if not condition.get("computed")
    ]
    topup_conditions = [condition for condition in CONDITIONS if condition.get("topup_dir")]
    accuracy_conditions = [
        condition for condition in CONDITIONS if condition.get("accuracy_match_for")
    ]
    condition_by_key = {condition["key"]: condition for condition in CONDITIONS}

    for repeat in REPEATS:
        repeat_dir = root / f"repeat_{repeat:02d}"
        static_rows = {}
        static_samples = {}
        for condition in static_conditions:
            condition_dir = repeat_dir.joinpath(*condition["path"])
            static_rows[condition["key"]] = summary_rows(
                condition_dir, condition["method"]
            )
            static_samples[condition["key"]] = samples_by_prompt(
                condition_dir / "samples.jsonl"
            )

        fixed = static_rows["fixed"]
        topup_samples_by_condition = {
            condition["key"]: topup_samples_by_prompt(repeat_dir / condition["topup_dir"])
            for condition in topup_conditions
        }
        available_topup_keys = {
            key for key, samples in topup_samples_by_condition.items() if samples
        }

        prompts = sorted(fixed)
        for k in PASS_KS:
            topup_rows_by_condition = {}
            for condition in topup_conditions:
                if condition["key"] not in available_topup_keys:
                    continue
                base_key = condition["base_key"]
                topup_rows_by_condition[condition["key"]] = compute_budget_topup_rows(
                    repeat,
                    k,
                    prompts,
                    fixed,
                    static_rows.get(base_key, {}),
                    static_samples.get(base_key, {}),
                    topup_samples_by_condition[condition["key"]],
                    condition,
                )
            for condition in accuracy_conditions:
                topup_condition = condition_by_key[condition["accuracy_match_for"]]
                if topup_condition["key"] not in available_topup_keys:
                    continue
                base_key = topup_condition["base_key"]
                accuracy_prompts = sorted(set(fixed) & set(static_rows.get(base_key, {})))
                accuracy_entries[(k, condition["key"])].extend(
                    build_accuracy_match_entries(
                        repeat,
                        k,
                        accuracy_prompts,
                        fixed,
                        static_rows.get(base_key, {}),
                        topup_samples_by_condition[topup_condition["key"]],
                        condition,
                    )
                )
            for prompt_index in prompts:
                target_cost = float(fixed[prompt_index][f"cost_at_{k}"])
                for condition in CONDITIONS:
                    key = condition["key"]
                    if condition.get("accuracy_match_for"):
                        continue
                    if condition.get("topup_dir"):
                        row = topup_rows_by_condition.get(key, {}).get(prompt_index)
                        if row:
                            prompt_rows.append(row)
                        continue

                    row = static_rows[key].get(prompt_index)
                    if not row:
                        continue
                    prompt_rows.append(
                        {
                            "repeat": repeat,
                            "prompt_index": prompt_index,
                            "pass_k": k,
                            "condition": key,
                            "label": condition["label"],
                            "pass": float(bool(row[f"pass_at_{k}"])),
                            "cost": float(row[f"cost_at_{k}"]),
                            "target_fixed_cost": target_cost,
                            "extra_generations": 0,
                            "effective_k": k,
                        }
                    )

    for (k, condition_key), entries in sorted(accuracy_entries.items()):
        condition = condition_by_key[condition_key]
        prompt_rows.extend(compute_global_accuracy_match_rows(k, entries, condition))

    return prompt_rows, build_repeat_rows(prompt_rows)


def build_extra_sources(
    prompt_index: int,
    k: int,
    base_samples: dict[int, list[dict]],
    topup_samples: dict[int, list[dict]],
) -> list[dict]:
    return [
        row
        for row in base_samples.get(prompt_index, [])
        if int(row["sample_index"]) >= k
    ] + topup_samples.get(prompt_index, [])


def build_accuracy_match_entries(
    repeat: int,
    k: int,
    prompts: list[int],
    fixed: dict[int, dict],
    base_rows: dict[int, dict],
    topup_samples: dict[int, list[dict]],
    condition: dict,
) -> list[dict]:
    """Build per-prompt ladders using base Pass@k plus actual top-up samples only."""
    entries = []
    for prompt in prompts:
        fixed_row = fixed[prompt]
        base = base_rows[prompt]
        states = [
            {
                "cost": float(base[f"cost_at_{k}"]),
                "pass": float(bool(base[f"pass_at_{k}"])),
                "extra_generations": 0,
            }
        ]
        cost = states[0]["cost"]
        correct = bool(states[0]["pass"])
        for i, sample in enumerate(topup_samples.get(prompt, []), start=1):
            cost += sample_cost(sample)
            correct = correct or bool(sample.get("correct"))
            states.append(
                {
                    "cost": cost,
                    "pass": float(correct),
                    "extra_generations": i,
                }
            )
        entries.append(
            {
                "repeat": repeat,
                "prompt_index": prompt,
                "pass_k": k,
                "condition": condition["key"],
                "label": condition["label"],
                "target_fixed_cost": float(fixed_row[f"cost_at_{k}"]),
                "fixed_pass": float(bool(fixed_row[f"pass_at_{k}"])),
                "states": states,
            }
        )
    return entries


def compute_global_accuracy_match_rows(
    k: int, entries: list[dict], condition: dict
) -> list[dict]:
    """Choose one uniform top-up depth whose aggregate accuracy reaches fixed."""
    if not entries:
        return []
    fixed_accuracy = mean([entry["fixed_pass"] for entry in entries])
    max_extra = max(len(entry["states"]) - 1 for entry in entries)
    chosen_step = max_extra
    for step in range(max_extra + 1):
        accuracy = mean(
            [
                entry["states"][min(step, len(entry["states"]) - 1)]["pass"]
                for entry in entries
            ]
        )
        if accuracy >= fixed_accuracy:
            chosen_step = step
            break

    rows = []
    for entry in entries:
        state = entry["states"][min(chosen_step, len(entry["states"]) - 1)]
        rows.append(
            {
                "repeat": entry["repeat"],
                "prompt_index": entry["prompt_index"],
                "pass_k": k,
                "condition": condition["key"],
                "label": condition["label"],
                "pass": state["pass"],
                "cost": state["cost"],
                "target_fixed_cost": entry["target_fixed_cost"],
                "extra_generations": state["extra_generations"],
                "effective_k": k + state["extra_generations"],
            }
        )
    return rows


def build_repeat_rows(prompt_rows: list[dict]) -> list[dict]:
    repeat_rows = []
    for repeat in REPEATS:
        for k in PASS_KS:
            for condition in CONDITIONS:
                rows = [
                    row
                    for row in prompt_rows
                    if row["repeat"] == repeat
                    and row["pass_k"] == k
                    and row["condition"] == condition["key"]
                ]
                if rows:
                    repeat_rows.append(
                        {
                            "repeat": repeat,
                            "pass_k": k,
                            "condition": condition["key"],
                            "label": condition["label"],
                            "pass_mean": mean([row["pass"] for row in rows]),
                            "cost_mean": mean([row["cost"] for row in rows]),
                            "extra_generations_mean": mean(
                                [row["extra_generations"] for row in rows]
                            ),
                            "effective_k_mean": mean(
                                [row["effective_k"] for row in rows]
                            ),
                            "n_prompts": len(rows),
                        }
                    )
    return repeat_rows


def summarize(repeat_rows: list[dict]) -> list[dict]:
    summary = []
    for k in PASS_KS:
        for condition in CONDITIONS:
            rows = [
                row
                for row in repeat_rows
                if row["pass_k"] == k and row["condition"] == condition["key"]
            ]
            if not rows:
                continue
            summary.append(
                {
                    "pass_k": k,
                    "condition": condition["key"],
                    "label": condition["label"],
                    "pass_mean": mean([row["pass_mean"] for row in rows]),
                    "pass_std": stdev([row["pass_mean"] for row in rows]),
                    "cost_mean": mean([row["cost_mean"] for row in rows]),
                    "cost_std": stdev([row["cost_mean"] for row in rows]),
                    "extra_generations_mean": mean(
                        [row["extra_generations_mean"] for row in rows]
                    ),
                    "effective_k_mean": mean([row["effective_k_mean"] for row in rows]),
                    "n_repeats": len(rows),
                    "n_prompts_total": sum(int(row["n_prompts"]) for row in rows),
                }
            )
    return summary


def overlay_completed_pass32_checkpoint(summary: list[dict]) -> None:
    """Use the completed run's Pass@32 checkpoint for the final shared2 top-up.

    The top-up job's prompt rows are the authoritative artifact for Pass@32. The
    lower-k plots are backfilled from saved samples because no extra generation
    was required to get those estimates.
    """
    if not PASS32_CHECKPOINT.exists():
        return
    label_to_key = {
        "Fixed": "fixed",
        "Shared every 2": "shared2",
        "Shared every 2 + top-up": "shared2_topup",
    }
    with PASS32_CHECKPOINT.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        key = label_to_key.get(row["condition"])
        if not key:
            continue
        target = next(
            item
            for item in summary
            if item["pass_k"] == 32 and item["condition"] == key
        )
        target["pass_mean"] = float(row["mean_accuracy"])
        target["pass_std"] = float(row["std_accuracy"])
        target["cost_mean"] = float(row["mean_cost_tokens"])
        target["cost_std"] = float(row["std_cost_tokens"])
        target["n_repeats"] = int(float(row["num_repeats"]))
        target["n_prompts_total"] = int(float(row["total_prompt_rows"]))


def overlay_topup_pass32_from_prompt_rows(summary: list[dict], root: Path) -> None:
    by_key = {(int(row["pass_k"]), row["condition"]): row for row in summary}
    for condition in CONDITIONS:
        topup_dir = condition.get("topup_dir")
        if not topup_dir:
            continue
        per_repeat = []
        for repeat in REPEATS:
            rows = completed_memory_match_prompt_rows(
                root / f"repeat_{repeat:02d}" / topup_dir
            )
            if not rows:
                continue
            per_repeat.append(
                {
                    "pass_mean": mean(
                        [float(bool(row.get("after_correct"))) for row in rows]
                    ),
                    "cost_mean": mean(
                        [float(row.get("after_cost_tokens") or 0) for row in rows]
                    ),
                    "extra_generations_mean": mean(
                        [float(row.get("topup_count") or 0) for row in rows]
                    ),
                    "effective_k_mean": mean(
                        [float(row.get("effective_k") or 32) for row in rows]
                    ),
                    "n_prompts": len(rows),
                }
            )
        target = by_key.get((32, condition["key"]))
        if not target or not per_repeat:
            continue
        target["pass_mean"] = mean([row["pass_mean"] for row in per_repeat])
        target["pass_std"] = stdev([row["pass_mean"] for row in per_repeat])
        target["cost_mean"] = mean([row["cost_mean"] for row in per_repeat])
        target["cost_std"] = stdev([row["cost_mean"] for row in per_repeat])
        target["extra_generations_mean"] = mean(
            [row["extra_generations_mean"] for row in per_repeat]
        )
        target["effective_k_mean"] = mean(
            [row["effective_k_mean"] for row in per_repeat]
        )
        target["n_repeats"] = len(per_repeat)
        target["n_prompts_total"] = sum(int(row["n_prompts"]) for row in per_repeat)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_results_table(path: Path, rows: list[dict]) -> None:
    by_key = {(int(row["pass_k"]), row["condition"]): row for row in rows}
    lines = [
        "| Pass@k | Fixed pass | Fixed tokens | Shared2 pass | Shared2 tokens | Accuracy-match pass | Accuracy-match tokens | Accuracy-match delta vs fixed | Budget-match pass | Budget-match tokens | Budget-match delta vs fixed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in PASS_KS:
        fixed = by_key[(k, "fixed")]
        shared2 = by_key[(k, "shared2")]
        acc = by_key[(k, "shared2_accuracy_match")]
        budget = by_key[(k, "shared2_topup")]
        acc_delta = acc["cost_mean"] - fixed["cost_mean"]
        budget_delta = budget["cost_mean"] - fixed["cost_mean"]
        lines.append(
            "| "
            f"{k} | "
            f"{fixed['pass_mean']:.4f} | {fixed['cost_mean']:.0f} | "
            f"{shared2['pass_mean']:.4f} | {shared2['cost_mean']:.0f} | "
            f"{acc['pass_mean']:.4f} | {acc['cost_mean']:.0f} | {signed_token_delta(acc_delta)} | "
            f"{budget['pass_mean']:.4f} | {budget['cost_mean']:.0f} | {signed_token_delta(budget_delta)} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_results_latex(path: Path, rows: list[dict]) -> None:
    by_key = {(int(row["pass_k"]), row["condition"]): row for row in rows}
    lines = [
        "% Auto-generated by scripts/backfill_shared2_budget_plots.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Shared every 2 accuracy before and after matching.}",
        "\\begin{tabular}{rrrr}",
        "\\toprule",
        "$k$ & Fixed & Accuracy-matched & Budget-matched \\\\",
        "\\midrule",
    ]
    for k in PASS_KS:
        fixed = by_key[(k, "fixed")]
        acc = by_key[(k, "shared2_accuracy_match")]
        budget = by_key[(k, "shared2_topup")]
        lines.append(
            f"{k} & {latex_float(fixed['pass_mean'])} & "
            f"{latex_float(acc['pass_mean'])} & {latex_float(budget['pass_mean'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def condition_lookup() -> dict[str, dict]:
    return {condition["key"]: condition for condition in CONDITIONS}


def accuracy_match_key_by_topup() -> dict[str, str]:
    return {
        condition["accuracy_match_for"]: condition["key"]
        for condition in CONDITIONS
        if condition.get("accuracy_match_for")
    }


def comparison_label(key: str) -> str:
    return condition_lookup()[key]["label"]


def comparison_rows(
    rows: list[dict], specs: tuple[tuple[str, str], ...]
) -> tuple[list[dict], list[dict]]:
    by_key = {(int(row["pass_k"]), row["condition"]): row for row in rows}
    acc_by_topup = accuracy_match_key_by_topup()
    budget_rows = []
    accuracy_rows = []
    for k in PASS_KS:
        fixed = by_key[(k, "fixed")]
        for base_key, topup_key in specs:
            base = by_key.get((k, base_key))
            topup = by_key.get((k, topup_key))
            if base and topup:
                budget_rows.append(
                    {
                        "pass_k": k,
                        "base_key": base_key,
                        "topup_key": topup_key,
                        "comparison": comparison_label(topup_key),
                        "fixed_pass": fixed["pass_mean"],
                        "fixed_tokens": fixed["cost_mean"],
                        "before_condition": comparison_label(base_key),
                        "before_pass": base["pass_mean"],
                        "before_tokens": base["cost_mean"],
                        "after_pass": topup["pass_mean"],
                        "after_tokens": topup["cost_mean"],
                        "delta_tokens_vs_fixed": topup["cost_mean"]
                        - fixed["cost_mean"],
                        "extra_generations": topup["extra_generations_mean"],
                        "effective_k": topup["effective_k_mean"],
                    }
                )
            acc_key = acc_by_topup.get(topup_key)
            acc = by_key.get((k, acc_key)) if acc_key else None
            if base and acc:
                accuracy_rows.append(
                    {
                        "pass_k": k,
                        "base_key": base_key,
                        "topup_key": topup_key,
                        "comparison": comparison_label(topup_key),
                        "fixed_pass": fixed["pass_mean"],
                        "fixed_tokens": fixed["cost_mean"],
                        "before_condition": comparison_label(base_key),
                        "before_pass": base["pass_mean"],
                        "before_tokens": base["cost_mean"],
                        "accuracy_match_pass": acc["pass_mean"],
                        "accuracy_match_tokens": acc["cost_mean"],
                        "delta_tokens_vs_fixed": acc["cost_mean"]
                        - fixed["cost_mean"],
                        "extra_generations": acc["extra_generations_mean"],
                        "effective_k": acc["effective_k_mean"],
                    }
                )
    return budget_rows, accuracy_rows


def write_budget_match_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "| Pass@k | Comparison | Fixed pass | Fixed tokens | Before pass | Before tokens | Budget-matched pass | Budget-matched tokens | Delta vs fixed | Extra gens | Effective k |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['pass_k']} | "
            f"{row['comparison']} | "
            f"{row['fixed_pass']:.4f} | {row['fixed_tokens']:.0f} | "
            f"{row['before_pass']:.4f} | {row['before_tokens']:.0f} | "
            f"{row['after_pass']:.4f} | {row['after_tokens']:.0f} | "
            f"{signed_token_delta(row['delta_tokens_vs_fixed'])} | "
            f"{row['extra_generations']:.2f} | {row['effective_k']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_accuracy_match_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "| Pass@k | Comparison | Fixed pass | Fixed tokens | Before pass | Before tokens | Accuracy-matched pass | Accuracy-matched tokens | Delta vs fixed | Extra gens | Effective k |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['pass_k']} | "
            f"{row['comparison']} | "
            f"{row['fixed_pass']:.4f} | {row['fixed_tokens']:.0f} | "
            f"{row['before_pass']:.4f} | {row['before_tokens']:.0f} | "
            f"{row['accuracy_match_pass']:.4f} | "
            f"{row['accuracy_match_tokens']:.0f} | "
            f"{signed_token_delta(row['delta_tokens_vs_fixed'])} | "
            f"{row['extra_generations']:.2f} | {row['effective_k']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def latex_bold(value: str, bold: bool) -> str:
    return f"\\textbf{{{value}}}" if bold else value


def nearly_equal(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance


def budget_match_latex_table(rows: list[dict], caption_prefix: str, k: int) -> str:
    k_rows = [row for row in rows if int(row["pass_k"]) == k]
    if not k_rows:
        return ""
    best_pass = max(row["after_pass"] for row in k_rows)
    fixed_pass = k_rows[0]["fixed_pass"]
    lines = [
        "% Auto-generated by scripts/backfill_shared2_budget_plots.py",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{latex_escape(caption_prefix)}: budget-matched Pass@{k}.}}",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Base & Top-up & Accuracy & Delta from fixed baseline \\\\",
        "\\midrule",
        f"Fixed & None & {latex_float(fixed_pass)} & {signed_accuracy_delta(0.0)} \\\\",
    ]
    for row in k_rows:
        is_best = nearly_equal(row["after_pass"], best_pass)
        accuracy = latex_bold(latex_float(row["after_pass"]), is_best)
        delta = latex_bold(
            signed_accuracy_delta(row["after_pass"] - fixed_pass), is_best
        )
        lines.append(
            f"{latex_escape(simple_base_label(row['base_key']))} "
            f"& {latex_escape(simple_topup_label(row['topup_key']))} "
            f"& {accuracy} "
            f"& {delta} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def accuracy_match_latex_table(rows: list[dict], caption_prefix: str, k: int) -> str:
    k_rows = [row for row in rows if int(row["pass_k"]) == k]
    if not k_rows:
        return ""
    eligible_rows = [
        row for row in k_rows if row["accuracy_match_pass"] + 1e-12 >= row["fixed_pass"]
    ]
    best_tokens = min(
        row["accuracy_match_tokens"] for row in (eligible_rows or k_rows)
    )
    fixed_tokens = k_rows[0]["fixed_tokens"]
    lines = [
        "% Auto-generated by scripts/backfill_shared2_budget_plots.py",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{latex_escape(caption_prefix)}: accuracy-matched Pass@{k}.}}",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Base & Top-up & Token cost & Delta from fixed baseline \\\\",
        "\\midrule",
        f"Fixed & None & {latex_tokens(fixed_tokens)} & {signed_token_delta(0.0)} \\\\",
    ]
    for row in k_rows:
        is_best = nearly_equal(row["accuracy_match_tokens"], best_tokens)
        tokens = latex_bold(latex_tokens(row["accuracy_match_tokens"]), is_best)
        delta = latex_bold(signed_token_delta(row["delta_tokens_vs_fixed"]), is_best)
        lines.append(
            f"{latex_escape(simple_base_label(row['base_key']))} "
            f"& {latex_escape(simple_topup_label(row['topup_key']))} "
            f"& {tokens} "
            f"& {delta} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def write_budget_match_latex(path: Path, rows: list[dict], caption_prefix: str) -> None:
    tables = []
    for k in LATEX_TABLE_KS:
        table = budget_match_latex_table(rows, caption_prefix, k)
        if table:
            tables.append(table)
            (path.parent / f"budget_matched_pass{k}.tex").write_text(table + "\n")
    path.write_text("\n\n".join(tables) + "\n")


def write_accuracy_match_latex(path: Path, rows: list[dict], caption_prefix: str) -> None:
    tables = []
    for k in LATEX_TABLE_KS:
        table = accuracy_match_latex_table(rows, caption_prefix, k)
        if table:
            tables.append(table)
            (path.parent / f"accuracy_matched_pass{k}.tex").write_text(table + "\n")
    path.write_text("\n\n".join(tables) + "\n")


def write_comparison_tables(
    out_dir: Path,
    rows: list[dict],
    specs: tuple[tuple[str, str], ...],
    caption_prefix: str,
    include_aux_files: bool = True,
) -> None:
    budget_rows, accuracy_rows = comparison_rows(rows, specs)
    if include_aux_files:
        write_csv(out_dir / "budget_matched_stats.csv", budget_rows)
        write_csv(out_dir / "accuracy_matched_stats.csv", accuracy_rows)
        write_budget_match_markdown(out_dir / "budget_matched_stats.md", budget_rows)
        write_accuracy_match_markdown(out_dir / "accuracy_matched_stats.md", accuracy_rows)
    else:
        for pattern in ("*.csv", "*.md"):
            for path in out_dir.glob(pattern):
                path.unlink()
    write_budget_match_latex(
        out_dir / "budget_matched_stats.tex", budget_rows, caption_prefix
    )
    write_accuracy_match_latex(
        out_dir / "accuracy_matched_stats.tex", accuracy_rows, caption_prefix
    )


def latex_escape(value: str) -> str:
    replacements = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def short_comparison_label(value: str) -> str:
    return (
        str(value)
        .replace("Shared every 2 + ", "")
        .replace("Shared every 4 + ", "")
        .replace("Shared every 8 + ", "")
        .replace("Shared every 16 + ", "")
    )


def simple_base_label(key: str) -> str:
    return {
        "fixed": "Fixed",
        "shared2": "Shared 2",
        "shared4": "Shared 4",
        "shared8": "Shared 8",
        "shared16": "Shared 16",
        "shared32": "Shared 32",
    }.get(key, comparison_label(key))


def simple_topup_label(key: str) -> str:
    return {
        "shared2_fixed_topup": "Fixed",
        "shared2_topup": "Shared 2",
        "shared2_shared4_topup": "Shared 4",
        "shared2_shared8_topup": "Shared 8",
        "shared4_fixed_topup": "Fixed",
        "shared4_shared2_topup": "Shared 2",
        "shared4_shared4_topup": "Shared 4",
        "shared4_shared8_topup": "Shared 8",
        "shared8_fixed_topup": "Fixed",
        "shared8_shared2_topup": "Shared 2",
        "shared8_shared4_topup": "Shared 4",
        "shared8_shared8_topup": "Shared 8",
        "shared16_shared16_topup": "Shared 16",
    }.get(key, short_comparison_label(comparison_label(key)))


def latex_float(value: float) -> str:
    return f"{value:.4f}"


def latex_tokens(value: float) -> str:
    return f"{value / 1000:.1f}k"


def write_latex_tables(path: Path, rows: list[dict]) -> None:
    by_key = {(int(row["pass_k"]), row["condition"]): row for row in rows}
    lines = [
        "% Auto-generated by scripts/backfill_shared2_budget_plots.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Shared every 2 with shared2 top-up at approximately the same token budget as fixed.}",
        "\\begin{tabular}{rrrrr}",
        "\\toprule",
        "$k$ & Fixed acc. & Fixed tokens & Top-up acc. & $\\Delta$ tokens \\\\",
        "\\midrule",
    ]
    for k in PASS_KS:
        fixed = by_key[(k, "fixed")]
        budget = by_key[(k, "shared2_topup")]
        delta = budget["cost_mean"] - fixed["cost_mean"]
        lines.append(
            f"{k} & {latex_float(fixed['pass_mean'])} & {latex_tokens(fixed['cost_mean'])} "
            f"& {latex_float(budget['pass_mean'])} & {signed_token_delta(delta)} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Token cost when shared every 2 with shared2 top-up first matches fixed accuracy.}",
        "\\begin{tabular}{rrrrr}",
        "\\toprule",
        "$k$ & Fixed acc. & Accuracy-matched acc. & Accuracy-matched tokens & Savings vs. fixed \\\\",
        "\\midrule",
    ]
    for k in PASS_KS:
        fixed = by_key[(k, "fixed")]
        acc = by_key[(k, "shared2_accuracy_match")]
        delta = acc["cost_mean"] - fixed["cost_mean"]
        lines.append(
            f"{k} & {latex_float(fixed['pass_mean'])} & {latex_float(acc['pass_mean'])} "
            f"& {latex_tokens(acc['cost_mean'])} & {signed_token_delta(delta)} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def first_k_state(rows: list[dict], k: int) -> tuple[float, bool]:
    cost = 0.0
    correct = False
    for row in sorted(rows, key=lambda item: int(item["sample_index"]))[:k]:
        cost += sample_cost(row)
        correct = correct or bool(row.get("correct"))
    return cost, correct


def compute_shared2_before_after(root: Path, pass_ks: tuple[int, ...]) -> list[dict]:
    out = []
    for k in pass_ks:
        per_prompt = []
        for repeat in REPEATS:
            repeat_dir = root / f"repeat_{repeat:02d}"
            fixed_samples = samples_by_prompt(
                repeat_dir / "bundle_fixed" / "fixed_trace_independent" / "samples.jsonl"
            )
            shared2_samples = samples_by_prompt(
                repeat_dir / "bundle_shared_low" / "shared_every_02" / "samples.jsonl"
            )
            shared2_topups = topup_samples_by_prompt(
                repeat_dir / "memory_match_topup_shared2_sharedgen_fast"
            )
            for prompt in sorted(set(fixed_samples) & set(shared2_samples)):
                target_cost, _ = first_k_state(fixed_samples[prompt], k)
                before_cost, before_correct = first_k_state(shared2_samples[prompt], k)
                after_cost = before_cost
                after_correct = before_correct
                extras = build_extra_sources(prompt, k, shared2_samples, shared2_topups)
                for sample in extras:
                    if after_cost >= target_cost:
                        break
                    after_cost += sample_cost(sample)
                    after_correct = after_correct or bool(sample.get("correct"))
                per_prompt.append(
                    {
                        "pass_k": k,
                        "fixed": float(
                            first_k_state(fixed_samples[prompt], k)[1]
                        ),
                        "before": float(before_correct),
                        "after": float(after_correct),
                    }
                )
        out.append(
            {
                "pass_k": k,
                "fixed": mean([row["fixed"] for row in per_prompt]),
                "before": mean([row["before"] for row in per_prompt]),
                "after": mean([row["after"] for row in per_prompt]),
            }
        )
    return out


def overlay_before_after_from_summary(before_after_rows: list[dict], summary_rows: list[dict]) -> None:
    by_key = {(int(row["pass_k"]), row["condition"]): row for row in summary_rows}
    for row in before_after_rows:
        k = int(row["pass_k"])
        if (k, "shared2") in by_key and (k, "shared2_topup") in by_key:
            row["fixed"] = by_key[(k, "fixed")]["pass_mean"]
            row["before"] = by_key[(k, "shared2")]["pass_mean"]
            row["after"] = by_key[(k, "shared2_topup")]["pass_mean"]


def write_before_after_latex(path: Path, rows: list[dict]) -> None:
    lines = [
        "% Auto-generated by scripts/backfill_shared2_budget_plots.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Shared every 2 accuracy before and after shared2 top-up to the fixed token budget.}",
        "\\begin{tabular}{rrrr}",
        "\\toprule",
        "$k$ & Fixed & Before top-up & After shared2 top-up \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['pass_k']} & {latex_float(row['fixed'])} & "
            f"{latex_float(row['before'])} & {latex_float(row['after'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def write_same_accuracy_delta_latex(path: Path, rows: list[dict]) -> None:
    by_key = {(int(row["pass_k"]), row["condition"]): row for row in rows}
    lines = [
        "% Auto-generated by scripts/backfill_shared2_budget_plots.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Token cost for shared every 2 with shared2 top-up when it reaches fixed accuracy.}",
        "\\begin{tabular}{rrrr}",
        "\\toprule",
        "$k$ & Fixed tokens & Accuracy-matched tokens & Token difference \\\\",
        "\\midrule",
    ]
    for k in PASS_KS:
        fixed = by_key[(k, "fixed")]
        acc = by_key[(k, "shared2_accuracy_match")]
        delta = acc["cost_mean"] - fixed["cost_mean"]
        lines.append(
            f"{k} & {latex_tokens(fixed['cost_mean'])} & "
            f"{latex_tokens(acc['cost_mean'])} & {signed_token_delta(delta)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def nice_token(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value / 1000:.0f}k"
    return f"{value:.0f}"


def show_std_bars(condition: dict) -> bool:
    return not condition.get("accuracy_match_for")


def signed_token_delta(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value) / 1000:.1f}k"


def signed_accuracy_delta(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.4f}"


def render_marker(
    parts: list[str], marker: str, x: float, y: float, color: str, size: float = 7.0
) -> None:
    if marker == "circle":
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" fill="{color}"/>')
    elif marker == "smallcircle":
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size * 0.72:.2f}" '
            f'fill="{color}" stroke="white" stroke-width="1.2"/>'
        )
    elif marker == "square":
        s = size * 1.8
        parts.append(
            f'<rect x="{x - s/2:.2f}" y="{y - s/2:.2f}" width="{s:.2f}" '
            f'height="{s:.2f}" fill="{color}"/>'
        )
    elif marker == "diamond":
        s = size * 1.18
        points = [
            (x, y - s),
            (x + s, y),
            (x, y + s),
            (x - s, y),
        ]
        points_attr = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
        parts.append(
            f'<polygon points="{points_attr}" fill="white" stroke="{color}" '
            f'stroke-width="2.2"/>'
        )
    elif marker == "triangle":
        s = size * 1.7
        points = [(x, y - s), (x + s, y + s), (x - s, y + s)]
        points_attr = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
        parts.append(f'<polygon points="{points_attr}" fill="{color}"/>')
    elif marker == "star":
        s = size * 1.55
        parts.append(
            f'<path d="M{x:.2f},{y-s:.2f} L{x+2.8:.2f},{y-3.2:.2f} '
            f'L{x+s:.2f},{y-3.2:.2f} L{x+4.3:.2f},{y+1.8:.2f} '
            f'L{x+6.0:.2f},{y+s:.2f} L{x:.2f},{y+5.0:.2f} '
            f'L{x-6.0:.2f},{y+s:.2f} L{x-4.3:.2f},{y+1.8:.2f} '
            f'L{x-s:.2f},{y-3.2:.2f} L{x-2.8:.2f},{y-3.2:.2f} Z" '
            f'fill="{color}"/>'
        )
    elif marker == "plus":
        s = size * 1.55
        parts.append(
            f'<path d="M{x-s:.2f},{y:.2f} L{x+s:.2f},{y:.2f} '
            f'M{x:.2f},{y-s:.2f} L{x:.2f},{y+s:.2f}" stroke="{color}" '
            f'stroke-width="2.4" stroke-linecap="round"/>'
        )
    elif marker == "x":
        s = size * 1.45
        parts.append(
            f'<path d="M{x-s:.2f},{y-s:.2f} L{x+s:.2f},{y+s:.2f} '
            f'M{x-s:.2f},{y+s:.2f} L{x+s:.2f},{y-s:.2f}" stroke="{color}" '
            f'stroke-width="2.4" stroke-linecap="round"/>'
        )


def render_svg(
    path: Path,
    rows: list[dict],
    k: int,
    condition_keys: tuple[str, ...] | None = None,
    arrow_pairs: tuple[tuple[str, str], ...] = (),
    title: str | None = None,
    include_budget_guides: bool = False,
    marker_size: float = 5.4,
) -> None:
    all_data = [row for row in rows if row["pass_k"] == k]
    all_by_key = {row["condition"]: row for row in all_data}
    if condition_keys is None:
        condition_keys = tuple(condition["key"] for condition in CONDITIONS)
    data = [all_by_key[key] for key in condition_keys if key in all_by_key]
    by_key = {row["condition"]: row for row in data}
    lookup = condition_lookup()
    legend_conditions = [
        lookup[key]
        for key in condition_keys
        if key in by_key and lookup[key].get("legend", True)
    ]

    width, height = 1600, 920
    left, right, top, bottom = 130, 520, 82, 118
    plot_w = width - left - right
    plot_h = height - top - bottom

    min_x = min(row["cost_mean"] - row["cost_std"] for row in data)
    max_x = max(row["cost_mean"] + row["cost_std"] for row in data)
    min_y = min(row["pass_mean"] - row["pass_std"] for row in data)
    max_y = max(row["pass_mean"] + row["pass_std"] for row in data)
    x_span = max(max_x - min_x, 1.0)
    y_span = max(max_y - min_y, 0.005)
    x_pad = max(x_span * 0.16, 3500.0)
    y_pad = max(y_span * 0.28, 0.012)
    x_round = 5000 if x_span <= 50000 else 10000
    y_round = 0.01 if y_span <= 0.08 else 0.025
    x0 = math.floor((min_x - x_pad) / x_round) * x_round
    x1 = math.ceil((max_x + x_pad) / x_round) * x_round
    y0 = max(0.0, math.floor((min_y - y_pad) / y_round) * y_round)
    y1 = min(1.0, math.ceil((max_y + y_pad) / y_round) * y_round)

    def sx(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * plot_w

    def sy(y: float) -> float:
        return top + (y1 - y) / (y1 - y0) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<defs>",
    ]
    for i, (_, target_key) in enumerate(arrow_pairs):
        color = lookup.get(target_key, {}).get("color", "#333333")
        parts.append(
        f'<marker id="arrow{i}" markerWidth="9" markerHeight="9" refX="8" '
            'refY="3" orient="auto" markerUnits="strokeWidth">'
            f'<path d="M0,0 L0,6 L8,3 z" fill="{color}"/></marker>'
        )
    parts += [
        "</defs>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:"Latin Modern Roman","Computer Modern","CMU Serif","Times New Roman",serif;}'
        '.tick{font-size:17px;fill:#111}.label{font-size:25px;fill:#111}'
        '.title{font-size:26px;font-weight:600;fill:#111}.legend{font-size:17px;fill:#111}'
        "</style>",
    ]

    x_ticks = []
    x_range = x1 - x0
    if x_range <= 50000:
        step = 10000
    elif x_range <= 140000:
        step = 25000
    else:
        step = 50000
    tick = math.ceil(x0 / step) * step
    while tick <= x1:
        x_ticks.append(tick)
        tick += step
    y_range = y1 - y0
    if y_range <= 0.08:
        y_step = 0.01
    elif y_range <= 0.16:
        y_step = 0.025
    else:
        y_step = 0.05
    y_ticks = []
    tick_y = math.ceil(y0 / y_step) * y_step
    while tick_y <= y1 + 1e-9:
        y_ticks.append(round(tick_y, 3))
        tick_y += y_step

    for tick in x_ticks:
        x = sx(tick)
        parts.append(
            f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{top+plot_h}" '
            'stroke="#eeeeee" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.2f}" y="{top+plot_h+36}" '
            f'text-anchor="middle">{nice_token(tick)}</text>'
        )
    for tick in y_ticks:
        y = sy(tick)
        parts.append(
            f'<line x1="{left}" x2="{left+plot_w}" y1="{y:.2f}" y2="{y:.2f}" '
            'stroke="#eeeeee" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text class="tick" x="{left-22}" y="{y+6:.2f}" '
            f'text-anchor="end">{tick:.3f}</text>'
        )

    parts.append(
        f'<line x1="{left}" x2="{left+plot_w}" y1="{top+plot_h}" y2="{top+plot_h}" '
        'stroke="#111" stroke-width="1.1"/>'
    )
    parts.append(
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top+plot_h}" '
        'stroke="#111" stroke-width="1.1"/>'
    )
    parts.append(
        f'<text class="title" x="{left + plot_w / 2:.2f}" y="50" '
        f'text-anchor="middle">{html.escape(title or f"AIME Train: Memory Usage vs. Pass@{k}")}</text>'
    )
    parts.append(
        f'<text class="label" x="{left + plot_w / 2:.2f}" y="{height - 38}" '
        'text-anchor="middle">Memory Usage (tokens)</text>'
    )
    parts.append(
        f'<text class="label" transform="translate(50 {top + plot_h / 2:.2f}) rotate(-90)" '
        f'text-anchor="middle">Pass@{k}</text>'
    )

    if include_budget_guides and "fixed" in by_key:
        fixed_x = sx(by_key["fixed"]["cost_mean"])
        parts.append(
            f'<line x1="{fixed_x:.2f}" x2="{fixed_x:.2f}" y1="{top}" '
            f'y2="{top + plot_h}" stroke="#4C78A8" stroke-width="1.0" '
            'stroke-dasharray="4 5" opacity="0.45"/>'
        )

    for i, (base_key, target_key) in enumerate(arrow_pairs):
        if base_key not in by_key or target_key not in by_key:
            continue
        base = by_key[base_key]
        target = by_key[target_key]
        x_base = sx(base["cost_mean"])
        y_base = sy(base["pass_mean"])
        x_target = sx(target["cost_mean"])
        y_target = sy(target["pass_mean"])
        dx = x_target - x_base
        dy = y_target - y_base
        length = math.hypot(dx, dy)
        if length <= 1:
            continue
        start_gap = marker_size * 2.0
        end_gap = marker_size * 2.2
        x_start = x_base + dx / length * start_gap
        y_start = y_base + dy / length * start_gap
        x_end = x_target - dx / length * end_gap
        y_end = y_target - dy / length * end_gap
        color = lookup[target_key]["color"]
        parts.append(
            f'<line x1="{x_start:.2f}" y1="{y_start:.2f}" x2="{x_end:.2f}" '
            f'y2="{y_end:.2f}" stroke="{color}" stroke-width="1.6" '
            f'stroke-dasharray="5 4" marker-end="url(#arrow{i})" opacity="0.85"/>'
        )

    for condition in legend_conditions:
        row = by_key.get(condition["key"])
        if not row:
            continue
        x = sx(row["cost_mean"])
        y = sy(row["pass_mean"])
        color = condition["color"]
        xerr = row["cost_std"] if show_std_bars(condition) else 0.0
        yerr = row["pass_std"] if show_std_bars(condition) else 0.0
        if xerr:
            parts.append(
                f'<line x1="{sx(row["cost_mean"] - xerr):.2f}" '
                f'x2="{sx(row["cost_mean"] + xerr):.2f}" y1="{y:.2f}" y2="{y:.2f}" '
                f'stroke="{color}" stroke-width="1.0" opacity="0.62"/>'
            )
        if yerr:
            parts.append(
                f'<line x1="{x:.2f}" x2="{x:.2f}" '
                f'y1="{sy(row["pass_mean"] - yerr):.2f}" '
                f'y2="{sy(row["pass_mean"] + yerr):.2f}" '
                f'stroke="{color}" stroke-width="1.0" opacity="0.62"/>'
            )
        render_marker(parts, condition["marker"], x, y, color, marker_size)

    legend_x = left + plot_w + 42
    legend_y = top + 20
    row_h = 31 if len(legend_conditions) > 11 else 38
    legend_size = 4.6 if len(legend_conditions) > 11 else marker_size
    parts.append(
        f'<rect x="{legend_x - 18}" y="{legend_y - 27}" width="{right - 58}" '
        f'height="{row_h * len(legend_conditions) + 12}" fill="white" opacity="0.96"/>'
    )
    for i, condition in enumerate(legend_conditions):
        x = legend_x
        y = legend_y + i * row_h
        render_marker(parts, condition["marker"], x, y - 5, condition["color"], legend_size)
        parts.append(
            f'<text class="legend" x="{x + 24}" y="{y + 1}">'
            f'{html.escape(condition["label"])}</text>'
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ALL_POINTS_NO_ARROWS_OUT.mkdir(parents=True, exist_ok=True)
    NO_ARROW_ACCURACY_OUT.mkdir(parents=True, exist_ok=True)
    NO_ARROW_BUDGET_OUT.mkdir(parents=True, exist_ok=True)
    NO_ARROW_COMBINED_OUT.mkdir(parents=True, exist_ok=True)
    SHARED2_TOPUP_COMPARISON_OUT.mkdir(parents=True, exist_ok=True)
    SHARED2_SHARED4_COMPARISON_OUT.mkdir(parents=True, exist_ok=True)
    GRID_SEARCH_COMPARISON_OUT.mkdir(parents=True, exist_ok=True)
    LATEST_SHARED2_TOPUP_COMPARISON_OUT.mkdir(parents=True, exist_ok=True)
    LATEST_SHARED2_SHARED4_COMPARISON_OUT.mkdir(parents=True, exist_ok=True)
    LATEST_GRID_SEARCH_COMPARISON_OUT.mkdir(parents=True, exist_ok=True)
    prompt_rows, repeat_rows = compute_rows(ROOT)
    summary_rows_out = summarize(repeat_rows)
    overlay_completed_pass32_checkpoint(summary_rows_out)
    overlay_topup_pass32_from_prompt_rows(summary_rows_out, ROOT)
    write_csv(OUT / "prompt_rows.csv", prompt_rows)
    write_csv(OUT / "repeat_rows.csv", repeat_rows)
    write_csv(OUT / "summary.csv", summary_rows_out)
    (OUT / "summary.json").write_text(json.dumps(summary_rows_out, indent=2) + "\n")
    write_results_table(OUT / "results_table.md", summary_rows_out)
    write_results_latex(OUT / "results_table.tex", summary_rows_out)
    write_latex_tables(OUT / "latex_tables.tex", summary_rows_out)
    write_same_accuracy_delta_latex(
        OUT / "same_accuracy_token_difference.tex", summary_rows_out
    )
    before_after_rows = compute_shared2_before_after(ROOT, TABLE_PASS_KS)
    overlay_before_after_from_summary(before_after_rows, summary_rows_out)
    write_csv(OUT / "before_after_budget_match.csv", before_after_rows)
    write_before_after_latex(OUT / "before_after_budget_match.tex", before_after_rows)
    write_comparison_tables(
        SHARED2_TOPUP_COMPARISON_OUT,
        summary_rows_out,
        SHARED2_TOPUP_SPECS,
        "Shared every 2 top-up generator comparison",
    )
    write_comparison_tables(
        LATEST_SHARED2_TOPUP_COMPARISON_OUT,
        summary_rows_out,
        SHARED2_TOPUP_SPECS,
        "Shared every 2 top-up generator comparison",
        include_aux_files=False,
    )
    write_comparison_tables(
        SHARED2_SHARED4_COMPARISON_OUT,
        summary_rows_out,
        SHARED2_SHARED4_TOPUP_SPECS,
        "Shared every 2 vs. shared every 4 top-up comparison",
    )
    write_comparison_tables(
        LATEST_SHARED2_SHARED4_COMPARISON_OUT,
        summary_rows_out,
        SHARED2_SHARED4_TOPUP_SPECS,
        "Shared every 2 vs. shared every 4 top-up comparison",
        include_aux_files=False,
    )
    write_comparison_tables(
        GRID_SEARCH_COMPARISON_OUT,
        summary_rows_out,
        GRID_SEARCH_TOPUP_SPECS,
        "Top-up grid search comparison",
    )
    write_comparison_tables(
        LATEST_GRID_SEARCH_COMPARISON_OUT,
        summary_rows_out,
        GRID_SEARCH_TOPUP_SPECS,
        "Top-up grid search comparison",
        include_aux_files=False,
    )

    shared2_topup_condition_keys = (
        "fixed",
        "shared2",
        "shared2_fixed_topup",
        "shared2_fixed_accuracy_match",
        "shared2_topup",
        "shared2_accuracy_match",
        "shared2_shared4_topup",
        "shared2_shared4_accuracy_match",
        "shared2_shared8_topup",
        "shared2_shared8_accuracy_match",
    )
    shared2_shared4_condition_keys = (
        "fixed",
        "shared2",
        "shared2_topup",
        "shared2_accuracy_match",
        "shared4",
        "shared4_shared4_topup",
        "shared4_shared4_accuracy_match",
    )
    grid_search_condition_keys = (
        "fixed",
        "shared2",
        "shared2_fixed_topup",
        "shared2_fixed_accuracy_match",
        "shared2_topup",
        "shared2_accuracy_match",
        "shared2_shared4_topup",
        "shared2_shared4_accuracy_match",
        "shared2_shared8_topup",
        "shared2_shared8_accuracy_match",
        "shared4",
        "shared4_fixed_topup",
        "shared4_fixed_accuracy_match",
        "shared4_shared2_topup",
        "shared4_shared2_accuracy_match",
        "shared4_shared4_topup",
        "shared4_shared4_accuracy_match",
        "shared4_shared8_topup",
        "shared4_shared8_accuracy_match",
        "shared8",
        "shared8_fixed_topup",
        "shared8_fixed_accuracy_match",
        "shared8_shared2_topup",
        "shared8_shared2_accuracy_match",
        "shared8_shared4_topup",
        "shared8_shared4_accuracy_match",
        "shared8_shared8_topup",
        "shared8_shared8_accuracy_match",
        "shared16",
        "shared16_shared16_topup",
        "shared16_shared16_accuracy_match",
    )
    for k in PASS_KS:
        render_svg(
            OUT / f"memory_vs_pass{k}.svg",
            summary_rows_out,
            k,
            title=f"AIME Train: Memory Usage vs. Pass@{k}",
        )
        render_svg(
            NO_ARROW_ACCURACY_OUT / f"memory_vs_pass{k}.svg",
            summary_rows_out,
            k,
            condition_keys=ACCURACY_MATCH_CONDITION_KEYS,
            title=f"Accuracy-Matched Points: Pass@{k}",
        )
        render_svg(
            NO_ARROW_BUDGET_OUT / f"memory_vs_pass{k}.svg",
            summary_rows_out,
            k,
            condition_keys=BUDGET_MATCH_CONDITION_KEYS,
            title=f"Budget-Matched Points: Pass@{k}",
        )
        render_svg(
            NO_ARROW_COMBINED_OUT / f"memory_vs_pass{k}.svg",
            summary_rows_out,
            k,
            title=f"All Matched Points: Pass@{k}",
        )
        render_svg(
            SHARED2_TOPUP_COMPARISON_OUT / f"memory_vs_pass{k}.svg",
            summary_rows_out,
            k,
            condition_keys=shared2_topup_condition_keys,
            arrow_pairs=SHARED2_TOPUP_SPECS,
            title=f"Shared2 Top-Up Generator Comparison: Pass@{k}",
            include_budget_guides=True,
        )
        render_svg(
            LATEST_SHARED2_TOPUP_COMPARISON_OUT / f"memory_vs_pass{k}.svg",
            summary_rows_out,
            k,
            condition_keys=shared2_topup_condition_keys,
            arrow_pairs=SHARED2_TOPUP_SPECS,
            title=f"Shared2 Top-Up Generator Comparison: Pass@{k}",
            include_budget_guides=True,
        )
        render_svg(
            SHARED2_SHARED4_COMPARISON_OUT / f"memory_vs_pass{k}.svg",
            summary_rows_out,
            k,
            condition_keys=shared2_shared4_condition_keys,
            arrow_pairs=SHARED2_SHARED4_TOPUP_SPECS,
            title=f"Shared2 vs. Shared4 Top-Up: Pass@{k}",
            include_budget_guides=True,
        )
        render_svg(
            LATEST_SHARED2_SHARED4_COMPARISON_OUT / f"memory_vs_pass{k}.svg",
            summary_rows_out,
            k,
            condition_keys=shared2_shared4_condition_keys,
            arrow_pairs=SHARED2_SHARED4_TOPUP_SPECS,
            title=f"Shared2 vs. Shared4 Top-Up: Pass@{k}",
            include_budget_guides=True,
        )
        render_svg(
            GRID_SEARCH_COMPARISON_OUT / f"memory_vs_pass{k}.svg",
            summary_rows_out,
            k,
            condition_keys=grid_search_condition_keys,
            arrow_pairs=GRID_SEARCH_TOPUP_SPECS,
            title=f"Top-Up Grid Search: Pass@{k}",
            include_budget_guides=True,
        )
        render_svg(
            LATEST_GRID_SEARCH_COMPARISON_OUT / f"memory_vs_pass{k}.svg",
            summary_rows_out,
            k,
            condition_keys=grid_search_condition_keys,
            arrow_pairs=GRID_SEARCH_TOPUP_SPECS,
            title=f"Top-Up Grid Search: Pass@{k}",
            include_budget_guides=True,
        )


if __name__ == "__main__":
    main()
