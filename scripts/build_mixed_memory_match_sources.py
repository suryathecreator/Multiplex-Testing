#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


PASS_AT_KS = [1, 2, 4, 8, 16, 32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build synthetic memory_match_topup source dirs for the Fixed+Shared mix "
            "condition. The synthetic shared_every_32 rows represent the mixed "
            "fixed@16 + shared@16 condition, while fixed_trace_independent is fixed@32."
        )
    )
    parser.add_argument("--root", required=True, help="Fixed/shared mix experiment root.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=32)
    parser.add_argument(
        "--output-name",
        default="memory_match_source_fixed_shared_mix",
        help="Per-repeat synthetic source directory name.",
    )
    return parser.parse_args()


def load_mix_aggregate_module(repo_root: Path) -> Any:
    script = repo_root / "scripts" / "aggregate_aime_train100_fixed_shared_mix4096.py"
    spec = importlib.util.spec_from_file_location("mix_aggregate", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bool_or_none(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def clean_summary_rows(rows: List[Dict[str, Any]], method: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for k in PASS_AT_KS:
        usable = [
            row
            for row in rows
            if row.get(f"pass_at_{k}") is not None and row.get(f"cost_at_{k}") is not None
        ]
        out.append(
            {
                "method": method,
                "k": k,
                "num_prompts": len(usable),
                "pass_at_k": mean([1.0 if row[f"pass_at_{k}"] else 0.0 for row in usable]),
                "avg_cost_tokens": mean([float(row[f"cost_at_{k}"]) for row in usable]),
                "num_failures": 0,
            }
        )
    return out


def write_condition(condition_dir: Path, rows: List[Dict[str, Any]], method: str) -> None:
    condition_dir.mkdir(parents=True, exist_ok=True)
    # The memory-match loader requires one of the raw-output marker files to exist,
    # but will use summary.json directly when clean_summary_rows has observed rows.
    (condition_dir / "samples.jsonl").touch()
    payload = {
        "clean_pass_at_ks": PASS_AT_KS,
        "clean_summary_rows": clean_summary_rows(rows, method),
        "summary_rows": clean_summary_rows(rows, method),
        "all_per_prompt_rows": rows,
        "per_prompt_rows": rows,
        "coverage": {
            "method_prompts_with_full_k": {method: len(rows)},
            "matched_prompts_with_full_k": len(rows),
        },
    }
    (condition_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (condition_dir / "manifest.json").write_text(
        json.dumps({"synthetic_memory_match_source": True, "method": method}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    mix = load_mix_aggregate_module(repo_root)
    max_k = int(args.max_k)
    component_k = max_k // 2
    if max_k != 32:
        raise ValueError("This source builder currently expects --max-k 32.")

    for repeat_index in range(args.repeats):
        repeat_dir = root / f"repeat_{repeat_index:02d}"
        fixed_rows_by_prompt = mix.load_condition_rows(repeat_dir, "fixed")
        shared_rows_by_prompt = mix.load_condition_rows(repeat_dir, "shared32")
        common_prompts = sorted(set(fixed_rows_by_prompt) & set(shared_rows_by_prompt))

        fixed_rows: List[Dict[str, Any]] = []
        mixed_rows: List[Dict[str, Any]] = []
        for prompt_index in common_prompts:
            fixed = fixed_rows_by_prompt[prompt_index]
            shared = shared_rows_by_prompt[prompt_index]
            if not mix.row_has_full_k(fixed, max_k) or not mix.row_has_full_k(shared, max_k):
                continue

            fixed_row = dict(fixed)
            fixed_row["method"] = "baseline_independent"
            fixed_rows.append(fixed_row)

            mixed_row: Dict[str, Any] = {
                "method": "shared_trace_group_32",
                "prompt_index": prompt_index,
                "sample_count": max_k,
                "usable_sample_count": max_k,
                "excluded_sample_count": 0,
                "success": True,
                "error": "",
                "excluded_reasons": "",
                "first_correct_k": None,
                "parent_completion_tokens": 0,
                "attempts_used": 0,
                "total_attempt_completion_tokens": 0,
                "total_attempt_latency_seconds": 0.0,
            }
            first_correct: Optional[int] = None
            for k in PASS_AT_KS:
                if k < 2:
                    mixed_row[f"pass_at_{k}"] = None
                    mixed_row[f"cost_at_{k}"] = None
                    continue
                half_k = k // 2
                if half_k * 2 != k:
                    mixed_row[f"pass_at_{k}"] = None
                    mixed_row[f"cost_at_{k}"] = None
                    continue
                fixed_correct = bool_or_none(fixed.get(f"pass_at_{half_k}"))
                shared_correct = bool_or_none(shared.get(f"pass_at_{half_k}"))
                fixed_cost = float_or_none(fixed.get(f"cost_at_{half_k}"))
                shared_cost = float_or_none(shared.get(f"cost_at_{half_k}"))
                if fixed_correct is None or shared_correct is None or fixed_cost is None or shared_cost is None:
                    mixed_row[f"pass_at_{k}"] = None
                    mixed_row[f"cost_at_{k}"] = None
                    continue
                correct = fixed_correct or shared_correct
                mixed_row[f"pass_at_{k}"] = correct
                mixed_row[f"cost_at_{k}"] = fixed_cost + shared_cost
                if correct and first_correct is None:
                    first_correct = k
            mixed_row["first_correct_k"] = first_correct
            mixed_rows.append(mixed_row)

        source_dir = repeat_dir / args.output_name
        write_condition(source_dir / "fixed_trace_independent", fixed_rows, "baseline_independent")
        write_condition(source_dir / "shared_every_32", mixed_rows, "shared_trace_group_32")
        print(
            f"[mixed-source] repeat={repeat_index:02d} source={source_dir} prompts={len(mixed_rows)}"
        )


if __name__ == "__main__":
    main()
