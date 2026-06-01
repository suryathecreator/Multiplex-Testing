#!/usr/bin/env python3
"""Create a deterministic DeepScaleR parquet subset for short RL runs."""

import argparse
import json
import random
import shutil
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="deepscaler/hdfs_data/train.parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--num-examples", default="64")
    parser.add_argument("--seed", type=int, default=26010808)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    requested_examples = str(args.num_examples).strip()
    use_all = requested_examples.lower() in {"all", "full", "-1"}

    if not use_all:
        try:
            num_examples = int(requested_examples)
        except ValueError as exc:
            raise ValueError("--num-examples must be a positive integer or ALL") from exc
        if num_examples <= 0:
            raise ValueError("--num-examples must be positive")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing source parquet: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if use_all:
        if output_path.exists() or output_path.is_symlink():
            output_path.unlink()
        try:
            output_path.symlink_to(input_path.resolve())
            output_kind = "symlink"
        except OSError:
            shutil.copy2(input_path, output_path)
            output_kind = "copy"
        manifest = {
            "created_at_unix": time.time(),
            "source_parquet": str(input_path),
            "output_parquet": str(output_path),
            "num_source_rows": None,
            "num_examples": "ALL",
            "selection_seed": None,
            "selected_source_indices": "ALL",
            "columns": None,
            "output_kind": output_kind,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[subset] using all rows from {input_path} via {output_kind} at {output_path}")
        print(f"[subset] manifest={manifest_path}")
        return

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas is required to create the DeepScaleR subset. "
            "Run this after scripts/bootstrap_runtime_overlay.py has prepared the runtime."
        ) from exc

    df = pd.read_parquet(input_path)
    if num_examples > len(df):
        raise ValueError(f"Requested {num_examples} examples, but {input_path} has only {len(df)} rows")

    rng = random.Random(args.seed)
    selected_indices = sorted(rng.sample(range(len(df)), num_examples))
    subset = df.iloc[selected_indices].copy()

    subset.to_parquet(output_path, index=False)

    manifest = {
        "created_at_unix": time.time(),
        "source_parquet": str(input_path),
        "output_parquet": str(output_path),
        "num_source_rows": int(len(df)),
        "num_examples": int(num_examples),
        "selection_seed": int(args.seed),
        "selected_source_indices": selected_indices,
        "columns": list(subset.columns),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[subset] wrote {len(subset)} rows to {output_path}")
    print(f"[subset] manifest={manifest_path}")


if __name__ == "__main__":
    main()
