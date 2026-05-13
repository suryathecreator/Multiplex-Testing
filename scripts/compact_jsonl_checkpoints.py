#!/usr/bin/env python
"""Compact experiment JSONL checkpoints in place.

This keeps the scalar fields needed for resume, summaries, and plots while
dropping full completions and large fork token/id payloads.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    compacted = dict(record)
    for key in ("text", "generated_suffix", "shared_trace_text"):
        if key in compacted:
            compacted[key] = ""
    if "score_debug" in compacted:
        compacted["score_debug"] = None
    for key in ("branch_input_ids", "cacheable_input_ids", "uncached_tail_input_ids"):
        if key in compacted:
            compacted[key] = []
    for key in ("response", "output_text", "output_preview"):
        value = compacted.get(key)
        if isinstance(value, str) and len(value) > 500:
            compacted[key] = value[:500] + "...[compact-jsonl-truncated]"
    compacted["compact_jsonl"] = True
    return compacted


def compact_jsonl_file(path: Path) -> tuple[int, int, int]:
    original_size = path.stat().st_size
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(compact_record(json.loads(line)))

    tmp_path = path.with_name(path.name + ".compact-tmp")
    with tmp_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=True) + "\n")
    os.replace(tmp_path, path)
    return len(rows), original_size, path.stat().st_size


def iter_jsonl_files(root: Path):
    names = {
        "samples.jsonl",
        "attempts.jsonl",
        "shared_trace_parents.jsonl",
        "events.jsonl",
    }
    if root.is_file() and root.name in names:
        yield root
        return
    for name in names:
        yield from root.rglob(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    seen: set[Path] = set()
    total_before = 0
    total_after = 0
    for root in args.paths:
        for path in iter_jsonl_files(root):
            path = path.resolve()
            if path in seen or not path.exists():
                continue
            seen.add(path)
            rows, before, after = compact_jsonl_file(path)
            total_before += before
            total_after += after
            print(f"{path}: {rows} rows, {before} -> {after} bytes")
    print(f"total: {total_before} -> {total_after} bytes")


if __name__ == "__main__":
    main()
