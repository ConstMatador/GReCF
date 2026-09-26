#!/usr/bin/env python3
"""Merge source-attribution Judge shards without changing task order."""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()

    task_ids = [json.loads(line)["task_id"] for line in args.tasks_jsonl.open() if line.strip()]
    expected = set(task_ids)
    rows: dict[str, dict] = {}
    invalid = 0
    for name in sorted(glob.glob(args.input_glob)):
        for line in Path(name).open(encoding="utf-8", errors="replace"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if row.get("task_id") in expected:
                rows[row["task_id"]] = row
    missing = [task_id for task_id in task_ids if task_id not in rows]
    if missing:
        raise SystemExit(f"missing {len(missing)} tasks; sample={missing[:10]}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for task_id in task_ids:
            handle.write(json.dumps(rows[task_id], ensure_ascii=False) + "\n")
    counts = Counter(row["judge"]["label"] for row in rows.values())
    total = len(rows)
    summary = {
        "rows": total,
        "counts": dict(counts),
        "rates": {label: count / total for label, count in counts.items()},
        "invalid_lines_skipped": invalid,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
