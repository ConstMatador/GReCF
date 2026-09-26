#!/usr/bin/env python3
"""Merge LLM Judge shards and write the project-standard summary."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()
    task_ids = [json.loads(line)["task_id"] for line in args.tasks_jsonl.open() if line.strip()]
    expected = set(task_ids)
    by_id = {}
    invalid = 0
    for shard in sorted(args.raw_dir.glob("judge_shard*.jsonl")):
        for line in shard.open(encoding="utf-8", errors="replace"):
            try:
                row = json.loads(line)
            except Exception:
                invalid += 1
                continue
            if row.get("task_id") in expected:
                by_id[row["task_id"]] = row
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"Qwen judge missing {len(missing)} tasks: {missing[:10]}")
    with (args.raw_dir / "judge_merged.jsonl").open("w", encoding="utf-8") as handle:
        for task_id in task_ids:
            handle.write(json.dumps(by_id[task_id], ensure_ascii=False) + "\n")
    rows = list(by_id.values())
    counts = Counter(row["judge"]["label"] for row in rows)
    history_failures = sum(not bool((row.get("history_stage") or {}).get("parse_ok")) for row in rows)
    cf_rows = [row for row in rows if row.get("cf_stage") is not None]
    cf_failures = sum(not bool((row.get("cf_stage") or {}).get("parse_ok")) for row in cf_rows)
    true_cf = counts.get("true_cf_expansion", 0)
    other = counts.get("other_or_uncertain", 0)
    summary = {
        "rows": len(rows),
        "counts": dict(counts),
        "rates": {key: value / len(rows) for key, value in counts.items()},
        "cf_support_rate": true_cf / max(true_cf + other, 1),
        "invalid_lines_skipped": invalid,
        "parse_reliability": {
            "total_rows": len(rows),
            "history_parse_failures": history_failures,
            "history_parse_failure_rate": history_failures / max(len(rows), 1),
            "cf_stage_invocations": len(cf_rows),
            "cf_parse_failures": cf_failures,
            "cf_parse_failure_rate": cf_failures / max(len(cf_rows), 1),
        },
        "history_match_mode": next(
            (row.get("history_match_mode") for row in rows if row.get("history_match_mode")),
            "legacy_unspecified",
        ),
        "cf_match_mode": next(
            (row.get("cf_match_mode") for row in rows if row.get("cf_match_mode")),
            "legacy_unspecified",
        ),
    }
    (args.raw_dir / "judge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
