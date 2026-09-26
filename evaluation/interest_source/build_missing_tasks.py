#!/usr/bin/env python3
"""Extract tasks that have no valid row in one or more Judge JSONL outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--require-parse-ok",
        action="store_true",
        help="Retry rows whose two-stage Judge output did not parse successfully.",
    )
    args = parser.parse_args()

    tasks = [json.loads(line) for line in args.tasks_jsonl.open() if line.strip()]
    completed: set[str] = set()
    invalid_lines = 0
    for path in sorted(args.raw_dir.glob("judge_shard*.jsonl")):
        for line in path.open(encoding="utf-8", errors="replace"):
            try:
                row = json.loads(line)
            except Exception:
                invalid_lines += 1
                continue
            task_id = row.get("task_id")
            if task_id and (not args.require_parse_ok or row.get("parse_ok") is True):
                completed.add(str(task_id))

    missing = [task for task in tasks if str(task["task_id"]) not in completed]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for task in missing:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "completed": len(completed),
                "missing": len(missing),
                "invalid_lines": invalid_lines,
                "output": str(args.output_jsonl),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
