#!/usr/bin/env python3
"""Aggregate binary human verification of source-attribution labels.

Input CSV columns: dataset, approach, user_id, image_id, annotator_id, correct. The `correct`
column accepts 1/0, true/false, or yes/no. No annotations are included in this
anonymous source release.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_correct(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"unsupported correctness value: {value!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-users", type=int, default=100)
    parser.add_argument("--expected-images-per-user", type=int, default=10)
    parser.add_argument("--expected-volunteers", type=int, default=40)
    parser.add_argument("--expected-groups", type=int, default=21)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    required = {"dataset", "approach", "user_id", "image_id", "annotator_id", "correct"}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")
    frame["correct"] = frame["correct"].map(parse_correct)
    volunteers = frame["annotator_id"].astype(str).nunique()
    if volunteers != args.expected_volunteers:
        raise SystemExit(f"volunteers={volunteers}, expected={args.expected_volunteers}")

    groups = []
    for (dataset, approach), group in frame.groupby(["dataset", "approach"], sort=True):
        users = group["user_id"].nunique()
        expected_rows = args.expected_users * args.expected_images_per_user
        per_user = group.groupby("user_id").size()
        if (
            users != args.expected_users
            or len(group) != expected_rows
            or not bool((per_user == args.expected_images_per_user).all())
        ):
            raise SystemExit(
                f"{dataset}/{approach}: users={users}, rows={len(group)}, "
                f"expected={args.expected_users}/{expected_rows}"
            )
        groups.append(
            {
                "dataset": str(dataset),
                "approach": str(approach),
                "rows": int(len(group)),
                "accuracy": float(group["correct"].mean()),
            }
        )
    output = {
        "volunteers": volunteers,
        "groups": groups,
        "macro_accuracy": sum(row["accuracy"] for row in groups) / max(len(groups), 1),
    }
    if len(groups) != args.expected_groups:
        raise SystemExit(f"groups={len(groups)}, expected={args.expected_groups}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
