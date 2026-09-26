#!/usr/bin/env python3
"""Aggregate generated-image CLIP similarity to held-out test preferences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-metrics", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.generation_metrics)
    required = {"user_id", "test_maxsim"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"generation metrics missing columns: {sorted(missing)}")
    user_means = frame.groupby("user_id", sort=True)["test_maxsim"].mean()
    summary = {
        "method": args.method,
        "metric": "CLIP Similarity",
        "images": int(len(frame)),
        "users": int(frame["user_id"].nunique()),
        "mean": float(frame["test_maxsim"].mean()),
        "user_macro_mean": float(user_means.mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
