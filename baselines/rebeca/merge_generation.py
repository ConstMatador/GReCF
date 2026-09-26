#!/usr/bin/env python3
"""Merge REBECA generation shards while preserving row/feature alignment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def shard_index(path: Path) -> int:
    return int(path.stem.replace("generation_metrics_shard", ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--expected-rows", type=int, default=0)
    args = parser.parse_args()

    tables = args.output_dir / "tables"
    parts = sorted(tables.glob("generation_metrics_shard*.csv"), key=shard_index)
    if not parts:
        raise SystemExit(f"no generation shards under {tables}")
    frames: list[pd.DataFrame] = []
    feature_parts: list[np.ndarray] = []
    for part_index, part in enumerate(parts):
        frame = pd.read_csv(part)
        feature_path = part.with_name(
            part.name.replace("generation_metrics", "generated_clip_features").replace(".csv", ".float32.npy")
        )
        features = np.load(feature_path).astype(np.float32)
        if len(frame) != len(features):
            raise SystemExit(f"row/feature mismatch for {part}: {len(frame)} vs {len(features)}")
        frame["_feature_part"] = part_index
        frame["_feature_row"] = np.arange(len(frame))
        frames.append(frame)
        feature_parts.append(features)

    merged = pd.concat(frames, ignore_index=True)
    order = merged.sort_values(["user_index", "seed_offset"], kind="stable").index.to_numpy()
    features = np.stack(
        [
            feature_parts[int(merged.loc[index, "_feature_part"])][int(merged.loc[index, "_feature_row"])]
            for index in order
        ]
    ).astype(np.float32)
    merged = merged.loc[order].drop(columns=["_feature_part", "_feature_row"]).reset_index(drop=True)
    merged["method"] = args.method
    if args.expected_rows and len(merged) != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} rows, found {len(merged)}")
    if merged.duplicated(["user_index", "seed_offset"]).any():
        raise SystemExit("duplicate (user_index, seed_offset) rows")

    csv_tmp = tables / "generation_metrics.csv.tmp"
    npy_tmp = tables / "generated_clip_features.float32.npy.tmp"
    merged.to_csv(csv_tmp, index=False)
    with npy_tmp.open("wb") as handle:
        np.save(handle, features)
    os.replace(csv_tmp, tables / "generation_metrics.csv")
    os.replace(npy_tmp, tables / "generated_clip_features.float32.npy")
    summary = {
        "rows": int(len(merged)),
        "users": int(merged["user_index"].nunique()),
        "shards": len(parts),
        "method": args.method,
        "feature_alignment": "explicit shard-row remap after user/seed sort",
    }
    (tables / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
