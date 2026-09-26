#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--expected-users", type=int, default=100)
    parser.add_argument("--expected-rows", type=int, default=1000)
    args = parser.parse_args()
    table_dir = args.generation_dir / "tables"
    parts = sorted(table_dir.glob("generation_metrics_shard*.csv"))
    if len(parts) != args.shards:
        raise ValueError(f"expected {args.shards} shards, found {len(parts)}")
    frames = []
    arrays = []
    for part in parts:
        shard = part.stem.removeprefix("generation_metrics_shard")
        frame = pd.read_csv(part)
        array = np.load(table_dir / f"generated_clip_features_shard{shard}.float32.npy")
        if len(frame) != len(array):
            raise ValueError(f"row/feature mismatch for {part}")
        frame["_part"] = len(arrays)
        frame["_row"] = np.arange(len(frame))
        frames.append(frame)
        arrays.append(array)
    merged = pd.concat(frames, ignore_index=True)
    order = merged.sort_values(["user_index", "seed_offset", "variant"]).index.to_numpy()
    features = np.stack(
        [arrays[int(merged.loc[index, "_part"])][int(merged.loc[index, "_row"])] for index in order]
    ).astype(np.float32)
    merged = merged.loc[order].drop(columns=["_part", "_row"]).reset_index(drop=True)
    if len(merged) != args.expected_rows or merged["user_id"].nunique() != args.expected_users:
        raise ValueError(f"unexpected merged generation shape: rows={len(merged)}, users={merged['user_id'].nunique()}")
    merged.to_csv(table_dir / "generation_metrics.csv", index=False)
    np.save(table_dir / "generated_clip_features.float32.npy", features)
    manifest = merged.copy()
    manifest["eval_method"] = manifest["method"]
    manifest["resolved_image_path"] = manifest["image_path"]
    manifest.to_csv(table_dir / "generated_image_clip_features_manifest.csv", index=False)
    summary = {"rows": len(merged), "users": int(merged["user_id"].nunique()), "shards": args.shards}
    (table_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
