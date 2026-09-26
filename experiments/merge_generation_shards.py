#!/usr/bin/env python3
"""Merge standard generation shards with explicit row/feature alignment."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


SHARD_PATTERN = re.compile(r"generation_metrics_shard(\d+)\.csv$")


def shard_index(path: Path) -> int:
    match = SHARD_PATTERN.search(path.name)
    if match is None:
        raise ValueError(path)
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args()

    tables = args.generation_dir / "tables"
    parts = sorted(tables.glob("generation_metrics_shard*.csv"), key=shard_index)
    if len(parts) != args.expected_shards:
        raise SystemExit(f"expected {args.expected_shards} shards, found {len(parts)}")
    frames: list[pd.DataFrame] = []
    arrays: list[np.ndarray] = []
    for part_index, part in enumerate(parts):
        frame = pd.read_csv(part)
        feature_path = part.with_name(
            part.name.replace("generation_metrics", "generated_clip_features").replace(".csv", ".float32.npy")
        )
        features = np.load(feature_path).astype(np.float32)
        if len(frame) != len(features):
            raise SystemExit(f"row/feature mismatch for {part}: {len(frame)} vs {len(features)}")
        frame["_part"] = part_index
        frame["_row"] = np.arange(len(frame))
        frames.append(frame)
        arrays.append(features)
    merged = pd.concat(frames, ignore_index=True)
    order = merged.sort_values(["user_index", "seed_offset", "variant"], kind="stable").index.to_numpy()
    features = np.stack(
        [arrays[int(merged.loc[index, "_part"])][int(merged.loc[index, "_row"])] for index in order]
    ).astype(np.float32)
    merged = merged.loc[order].drop(columns=["_part", "_row"]).reset_index(drop=True)
    if len(merged) != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} rows, found {len(merged)}")
    if merged.duplicated(["user_index", "seed_offset", "variant"]).any():
        raise SystemExit("duplicate generation rows")

    csv_tmp = tables / "generation_metrics.csv.tmp"
    npy_tmp = tables / "generated_clip_features.float32.npy.tmp"
    merged.to_csv(csv_tmp, index=False)
    with npy_tmp.open("wb") as handle:
        np.save(handle, features)
    os.replace(csv_tmp, tables / "generation_metrics.csv")
    os.replace(npy_tmp, tables / "generated_clip_features.float32.npy")
    manifest = pd.DataFrame(
        {
            "eval_method": merged["method"].astype(str),
            "user_id": merged["user_id"].astype(str),
            "seed_offset": merged["seed_offset"].astype(int),
            "resolved_image_path": merged["image_path"].astype(str),
        }
    )
    manifest.to_csv(tables / "generated_image_clip_features_manifest.csv", index=False)
    summary = {
        "rows": len(merged),
        "users": int(merged["user_index"].nunique()),
        "shards": len(parts),
        "feature_alignment": "explicit shard-row remap after user/seed/variant sort",
    }
    (tables / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
