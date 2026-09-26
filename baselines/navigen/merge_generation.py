#!/usr/bin/env python3
"""Merge NaviGen generation shards while preserving feature/table row alignment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-name", default="NaviGen")
    parser.add_argument("--expected-users", type=int, default=1000)
    parser.add_argument("--images-per-user", type=int, default=10)
    args = parser.parse_args()
    tables = args.output_dir / "tables"
    metric_parts = sorted(tables.glob("generation_metrics_shard*.csv"))
    if not metric_parts:
        raise SystemExit(f"no generation shards under {tables}")
    combined = []
    instructions = []
    for part in metric_parts:
        shard = part.stem.rsplit("_shard", 1)[-1]
        frame = pd.read_csv(part)
        features = np.load(tables / f"generated_clip_features_shard{shard}.float32.npy")
        if len(frame) != len(features):
            raise ValueError(f"row/feature mismatch in shard {shard}")
        frame["_feature"] = [value for value in features]
        combined.append(frame)
        instructions.extend(
            json.loads(line)
            for line in (tables / f"instructions_shard{shard}.jsonl").open(encoding="utf-8")
            if line.strip()
        )
    merged = pd.concat(combined, ignore_index=True).sort_values(["user_index", "seed_offset"])
    features = np.stack(merged.pop("_feature").tolist()).astype(np.float32)
    merged = merged.reset_index(drop=True)
    expected_rows = args.expected_users * args.images_per_user
    if len(merged) != expected_rows or merged["user_index"].nunique() != args.expected_users:
        raise ValueError(
            f"formal generation incomplete: rows={len(merged)}, users={merged['user_index'].nunique()}"
        )
    counts = merged.groupby("user_index").size()
    if not bool((counts == args.images_per_user).all()):
        raise ValueError("not every user has the formal image budget")
    merged.to_csv(tables / "generation_metrics.csv", index=False)
    np.save(tables / "generated_clip_features.float32.npy", features)
    instructions = sorted(instructions, key=lambda row: int(row["user_index"]))
    with (tables / "instructions.jsonl").open("w", encoding="utf-8") as handle:
        for row in instructions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = pd.DataFrame(
        {
            "eval_method": merged["method"],
            "user_id": merged["user_id"],
            "seed_offset": merged["seed_offset"],
            "resolved_image_path": merged["image_path"],
            "route": "direct",
            "route_id": 0,
        }
    )
    manifest.to_csv(tables / "generated_image_clip_features_manifest.csv", index=False)
    summary = {
        "method": args.method_name,
        "rows": len(merged),
        "users": int(merged["user_index"].nunique()),
        "images_per_user": args.images_per_user,
        "feature_shape": list(features.shape),
    }
    (tables / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
