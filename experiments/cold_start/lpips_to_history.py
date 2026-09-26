#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image


def image_tensor(path: str, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    values = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(values).permute(2, 0, 1)


def min_distances(model, queries: torch.Tensor, references: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    result = []
    with torch.inference_mode():
        for query in queries:
            distances = []
            for start in range(0, len(references), batch_size):
                reference_batch = references[start : start + batch_size].to(device, non_blocking=True)
                query_batch = query.unsqueeze(0).expand(len(reference_batch), -1, -1, -1).to(device, non_blocking=True)
                distances.append(model(query_batch, reference_batch).flatten().cpu())
            result.append(float(torch.cat(distances).min()))
    return np.asarray(result, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        parts = [args.output_dir / f"lpips_to_history_per_image_shard{index}.csv" for index in range(args.num_shards)]
        if not all(path.is_file() for path in parts):
            raise FileNotFoundError("not all LPIPS shards are complete")
        result = pd.concat((pd.read_csv(path) for path in parts), ignore_index=True)
        result.to_csv(args.output_dir / "lpips_to_history_per_image.csv", index=False)
        user_means = result.groupby("user_id")["lpips_to_input_history_min"].mean()
        summary = {
            "images": len(result),
            "users": int(result["user_id"].nunique()),
            "lpips_to_input_history_min_mean": float(result["lpips_to_input_history_min"].mean()),
            "user_mean": float(user_means.mean()),
            "user_std": float(user_means.std(ddof=0)),
            "image_size": args.image_size,
            "reference": "all images used as conditioning history for the current round",
        }
        (args.output_dir / "lpips_to_history_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary), flush=True)
        return 0

    frame = pd.read_csv(args.generation_dir / "tables/generation_metrics.csv")
    users = np.asarray(sorted(frame["user_index"].unique()), dtype=np.int64)
    shard_users = set(users[args.shard_index :: args.num_shards].tolist())
    frame = frame.loc[frame["user_index"].isin(shard_users)].copy()
    history = pd.read_parquet(args.context_root / "cold_history.parquet")
    device = torch.device(args.device)
    model = lpips.LPIPS(net="alex").to(device).eval()
    model.requires_grad_(False)
    rows = []
    for processed, (user_index, group) in enumerate(frame.groupby("user_index", sort=True), 1):
        user_id = str(group["user_id"].iloc[0])
        history_paths = history.loc[history["user_id"].astype(str).eq(user_id), "path"].astype(str).tolist()
        references = torch.stack([image_tensor(path, args.image_size) for path in history_paths])
        queries = torch.stack([image_tensor(path, args.image_size) for path in group["image_path"].astype(str)])
        distances = min_distances(model, queries, references, args.batch_size, device)
        for row, distance in zip(group.itertuples(index=False), distances.tolist(), strict=True):
            rows.append(
                {
                    "user_index": int(user_index),
                    "user_id": user_id,
                    "seed_offset": int(row.seed_offset),
                    "history_size": len(history_paths),
                    "lpips_to_input_history_min": distance,
                }
            )
        if processed % 25 == 0:
            print(json.dumps({"users": processed, "images": len(rows)}), flush=True)
    pd.DataFrame(rows).to_csv(args.output_dir / f"lpips_to_history_per_image_shard{args.shard_index}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
