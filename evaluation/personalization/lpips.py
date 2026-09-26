#!/usr/bin/env python3
"""Compute minimum LPIPS distance to each user's held-out test images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image

from grecf.data import load_public_dataset


def image_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    values = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(values).permute(2, 0, 1)


def resolve_path(value: str, metrics_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for parent in (Path.cwd(), *metrics_path.resolve().parents):
        candidate = parent / path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path)


def minimum_distances(
    model,
    queries: torch.Tensor,
    references: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    output: list[float] = []
    with torch.inference_mode():
        for query in queries:
            distances = []
            for start in range(0, len(references), batch_size):
                reference = references[start : start + batch_size].to(device)
                repeated = query.unsqueeze(0).expand(len(reference), -1, -1, -1).to(device)
                distances.append(model(repeated, reference).flatten().cpu())
            output.append(float(torch.cat(distances).min()))
    return np.asarray(output, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-metrics", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset = load_public_dataset(args.dataset_root)
    frame = pd.read_csv(args.generation_metrics)
    required = {"user_index", "user_id", "image_path"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"generation metrics missing columns: {sorted(missing)}")

    device = torch.device(args.device)
    model = lpips.LPIPS(net="alex").to(device).eval()
    model.requires_grad_(False)
    records: list[dict[str, object]] = []
    for user, group in frame.groupby("user_index", sort=True):
        user = int(user)
        references = torch.stack(
            [image_tensor(Path(dataset.image_paths[int(item)]), args.image_size) for item in dataset.test[user]]
        )
        queries = torch.stack(
            [image_tensor(resolve_path(str(path), args.generation_metrics), args.image_size) for path in group["image_path"]]
        )
        distances = minimum_distances(model, queries, references, args.batch_size, device)
        for row, distance in zip(group.itertuples(index=False), distances.tolist(), strict=True):
            records.append(
                {
                    "method": args.method,
                    "user_index": user,
                    "user_id": str(row.user_id),
                    "image_path": str(row.image_path),
                    "lpips": float(distance),
                }
            )

    per_image = pd.DataFrame(records)
    user_means = per_image.groupby("user_id", sort=True)["lpips"].mean()
    summary = {
        "method": args.method,
        "metric": "LPIPS",
        "images": int(len(per_image)),
        "users": int(per_image["user_id"].nunique()),
        "mean": float(per_image["lpips"].mean()),
        "user_macro_mean": float(user_means.mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(args.output.with_suffix(".per_image.csv"), index=False)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
