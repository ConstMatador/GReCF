#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def user_id_from_filename(filename: str) -> str:
    """Read the user identifier preceding the generated-image suffix."""
    return filename.split("_seed", 1)[0]


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], preprocess) -> None:
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.preprocess(image.convert("RGB"))
        return tensor, str(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--predictor-weights",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--clip-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.image_dir.glob("*.png"))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no PNG images found under {args.image_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = open_clip.load_openai_model(
        str(args.clip_checkpoint),
        precision="fp16" if device.type == "cuda" else "fp32",
        device=device,
    )
    preprocess = open_clip.image_transform(
        model.visual.image_size,
        is_train=False,
        mean=open_clip.OPENAI_DATASET_MEAN,
        std=open_clip.OPENAI_DATASET_STD,
        resize_mode="shortest",
        interpolation="bicubic",
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    predictor = torch.nn.Linear(768, 1)
    state = torch.load(args.predictor_weights, map_location="cpu", weights_only=True)
    predictor.load_state_dict(state)
    predictor.to(device=device, dtype=torch.float32).eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)

    loader = DataLoader(
        ImageDataset(paths, preprocess),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    records: list[dict[str, object]] = []
    processed = 0
    with torch.inference_mode():
        for images, batch_paths in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                features = model.encode_image(images, normalize=True)
            scores = predictor(features.float()).squeeze(1).cpu().numpy()
            for path_text, score in zip(batch_paths, scores, strict=True):
                filename = Path(path_text).name
                records.append(
                    {
                        "method": args.method,
                        "image_path": path_text,
                        "filename": filename,
                        "user_id": user_id_from_filename(filename),
                        "laion_aesthetic_score": float(score),
                    }
                )
            processed += len(scores)
            print(
                json.dumps(
                    {"processed": processed, "total": len(paths)},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    frame = pd.DataFrame(records)
    frame.to_csv(args.output_dir / "laion_aesthetic_per_image.csv", index=False)
    scores = frame["laion_aesthetic_score"].to_numpy(dtype=np.float64)
    user_means = (
        frame.groupby("user_id", sort=True)["laion_aesthetic_score"]
        .mean()
        .to_numpy(dtype=np.float64)
    )
    summary = {
        "method": args.method,
        "metric": "LAION Aesthetic Predictor V1",
        "clip_backbone": "ViT-L-14/openai",
        "clip_checkpoint": str(args.clip_checkpoint),
        "clip_checkpoint_sha256": sha256(args.clip_checkpoint),
        "predictor_weights": str(args.predictor_weights),
        "predictor_sha256": sha256(args.predictor_weights),
        "images": int(len(frame)),
        "users": int(frame["user_id"].nunique()),
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=0)),
        "median": float(np.median(scores)),
        "p05": float(np.quantile(scores, 0.05)),
        "p25": float(np.quantile(scores, 0.25)),
        "p75": float(np.quantile(scores, 0.75)),
        "p95": float(np.quantile(scores, 0.95)),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "fraction_ge_5": float((scores >= 5.0).mean()),
        "fraction_ge_6": float((scores >= 6.0).mean()),
        "user_mean": float(user_means.mean()),
        "user_std": float(user_means.std(ddof=0)),
    }
    (args.output_dir / "laion_aesthetic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
