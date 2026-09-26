#!/usr/bin/env python3
"""Build or merge MovieLens CLIP and SD1.5 latent caches."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from grecf.data import id_order_sha256, load_public_dataset


class ImageDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        indices: np.ndarray,
        kind: str,
        latent_resize_mode: str,
    ) -> None:
        self.paths = paths
        self.indices = indices
        self.kind = kind
        self.latent_resize_mode = latent_resize_mode

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, offset: int):
        index = int(self.indices[offset])
        with Image.open(self.paths[index]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if self.kind == "clip":
                return index, image
            if self.latent_resize_mode == "center_crop":
                image = ImageOps.fit(
                    image,
                    (512, 512),
                    method=Image.Resampling.BICUBIC,
                    centering=(0.5, 0.5),
                )
            else:
                image = image.resize((512, 512), Image.Resampling.BICUBIC)
            array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
            return index, torch.from_numpy(array).permute(2, 0, 1)


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("clip", "latent"), required=True)
    parser.add_argument("--model", type=Path, default=Path("/path/to/models/stable-diffusion-v1-5"))
    parser.add_argument("--clip-model", type=Path, default=Path("/path/to/models/clip-vit-base-patch32"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--latent-resize-mode",
        choices=("stretch", "center_crop"),
        default="stretch",
        help="SD latent preprocessing. center_crop preserves aspect ratio before the 512px crop.",
    )
    parser.add_argument("--merge-shards", nargs="*", type=Path, default=None)
    return parser.parse_args()


def merge(args: argparse.Namespace) -> int:
    dataset = load_public_dataset(args.dataset_root)
    order_sha = id_order_sha256(dataset.image_ids)
    shards = sorted(args.merge_shards or [])
    if not shards:
        raise ValueError("--merge-shards requires at least one shard")
    arrays = [np.load(path, mmap_mode="r") for path in shards]
    expected_tail = (512,) if args.kind == "clip" else (4, 64, 64)
    if sum(len(array) for array in arrays) != dataset.num_items:
        raise ValueError("cache shard rows do not cover the dataset exactly")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=np.float16, shape=(dataset.num_items, *expected_tail)
    )
    cursor = 0
    for array in arrays:
        if array.shape[1:] != expected_tail:
            raise ValueError(f"unexpected shard shape {array.shape}")
        merged[cursor : cursor + len(array)] = array
        cursor += len(array)
    merged.flush()
    write_json(args.output.with_suffix(".json"), {
        "complete": True,
        "rows": dataset.num_items,
        "shape": [dataset.num_items, *expected_tail],
        "dtype": "float16",
        "image_id_order_sha256": order_sha,
        "dataset_root": str(dataset.root),
        "kind": args.kind,
        "model_path": str(args.clip_model if args.kind == "clip" else args.model),
        "public_only": True,
        "latent_resize_mode": args.latent_resize_mode if args.kind == "latent" else None,
    })
    print(json.dumps({"stage": "merged", "kind": args.kind, "rows": dataset.num_items, "output": str(args.output)}), flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.merge_shards is not None:
        return merge(args)
    dataset = load_public_dataset(args.dataset_root)
    order_sha = id_order_sha256(dataset.image_ids)
    start = max(0, args.start)
    end = dataset.num_items if args.end <= 0 else min(args.end, dataset.num_items)
    if not 0 <= start < end <= dataset.num_items:
        raise ValueError(f"invalid range [{start}, {end}) for {dataset.num_items} items")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "clip":
        from transformers import CLIPImageProcessor, CLIPModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        processor = CLIPImageProcessor.from_pretrained(args.clip_model, local_files_only=True)
        model = CLIPModel.from_pretrained(args.clip_model, dtype=dtype, local_files_only=True).to(device).eval()
        model.requires_grad_(False)

        def collate(batch):
            indices, images = zip(*batch)
            inputs = processor(images=list(images), return_tensors="pt")
            return torch.tensor(indices, dtype=torch.long), inputs["pixel_values"]
        tail = (512,)
    else:
        from diffusers import AutoencoderKL

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = AutoencoderKL.from_pretrained(
            args.model, subfolder="vae", torch_dtype=dtype, local_files_only=True
        ).to(device).eval()
        model.requires_grad_(False)
        scaling = float(model.config.scaling_factor)
        collate = None
        tail = (4, 64, 64)

    indices = np.arange(start, end, dtype=np.int64)
    loader = DataLoader(
        ImageDataset(dataset.image_paths, indices, args.kind, args.latent_resize_mode),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
        collate_fn=collate,
    )
    output = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.float16, shape=(len(indices), *tail))
    started = time.time()
    completed = 0
    for batch_no, (batch_indices, batch_images) in enumerate(loader, start=1):
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
            if args.kind == "clip":
                values = model.get_image_features(pixel_values=batch_images.to(device, dtype=dtype, non_blocking=True))
                values = torch.nn.functional.normalize(values.float(), dim=-1)
            else:
                values = model.encode(batch_images.to(device, dtype=dtype, non_blocking=True)).latent_dist.mean * scaling
        local = batch_indices.numpy() - start
        output[local] = values.float().cpu().numpy().astype(np.float16)
        completed += len(local)
        if batch_no % args.log_every == 0 or completed == len(indices):
            output.flush()
            print(json.dumps({"kind": args.kind, "completed": completed, "total": len(indices), "elapsed": round(time.time() - started, 1)}), flush=True)
    output.flush()
    write_json(args.output.with_suffix(".json"), {
        "complete": True,
        "rows": len(indices),
        "shape": [len(indices), *tail],
        "dtype": "float16",
        "image_id_order_sha256": order_sha,
        "dataset_root": str(dataset.root),
        "kind": args.kind,
        "start": start,
        "end": end,
        "model_path": str(args.clip_model if args.kind == "clip" else args.model),
        "scaling_factor": scaling if args.kind == "latent" else None,
        "public_only": True,
        "latent_resize_mode": args.latent_resize_mode if args.kind == "latent" else None,
    })
    print(json.dumps({"stage": "done", "kind": args.kind, "rows": len(indices), "output": str(args.output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
