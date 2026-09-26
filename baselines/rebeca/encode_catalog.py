#!/usr/bin/env python3
"""Encode catalog images into the IP-Adapter SD1.5 embedding space.

The upstream REBECA prior generates embeddings that are decoded by
StableDiffusionPipeline + IP-Adapter, so training targets must live in that
pipeline's `encode_image` space (this is what upstream's preprocessing notebooks
produce). This script runs the same path over every catalog image and stores one
float16 tensor ordered by GReCF's image identifiers.

Multi-GPU: run with --num-shards/--shard-index like experiments/sd15/generate.py,
then merge with --merge-shards.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

SD15_MODEL = Path("/path/to/models/stable-diffusion-v1-5")
IP_ADAPTER_DIR = Path("/path/to/models/IP-Adapter")
IP_ADAPTER_WEIGHT = "ip-adapter_sd15.safetensors"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def open_rgb(path: Path):
    from PIL import Image
    with Image.open(path) as handle:
        return handle.convert("RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge-shards", action="store_true",
                        help="skip encoding; concatenate rebeca_catalog_shards/shard*.pt into --output-file")
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/path/to/datasets/CIGR"))
    parser.add_argument("--sd15-model", type=Path, default=SD15_MODEL)
    parser.add_argument("--ip-adapter-dir", type=Path, default=IP_ADAPTER_DIR)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--decode-workers", type=int, default=32,
                        help="threads for PNG decoding; catalog images are 1024x1024 and CPU-bound")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-file", type=Path, required=True,
                        help="storage-side .pt path for the [num_items, D] float16 matrix")
    return parser.parse_args()


def merge_shards(output_file: Path) -> int:
    parts_dir = output_file.parent / "rebeca_catalog_shards"
    parts = sorted(parts_dir.glob("shard*.pt"))
    if not parts:
        raise SystemExit(f"no shard files under {parts_dir}")
    merged = torch.cat([torch.load(part, weights_only=True) for part in parts], dim=0).contiguous()
    torch.save(merged, output_file)
    print(f"merged {len(parts)} shards -> rows={merged.shape[0]} dim={merged.shape[1]}")
    return 0


def encode(args: argparse.Namespace, device: str) -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from grecf.data import load_public_dataset

    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(args.sd15_model, safety_checker=None).to(device)
    pipe.load_ip_adapter(str(args.ip_adapter_dir), subfolder="models", weight_name=IP_ADAPTER_WEIGHT)
    pipe.safety_checker = None
    encode = pipe.encode_image

    dataset = load_public_dataset(args.dataset_root)
    shard_start = dataset.num_items * args.shard_index // args.num_shards
    shard_stop = dataset.num_items * (args.shard_index + 1) // args.num_shards
    paths = dataset.image_paths[shard_start:shard_stop]

    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    chunks: list[torch.Tensor] = []
    done = 0
    began = time.time()
    # PNG decode of the 1024x1024 catalog images is CPU-bound and would leave the
    # GPU idle; decode batches ahead of the encoder in a thread pool instead.
    with ThreadPoolExecutor(max_workers=args.decode_workers) as pool:
        pending: deque = deque()
        next_index = 0
        while next_index < len(paths) or pending:
            while next_index < len(paths) and len(pending) < 2:
                start, stop = next_index, min(next_index + args.batch_size, len(paths))
                pending.append(pool.map(open_rgb, paths[start:stop]))
                next_index = stop
            images = list(pending.popleft())
            with torch.inference_mode():
                embeds = encode(images, device=device, num_images_per_prompt=1)
            # embeds may be a tuple (embeds, negative_embeds); keep the positive half only
            tensor = embeds[0] if isinstance(embeds, tuple) else embeds
            chunks.append(tensor.float().cpu())
            done += len(images)
            if done % (args.batch_size * 20) < args.batch_size:
                rate = done / max(time.time() - began, 1e-9)
                print(f"[shard {args.shard_index}/{args.num_shards}] {done}/{len(paths)} ({rate:.1f} img/s)", flush=True)

    features = torch.cat(chunks, dim=0).to(torch.float16)
    # Squeeze sequence axes so the stored tensor is exactly [rows, feature_dim * tokens]
    features = features.reshape(features.shape[0], -1)

    if args.num_shards > 1:
        parts_dir = args.output_file.parent / "rebeca_catalog_shards"
        parts_dir.mkdir(parents=True, exist_ok=True)
        torch.save(features, parts_dir / f"shard{args.shard_index}.pt")
    else:
        torch.save(features, args.output_file)

    write_json(args.output_file.with_suffix(".json"), {
        "num_items": int(dataset.num_items),
        "shard": [int(args.shard_index), int(args.num_shards)],
        "range": [int(shard_start), int(shard_stop)],
        "rows_in_shard": int(features.shape[0]),
        "row_dim": int(features.shape[1]),
        "dtype": "float16",
        "sd15_model": str(args.sd15_model),
        "ip_adapter_weight": IP_ADAPTER_WEIGHT,
    })
    print(f"encoded rows={features.shape[0]} dim={features.shape[1]}")
    return 0


def main() -> int:
    args = parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.merge_shards:
        return merge_shards(args.output_file)
    return encode(args, device)


if __name__ == "__main__":
    raise SystemExit(main())
