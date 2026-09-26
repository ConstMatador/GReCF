#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "baselines" / "rebeca"))
from generate import CONTRACT_COLUMNS, NAN_FIELDS  # noqa: E402

from grecf.data import id_order_sha256, load_cache, load_public_dataset  # noqa: E402
from grecf.image_embedder import DifferentiableCLIPImageEmbedder  # noqa: E402
from grecf.pixart import PixArtPrefixAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-context", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument("--clip-model", type=Path, required=True)
    parser.add_argument("--interest-cache-dir", type=Path, required=True)
    parser.add_argument("--selected-users-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-name", default="GReCF (PixArt)")
    parser.add_argument("--images-per-user", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def weighted_schedule(weights: np.ndarray, mask: np.ndarray, count: int) -> list[int]:
    active = np.flatnonzero(mask.astype(bool))
    probabilities = weights[active].astype(np.float64)
    probabilities = probabilities / probabilities.sum()
    allocation = np.zeros(len(active), dtype=np.int64)
    if count >= len(active):
        allocation += 1
        remaining = count - len(active)
    else:
        remaining = count
    raw = probabilities * remaining
    allocation += np.floor(raw).astype(np.int64)
    missing = count - int(allocation.sum())
    allocation[np.argsort(-(raw - np.floor(raw)), kind="stable")[:missing]] += 1
    result: list[int] = []
    while len(result) < count:
        for local, item in enumerate(active):
            if allocation[local] > 0:
                result.append(int(item))
                allocation[local] -= 1
    return result


def main() -> int:
    args = parse_args()
    from diffusers import PixArtAlphaPipeline
    from transformers import CLIPImageProcessor, CLIPModel

    device = torch.device("cuda")
    dtype = torch.bfloat16
    dataset = load_public_dataset(args.dataset_root)
    selected = json.loads(args.selected_users_json.read_text(encoding="utf-8"))
    members = selected.get("user_indices", selected.get("all_user_indices"))
    users = sorted(int(value) for value in members)
    start = len(users) * args.shard_index // args.num_shards
    stop = len(users) * (args.shard_index + 1) // args.num_shards
    users = users[start:stop]

    order_sha = id_order_sha256(dataset.image_ids)
    clip = np.asarray(load_cache(args.clip_cache, dataset.num_items, order_sha, (512,)), dtype=np.float32)
    prototypes = np.load(args.interest_cache_dir / "adaptive_b_prototypes.float32.npy", mmap_mode="r")
    weights = np.load(args.interest_cache_dir / "adaptive_b_weights.float32.npy", mmap_mode="r")
    masks = np.load(args.interest_cache_dir / "adaptive_b_mask.bool.npy", mmap_mode="r")
    auxiliary = np.load(args.interest_cache_dir / "adaptive_b_auxiliary.float32.npy", mmap_mode="r")
    context_state = torch.load(args.base_context, map_location="cpu", weights_only=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    adapter = PixArtPrefixAdapter(
        context_state["prompt_embeds"].float(),
        context_state["prompt_attention_mask"].bool(),
        auxiliary_center=torch.from_numpy(np.asarray(auxiliary).mean(axis=0, keepdims=True)),
    ).to(device).eval()
    adapter.load_state_dict(checkpoint["adapter"])

    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model,
        text_encoder=None,
        tokenizer=None,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    processor = CLIPImageProcessor.from_pretrained(args.clip_model, local_files_only=True)
    clip_model = CLIPModel.from_pretrained(
        args.clip_model,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).to(device).eval()
    image_size = int(processor.crop_size["height"])
    embedder = DifferentiableCLIPImageEmbedder(
        clip_model, image_size, processor.image_mean, processor.image_std
    ).to(device)
    catalog = torch.from_numpy(clip).to(device)

    images_dir = args.output_dir / "images"
    tables_dir = args.output_dir / "tables"
    images_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    features: list[np.ndarray] = []
    began = time.time()

    for position, user in enumerate(users, start=1):
        schedule = weighted_schedule(weights[user], masks[user], args.images_per_user)
        user_prototypes = torch.from_numpy(
            np.repeat(np.asarray(prototypes[user:user + 1]), args.images_per_user, axis=0).copy()
        ).to(device)
        user_masks = torch.from_numpy(
            np.repeat(np.asarray(masks[user:user + 1]), args.images_per_user, axis=0).copy()
        ).to(device)
        user_auxiliary = torch.from_numpy(
            np.repeat(np.asarray(auxiliary[user:user + 1]), args.images_per_user, axis=0).copy()
        ).to(device)
        selected_interests = user_prototypes[
            torch.arange(args.images_per_user, device=device),
            torch.tensor(schedule, device=device),
        ]
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            context, context_mask, _ = adapter(
                user_prototypes, user_masks, user_auxiliary, selected_interests
            )
            negative_context, negative_mask = adapter.unconditional(args.images_per_user)
        latent_rows = []
        for seed_offset in range(args.images_per_user):
            seed = args.seed + user * 10_000 + seed_offset
            generator = torch.Generator(device=device).manual_seed(seed)
            latent_rows.append(torch.randn((1, 4, 64, 64), generator=generator, device=device, dtype=dtype))
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            generated = pipe(
                prompt=None,
                negative_prompt=None,
                prompt_embeds=context.to(dtype),
                prompt_attention_mask=context_mask,
                negative_prompt_embeds=negative_context.to(dtype),
                negative_prompt_attention_mask=negative_mask,
                latents=torch.cat(latent_rows),
                height=512,
                width=512,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
            ).images
        for seed_offset, image in enumerate(generated):
            array = np.asarray(image).copy()
            pixel = torch.from_numpy(array).to(device=device, dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1
            with torch.inference_mode():
                feature = embedder(pixel.unsqueeze(0)).float().cpu().numpy()[0]
            features.append(feature)
            path = images_dir / f"{dataset.user_ids[user]}_seed{seed_offset}_personalized.png"
            image.save(path)
            train_scores = clip[dataset.train[user]] @ feature
            test_scores = clip[dataset.test[user]] @ feature
            validation_scores = clip[dataset.validation[user]] @ feature
            top = lambda values: float(np.sort(values)[-min(5, len(values)):].mean())
            rows.append({
                **NAN_FIELDS,
                "method": args.method_name,
                "user_index": user,
                "user_id": str(dataset.user_ids[user]),
                "seed": args.seed + user * 10_000 + seed_offset,
                "seed_offset": seed_offset,
                "interest_index": schedule[seed_offset],
                "interest_count": int(masks[user].sum()),
                "wrong_interest_count": 0,
                "variant": "personalized",
                "image_path": str(path),
                "train_maxsim": float(train_scores.max()),
                "test_maxsim": float(test_scores.max()),
                "train_top5": top(train_scores),
                "validation_top5": top(validation_scores),
                "test_top5": top(test_scores),
                "assigned_train_topic_index": -1,
                "assigned_train_topic_similarity": float("nan"),
                "nearest_catalog_index": -1,
                "nearest_catalog_image_id": "",
                "nearest_catalog_similarity": float("nan"),
            })
        if position % 10 == 0 or position == len(users):
            print(json.dumps({
                "shard": args.shard_index,
                "users_done": position,
                "users_total": len(users),
                "users_per_sec": round(position / max(time.time() - began, 1e-9), 4),
            }), flush=True)

    suffix = "" if args.num_shards == 1 else f"_shard{args.shard_index}"
    pd.DataFrame(rows, columns=CONTRACT_COLUMNS).to_csv(
        tables_dir / f"generation_metrics{suffix}.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )
    np.save(
        tables_dir / f"generated_clip_features{suffix}.float32.npy",
        np.stack(features).astype(np.float32),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
