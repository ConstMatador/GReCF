#!/usr/bin/env python3
"""ViPer adaptation step 2: personalized generation via preference-embedding
injection (adapter of baselines/rebeca/generate.py).

Core ViPer route kept intact: per-user structured liked/disliked attributes
(VP+ / VP-) are encoded by the SD text encoder and injected into the prompt
embedding, p = E(prompt) + beta * (E(VP+) - E(VP-)); no diffusion weights are
trained. Everything project-specific follows the shared protocol: SD1.5
decode, cohort, 10 images/user, seed = seed + user*10_000 + seed_offset,
per-seed initial latents, and the byte-identical CLIP eval feature path.
Outputs land in the standard evaluation contract (sharded).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET = Path("/path/to/datasets/CIGR")
DEFAULT_SD15 = Path("/path/to/models/stable-diffusion-v1-5")
DEFAULT_CLIP_MODEL = Path("/path/to/models/clip-vit-base-patch32")

sys.path.insert(0, str(REPO_ROOT))
from grecf.data import load_public_dataset  # noqa: E402
from grecf.image_embedder import DifferentiableCLIPImageEmbedder  # noqa: E402

# Reuse the exact contract column set / sentinel fields from the REBECA adapter.
sys.path.insert(0, str(REPO_ROOT / "baselines" / "rebeca"))
from generate import CONTRACT_COLUMNS, NAN_FIELDS  # noqa: E402  (rebeca adapter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preferences-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sd15-model", type=Path, default=DEFAULT_SD15)
    parser.add_argument("--clip-model", type=Path, default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument("--selected-users-json", type=Path, required=True)
    parser.add_argument("--method-name", default="ViPer")
    parser.add_argument("--images-per-user", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.5, help="ViPer personalization strength")
    parser.add_argument("--base-prompt", default="a photo")
    parser.add_argument("--pipe-cfg", type=float, default=4.5)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def encode_text(pipe, texts: list[str], device) -> torch.Tensor:
    """Token-level CLIP text embeddings [len, 77, 768], matching prompt_embeds."""
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    tokens = tokenizer(texts, padding="max_length", max_length=tokenizer.model_max_length,
                       truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        outputs = text_encoder(tokens)
    return outputs[0].detach()


def join_keywords(keywords: list[str], limit: int = 16) -> str:
    clean = [str(k).strip().strip(".").replace(",", " ") for k in keywords if str(k).strip()]
    return ", ".join(clean[:limit])


def main() -> int:
    args = parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    from diffusers import StableDiffusionPipeline
    from transformers import CLIPImageProcessor, CLIPModel

    dataset = load_public_dataset(args.dataset_root)
    selected = json.loads(args.selected_users_json.read_text())
    members = selected["user_indices"] if "user_indices" in selected else selected["all_user_indices"]
    cohort_users = [int(u) for u in sorted(members)]
    start = len(cohort_users) * args.shard_index // args.num_shards
    stop = len(cohort_users) * (args.shard_index + 1) // args.num_shards
    shard_users = cohort_users[start:stop]

    preferences: dict[int, dict] = {}
    if args.preferences_jsonl.exists():
        for line in args.preferences_jsonl.open(encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            preferences[int(row["user_index"])] = row

    from grecf.data import id_order_sha256, load_cache
    order_sha = id_order_sha256(dataset.image_ids)
    clip_features = np.asarray(
        load_cache(args.clip_cache, dataset.num_items, order_sha, (512,)), dtype=np.float32
    )

    image_size = int(CLIPImageProcessor.from_pretrained(args.clip_model, local_files_only=True).crop_size["height"])
    clip_model = CLIPModel.from_pretrained(args.clip_model, torch_dtype=torch.float32, local_files_only=True).to(device).eval()
    processor_mean = CLIPImageProcessor.from_pretrained(args.clip_model, local_files_only=True).image_mean
    processor_std = CLIPImageProcessor.from_pretrained(args.clip_model, local_files_only=True).image_std
    embedder = DifferentiableCLIPImageEmbedder(clip_model, image_size, processor_mean, processor_std).to(device)

    pipe = StableDiffusionPipeline.from_pretrained(args.sd15_model, safety_checker=None).to(device, torch_dtype=torch.float16)
    pipe.safety_checker = None

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    features_out: list[np.ndarray] = []

    catalog = torch.from_numpy(clip_features).to(device)
    sims_of = lambda items: (
        catalog[torch.from_numpy(np.asarray(items, dtype=np.int64)).to(device)]
        @ torch.from_numpy(feature).to(device)
    )
    top = lambda scores: float(torch.topk(scores, k=min(5, len(scores))).values.mean()) if len(scores) else float("nan")

    negative_embed = encode_text(pipe, [""], device)
    began = time.time()
    for position, user in enumerate(shard_users):
        pref = preferences.get(user, {})
        liked = join_keywords(pref.get("liked_attributes", []))
        disliked = join_keywords(pref.get("disliked_attributes", []))

        base = encode_text(pipe, [args.base_prompt], device)
        inject = torch.zeros_like(base)
        if liked:
            inject = inject + encode_text(pipe, [f"a photo of {liked}"], device)
        if disliked:
            inject = inject - encode_text(pipe, [f"a photo of {disliked}"], device)
        prompt_embed = (base + args.beta * inject).to(dtype=torch.float16)
        prompt_embeds = prompt_embed.repeat(args.images_per_user, 1, 1)
        negative_embeds = negative_embed.repeat(args.images_per_user, 1, 1).to(dtype=torch.float16)

        latents = []
        for seed_offset in range(args.images_per_user):
            seed = args.seed + user * 10_000 + seed_offset
            generator = torch.Generator(device=device).manual_seed(seed)
            latents.append(torch.randn((1, 4, 64, 64), generator=generator, device=device, dtype=torch.float16))

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            images = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                guidance_scale=args.pipe_cfg,
                num_inference_steps=args.num_inference_steps,
                latents=torch.cat(latents, dim=0),
            ).images
        assert len(images) == args.images_per_user

        for seed_offset, result in enumerate(images):
            seed = args.seed + user * 10_000 + seed_offset
            pixel = torch.from_numpy(np.asarray(result)).to(device=device, dtype=torch.float32)
            pixel = pixel.permute(2, 0, 1) / 127.5 - 1.0
            with torch.no_grad():
                feature = embedder(pixel.unsqueeze(0)).float().cpu().numpy()[0]
            features_out.append(feature)

            train_items = dataset.train[user]
            test_items = dataset.test[user]
            validation_items = dataset.validation[user]
            row = {
                **NAN_FIELDS,
                "method": args.method_name,
                "user_index": int(user),
                "user_id": str(dataset.user_ids[user]),
                "seed": int(seed),
                "seed_offset": int(seed_offset),
                "interest_count": 0,
                "wrong_interest_count": 0,
                "variant": "personalized",
                "image_path": "",
                "train_maxsim": float(sims_of(train_items).max()) if len(train_items) else float("nan"),
                "test_maxsim": float(sims_of(test_items).max()) if len(test_items) else float("nan"),
                "assigned_train_topic_index": -1,
                "assigned_train_topic_similarity": float("nan"),
                "train_top5": top(sims_of(train_items)),
                "validation_top5": top(sims_of(validation_items)),
                "test_top5": top(sims_of(test_items)),
                "nearest_catalog_index": -1,
                "nearest_catalog_image_id": "",
                "nearest_catalog_similarity": float("nan"),
            }
            filename = f"{dataset.user_ids[user]}_seed{seed_offset}_personalized.png"
            result.save(images_dir / filename)
            row["image_path"] = str(images_dir / filename)
            rows.append(row)
        done = position + 1
        if done % 10 == 0 or done == len(shard_users):
            rate = done / max(time.time() - began, 1e-9)
            print(json.dumps({"shard": args.shard_index, "users_done": done,
                              "users_total": len(shard_users), "users_per_sec": round(rate, 3),
                              "user": user, "liked_kw": len(pref.get("liked_attributes", [])),
                              "disliked_kw": len(pref.get("disliked_attributes", []))}), flush=True)

    suffix = "" if args.num_shards == 1 else f"_shard{args.shard_index}"
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=CONTRACT_COLUMNS)
    frame.to_csv(tables_dir / f"generation_metrics{suffix}.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    np.save(tables_dir / f"generated_clip_features{suffix}.float32.npy",
            np.stack(features_out).astype(np.float32))
    (tables_dir / f"generation_summary{suffix}.json").write_text(json.dumps({
        "rows": len(rows), "users_shard": len(shard_users), "range": [start, stop],
        "beta": args.beta, "base_prompt": args.base_prompt, "pipe_cfg": args.pipe_cfg,
        "steps": args.num_inference_steps, "seed": args.seed, "method": args.method_name,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
