#!/usr/bin/env python3
"""Generate personalized images with the trained REBECA prior (adapter step 3).

Pipeline mirrors the original REBECA image-generation procedure: sample
embeddings from the user-conditioned prior with classifier-free guidance, decode
each embedding through SD15 + IP-Adapter, then re-encode. Adaptations:

- The user cohort, images-per-user budget and per-image generator rule follow
  experiments/sd15/generate.py so all methods share one protocol.
- Eval features are computed with this project's DifferentiableCLIPImageEmbedder
  on the decoded VAE tensor — byte-for-byte the same path as the main method —
  and rows land in the standard evaluation contract:
  tables/generation_metrics.csv + tables/generated_clip_features.float32.npy.

Writes shards that merge_generation-style consolidation joins; run with
--num-shards == 1 to emit final contract files directly.
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
DEFAULT_IP_ADAPTER = Path("/path/to/models/IP-Adapter")
DEFAULT_CLIP_MODEL = Path("/path/to/models/clip-vit-base-patch32")

CONTRACT_COLUMNS = [
    "method", "user_index", "user_id", "neighbor_index", "neighbor_id",
    "neighbor_train_overlap", "neighbor_train_jaccard", "wrong_user_index",
    "wrong_user_id", "seed", "seed_offset", "interest_index", "route",
    "route_id", "route_mode", "expansion_prob", "wrong_interest_index",
    "interest_count", "wrong_interest_count", "interest_weight",
    "anchor_item_index", "anchor_image_id", "variant", "image_path",
    "train_maxsim", "test_maxsim", "assigned_train_topic_index",
    "assigned_train_topic_similarity", "v6_fusion_mode", "v6_neighbor_selection",
    "v6_neighbor_topk", "v6_bridge_candidate_topk", "v6_neighbor_scale",
    "v6_interest_scale", "train_top5", "validation_top5", "test_top5",
    "neighbor_only_top5", "popularity_matched_top5", "nearest_catalog_index",
    "nearest_catalog_image_id", "nearest_catalog_similarity",
]

NAN_FIELDS = {
    "neighbor_index": -1, "neighbor_id": "", "neighbor_train_overlap": float("nan"),
    "neighbor_train_jaccard": float("nan"), "wrong_user_index": -1, "wrong_user_id": "",
    "interest_index": -1, "route": "prior", "route_id": -1,
    "route_mode": "rebeca_prior", "expansion_prob": 0.0,
    "wrong_interest_index": -1, "interest_weight": float("nan"),
    "anchor_item_index": -1, "anchor_image_id": "",
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True,
                        help="train_prior.py output dir containing weights/{prior.pth, config.json}")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sd15-model", type=Path, default=DEFAULT_SD15)
    parser.add_argument("--ip-adapter-dir", type=Path, default=DEFAULT_IP_ADAPTER)
    parser.add_argument("--clip-model", type=Path, default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--clip-cache", type=Path, required=True,
                        help="shared CLIP cache for train/test MaxSim columns")
    parser.add_argument("--selected-users-json", type=Path, required=True)
    parser.add_argument("--method-name", default="REBECA")
    parser.add_argument("--images-per-user", type=int, default=10)
    parser.add_argument("--rebeca-guidance", type=float, default=7.0, help="prior-space CFG")
    parser.add_argument("--pipe-cfg", type=float, default=4.5, help="SD pipeline CFG")
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, str(args.upstream_root))
    sys.path.insert(0, str(REPO_ROOT))

    from grecf.data import id_order_sha256, load_cache, load_public_dataset
    from grecf.image_embedder import DifferentiableCLIPImageEmbedder

    config = json.loads((args.weights_dir / "weights" / "config.json").read_text())
    kwargs = config["model_kwargs"]
    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from transformers import CLIPImageProcessor, CLIPModel

    from sampling import sample_from_diffusion
    from prior_models import RebecaDiffusionPrior

    dataset = load_public_dataset(args.dataset_root)

    selected = json.loads(args.selected_users_json.read_text())
    members = selected["user_indices"] if "user_indices" in selected else selected["all_user_indices"]
    cohort_users = [int(u) for u in sorted(members)]
    start = len(cohort_users) * args.shard_index // args.num_shards
    stop = len(cohort_users) * (args.shard_index + 1) // args.num_shards
    shard_users = cohort_users[start:stop]

    order_sha = id_order_sha256(dataset.image_ids)
    clip_features = np.asarray(
        load_cache(args.clip_cache, dataset.num_items, order_sha, (512,)), dtype=np.float32
    )

    image_size = int(CLIPImageProcessor.from_pretrained(args.clip_model, local_files_only=True).crop_size["height"])
    clip_model = CLIPModel.from_pretrained(args.clip_model, torch_dtype=torch.float32, local_files_only=True).to(device).eval()
    processor_mean = CLIPImageProcessor.from_pretrained(args.clip_model, local_files_only=True).image_mean
    processor_std = CLIPImageProcessor.from_pretrained(args.clip_model, local_files_only=True).image_std
    embedder = DifferentiableCLIPImageEmbedder(clip_model, image_size, processor_mean, processor_std).to(device)

    pipe = StableDiffusionPipeline.from_pretrained(args.sd15_model, safety_checker=None).to(device)
    ip_config = config.get("ip_adapter") or {}
    pipe.load_ip_adapter(
        ip_config.get("dir", str(DEFAULT_IP_ADAPTER)),
        subfolder=ip_config.get("subfolder", "models"),
        weight_name=ip_config.get("weight_name", "ip-adapter_sd15.safetensors"),
    )
    pipe.safety_checker = None

    prior = RebecaDiffusionPrior(**kwargs).to(device)
    prior.load_state_dict(torch.load(args.weights_dir / "weights" / "prior.pth", weights_only=True))
    prior.eval()
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_train_timesteps"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False,
        prediction_type="sample",
    )

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

    num_null_user = int(kwargs["num_users"])
    dim = int(kwargs["img_embed_dim"])
    began = time.time()
    for position, user in enumerate(shard_users):
        seed_offsets = range(args.images_per_user)
        seed_tensor = torch.ones(args.images_per_user, dtype=torch.long, device=device)  # like=1 conditioning
        user_tensor = torch.full((args.images_per_user,), int(user), dtype=torch.long, device=device)
        null_user = torch.full_like(user_tensor, num_null_user)
        null_score = torch.full_like(seed_tensor, 2)

        # Prior-space sampling has no explicit generator in upstream; seeding the
        # global CUDA RNG per user keeps results reproducible and shard-stable.
        torch.manual_seed(args.seed + user * 10_000)
        with torch.no_grad():
            sampled = sample_from_diffusion(
                model=prior,
                user_ids_cond=user_tensor,
                scores_cond=seed_tensor,
                user_ids_uncond=null_user,
                scores_uncond=null_score,
                img_embedding_size=dim,
                scheduler=noise_scheduler,
                guidance_scale=args.rebeca_guidance,
                device=str(device),
            )

        # One batched decode for the whole user: each image keeps its own
        # per-seed initial latent, so results stay reproducible while avoiding
        # ten separate pipeline calls. diffusers expects one list entry per
        # IP-Adapter projection layer with the batch on dim 0, split as
        # [negatives, positives] so its CFG chunk(2) lands correctly.
        latents = []
        for seed_offset in seed_offsets:
            seed = args.seed + user * 10_000 + seed_offset
            latent_generator = torch.Generator(device=device).manual_seed(seed)
            latents.append(torch.randn((1, 4, 64, 64), generator=latent_generator,
                                       device=device, dtype=torch.float16))
        pos = sampled.to(device=device, dtype=torch.float16).view(-1, 1, sampled.shape[-1])
        neg = torch.zeros_like(pos)
        image_embeds = [torch.cat([neg, pos], dim=0)]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            images = pipe(
                prompt=[""] * args.images_per_user,
                negative_prompt=[""] * args.images_per_user,
                guidance_scale=args.pipe_cfg,
                ip_adapter_image_embeds=image_embeds,
                num_inference_steps=args.num_inference_steps,
                latents=torch.cat(latents, dim=0),
            ).images
        assert len(images) == args.images_per_user

        for seed_offset, result in zip(seed_offsets, images):
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
        if done % 20 == 0 or done == len(shard_users):
            rate = done / max(time.time() - began, 1e-9)
            print(f"[shard {args.shard_index}/{args.num_shards}] users {done}/{len(shard_users)} ({rate:.2f} u/s)", flush=True)

    suffix = "" if args.num_shards == 1 else f"_shard{args.shard_index}"
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=CONTRACT_COLUMNS)
    frame.to_csv(tables_dir / f"generation_metrics{suffix}.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    np.save(tables_dir / f"generated_clip_features{suffix}.float32.npy",
            np.stack(features_out).astype(np.float32))
    write_json(tables_dir / f"generation_summary{suffix}.json", {
        "rows": len(rows),
        "users_shard": len(shard_users),
        "range": [start, stop],
        "rebeca_guidance": args.rebeca_guidance,
        "pipe_cfg": args.pipe_cfg,
        "steps": args.num_inference_steps,
        "seed": args.seed,
        "method": args.method_name,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
