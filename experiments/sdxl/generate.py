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
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "baselines" / "rebeca"))
from generate import CONTRACT_COLUMNS, NAN_FIELDS  # noqa: E402

from grecf.data import id_order_sha256, load_cache, load_public_dataset  # noqa: E402
from grecf.image_embedder import DifferentiableCLIPImageEmbedder  # noqa: E402
from grecf.preference_ip_adapter import (  # noqa: E402
    install_preference_ip_trainable_kv_processors,
    load_preference_ip_processor_state_dict,
)
from grecf.sdxl import SDXLPreferenceAdapter  # noqa: E402


def load_interests(path: Path):
    return {
        "prototypes": np.load(path / "adaptive_b_prototypes.float32.npy", mmap_mode="r"),
        "weights": np.load(path / "adaptive_b_weights.float32.npy", mmap_mode="r"),
        "mask": np.load(path / "adaptive_b_mask.bool.npy", mmap_mode="r"),
        "auxiliary": np.load(path / "adaptive_b_auxiliary.float32.npy", mmap_mode="r"),
    }


def schedule(weights, mask, count=10):
    active = np.flatnonzero(np.asarray(mask, dtype=bool))
    prob = np.asarray(weights[active], dtype=np.float64)
    prob /= max(prob.sum(), 1e-12)
    allocation = np.zeros(len(active), dtype=np.int64)
    if count >= len(active):
        allocation += 1
        remaining = count - len(active)
    else:
        remaining = count
    raw = prob * remaining
    allocation += np.floor(raw).astype(np.int64)
    missing = count - int(allocation.sum())
    if missing:
        allocation[np.argsort(-(raw - np.floor(raw)), kind="stable")[:missing]] += 1
    result = []
    while len(result) < count:
        for i, value in enumerate(active):
            if allocation[i] > 0:
                result.append(int(value)); allocation[i] -= 1
    return result


def main() -> int:
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
    parser.add_argument("--method-name", default="GReCF (SDXL1.0)")
    parser.add_argument("--images-per-user", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()

    from diffusers import StableDiffusionXLPipeline
    from transformers import CLIPImageProcessor, CLIPModel

    device = torch.device("cuda")
    dtype = torch.bfloat16
    dataset = load_public_dataset(args.dataset_root)
    selected = json.loads(args.selected_users_json.read_text(encoding="utf-8"))
    user_indices = sorted(int(x) for x in selected.get("user_indices", selected.get("all_user_indices")))
    lo = len(user_indices) * args.shard_index // args.num_shards
    hi = len(user_indices) * (args.shard_index + 1) // args.num_shards
    user_indices = user_indices[lo:hi]
    order_sha = id_order_sha256(dataset.image_ids)
    clip = np.asarray(load_cache(args.clip_cache, dataset.num_items, order_sha, (512,)), dtype=np.float32)
    interest = load_interests(args.interest_cache_dir)
    state = torch.load(args.base_context, map_location="cpu", weights_only=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    adapter = SDXLPreferenceAdapter(
        state["prompt_embeds"], state["prompt_attention_mask"],
        torch.from_numpy(np.asarray(interest["auxiliary"], dtype=np.float32).mean(axis=0, keepdims=True)),
    ).to(device).eval()
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=dtype, local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    install_preference_ip_trainable_kv_processors(pipe.unet, layer_gates=True)
    load_preference_ip_processor_state_dict(pipe.unet, checkpoint["preference_ip_processor_state"])
    pipe.unet.requires_grad_(False)
    processor = CLIPImageProcessor.from_pretrained(args.clip_model, local_files_only=True)
    clip_model = CLIPModel.from_pretrained(
        args.clip_model, torch_dtype=torch.float32, local_files_only=True
    ).to(device).eval()
    embedder = DifferentiableCLIPImageEmbedder(clip_model, int(processor.crop_size["height"]), processor.image_mean, processor.image_std).to(device)
    prototypes = torch.from_numpy(np.asarray(interest["prototypes"], dtype=np.float32)).to(device)
    masks = torch.from_numpy(np.asarray(interest["mask"], dtype=bool)).to(device)
    auxiliary = torch.from_numpy(np.asarray(interest["auxiliary"], dtype=np.float32)).to(device)
    weights = interest["weights"]
    output = args.output_dir
    images_dir, tables_dir = output / "images", output / "tables"
    images_dir.mkdir(parents=True, exist_ok=True); tables_dir.mkdir(parents=True, exist_ok=True)
    rows, features = [], []
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    for user_position, user in enumerate(user_indices, start=1):
        indices = schedule(weights[user], interest["mask"][user], args.images_per_user)
        user_proto = torch.from_numpy(np.repeat(np.asarray(interest["prototypes"][user:user + 1]), args.images_per_user, axis=0).copy()).to(device)
        user_mask = torch.from_numpy(np.repeat(np.asarray(interest["mask"][user:user + 1]), args.images_per_user, axis=0).copy()).to(device)
        user_aux = torch.from_numpy(np.repeat(np.asarray(interest["auxiliary"][user:user + 1]), args.images_per_user, axis=0).copy()).to(device)
        chosen = user_proto[torch.arange(args.images_per_user, device=device), torch.tensor(indices, device=device)]
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            context, _, attention_kwargs = adapter(user_proto, user_mask, user_aux, chosen)
            negative_context = state["negative_prompt_embeds"].to(device=device, dtype=dtype).expand(args.images_per_user, -1, -1)
            negative_pooled = state["negative_pooled_prompt_embeds"].to(device=device, dtype=dtype).expand(args.images_per_user, -1)
            negative_kwargs = adapter.zero_attention_kwargs(args.images_per_user, device, dtype)
            guidance_kwargs = {
                key: torch.cat((negative_kwargs[key], attention_kwargs[key]), dim=0)
                if key.endswith("_tokens") or key.endswith("_mask")
                else attention_kwargs[key]
                for key in attention_kwargs
            }
        latents = []
        for seed_offset in range(args.images_per_user):
            g = torch.Generator(device=device).manual_seed(args.seed + user * 10_000 + seed_offset)
            latents.append(torch.randn((1, 4, 64, 64), generator=g, device=device, dtype=dtype))
        scheduler = pipe.scheduler
        scheduler.set_timesteps(args.steps, device=device)
        latent = torch.cat(latents) * scheduler.init_noise_sigma
        time_ids = torch.tensor([512, 512, 0, 0, 512, 512], dtype=dtype, device=device).view(1, 6).expand(args.images_per_user, -1)
        added = {"text_embeds": torch.cat((negative_pooled, state["pooled_prompt_embeds"].to(device=device, dtype=dtype).expand(args.images_per_user, -1)), dim=0), "time_ids": torch.cat((time_ids, time_ids), dim=0)}
        encoder_states = torch.cat((negative_context, context.to(device=device, dtype=dtype)), dim=0)
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for timestep in scheduler.timesteps:
                model_input = torch.cat((latent, latent), dim=0)
                scaled = scheduler.scale_model_input(model_input, timestep)
                prediction = pipe.unet(
                    scaled, timestep.expand(model_input.shape[0]),
                    encoder_hidden_states=encoder_states,
                    encoder_attention_mask=state["negative_prompt_attention_mask"].to(device).expand(2 * args.images_per_user, -1),
                    added_cond_kwargs=added,
                    cross_attention_kwargs=guidance_kwargs,
                ).sample
                uncond, cond = prediction.chunk(2)
                latent = scheduler.step(uncond + args.guidance_scale * (cond - uncond), timestep, latent).prev_sample
            decoded = pipe.vae.decode(latent / float(pipe.vae.config.scaling_factor)).sample
            generated = [
                Image.fromarray(
                    decoded[i].float().cpu().add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                )
                for i in range(args.images_per_user)
            ]
        for seed_offset, image in enumerate(generated):
            path = images_dir / f"{dataset.user_ids[user]}_seed{seed_offset}_personalized.png"
            image.save(path)
            pixel = torch.from_numpy(np.asarray(image).copy()).to(device, dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1
            with torch.inference_mode():
                feature = embedder(pixel.unsqueeze(0)).float().cpu().numpy()[0]
            features.append(feature)
            train_scores = clip[dataset.train[user]] @ feature
            test_scores = clip[dataset.test[user]] @ feature
            row = {**NAN_FIELDS, "method": args.method_name, "user_index": user, "user_id": str(dataset.user_ids[user]), "seed": args.seed + user * 10_000 + seed_offset, "seed_offset": seed_offset, "interest_index": indices[seed_offset], "interest_count": int(interest["mask"][user].sum()), "variant": "personalized", "image_path": str(path), "train_maxsim": float(train_scores.max()), "test_maxsim": float(test_scores.max()), "assigned_train_topic_index": -1, "assigned_train_topic_similarity": float("nan")}
            rows.append(row)
        if user_position == 1 or user_position % 10 == 0:
            print(json.dumps({
                "shard": args.shard_index,
                "users_complete": user_position,
                "users_total": len(user_indices),
                "images_complete": len(rows),
                "elapsed_seconds": round(time.time() - started, 2),
                "peak_gpu_gib": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 2),
            }), flush=True)
    suffix = "" if args.num_shards == 1 else f"_shard{args.shard_index}"
    pd.DataFrame(rows, columns=CONTRACT_COLUMNS).to_csv(tables_dir / f"generation_metrics{suffix}.csv", index=False)
    np.save(tables_dir / f"generated_clip_features{suffix}.float32.npy", np.stack(features).astype(np.float32))
    print(json.dumps({"shard": args.shard_index, "users": len(user_indices), "images": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
