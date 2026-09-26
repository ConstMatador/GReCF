#!/usr/bin/env python3
"""Decode NaviGen instructions with the shared SD1.5 evaluation backbone."""
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

from grecf.data import id_order_sha256, load_cache, load_public_dataset  # noqa: E402
from grecf.image_embedder import DifferentiableCLIPImageEmbedder  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "baselines/rebeca"))
from generate import CONTRACT_COLUMNS, NAN_FIELDS  # noqa: E402


DEFAULT_DATASET = Path("/path/to/datasets/CIGR")
DEFAULT_SD15 = Path("/path/to/models/stable-diffusion-v1-5")
DEFAULT_CLIP_MODEL = Path("/path/to/models/clip-vit-base-patch32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sd15-model", type=Path, default=DEFAULT_SD15)
    parser.add_argument("--clip-model", type=Path, default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument("--selected-users-json", type=Path, required=True)
    parser.add_argument("--method-name", default="NaviGen")
    parser.add_argument("--images-per-user", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--negative-prompt", default="low quality, blurry, distorted, artifacts, text, watermark")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_predictions(path: Path) -> dict[int, dict]:
    predictions: dict[int, dict] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            source = row.get("source_row") or {}
            prediction = row.get("prediction") or row.get("parsed_json") or {}
            user = source.get("user_index")
            instruction = str(prediction.get("target_ins", "")).strip()
            tid = prediction.get("target_tid") or []
            if user is None or not instruction:
                raise ValueError(f"missing user/instruction in {path}:{line_number}")
            predictions[int(user)] = {
                "target_ins": instruction,
                "target_tid": [str(value) for value in tid],
                "raw": row,
            }
    return predictions


def cohort_users(path: Path, num_users: int) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("user_indices", payload.get("all_user_indices"))
    if values is None:
        raise ValueError(f"cohort JSON has no user indices: {path}")
    users = sorted(int(value) for value in values)
    if not users or len(users) != len(set(users)):
        raise ValueError(f"cohort must contain non-empty unique user indices, got {len(users)}")
    invalid = [user for user in users if not 0 <= user < num_users]
    if invalid:
        raise ValueError(f"cohort has invalid user indices: {invalid[:10]}")
    return users


def main() -> int:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard configuration")
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    from diffusers import StableDiffusionPipeline
    from transformers import CLIPImageProcessor, CLIPModel

    dataset = load_public_dataset(args.dataset_root)
    users = cohort_users(args.selected_users_json, dataset.num_users)
    start = len(users) * args.shard_index // args.num_shards
    stop = len(users) * (args.shard_index + 1) // args.num_shards
    shard_users = users[start:stop]
    predictions = read_predictions(args.predictions_jsonl)
    missing = sorted(set(shard_users).difference(predictions))
    if missing:
        raise ValueError(f"missing predictions for {len(missing)} shard users: {missing[:10]}")

    catalog_features = np.asarray(
        load_cache(
            args.clip_cache,
            dataset.num_items,
            id_order_sha256(dataset.image_ids),
            (512,),
        ),
        dtype=np.float32,
    )
    catalog = torch.from_numpy(catalog_features).to(device)

    processor = CLIPImageProcessor.from_pretrained(args.clip_model, local_files_only=True)
    image_size = int(processor.crop_size["height"])
    clip_model = CLIPModel.from_pretrained(
        args.clip_model, torch_dtype=torch.float32, local_files_only=True
    ).to(device).eval()
    embedder = DifferentiableCLIPImageEmbedder(
        clip_model, image_size, processor.image_mean, processor.image_std
    ).to(device)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd15_model,
        safety_checker=None,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to(device)
    pipe.safety_checker = None

    images_dir = args.output_dir / "images"
    tables_dir = args.output_dir / "tables"
    images_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    features_out: list[np.ndarray] = []
    instructions_out: list[dict[str, object]] = []
    began = time.time()

    for position, user in enumerate(shard_users):
        prediction = predictions[user]
        instruction = prediction["target_ins"]
        latents = []
        for seed_offset in range(args.images_per_user):
            seed = args.seed + user * 10_000 + seed_offset
            generator = torch.Generator(device=device).manual_seed(seed)
            latents.append(torch.randn((1, 4, 64, 64), generator=generator, device=device, dtype=torch.float16))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            generated = pipe(
                prompt=[instruction] * args.images_per_user,
                negative_prompt=[args.negative_prompt] * args.images_per_user,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                latents=torch.cat(latents, dim=0),
            ).images

        instructions_out.append(
            {
                "user_index": user,
                "user_id": dataset.user_ids[user],
                "target_tid": prediction["target_tid"],
                "target_ins": instruction,
            }
        )
        for seed_offset, image in enumerate(generated):
            seed = args.seed + user * 10_000 + seed_offset
            pixel = torch.from_numpy(np.asarray(image)).to(device=device, dtype=torch.float32)
            pixel = pixel.permute(2, 0, 1) / 127.5 - 1.0
            with torch.inference_mode():
                feature = embedder(pixel.unsqueeze(0)).float().cpu().numpy()[0]
            features_out.append(feature)

            def similarities(items) -> torch.Tensor:
                indices = torch.from_numpy(np.asarray(items, dtype=np.int64)).to(device)
                return catalog[indices] @ torch.from_numpy(feature).to(device)

            def top5(values: torch.Tensor) -> float:
                return float(torch.topk(values, k=min(5, len(values))).values.mean()) if len(values) else float("nan")

            train_scores = similarities(dataset.train[user])
            validation_scores = similarities(dataset.validation[user])
            test_scores = similarities(dataset.test[user])
            filename = f"{dataset.user_ids[user]}_seed{seed_offset}_personalized.png"
            image_path = images_dir / filename
            image.save(image_path)
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
                "image_path": str(image_path),
                "train_maxsim": float(train_scores.max()),
                "test_maxsim": float(test_scores.max()),
                "assigned_train_topic_index": -1,
                "assigned_train_topic_similarity": float("nan"),
                "train_top5": top5(train_scores),
                "validation_top5": top5(validation_scores),
                "test_top5": top5(test_scores),
                "nearest_catalog_index": -1,
                "nearest_catalog_image_id": "",
                "nearest_catalog_similarity": float("nan"),
            }
            rows.append(row)
        completed = position + 1
        if completed % 10 == 0 or completed == len(shard_users):
            elapsed = max(time.time() - began, 1e-9)
            print(
                json.dumps(
                    {
                        "shard": args.shard_index,
                        "users_done": completed,
                        "users_total": len(shard_users),
                        "users_per_sec": round(completed / elapsed, 3),
                    }
                ),
                flush=True,
            )

    suffix = "" if args.num_shards == 1 else f"_shard{args.shard_index}"
    pd.DataFrame(rows, columns=CONTRACT_COLUMNS).to_csv(
        tables_dir / f"generation_metrics{suffix}.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )
    np.save(
        tables_dir / f"generated_clip_features{suffix}.float32.npy",
        np.stack(features_out).astype(np.float32),
    )
    with (tables_dir / f"instructions{suffix}.jsonl").open("w", encoding="utf-8") as handle:
        for row in instructions_out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "rows": len(rows),
        "users": len(shard_users),
        "user_range": [start, stop],
        "method": args.method_name,
        "seed": args.seed,
        "steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "instruction_budget": "one deterministic NaviGen instruction per user, ten shared-protocol image seeds",
    }
    (tables_dir / f"generation_summary{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
