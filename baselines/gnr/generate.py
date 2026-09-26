#!/usr/bin/env python3
"""GNR step 3: cohort generation with the SFT'd Janus-Pro-1B.

For each cohort user the history is their first k chronological train
positives (deterministic); the SFT model generates images_per_user next-item
images with the paper's CFG decoding (cond = history conversation, uncond =
padded input). Contract files follow the shared evaluation protocol; CLIP
features use the byte-identical DifferentiableCLIPImageEmbedder path.
"""
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
from generate import CONTRACT_COLUMNS, NAN_FIELDS  # noqa: E402  (rebeca adapter)
from grecf.data import id_order_sha256, load_cache, load_public_dataset  # noqa: E402
from grecf.image_embedder import DifferentiableCLIPImageEmbedder  # noqa: E402

INSTRUCTION = (
    "\nBased on the user's interaction history above, "
    "generate the image of the next item this user would like."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--janus-root", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=Path("/path/to/models/Janus-Pro-1B"))
    parser.add_argument("--sft-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/path/to/datasets/CIGR"))
    parser.add_argument("--clip-model", type=Path,
                        default=Path("/path/to/models/clip-vit-base-patch32"))
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument("--selected-users-json", type=Path, required=True)
    parser.add_argument("--method-name", default="GNR")
    parser.add_argument("--images-per-user", type=int, default=10)
    parser.add_argument("--history-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-weight", type=float, default=5.0)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--image-root-old", default=None,
                        help="replace this path prefix in catalog image paths (IO cache)")
    parser.add_argument("--image-root-new", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def generate_user_images(model, processor, history_images, args, generator, device):
    conversation = [{
        "role": "User",
        "content": "<image_placeholder>" * len(history_images) + INSTRUCTION,
        "images": history_images,
    }, {"role": "Assistant", "content": ""}]
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=[{"role": "User", "content": conversation[0]["content"].strip()},
                       {"role": "Assistant", "content": ""}],
        sft_format=processor.sft_format, system_prompt="")
    prompt = sft_format + processor.image_start_tag
    input_ids = torch.LongTensor(processor.tokenizer.encode(prompt))
    prompt_len = len(input_ids)

    n = args.images_per_user
    tokens = torch.zeros((n * 2, prompt_len), dtype=torch.int, device=device)
    for i in range(n * 2):
        tokens[i, :] = input_ids.to(device)
        if i % 2 != 0:
            tokens[i, 1:-1] = processor.pad_id
    inputs_embeds = model.language_model.get_input_embeddings()(tokens)

    image_token_num = 576
    generated = torch.zeros((n, image_token_num), dtype=torch.int, device=device)
    outputs = None
    for i in range(image_token_num):
        outputs = model.language_model.model(
            inputs_embeds=inputs_embeds, use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None)
        logits = model.gen_head(outputs.last_hidden_state[:, -1, :])
        logit_cond, logit_uncond = logits[0::2], logits[1::2]
        logits = logit_uncond + args.cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / args.temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=generator)
        generated[:, i] = next_token.squeeze(-1)
        pair = next_token.repeat_interleave(2, dim=0).view(-1)
        inputs_embeds = model.prepare_gen_img_embeds(pair).unsqueeze(1)

    patch = args.image_size // 16
    dec = model.gen_vision_model.decode_code(
        generated.to(torch.int), shape=[n, 8, patch, patch])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    return np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.janus_root))
    from janus.models import MultiModalityCausalLM, VLChatProcessor

    device = args.device or "cuda:0"

    dataset = load_public_dataset(args.dataset_root)
    selected = json.loads(args.selected_users_json.read_text())
    members = selected["user_indices"] if "user_indices" in selected else selected["all_user_indices"]
    cohort_users = [int(u) for u in sorted(members)]
    start = len(cohort_users) * args.shard_index // args.num_shards
    stop = len(cohort_users) * (args.shard_index + 1) // args.num_shards
    shard_users = cohort_users[start:stop]

    processor = VLChatProcessor.from_pretrained(args.model)

    def remap(path: str) -> str:
        if args.image_root_old and args.image_root_new and path.startswith(args.image_root_old):
            return args.image_root_new + path[len(args.image_root_old):]
        return path

    model = MultiModalityCausalLM.from_pretrained(args.model, trust_remote_code=True)
    state = torch.load(args.sft_checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(
        {k: v.to(torch.float32) for k, v in state.items()}, strict=False)
    if missing:
        raise SystemExit(f"missing keys in checkpoint: {missing[:5]}")
    model = model.to(torch.bfloat16).to(device).eval()

    from transformers import CLIPImageProcessor, CLIPModel
    order_sha = id_order_sha256(dataset.image_ids)
    clip_features = np.asarray(
        load_cache(args.clip_cache, dataset.num_items, order_sha, (512,)), dtype=np.float32)
    image_size = int(CLIPImageProcessor.from_pretrained(
        args.clip_model, local_files_only=True).crop_size["height"])
    clip_model = CLIPModel.from_pretrained(
        args.clip_model, torch_dtype=torch.float32, local_files_only=True).to(device).eval()
    mean = CLIPImageProcessor.from_pretrained(args.clip_model, local_files_only=True).image_mean
    std = CLIPImageProcessor.from_pretrained(args.clip_model, local_files_only=True).image_std
    embedder = DifferentiableCLIPImageEmbedder(clip_model, image_size, mean, std).to(device)

    catalog = torch.from_numpy(clip_features).to(device)

    def sims_of(items, feature):
        idx = torch.from_numpy(np.asarray(items, dtype=np.int64)).to(device)
        return catalog[idx] @ torch.from_numpy(feature).to(device)

    def top(scores):
        return float(torch.topk(scores, k=min(5, len(scores))).values.mean()) if len(scores) else float("nan")

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    features_out: list[np.ndarray] = []
    began = time.time()

    for position, user in enumerate(shard_users):
        train_items = list(dataset.train[user])
        k = min(args.history_k, len(train_items))
        history_paths = [remap(str(dataset.image_paths[item])) for item in train_items[:k]]

        user_seed = args.seed + user * 10_000
        generator = torch.Generator(device=device).manual_seed(user_seed)
        images = generate_user_images(model, processor, history_paths, args, generator, device)

        for seed_offset, pixels in enumerate(images):
            pixel = torch.from_numpy(pixels).to(device=device, dtype=torch.float32)
            pixel = pixel.permute(2, 0, 1) / 127.5 - 1.0
            with torch.no_grad():
                feature = embedder(pixel.unsqueeze(0)).float().cpu().numpy()[0]
            features_out.append(feature)

            filename = f"{dataset.user_ids[user]}_seed{seed_offset}_personalized.png"
            Image.fromarray(pixels).save(images_dir / filename)
            rows.append({
                **NAN_FIELDS,
                "method": args.method_name,
                "user_index": int(user),
                "user_id": str(dataset.user_ids[user]),
                "seed": int(user_seed),
                "seed_offset": int(seed_offset),
                "interest_count": 0,
                "wrong_interest_count": 0,
                "variant": "personalized",
                "image_path": str(images_dir / filename),
                "train_maxsim": float(sims_of(dataset.train[user], feature).max()) if len(dataset.train[user]) else float("nan"),
                "test_maxsim": float(sims_of(dataset.test[user], feature).max()) if len(dataset.test[user]) else float("nan"),
                "assigned_train_topic_index": -1,
                "assigned_train_topic_similarity": float("nan"),
                "train_top5": top(sims_of(dataset.train[user], feature)),
                "validation_top5": top(sims_of(dataset.validation[user], feature)),
                "test_top5": top(sims_of(dataset.test[user], feature)),
                "nearest_catalog_index": -1,
                "nearest_catalog_image_id": "",
                "nearest_catalog_similarity": float("nan"),
            })
        done = position + 1
        if done % 10 == 0 or done == len(shard_users):
            rate = done / max(time.time() - began, 1e-9)
            print(json.dumps({"shard": args.shard_index, "users_done": done,
                              "users_total": len(shard_users), "users_per_sec": round(rate, 3),
                              "user": user}), flush=True)

    suffix = "" if args.num_shards == 1 else f"_shard{args.shard_index}"
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=CONTRACT_COLUMNS)
    frame.to_csv(tables_dir / f"generation_metrics{suffix}.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    np.save(tables_dir / f"generated_clip_features{suffix}.float32.npy",
            np.stack(features_out).astype(np.float32))
    (tables_dir / f"generation_summary{suffix}.json").write_text(json.dumps({
        "rows": len(rows), "users_shard": len(shard_users), "range": [start, stop],
        "history_k": args.history_k, "temperature": args.temperature,
        "cfg_weight": args.cfg_weight, "image_size": args.image_size,
        "seed_rule": "one user-level torch.Generator seed = seed + user*10000; "
                     "the user's 10 images are sampled from one stream",
        "sft_checkpoint": str(args.sft_checkpoint), "method": args.method_name,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
