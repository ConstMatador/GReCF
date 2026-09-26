#!/usr/bin/env python3
"""ViPer adaptation step 1: visual preference extraction for each cohort user.

ViPer's Visual Preference Extractor takes a set of images plus the user's
free-form like/dislike comments and emits structured liked/disliked visual
attributes (VP+ / VP-). The datasets have no user comments, so the
adaptation replaces the comment channel with implicit interaction feedback:
8 evenly-sampled train positives labelled as "the user liked" and 4
deterministic random catalog images (train/val/test positives excluded)
labelled as "not chosen by the user". A VLM is then prompted in the VPE
role to output keyword-format attributes only.

Output rows (append-resumable JSONL):
  {user_index, user_id, liked_images: [item ids], neutral_images: [item ids],
   liked_attributes: [...], disliked_attributes: [...], raw_output}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation" / "interest_source"))
from grecf.data import load_public_dataset  # noqa: E402
from run_judge import parse_stage_outputs, open_image  # noqa: E402

DEFAULT_DATASET = Path("/path/to/datasets/CIGR")
DEFAULT_MODEL = Path("/path/to/models/Qwen3_VL_32B")

PREF_PROMPT = """
You are a visual preference extractor (ViPer-style).

A user interacted with an image catalog. Below you see two groups of catalog images:
- LIKED images: the user chose / positively interacted with these.
- NOT-CHOSEN images: catalog images the user did not choose.

Task:
Extract the user's visual preference profile as keyword-format attributes, inferred
jointly from the images and the like / not-chosen feedback. Cover subject matter,
scene, style, color palette, composition, mood, and texture when supported by the
images. Use short keywords only (no sentences). Only include attributes with clear
evidence; the not-chosen group only supports weak negative evidence, so keep
disliked_attributes conservative.

Output valid JSON only.

JSON schema:
{
  "liked_attributes": ["example keyword"],
  "disliked_attributes": ["example keyword"]
}

Field constraints:
- 8 to 16 keywords per list, keyword format, no sentences.
- If negative evidence is weak, return an empty disliked_attributes list.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--selected-users-json", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--positives-per-user", type=int, default=8)
    parser.add_argument("--negatives-per-user", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def pick_user_images(dataset, user: int, args: argparse.Namespace) -> tuple[list[int], list[int]]:
    train_items = list(dataset.train[user])
    if not train_items:
        return [], []
    positions = np.linspace(0, len(train_items) - 1, min(args.positives_per_user, len(train_items)))
    positives = sorted({int(train_items[int(round(p))]) for p in positions})

    excluded = set(dataset.train[user]) | set(dataset.validation[user]) | set(dataset.test[user])
    rng = np.random.default_rng(args.seed + int(user))
    negatives: list[int] = []
    while len(negatives) < args.negatives_per_user:
        candidate = int(rng.integers(0, dataset.num_items))
        if candidate in excluded or candidate in negatives:
            continue
        negatives.append(candidate)
    return positives, negatives


def build_content(dataset, positives: list[int], negatives: list[int], args: argparse.Namespace) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": PREF_PROMPT.strip()}]
    content.append({"type": "text", "text": "\nLIKED images (the user chose these):"})
    for i, item in enumerate(positives, 1):
        content.append({"type": "text", "text": f"L{i}:"})
        content.append({"type": "image", "image": open_image(dataset.image_paths[item], args.image_size)})
    content.append({"type": "text", "text": "\nNOT-CHOSEN images (catalog images the user did not choose):"})
    for i, item in enumerate(negatives, 1):
        content.append({"type": "text", "text": f"N{i}:"})
        content.append({"type": "image", "image": open_image(dataset.image_paths[item], args.image_size)})
    content.append({"type": "text", "text": "\nOutput JSON only."})
    return content


def run_batch(model, processor, device, tasks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    messages = [[{"role": "user", "content": task["content"]}] for task in tasks]
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", padding=True,
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
    trimmed = generated[:, inputs.input_ids.shape[1]:]
    raw_texts = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return parse_stage_outputs(raw_texts, "liked_attributes")


def main() -> int:
    args = parse_args()
    dataset = load_public_dataset(args.dataset_root)
    selected = json.loads(args.selected_users_json.read_text())
    members = selected["user_indices"] if "user_indices" in selected else selected["all_user_indices"]
    cohort_users = sorted(int(u) for u in members)
    start = len(cohort_users) * args.shard_index // args.num_shards
    stop = len(cohort_users) * (args.shard_index + 1) // args.num_shards
    shard_users = cohort_users[start:stop]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done: set[int] = set()
    if args.output_jsonl.exists():
        with args.output_jsonl.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "user_index" in row:
                    done.add(int(row["user_index"]))
    tasks: list[dict[str, Any]] = []
    for user in shard_users:
        if user in done:
            continue
        positives, negatives = pick_user_images(dataset, user, args)
        if not positives:
            continue
        tasks.append({
            "user_index": user,
            "user_id": str(dataset.user_ids[user]),
            "positives": positives,
            "negatives": negatives,
            "content": build_content(dataset, positives, negatives, args),
        })
    if not tasks:
        print(json.dumps({"skipped_completed": len(done), "remaining": 0, "shard_index": args.shard_index}), flush=True)
        return 0

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa",
    ).eval().to("cuda")
    device = model.device

    began = time.time()
    processed = 0
    with args.output_jsonl.open("a", encoding="utf-8") as handle:
        index = 0
        while index < len(tasks):
            batch = tasks[index:index + args.batch_size]
            try:
                results = run_batch(model, processor, device, batch, args)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if args.batch_size <= 1:
                    raise
                args.batch_size = max(1, args.batch_size // 2)
                print(json.dumps({"event": "oom_reduce_batch", "new_batch_size": args.batch_size}), flush=True)
                continue
            for task, result in zip(batch, results):
                judge = result.get("judge") or {}
                row = {
                    "user_index": task["user_index"],
                    "user_id": task["user_id"],
                    "liked_images": task["positives"],
                    "neutral_images": task["negatives"],
                    "liked_attributes": judge.get("liked_attributes") or [],
                    "disliked_attributes": judge.get("disliked_attributes") or [],
                    "parse_ok": bool(result.get("parse_ok")),
                    "raw_output": result.get("raw_output", ""),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            index += len(batch)
            processed += len(batch)
            if processed % args.log_every < len(batch) or index >= len(tasks):
                rate = processed / max(time.time() - began, 1e-9)
                print(json.dumps({"processed": processed, "remaining": len(tasks) - processed,
                                  "users_per_sec": round(rate, 3)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
