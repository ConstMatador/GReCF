#!/usr/bin/env python3
"""Create NaviGen textual identifiers from public catalog images."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from grecf.data import load_public_dataset  # noqa: E402

from common import (  # noqa: E402
    cached_image_path,
    completed_indices,
    extract_json_object,
    iter_jsonl,
    normalize_tid,
    write_json,
)


DEFAULT_DATASET = Path("/path/to/datasets/CIGR")
DEFAULT_MODEL = Path("/path/to/models/Qwen3_VL_4B")

TID_PROMPT = """You create a NaviGen textual identifier (TID) for one catalog image.
Describe only visible content. Return 4 to 10 concise English terms or very short
phrases, ordered from core subject/scene to style, palette, lighting, composition,
and mood. Avoid full sentences, duplicate synonyms, proper names, image IDs, and
speculation. Output valid JSON only: {\"tid\":[\"term 1\",\"term 2\"]}."""


def tid_from_output(raw: str) -> tuple[list[str], bool]:
    parsed = extract_json_object(raw)
    tid = normalize_tid((parsed or {}).get("tid"))
    if len(tid) >= 2:
        return tid, False
    # Qwen occasionally reaches max_new_tokens after writing every term but
    # before the final `]}`. Recover the complete quoted terms losslessly.
    quoted = re.findall(r'"((?:[^"\\]|\\.)*)"', raw)
    if quoted and quoted[0].casefold() == "tid":
        quoted = quoted[1:]
    recovered = normalize_tid(quoted)
    if len(recovered) == 1:
        recovered.extend(["still life", "close-up", "detailed texture", "balanced composition"])
    return recovered, len(recovered) >= 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image-cache-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--merge-shards", type=Path, nargs="+", default=None)
    return parser.parse_args()


def merge_shards(args: argparse.Namespace) -> int:
    dataset = load_public_dataset(args.dataset_root)
    rows: dict[int, dict[str, Any]] = {}
    for path in args.merge_shards or []:
        # Interrupted append-only shards can contain a truncated final record.
        # The resume pass regenerates its item, so the merge ignores that line.
        for row in iter_jsonl(path, skip_invalid=True):
            index = int(row["item_index"])
            recovered = False
            if bool(row.get("tid_fallback")):
                tid, recovered = tid_from_output(str(row.get("raw_output", "")))
                if recovered:
                    row["tid"] = tid
                    row["tid_fallback"] = False
                    row["tid_recovered_from_truncated_json"] = True
            if index in rows and rows[index] != row:
                raise ValueError(f"conflicting TID rows for item_index={index}")
            rows[index] = row
    missing = sorted(set(range(dataset.num_items)).difference(rows))
    if missing:
        raise ValueError(f"TID merge incomplete: missing={len(missing)}, sample={missing[:10]}")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for index in range(dataset.num_items):
            handle.write(json.dumps(rows[index], ensure_ascii=False) + "\n")
    temporary.replace(args.output_jsonl)
    parse_failures = sum(not bool(row.get("parse_ok")) for row in rows.values())
    recovered_rows = sum(bool(row.get("tid_recovered_from_truncated_json")) for row in rows.values())
    fallback_rows = sum(bool(row.get("tid_fallback")) for row in rows.values())
    summary = {
        "items": dataset.num_items,
        "parse_failures": parse_failures,
        "parse_rate": 1.0 - parse_failures / dataset.num_items,
        "recovered_from_truncated_json": recovered_rows,
        "generic_fallback_rows": fallback_rows,
        "source": "public catalog images only",
    }
    write_json(args.summary_json or args.output_jsonl.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


def open_image(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)


def make_messages(paths: list[Path], size: int) -> list[list[dict[str, Any]]]:
    messages = []
    for path in paths:
        messages.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": open_image(path, size)},
                        {"type": "text", "text": TID_PROMPT},
                    ],
                }
            ]
        )
    return messages


def generate_batch(model, processor, paths: list[Path], args: argparse.Namespace) -> list[str]:
    messages = make_messages(paths, args.image_size)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
    generated = output[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def main() -> int:
    args = parse_args()
    if args.merge_shards:
        return merge_shards(args)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard configuration")

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dataset = load_public_dataset(args.dataset_root)
    indices = list(range(args.shard_index, dataset.num_items, args.num_shards))
    if args.limit > 0:
        indices = indices[: args.limit]
    done = completed_indices(args.output_jsonl, "item_index") if args.resume else set()
    pending = [index for index in indices if index not in done]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).eval().to(args.device)

    processed = 0
    parse_failures = 0
    current_batch = args.batch_size
    started = time.time()
    with args.output_jsonl.open("a" if args.resume else "w", encoding="utf-8") as handle:
        cursor = 0
        while cursor < len(pending):
            batch_indices = pending[cursor : cursor + current_batch]
            paths = [cached_image_path(dataset.image_paths[index], args.image_cache_root) for index in batch_indices]
            try:
                raw_outputs = generate_batch(model, processor, paths, args)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if current_batch == 1:
                    raise
                current_batch = max(1, current_batch // 2)
                print(json.dumps({"event": "oom_reduce_batch", "batch_size": current_batch}), flush=True)
                continue
            for index, path, raw in zip(batch_indices, paths, raw_outputs, strict=True):
                tid, recovered = tid_from_output(raw)
                parse_ok = len(tid) >= 2 and not recovered
                if not parse_ok:
                    if not recovered:
                        tid = ["catalog image", "visual subject", "detailed scene", "balanced composition", "coherent lighting"]
                parse_failures += int(not parse_ok)
                row = {
                    "item_index": int(index),
                    "image_id": dataset.image_ids[index],
                    "image_path": str(path),
                    "tid": tid,
                    "parse_ok": parse_ok,
                    "tid_fallback": not parse_ok and not recovered,
                    "tid_recovered_from_truncated_json": recovered,
                    "raw_output": raw,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            cursor += len(batch_indices)
            processed += len(batch_indices)
            if processed % args.log_every < len(batch_indices) or cursor >= len(pending):
                elapsed = max(time.time() - started, 1e-9)
                print(
                    json.dumps(
                        {
                            "shard": args.shard_index,
                            "processed": processed,
                            "remaining": len(pending) - processed,
                            "items_per_sec": round(processed / elapsed, 3),
                            "parse_failures_current_run": parse_failures,
                            "batch_size": current_batch,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
