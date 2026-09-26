#!/usr/bin/env python3
"""Generate one-round evolutionary CID2INS supervision with a local Qwen3 teacher."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import completed_indices, extract_json_object, iter_jsonl, normalize_tid, write_json  # noqa: E402


DEFAULT_MODEL = Path("/path/to/models/Qwen3_8B")

SYSTEM_PROMPT = """You are the local teacher for NaviGen. Given a user's ordered
visual TID history and a known training target TID, internally compare three possible
creative directions: conservative, balanced, and exploratory. Select the direction
that best preserves the target semantics while using history only as preference
support. Produce a concrete, visually generatable image instruction.

Return JSON only with exactly two fields:
{"reasoning":"one first-person paragraph explaining preference evolution and candidate selection",
 "target_ins":"a detailed English text-to-image instruction"}
The instruction should specify subject, scene, style, composition, lighting, palette,
and mood where supported. Do not mention users, recommendation, CID, TID, candidate
selection, or hidden target data in target_ins."""


def parse_teacher_output(raw: str) -> tuple[str, str, bool]:
    parsed = extract_json_object(raw)
    reasoning = str((parsed or {}).get("reasoning", "")).strip()
    instruction = str((parsed or {}).get("target_ins", "")).strip()
    if len(reasoning) >= 40 and len(instruction) >= 60:
        return reasoning, instruction, False

    recovered: dict[str, str] = {}
    for field in ("reasoning", "target_ins"):
        match = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
        if match:
            try:
                recovered[field] = json.loads('"' + match.group(1) + '"').strip()
            except json.JSONDecodeError:
                pass
    reasoning = recovered.get("reasoning", reasoning)
    instruction = recovered.get("target_ins", instruction)
    valid = len(reasoning) >= 40 and len(instruction) >= 60
    return reasoning, instruction, valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-history-items", type=int, default=20)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-id", action="append", default=None, help="Only process selected sample IDs")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--merge-shards", type=Path, nargs="+", default=None)
    parser.add_argument("--output-parquet", type=Path, default=None)
    return parser.parse_args()


def fallback(sample: dict[str, Any]) -> tuple[str, str]:
    target = ", ".join(normalize_tid(sample.get("target_tid")))
    history = sample.get("history_tids") or []
    recent = [term for row in history[-3:] for term in normalize_tid(row, maximum=3)]
    support = ", ".join(recent[:6])
    reasoning = (
        f"I infer the next visual direction from the recent preference evidence ({support}) and compare "
        f"conservative, balanced, and exploratory continuations. The balanced direction best preserves "
        f"the target semantics ({target}) while adding enough visual specificity for generation."
    )
    instruction = (
        f"Create a polished, high-quality image centered on {target}. Build a coherent scene with a clear "
        "primary subject, purposeful composition, detailed textures, balanced lighting, a harmonious color "
        "palette, and an atmosphere that makes the concept immediately readable and visually engaging."
    )
    return reasoning, instruction


def prompt_for(sample: dict[str, Any], max_history_items: int) -> str:
    history = sample.get("history_tids") or []
    history = history[-max_history_items:]
    lines = [f"{index + 1}. {', '.join(normalize_tid(row))}" for index, row in enumerate(history)]
    target = ", ".join(normalize_tid(sample.get("target_tid")))
    return "Historical visual TIDs in order:\n" + "\n".join(lines) + f"\n\nTraining target TID:\n{target}\n\nOutput JSON only."


def tokenize_batch(tokenizer, samples: list[dict[str, Any]], args: argparse.Namespace):
    messages = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_for(sample, args.max_history_items)},
        ]
        for sample in samples
    ]
    kwargs = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_tokens,
    )
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def merge(args: argparse.Namespace) -> int:
    if args.output_parquet is None:
        raise SystemExit("--output-parquet is required with --merge-shards")
    source = {str(row["sample_id"]): row for row in iter_jsonl(args.input_jsonl)}
    generated: dict[str, dict[str, Any]] = {}
    for path in args.merge_shards or []:
        for row in iter_jsonl(path):
            generated[str(row["sample_id"])] = row
    missing = sorted(set(source).difference(generated))
    if missing:
        raise ValueError(f"teacher merge incomplete: missing={len(missing)}, sample={missing[:5]}")
    rows = []
    for sample_id, sample in source.items():
        teacher = generated[sample_id]
        reasoning = str(teacher["reasoning"])
        instruction = str(teacher["target_ins"])
        used_fallback = bool(teacher.get("teacher_fallback"))
        recovered = False
        if used_fallback:
            recovered_reasoning, recovered_instruction, recovered = parse_teacher_output(
                str(teacher.get("raw_output", ""))
            )
            if recovered:
                reasoning = recovered_reasoning
                instruction = recovered_instruction
                used_fallback = False
        rows.append(
            {
                "sample_id": sample_id,
                "user_index": int(sample["user_index"]),
                "user_id": str(sample["user_id"]),
                "hist_sid": sample["hist_sid"],
                "target_tid": sample["target_tid"],
                "target_ins": instruction,
                "reasoning": reasoning,
                "teacher_fallback": used_fallback,
                "teacher_recovered_from_truncated_json": recovered,
            }
        )
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output_parquet, index=False)
    summary = {
        "rows": len(rows),
        "fallback_rows": sum(bool(row["teacher_fallback"]) for row in rows),
        "output_parquet": str(args.output_parquet),
    }
    write_json(args.output_parquet.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.merge_shards:
        return merge(args)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard configuration")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    all_samples = list(iter_jsonl(args.input_jsonl))
    indexed = list(enumerate(all_samples))[args.shard_index :: args.num_shards]
    if args.sample_id:
        selected_ids = set(args.sample_id)
        indexed = [(index, sample) for index, sample in indexed if str(sample.get("sample_id")) in selected_ids]
    if args.limit > 0:
        indexed = indexed[: args.limit]
    done = completed_indices(args.output_jsonl, "source_index") if args.resume else set()
    pending = [(index, sample) for index, sample in indexed if index not in done]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).eval().to(args.device)

    current_batch = args.batch_size
    processed = 0
    fallback_count = 0
    cursor = 0
    started = time.time()
    with args.output_jsonl.open("a" if args.resume else "w", encoding="utf-8") as handle:
        while cursor < len(pending):
            batch = pending[cursor : cursor + current_batch]
            batch_samples = [sample for _, sample in batch]
            inputs = tokenize_batch(tokenizer, batch_samples, args).to(model.device)
            try:
                with torch.inference_mode():
                    output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                del inputs
                torch.cuda.empty_cache()
                if current_batch == 1:
                    raise
                current_batch = max(1, current_batch // 2)
                print(json.dumps({"event": "oom_reduce_batch", "batch_size": current_batch}), flush=True)
                continue
            texts = tokenizer.batch_decode(
                output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for (source_index, sample), raw in zip(batch, texts, strict=True):
                reasoning, instruction, recovered = parse_teacher_output(raw)
                used_fallback = not recovered and (len(reasoning) < 40 or len(instruction) < 60)
                if used_fallback:
                    reasoning, instruction = fallback(sample)
                    fallback_count += 1
                row = {
                    "source_index": int(source_index),
                    "sample_id": str(sample["sample_id"]),
                    "reasoning": reasoning,
                    "target_ins": instruction,
                    "teacher_fallback": used_fallback,
                    "teacher_recovered_from_truncated_json": recovered,
                    "raw_output": raw,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            cursor += len(batch)
            processed += len(batch)
            if processed % args.log_every < len(batch) or cursor >= len(pending):
                elapsed = max(time.time() - started, 1e-9)
                print(
                    json.dumps(
                        {
                            "shard": args.shard_index,
                            "processed": processed,
                            "remaining": len(pending) - processed,
                            "samples_per_sec": round(processed / elapsed, 3),
                            "fallback_current_run": fallback_count,
                            "batch_size": current_batch,
                        }
                    ),
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
