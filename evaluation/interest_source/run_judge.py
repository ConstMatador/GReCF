#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import AutoModelForMultimodalLM
except ImportError:
    AutoModelForMultimodalLM = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def completed_task_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            task_id = row.get("task_id")
            if task_id:
                done.add(str(task_id))
    return done


def open_image(path: str | Path, image_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image_size > 0:
        image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    return image


def json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def validate_bool(parsed: dict[str, Any], key: str) -> tuple[bool, str]:
    if key not in parsed or not isinstance(parsed[key], bool):
        return False, f"missing/non-bool {key}"
    return True, ""


def history_prompt(task: dict[str, Any], match_mode: str) -> str:
    h_ids = ", ".join(h["id"] for h in task["history_images"])
    g_id = task["generated_image"].get("id", "G01")
    if match_mode == "loose":
        rule = """
Important rule:
Historical matching must be broad. Answer YES if G has any reasonable connection to any H image at the broad-interest level.

A YES decision may be based on:
- same or similar broad subject category;
- same or similar object, animal, person, place, or scene category;
- same or similar activity, function, or usage;
- same or similar visual theme, semantic theme, or artistic style;
- natural extension, variant, combination, or neighboring interest of the user's historical interests.

Answer NO only when G has no reasonable connection to all H images in subject, scene, activity, function, semantic theme, or visual interest.
If uncertain but G can reasonably be interpreted as an extension or neighboring interest of the user's history, answer YES.
""".strip()
    elif match_mode == "strict":
        rule = """
Important rule for movie-poster history matching:
Historical matching must require a concrete visual-interest connection. Answer YES when G and at least one H image clearly share any of the following:
- the same recognizable subject category, such as spacecraft, monsters, animated animals, soldiers, detectives, musicians, or sports players;
- the same concrete activity or event, such as a battle, wedding, chase, performance, investigation, or journey;
- the same distinctive scene type or specific visual concept, even when object identity and composition differ.

The following evidence alone is NOT sufficient for YES:
- the same broad movie genre, such as drama, action, romance, comedy, horror, or science fiction;
- a generic person, face, couple, group, landscape, building, or poster layout without a shared role, activity, or scene;
- similar color palette, lighting, atmosphere, mood, typography, composition, or artistic style;
- a natural extension, neighboring interest, or broadly related semantic theme without a shared concrete visual concept.

Exact identity, franchise, scene, and composition are not required. Answer NO when the connection is only broad, stylistic, genre-level, or speculative. If unsure, answer NO.
""".strip()
    elif match_mode == "balanced":
        rule = """
Important rule for balanced movie-poster history matching:
Answer YES when G has a clear or reasonably strong visual-interest connection to at least one H image. Exact identity, franchise, object, scene, and composition are not required.

A YES decision may be based on:
- the same or a meaningfully related recognizable subject category, excluding a merely generic person or face;
- the same or a related concrete scene, activity, event, role, or object function;
- a natural variation within a specific visual-interest family, such as related space-travel scenes, creature-horror concepts, wedding scenes, musical performances, investigations, battles, or sports activities;
- a combination of related semantic content and related visual treatment that together indicates the same user interest.

The following evidence alone is NOT sufficient for YES:
- only the same broad movie genre;
- only a generic person, face, couple, group, landscape, building, or poster layout;
- only similar color, lighting, mood, typography, composition, or artistic style;
- a speculative neighboring interest without a concrete shared subject, scene, activity, role, or visual concept.

If uncertain, answer YES only when you can identify a concrete shared visual interest; otherwise answer NO.
""".strip()
    elif match_mode == "pixelrec_balanced":
        rule = """
Important rule for balanced video-cover history matching:
Answer YES when G has a clear or reasonably strong content-interest connection to at least one H image. Exact identity, creator, object, scene, and composition are not required.

A YES decision may be based on:
- the same or a meaningfully related recognizable subject category, such as a specific animal, game, sport, food, performance, vehicle, character type, or technology;
- the same or a related concrete scene, activity, event, role, or object function;
- a natural variation within a specific content-interest family, such as related gameplay, cooking, music-performance, animation, pet, sports, or film-and-television content;
- a combination of related semantic content and visual treatment that together indicates the same user interest.

The following evidence alone is NOT sufficient for YES:
- only a generic person, face, group, indoor scene, outdoor scene, or video-cover layout;
- only overlaid text, subtitles, logos, borders, color, lighting, composition, or editing style;
- only the same broad entertainment category;
- a speculative neighboring interest without a concrete shared subject, scene, activity, role, or visual concept.

Treat H as content the user previously interacted with, not as an explicit statement that every visual detail was liked. If uncertain, answer YES only when you can identify a concrete shared visual interest; otherwise answer NO.
""".strip()
    else:
        raise ValueError(match_mode)
    return f"""
You are a history-interest recall judge.

Task:
Decide whether the generated image {g_id} can be explained by the target user's historical interests.

Input:
- H: five historical images {"previously interacted with by" if match_mode == "pixelrec_balanced" else "liked by"} the target user, numbered as {h_ids}.
- G: one generated image, numbered as {g_id}.

{rule}

Output valid JSON only.

JSON schema:
{{
  "history_related": true,
  "matched_history_images": ["H01"]
}}

Field constraints:
- `history_related` must be true or false.
- `matched_history_images` may only contain IDs from: {h_ids}.
- If `history_related` is true, include at least one matched history image.
- If `history_related` is false, `matched_history_images` must be an empty list.
""".strip()


def cf_prompt(task: dict[str, Any], match_mode: str) -> str:
    c_ids = ", ".join(c["id"] for c in task["cf_images"])
    g_id = task["generated_image"].get("id", "G01")
    if match_mode == "loose":
        rule = """
Important rule:
CF matching must be broad.
Answer YES when G is related to any C image in broad visual-interest terms, including same or similar subject category, related scene category, related activity or object function, similar semantic theme, natural extension of the candidate interest, or same visual interest family.
Do not require exact object identity, exact scene, exact activity, exact number of objects, or exact visual composition.
Only answer NO when G is clearly unrelated to all C images at the broad-interest level.
If unsure, answer YES.
""".strip()
    elif match_mode == "strict":
        rule = """
Important rule:
CF matching must be strict.
Answer YES only when G and at least one C image clearly share the same core subject or core scene.
Do not answer YES only because of similar color, composition, atmosphere, background, or generic style.
If unsure, answer NO.
""".strip()
    else:
        raise ValueError(match_mode)
    return f"""
You are a collaborative-filtering visual judge.

Task:
Decide whether the generated image {g_id} is related to the collaborative-filtering candidate images C.

Input:
- C: five candidate images from similar users, numbered as {c_ids}.
- G: one generated image, numbered as {g_id}.

{rule}

Output valid JSON only.

JSON schema:
{{
  "cf_related": true,
  "matched_cf_images": ["C01"]
}}

Field constraints:
- `cf_related` must be true or false.
- `matched_cf_images` may only contain IDs from: {c_ids}.
- If `cf_related` is true, include at least one matched CF image.
- If `cf_related` is false, `matched_cf_images` must be an empty list.
""".strip()


def build_stage_content(
    task: dict[str, Any],
    stage: str,
    image_size: int,
    history_match_mode: str,
    cf_match_mode: str,
) -> list[dict[str, Any]]:
    if stage == "history":
        content: list[dict[str, Any]] = [
            {"type": "text", "text": history_prompt(task, history_match_mode)}
        ]
        content.append({"type": "text", "text": "\nHistorical images H:"})
        for image in task["history_images"]:
            content.append({"type": "text", "text": f"{image['id']} history image:"})
            content.append({"type": "image", "image": open_image(image["path"], image_size)})
    elif stage == "cf":
        content = [{"type": "text", "text": cf_prompt(task, cf_match_mode)}]
        content.append({"type": "text", "text": "\nCollaborative-filtering candidate images C:"})
        for image in task["cf_images"]:
            content.append({"type": "text", "text": f"{image['id']} CF candidate image:"})
            content.append({"type": "image", "image": open_image(image["path"], image_size)})
    else:
        raise ValueError(stage)
    content.append({"type": "text", "text": "\nGenerated image G:"})
    content.append({"type": "text", "text": f"{task['generated_image']['id']} generated image:"})
    content.append({"type": "image", "image": open_image(task["generated_image"]["path"], image_size)})
    content.append({"type": "text", "text": "\nOutput JSON only."})
    return content


def parse_stage_outputs(raw_texts: list[str], key: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for raw_text in raw_texts:
        try:
            parsed = json_from_text(raw_text)
            parse_ok, parse_error = validate_bool(parsed, key)
        except Exception as exc:  # noqa: BLE001
            parsed = {}
            parse_ok = False
            parse_error = f"{type(exc).__name__}: {exc}"
        results.append({"raw_output": raw_text, "judge": parsed, "parse_ok": parse_ok, "parse_error": parse_error})
    return results


def run_stage_batch(
    model,
    processor,
    input_device,
    tasks: list[dict[str, Any]],
    stage: str,
    image_size: int,
    max_new_tokens: int,
    history_match_mode: str,
    cf_match_mode: str,
) -> list[dict[str, Any]]:
    messages = [
        [{
            "role": "user",
            "content": build_stage_content(
                task, stage, image_size, history_match_mode, cf_match_mode
            ),
        }]
        for task in tasks
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(input_device)
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    prompt_len = inputs.input_ids.shape[1]
    trimmed = generated[:, prompt_len:]
    raw_texts = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    key = "history_related" if stage == "history" else "cf_related"
    return parse_stage_outputs(raw_texts, key)


def run_stage_adaptive(
    model,
    processor,
    input_device,
    tasks: list[dict[str, Any]],
    stage: str,
    image_size: int,
    max_new_tokens: int,
    history_match_mode: str,
    cf_match_mode: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    start = 0
    current_batch = max(1, int(batch_size))
    while start < len(tasks):
        batch = tasks[start : start + current_batch]
        try:
            results.extend(
                run_stage_batch(
                    model,
                    processor,
                    input_device,
                    batch,
                    stage,
                    image_size,
                    max_new_tokens,
                    history_match_mode,
                    cf_match_mode,
                )
            )
            start += len(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current_batch <= 1:
                raise
            current_batch = max(1, current_batch - min(4, current_batch - 1))
            print(json.dumps({"event": "oom_reduce_batch", "stage": stage, "new_batch_size": current_batch}), flush=True)
    return results


def final_label(history_result: dict[str, Any], cf_result: dict[str, Any] | None) -> tuple[str, bool]:
    if not history_result.get("parse_ok"):
        raise RuntimeError("history-stage output could not be parsed; no semantic label was assigned")
    history_yes = bool((history_result.get("judge") or {}).get("history_related"))
    if history_yes:
        return "direct_history_theme", False
    if cf_result is None or not cf_result.get("parse_ok"):
        raise RuntimeError("CF-stage output could not be parsed; no semantic label was assigned")
    cf_yes = bool((cf_result.get("judge") or {}).get("cf_related"))
    if cf_yes:
        return "true_cf_expansion", True
    return "other_or_uncertain", False


def main() -> int:
    parser = argparse.ArgumentParser(description="Batched two-stage History/CF source attribution.")
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--model-family",
        choices=("auto", "qwen3-vl", "auto-multimodal"),
        default="qwen3-vl",
    )
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa", choices=("sdpa", "flash_attention_2", "eager"))
    parser.add_argument(
        "--history-match-mode",
        default="loose",
        choices=("strict", "balanced", "pixelrec_balanced", "loose"),
    )
    parser.add_argument("--cf-match-mode", default="loose", choices=("strict", "loose"))
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    family = args.model_family
    if family == "auto":
        family = "qwen3-vl" if "qwen3-vl" in args.model.name.lower() else "auto-multimodal"
    model_class = Qwen3VLForConditionalGeneration if family == "qwen3-vl" else AutoModelForMultimodalLM
    if model_class is None:
        raise RuntimeError(f"the installed Transformers version does not support {family}")

    tasks_all = load_jsonl(args.tasks_jsonl)
    tasks = tasks_all[args.shard_index :: args.num_shards]
    if args.limit > 0:
        tasks = tasks[: args.limit]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = completed_task_ids(args.output_jsonl)
    tasks = [task for task in tasks if task["task_id"] not in done]

    # A resumed shard may already be complete. Avoid loading the judge just
    # to discover that there is no remaining work.
    if not tasks:
        print(
            json.dumps(
                {
                    "processed": 0,
                    "remaining": 0,
                    "skipped_completed": len(done),
                    "shard_index": args.shard_index,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    # Decoder-only batched generation must read the final real prompt token,
    # rather than a right-padding token, when prompt lengths differ.
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model = model_class.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    ).eval()
    input_device = getattr(model, "device", None) or next(model.parameters()).device

    started = time.time()
    processed = 0
    label_counts = {"direct_history_theme": 0, "true_cf_expansion": 0, "other_or_uncertain": 0}
    with args.output_jsonl.open("a", encoding="utf-8") as handle:
        for start in range(0, len(tasks), args.batch_size):
            batch = tasks[start : start + args.batch_size]
            t0 = time.time()
            history_results = run_stage_adaptive(
                model,
                processor,
                input_device,
                batch,
                "history",
                args.image_size,
                args.max_new_tokens,
                args.history_match_mode,
                args.cf_match_mode,
                args.batch_size,
            )
            if start == 0 and not any(result.get("parse_ok") for result in history_results):
                examples = [result.get("raw_output", "")[:160] for result in history_results[:3]]
                raise RuntimeError(
                    "all outputs in the first history batch failed JSON parsing; "
                    f"refusing to continue with a biased fallback: {examples}"
                )
            cf_positions = [
                idx
                for idx, history in enumerate(history_results)
                if history.get("parse_ok")
                and not bool((history.get("judge") or {}).get("history_related"))
            ]
            cf_by_pos: dict[int, dict[str, Any]] = {}
            if cf_positions:
                cf_tasks = [batch[idx] for idx in cf_positions]
                cf_results = run_stage_adaptive(
                    model,
                    processor,
                    input_device,
                    cf_tasks,
                    "cf",
                    args.image_size,
                    args.max_new_tokens,
                    args.history_match_mode,
                    args.cf_match_mode,
                    args.batch_size,
                )
                cf_by_pos = {pos: result for pos, result in zip(cf_positions, cf_results)}

            for local_idx, task in enumerate(batch):
                history = history_results[local_idx]
                cf = cf_by_pos.get(local_idx)
                history_yes = (
                    bool((history.get("judge") or {}).get("history_related"))
                    if history.get("parse_ok")
                    else False
                )
                label, is_true_cf = final_label(history, cf)
                label_counts[label] = label_counts.get(label, 0) + 1
                result = {
                    "task_id": task["task_id"],
                    "user_id": task["user_id"],
                    "user_index": task["user_index"],
                    "method": task["method"],
                    "generated_image": task["generated_image"],
                    "history_images": task["history_images"],
                    "cf_images": task["cf_images"],
                    "history_match_mode": args.history_match_mode,
                    "cf_match_mode": args.cf_match_mode,
                    "history_stage": history,
                    "cf_stage": cf,
                    "judge": {
                        "label": label,
                        "is_true_cf_expansion": is_true_cf,
                        "history_related": history_yes,
                        "cf_related": None if cf is None else bool((cf.get("judge") or {}).get("cf_related")),
                        "matched_history_images": (history.get("judge") or {}).get("matched_history_images", []),
                        "matched_cf_images": [] if cf is None else (cf.get("judge") or {}).get("matched_cf_images", []),
                        "reason": "",
                    },
                    "parse_ok": bool(history.get("parse_ok")) and (cf is None or bool(cf.get("parse_ok"))),
                    "elapsed_sec": time.time() - t0,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            processed += len(batch)
            if processed == len(batch) or processed % args.log_every == 0:
                elapsed = time.time() - started
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "remaining": len(tasks) - processed,
                            "avg_sec_per_image": elapsed / max(processed, 1),
                            "batch_size": args.batch_size,
                            "label_counts": label_counts,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
