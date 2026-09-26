#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from grecf.data import id_order_sha256, load_cache, load_public_dataset


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--judge-merged", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, action="append", required=True)
    parser.add_argument("--context-root", type=Path, action="append", required=True)
    parser.add_argument("--lpips-summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="GReCF (SD1.5) cold-start generation")
    args = parser.parse_args()
    if not (len(args.generation_dir) == len(args.context_root) == len(args.lpips_summary) == 5):
        raise ValueError("all per-round argument counts must match")

    dataset = load_public_dataset(args.base_dataset)
    judge_rows = [json.loads(line) for line in args.judge_merged.open(encoding="utf-8") if line.strip()]

    rounds = []
    per_round = zip(args.generation_dir, args.context_root, args.lpips_summary)
    for round_number, (generation_dir, context_root, lpips_path) in enumerate(per_round, 1):
        frame = pd.read_csv(generation_dir / "tables/generation_metrics.csv")
        generated = normalize(np.load(generation_dir / "tables/generated_clip_features.float32.npy", mmap_mode="r"))
        selected_judge = [row for row in judge_rows if int(row["generated_image"]["round"]) == round_number]
        if len(selected_judge) != 1000:
            raise ValueError(
                f"round {round_number}: expected 1,000 judged images, found {len(selected_judge)}"
            )
        counts = Counter(row["judge"]["label"] for row in selected_judge)
        interest_prefix = context_root / "interests/adaptive_b"
        prototypes_all = normalize(np.load(interest_prefix.with_name("adaptive_b_prototypes.float32.npy"), mmap_mode="r"))
        masks_all = np.load(interest_prefix.with_name("adaptive_b_mask.bool.npy"), mmap_mode="r")
        per_user = []
        for user_index, group in frame.groupby("user_index", sort=True):
            generated_rows = group.index.to_numpy()
            active = np.flatnonzero(masks_all[int(user_index)])
            prototypes = prototypes_all[int(user_index), active]
            similarity = generated[generated_rows] @ prototypes.T
            assigned = similarity.argmax(axis=1)
            histogram = np.bincount(assigned, minlength=len(active))
            per_user.append(
                {
                    "topics": len(active),
                    "hit": int((histogram > 0).sum()),
                    "budget": len(generated_rows),
                    "dominant": int(histogram.max()),
                }
            )
        total_topics = sum(row["topics"] for row in per_user)
        total_budget = sum(row["budget"] for row in per_user)
        lpips = json.loads(lpips_path.read_text(encoding="utf-8"))
        context_summary = json.loads((context_root / "summary.json").read_text(encoding="utf-8"))
        round_result = {
                "round": round_number,
                "images": len(selected_judge),
                "judge_counts": dict(counts),
                "judge_rates": {key: value / len(selected_judge) for key, value in counts.items()},
                "mean_nearest_current_history_similarity": float(
                    np.mean([row["generated_image"]["clip_nearest_history_similarity"] for row in selected_judge])
                ),
                "mean_nearest_cf_similarity": float(
                    np.mean([row["generated_image"]["clip_nearest_cf_similarity"] for row in selected_judge])
                ),
                "lpips_to_current_input_history": lpips["lpips_to_input_history_min_mean"],
                "current_history_topic_coverage_weighted": sum(row["hit"] for row in per_user) / total_topics,
                "current_history_topic_coverage_macro": float(np.mean([row["hit"] / row["topics"] for row in per_user])),
                "current_history_topic_dominance_weighted": sum(row["dominant"] for row in per_user) / total_budget,
        }
        if "history_balance_max_ratio" in context_summary:
            round_result["history_balance"] = {
                "max_theme_ratio": context_summary["history_balance_max_ratio"],
                "archive_images_per_user_min": context_summary["archive_images_per_user_min"],
                "archive_images_per_user_max": context_summary["archive_images_per_user_max"],
                "input_images_per_user_min": context_summary["history_images_per_user_min"],
                "input_images_per_user_max": context_summary["history_images_per_user_max"],
                "images_removed": context_summary["history_images_removed"],
                "users_trimmed": context_summary["users_trimmed"],
            }
        rounds.append(round_result)
    balance_ratio = max(
        (round_result.get("history_balance", {}).get("max_theme_ratio", 0.0) for round_result in rounds),
        default=0.0,
    )
    feedback = "all prior generated images are assumed liked and appended to history"
    if balance_ratio > 0.0:
        feedback = (
            "all prior generated images are retained in the archive; each round conditions on a deterministic "
            f"adaptive-theme-balanced subset with maximum/minimum theme count <= {balance_ratio:g}"
        )
    payload = {
        "method": args.method,
        "users": int(pd.read_csv(args.generation_dir[0] / "tables/generation_metrics.csv")["user_id"].nunique()),
        "rounds": rounds,
        "protocol": {
            "initial_images": "5-10",
            "initial_fine_topics": "1-3",
            "generated_images_per_round": 10,
            "feedback": feedback,
            "metric_reference": "all images used as conditioning history in the current round",
            "training_exposure": "cold-start users excluded from GReCF training",
            "judge": "Qwen3-VL-32B, CIGR History loose / CF loose",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
