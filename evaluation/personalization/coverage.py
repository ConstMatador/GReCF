#!/usr/bin/env python3
"""Evaluate generated-image coverage of adaptive held-out test interests."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from grecf.data import id_order_sha256, load_cache, load_public_dataset
from grecf.multi_interest import load_or_build_adaptive_interest_cache


def normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), eps)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument("--test-interest-cache-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variant", default="personalized")
    parser.add_argument("--max-interests", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--minimum-support", type=int, default=4)
    parser.add_argument("--minimum-support-fraction", type=float, default=0.05)
    parser.add_argument("--merge-cosine", type=float, default=0.90)
    parser.add_argument("--retained-gain", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def load_generation(generation_dir: Path, variant: str) -> tuple[pd.DataFrame, np.ndarray]:
    table = generation_dir / "tables/generation_metrics.csv"
    feature_path = generation_dir / "tables/generated_clip_features.float32.npy"
    if not table.is_file() or not feature_path.is_file():
        raise FileNotFoundError(f"missing generation contract under {generation_dir}")
    frame = pd.read_csv(table)
    features = np.load(feature_path).astype(np.float32)
    if len(frame) != len(features):
        raise ValueError(f"generation row/feature mismatch: {len(frame)} vs {len(features)}")
    keep = frame["variant"].astype(str).eq(variant).to_numpy()
    frame = frame.loc[keep].copy().reset_index(drop=True)
    features = normalize(features[keep])
    order = frame.sort_values(["user_index", "seed_offset"], kind="stable").index.to_numpy()
    return frame.loc[order].reset_index(drop=True), features[order]


def main() -> int:
    args = parse_args()
    dataset = load_public_dataset(args.dataset_root)
    catalog = np.asarray(
        load_cache(args.clip_cache, dataset.num_items, id_order_sha256(dataset.image_ids), (512,)),
        dtype=np.float32,
    )
    test_interests = load_or_build_adaptive_interest_cache(
        args.test_interest_cache_dir,
        dataset,
        catalog,
        max_interests=args.max_interests,
        folds=args.folds,
        minimum_support=args.minimum_support,
        minimum_support_fraction=args.minimum_support_fraction,
        merge_cosine=args.merge_cosine,
        retained_gain=args.retained_gain,
        seed=args.seed,
        source_split="test",
    )
    frame, generated = load_generation(args.generation_dir, args.variant)
    device = torch.device(args.device)

    per_user_rows: list[dict[str, object]] = []
    per_image_rows: list[dict[str, object]] = []
    for user_index, group in frame.groupby("user_index", sort=True):
        user = int(user_index)
        row_indices = group.index.to_numpy()
        test_count = int(test_interests["counts"][user])
        active = np.flatnonzero(np.asarray(test_interests["mask"][user], dtype=bool))[:test_count]
        if test_count <= 0 or len(active) != test_count:
            raise ValueError(f"invalid test interests for user {user}: M={test_count}, active={len(active)}")
        budget = 2 * test_count
        if len(row_indices) < budget:
            raise ValueError(f"user {user} has {len(row_indices)} generated images, needs {budget}")
        selected_rows = row_indices[:budget]
        selected_group = frame.loc[selected_rows]
        expected_offsets = list(range(budget))
        actual_offsets = selected_group["seed_offset"].astype(int).tolist()
        if actual_offsets != expected_offsets:
            raise ValueError(
                f"user {user} requires fixed seed offsets {expected_offsets}, got {actual_offsets}"
            )

        prototypes = normalize(np.asarray(test_interests["prototypes"][user, active]))
        similarities = (
            torch.from_numpy(generated[selected_rows]).to(device)
            @ torch.from_numpy(prototypes).to(device).T
        ).float().cpu().numpy()
        assigned_local = similarities.argmax(axis=1)
        assigned_interests = active[assigned_local]
        assigned_sims = similarities[np.arange(budget), assigned_local]
        hit_interests = np.unique(assigned_interests)
        histogram = {int(index): int((assigned_interests == index).sum()) for index in active}
        dominant = max(histogram.values()) if histogram else 0
        per_user_rows.append(
            {
                "method": args.method,
                "user_index": user,
                "user_id": str(group["user_id"].iloc[0]),
                "test_image_count": int(len(dataset.test[user])),
                "test_topic_count_m": test_count,
                "required_generation_count_2m": budget,
                "available_generation_count": int(len(row_indices)),
                "hit_test_topic_count": int(len(hit_interests)),
                "test_topic_coverage": float(len(hit_interests) / test_count),
                "dominant_test_topic_count": int(dominant),
                "test_topic_dominance": float(dominant / budget),
                "mean_assignment_similarity": float(assigned_sims.mean()),
                "assigned_test_topic_histogram": json.dumps(histogram, ensure_ascii=False),
            }
        )
        for local_index, (_, row) in enumerate(selected_group.iterrows()):
            per_image_rows.append(
                {
                    "method": args.method,
                    "user_index": user,
                    "user_id": str(row["user_id"]),
                    "seed_offset": int(row["seed_offset"]),
                    "image_path": str(row["image_path"]),
                    "assigned_test_topic_index": int(assigned_interests[local_index]),
                    "assigned_test_topic_similarity": float(assigned_sims[local_index]),
                }
            )

    per_user = pd.DataFrame(per_user_rows)
    per_image = pd.DataFrame(per_image_rows)
    if per_user.empty:
        raise ValueError("no Test Topic Coverage rows were produced")
    total_topics = int(per_user["test_topic_count_m"].sum())
    summary = {
        "method": args.method,
        "users": int(per_user["user_index"].nunique()),
        "source_split": "test",
        "total_test_images": int(per_user["test_image_count"].sum()),
        "total_test_topics_m": total_topics,
        "total_evaluated_generated_images_2m": int(per_user["required_generation_count_2m"].sum()),
        "test_topic_coverage_weighted": float(per_user["hit_test_topic_count"].sum() / total_topics),
        "test_topic_coverage_macro": float(per_user["test_topic_coverage"].mean()),
        "test_topic_dominance_weighted": float(
            per_user["dominant_test_topic_count"].sum()
            / per_user["required_generation_count_2m"].sum()
        ),
        "test_topic_dominance_macro": float(per_user["test_topic_dominance"].mean()),
        "mean_assignment_similarity": float(per_user["mean_assignment_similarity"].mean()),
        "budget_rule": "first 2*M_u fixed-seed generated images; test prototypes are evaluation-only",
        "adaptive_parameters": {
            "max_interests": args.max_interests,
            "folds": args.folds,
            "minimum_support": args.minimum_support,
            "minimum_support_fraction": args.minimum_support_fraction,
            "merge_cosine": args.merge_cosine,
            "retained_gain": args.retained_gain,
            "seed": args.seed,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_user.to_csv(args.output_dir / "test_topic_coverage_per_user.csv", index=False)
    per_image.to_csv(args.output_dir / "test_topic_coverage_per_image.csv", index=False)
    pd.DataFrame([summary]).to_csv(args.output_dir / "test_topic_coverage_summary.csv", index=False)
    write_json(args.output_dir / "test_topic_coverage_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
