#!/usr/bin/env python3
"""Build one two-stage source-attribution task per generated image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grecf.data import id_order_sha256, load_cache, load_public_dataset


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def top_indices(values: np.ndarray, count: int) -> np.ndarray:
    count = min(int(count), len(values))
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count == len(values):
        return np.argsort(-values, kind="stable")
    selected = np.argpartition(-values, count - 1)[:count]
    return selected[np.argsort(-values[selected], kind="stable")]


def load_selected_users(path: Path, dataset) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("user_indices", payload.get("all_user_indices"))
    if values is not None:
        return [int(value) for value in values]
    if "user_ids" in payload:
        by_id = {str(user_id): index for index, user_id in enumerate(dataset.user_ids)}
        return [by_id[str(value)] for value in payload["user_ids"]]
    raise ValueError(f"unsupported selected-users file: {path}")


def load_generated(
    feature_path: Path,
    manifest_path: Path,
    method: str,
    selected_user_ids: set[str],
    images_per_user: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    manifest = pd.read_csv(manifest_path)
    features = np.load(feature_path, mmap_mode="r")
    if len(manifest) != len(features):
        raise ValueError(f"manifest/features mismatch: {len(manifest)} vs {len(features)}")
    method_column = "eval_method" if "eval_method" in manifest.columns else "method"
    path_column = "resolved_image_path" if "resolved_image_path" in manifest.columns else "image_path"
    keep = (
        manifest[method_column].astype(str).eq(method)
        & manifest["user_id"].astype(str).isin(selected_user_ids)
        & (manifest["seed_offset"].astype(int) < images_per_user)
    )
    rows = np.flatnonzero(keep.to_numpy())
    frame = manifest.loc[keep].copy()
    frame["resolved_image_path"] = frame[path_column].astype(str)
    frame["feature_row"] = rows
    frame = frame.sort_values(["user_id", "seed_offset"], kind="stable").reset_index(drop=True)
    ordered_rows = frame["feature_row"].to_numpy(dtype=np.int64)
    return frame, normalize(np.asarray(features[ordered_rows], dtype=np.float32))


def collect_cf_pool(
    dataset,
    user: int,
    neighbors: np.ndarray,
    similarities: np.ndarray,
) -> tuple[np.ndarray, dict[int, float], dict[int, list[int]], dict[int, int]]:
    observed = set(int(item) for item in dataset.train[user])
    scores: dict[int, float] = {}
    supporters: dict[int, list[int]] = {}
    best_rank: dict[int, int] = {}
    for rank, (neighbor, similarity) in enumerate(zip(neighbors.tolist(), similarities.tolist()), 1):
        weight = max(float(similarity), 0.0) + 1e-3
        for raw_item in dataset.train[int(neighbor)]:
            item = int(raw_item)
            if item in observed:
                continue
            scores[item] = scores.get(item, 0.0) + weight
            supporters.setdefault(item, []).append(int(neighbor))
            best_rank[item] = min(best_rank.get(item, rank), rank)
    return np.asarray(sorted(scores), dtype=np.int64), scores, supporters, best_rank


def image_record(dataset, item: int, identifier: str, **metadata: Any) -> dict[str, Any]:
    return {
        "id": identifier,
        "image_id": str(dataset.image_ids[item]),
        "item_index": int(item),
        "path": str(dataset.image_paths[item]),
        **metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument("--user-rep-cache", type=Path, required=True)
    parser.add_argument("--selected-users-json", type=Path, required=True)
    parser.add_argument("--generated-features", type=Path, required=True)
    parser.add_argument("--generated-manifest", type=Path, required=True)
    parser.add_argument("--eval-method", default="GReCF")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--images-per-user", type=int, default=10)
    parser.add_argument("--top-neighbors", type=int, default=5)
    parser.add_argument("--history-images", type=int, default=5)
    parser.add_argument("--cf-images", type=int, default=5)
    args = parser.parse_args()

    dataset = load_public_dataset(args.dataset_root)
    users = load_selected_users(args.selected_users_json, dataset)
    if not users or len(users) != len(set(users)):
        raise ValueError("selected users must be non-empty and unique")
    selected_ids = {str(dataset.user_ids[user]) for user in users}

    catalog = normalize(
        np.asarray(
            load_cache(
                args.clip_cache,
                dataset.num_items,
                id_order_sha256(dataset.image_ids),
                (512,),
            ),
            dtype=np.float32,
        )
    )
    user_representations = normalize(np.asarray(np.load(args.user_rep_cache, mmap_mode="r")))
    generated_frame, generated_features = load_generated(
        args.generated_features,
        args.generated_manifest,
        args.eval_method,
        selected_ids,
        args.images_per_user,
    )
    expected = len(users) * args.images_per_user
    if len(generated_frame) != expected:
        raise RuntimeError(f"expected {expected} generated images, found {len(generated_frame)}")
    user_by_id = {str(dataset.user_ids[user]): user for user in users}
    generated_frame["user_index"] = generated_frame["user_id"].astype(str).map(user_by_id).astype(int)

    tasks: list[dict[str, Any]] = []
    for user in users:
        user_id = str(dataset.user_ids[user])
        user_scores = user_representations @ user_representations[user]
        user_scores[user] = -np.inf
        neighbors = top_indices(user_scores, args.top_neighbors)
        neighbor_scores = user_scores[neighbors]
        cf_items, cf_scores, supporters, best_rank = collect_cf_pool(
            dataset, user, neighbors, neighbor_scores
        )
        if len(cf_items) < args.cf_images:
            raise RuntimeError(
                f"user {user_id} has only {len(cf_items)} collaborative candidates"
            )

        history_items = np.asarray(dataset.train[user], dtype=np.int64)
        if len(history_items) < args.history_images:
            raise RuntimeError(f"user {user_id} has only {len(history_items)} history images")
        history_features = catalog[history_items]
        cf_features = catalog[cf_items]
        group = generated_frame.loc[generated_frame["user_index"].eq(user)].sort_values("seed_offset")
        if len(group) != args.images_per_user:
            raise RuntimeError(f"user {user_id} has {len(group)} generated images")

        for sample_index, (row_index, row) in enumerate(group.iterrows(), 1):
            generated = generated_features[row_index]
            history_similarity = history_features @ generated
            cf_similarity = cf_features @ generated
            history_local = top_indices(history_similarity, args.history_images)
            cf_local = top_indices(cf_similarity, args.cf_images)

            history_records = [
                image_record(
                    dataset,
                    int(history_items[local]),
                    f"H{position:02d}",
                    clip_similarity=float(history_similarity[local]),
                )
                for position, local in enumerate(history_local.tolist(), 1)
            ]
            cf_records = []
            for position, local in enumerate(cf_local.tolist(), 1):
                item = int(cf_items[local])
                cf_records.append(
                    image_record(
                        dataset,
                        item,
                        f"C{position:02d}",
                        clip_similarity=float(cf_similarity[local]),
                        support_user_count=len(set(supporters[item])),
                        support_score=float(cf_scores[item]),
                        best_neighbor_rank=int(best_rank[item]),
                    )
                )

            tasks.append(
                {
                    "task_id": f"{user_id}_G{sample_index:02d}",
                    "method": args.eval_method,
                    "condition_variant": "personalized",
                    "user_index": int(user),
                    "user_id": user_id,
                    "generated_image": {
                        "id": "G01",
                        "sample_index": sample_index,
                        "seed_offset": int(row["seed_offset"]),
                        "path": str(row["resolved_image_path"]),
                    },
                    "history_images": history_records,
                    "cf_images": cf_records,
                    "top_neighbors": [
                        {
                            "rank": rank,
                            "user_index": int(neighbor),
                            "user_id": str(dataset.user_ids[int(neighbor)]),
                            "similarity": float(similarity),
                        }
                        for rank, (neighbor, similarity) in enumerate(
                            zip(neighbors.tolist(), neighbor_scores.tolist()), 1
                        )
                    ],
                }
            )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    summary = {
        "tasks": len(tasks),
        "users": len(users),
        "images_per_user": args.images_per_user,
        "nearest_users": args.top_neighbors,
        "history_images_per_task": args.history_images,
        "cf_images_per_task": args.cf_images,
    }
    args.output_jsonl.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
