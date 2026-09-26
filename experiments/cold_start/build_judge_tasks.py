#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grecf.data import id_order_sha256, load_cache, load_public_dataset


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def top_indices(values: np.ndarray, count: int) -> np.ndarray:
    count = min(int(count), len(values))
    if count == len(values):
        return np.argsort(-values, kind="stable")
    selected = np.argpartition(-values, count - 1)[:count]
    return selected[np.argsort(-values[selected], kind="stable")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument("--base-user-reps", type=Path, required=True)
    parser.add_argument("--context-root", type=Path, action="append", required=True)
    parser.add_argument("--generation-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="GReCF (SD1.5) cold-start generation")
    parser.add_argument("--top-neighbors", type=int, default=5)
    parser.add_argument("--history-topk", type=int, default=5)
    parser.add_argument("--cf-topk", type=int, default=5)
    parser.add_argument("--limit-users", type=int, default=0)
    parser.add_argument("--selected-users-json", type=Path, default=None)
    args = parser.parse_args()
    if len(args.context_root) != len(args.generation_dir):
        raise ValueError("context-root and generation-dir counts must match")
    if len(args.context_root) != 5:
        raise ValueError("the paper protocol requires exactly five cold-start rounds")

    dataset = load_public_dataset(args.base_dataset)
    catalog = normalize(
        np.asarray(load_cache(args.clip_cache, dataset.num_items, id_order_sha256(dataset.image_ids), (512,)), dtype=np.float32)
    )
    seen_reps = normalize(np.asarray(np.load(args.base_user_reps, mmap_mode="r"), dtype=np.float32))
    item_index_by_id = {value: index for index, value in enumerate(dataset.image_ids)}

    generated_by_path: dict[str, np.ndarray] = {}
    generation_rounds = []
    for round_number, directory in enumerate(args.generation_dir, 1):
        frame = pd.read_csv(directory / "tables/generation_metrics.csv")
        features = normalize(np.load(directory / "tables/generated_clip_features.float32.npy", mmap_mode="r"))
        if len(frame) != len(features):
            raise ValueError(f"generation row/feature mismatch under {directory}")
        for row_index, path in enumerate(frame["image_path"].astype(str)):
            generated_by_path[path] = features[row_index]
        generation_rounds.append((round_number, frame, features))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    task_count = 0
    with args.output.open("w", encoding="utf-8") as output:
        for (round_number, frame, generated), context_root in zip(generation_rounds, args.context_root):
            context_dataset = load_public_dataset(context_root / "dataset")
            cold_reps = normalize(np.asarray(np.load(context_root / "user_reps.float32.npy", mmap_mode="r"), dtype=np.float32)[dataset.num_users :])
            history_table = pd.read_parquet(context_root / "cold_history.parquet")
            all_cold_ids = context_dataset.user_ids[dataset.num_users :]
            cold_offset_by_id = {user_id: offset for offset, user_id in enumerate(all_cold_ids)}
            if args.selected_users_json is not None:
                selected = json.loads(args.selected_users_json.read_text(encoding="utf-8"))
                cold_ids = [str(value) for value in selected["user_ids"]]
            else:
                cold_ids = all_cold_ids
            if args.limit_users > 0:
                cold_ids = cold_ids[: args.limit_users]
            for user_id in cold_ids:
                cold_offset = cold_offset_by_id[user_id]
                user_index = dataset.num_users + cold_offset
                group = frame.loc[frame["user_id"].astype(str).eq(user_id)].sort_values("seed_offset")
                if len(group) != 10:
                    raise ValueError(f"round {round_number}, user {user_id}: expected 10 images, found {len(group)}")
                if len(group) != 10:
                    raise ValueError(f"round {round_number} user {user_id} has {len(group)} generated rows")
                gen_rows = group.index.to_numpy(dtype=np.int64)
                gen_features = generated[gen_rows]

                user_history = history_table.loc[history_table["user_id"].astype(str).eq(user_id)].copy()
                history_features = []
                history_records = []
                initial_items: set[int] = set()
                for _, row in user_history.iterrows():
                    item_index = int(row["item_index"])
                    path = str(row["path"])
                    if item_index >= 0:
                        feature = catalog[item_index]
                        initial_items.add(item_index)
                        image_id = dataset.image_ids[item_index]
                    else:
                        feature = generated_by_path[path]
                        image_id = f"generated-r{int(row['round']):02d}-{int(row['rank']):02d}"
                    history_features.append(feature)
                    history_records.append({"path": path, "image_id": image_id})
                history_features_np = normalize(np.stack(history_features))

                similarities = seen_reps @ cold_reps[cold_offset]
                neighbor_indices = top_indices(similarities, args.top_neighbors)
                cf_scores_by_item: dict[int, float] = {}
                support_by_item: dict[int, list[int]] = {}
                best_rank: dict[int, int] = {}
                for rank, neighbor in enumerate(neighbor_indices.tolist(), 1):
                    weight = float(max(similarities[neighbor], 0.0)) + 1e-3
                    for item in dataset.train[neighbor].tolist():
                        item = int(item)
                        if item in initial_items:
                            continue
                        cf_scores_by_item[item] = cf_scores_by_item.get(item, 0.0) + weight
                        support_by_item.setdefault(item, []).append(neighbor)
                        best_rank[item] = min(best_rank.get(item, rank), rank)
                cf_items = np.asarray(sorted(cf_scores_by_item), dtype=np.int64)
                cf_features = catalog[cf_items]

                history_scores = gen_features @ history_features_np.T
                cf_similarity = gen_features @ cf_features.T
                for local_index, (row_index, row) in enumerate(group.iterrows(), 1):
                    history_order = top_indices(history_scores[local_index - 1], args.history_topk)
                    cf_order = top_indices(cf_similarity[local_index - 1], args.cf_topk)
                    history_images = [
                        {
                            "id": f"H{position + 1:02d}",
                            "image_id": history_records[index]["image_id"],
                            "path": history_records[index]["path"],
                            "max_clip_similarity_to_generated": float(history_scores[local_index - 1, index]),
                        }
                        for position, index in enumerate(history_order.tolist())
                    ]
                    cf_images = []
                    for position, cf_local in enumerate(cf_order.tolist()):
                        item = int(cf_items[cf_local])
                        cf_images.append(
                            {
                                "id": f"C{position + 1:02d}",
                                "image_id": dataset.image_ids[item],
                                "item_index": item,
                                "path": str(dataset.image_paths[item]),
                                "max_clip_similarity_to_generated": float(cf_similarity[local_index - 1, cf_local]),
                                "max_clip_similarity_to_user_history": float((catalog[item] @ history_features_np.T).max()),
                                "support_user_count": len(set(support_by_item[item])),
                                "support_score": cf_scores_by_item[item],
                                "best_neighbor_rank": best_rank[item],
                            }
                        )
                    generated_image = {
                        "id": f"G{local_index:02d}",
                        "round": round_number,
                        "seed_offset": int(row["seed_offset"]),
                        "path": str(row["image_path"]),
                        "route": "direct",
                        "route_id": 0,
                        "clip_nearest_history_similarity": float(history_scores[local_index - 1, history_order[0]]),
                        "clip_nearest_cf_similarity": float(cf_similarity[local_index - 1, cf_order[0]]),
                    }
                    task = {
                        "task_id": f"{user_id}_R{round_number:02d}_G{local_index:02d}_cold_start",
                        "method": args.method,
                        "condition_variant": "personalized",
                        "user_index": user_index,
                        "user_id": user_id,
                        "generated_image": generated_image,
                        "history_images": history_images,
                        "cf_images": cf_images,
                        "cf_pool_summary": {
                            "top_neighbors": args.top_neighbors,
                            "history_size": len(history_records),
                            "raw_cf_pool_size": len(cf_items),
                            "seen_users_only": True,
                        },
                        "judge_instruction_version": "cold_start_single_image_history_loose_cf_loose_v1",
                    }
                    output.write(json.dumps(task, ensure_ascii=False) + "\n")
                    task_count += 1
                if cold_offset % 100 == 0:
                    print(json.dumps({"round": round_number, "users": cold_offset, "tasks": task_count}), flush=True)

    summary = {
        "tasks": task_count,
        "users": len(cold_ids),
        "rounds": len(args.generation_dir),
        "images_per_user_per_round": 10,
        "top_neighbors": args.top_neighbors,
        "history_topk": args.history_topk,
        "cf_topk": args.cf_topk,
        "cf_neighbor_scope": "original 10000 seen users only",
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
