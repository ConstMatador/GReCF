#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from grecf.data import id_order_sha256, load_cache, load_public_dataset
from grecf.multi_interest import _adaptive_user_interests


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, path)


def load_previous_histories(
    directories: list[Path],
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[dict[str, object]]]]:
    features_by_user: dict[str, list[np.ndarray]] = {}
    records_by_user: dict[str, list[dict[str, object]]] = {}
    for round_number, directory in enumerate(directories, 1):
        table = pd.read_csv(directory / "tables/generation_metrics.csv")
        features = np.load(directory / "tables/generated_clip_features.float32.npy", mmap_mode="r")
        if len(table) != len(features):
            raise ValueError(f"generation row/feature mismatch under {directory}")
        for row_index, row in table.iterrows():
            user_id = str(row["user_id"])
            features_by_user.setdefault(user_id, []).append(np.asarray(features[row_index], dtype=np.float32))
            records_by_user.setdefault(user_id, []).append(
                {
                    "source": "generated",
                    "round": round_number,
                    "seed_offset": int(row["seed_offset"]),
                    "path": str(row["image_path"]),
                }
            )
    return features_by_user, records_by_user


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--cold-root", type=Path, required=True)
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument("--base-user-reps", type=Path, required=True)
    parser.add_argument("--base-interest-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--previous-round", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--balance-history-max-ratio",
        type=float,
        default=3.0,
        help="Cap each adaptive theme at this multiple of the smallest theme; <=0 disables balancing.",
    )
    parser.add_argument("--balance-seed", type=int, default=20260906)
    args = parser.parse_args()
    if 0.0 < args.balance_history_max_ratio < 1.0:
        raise ValueError("balance-history-max-ratio must be >= 1.0, or <= 0 to disable balancing")

    base = load_public_dataset(args.base_dataset)
    catalog = normalize(
        np.asarray(
            load_cache(args.clip_cache, base.num_items, id_order_sha256(base.image_ids), (512,)),
            dtype=np.float32,
        )
    )
    cold_public = args.cold_root / "public"
    cold_users = pd.read_parquet(cold_public / "users.parquet").sort_values("user_id").reset_index(drop=True)
    cold_edges = pd.read_parquet(cold_public / "interactions.parquet")
    if len(cold_users) != 100:
        raise ValueError(f"the paper protocol requires 100 cold-start users, found {len(cold_users)}")
    image_index = {image_id: index for index, image_id in enumerate(base.image_ids)}
    initial_items = {
        str(user_id): np.asarray([image_index[str(value)] for value in group["image_id"]], dtype=np.int64)
        for user_id, group in cold_edges.sort_values("interaction_id").groupby("user_id", sort=False)
    }
    invalid_initial = {
        user_id: len(items) for user_id, items in initial_items.items() if not 5 <= len(items) <= 10
    }
    if invalid_initial:
        raise ValueError(f"cold-start histories must contain 5-10 images: {invalid_initial}")
    previous_features, previous_records = load_previous_histories(args.previous_round)

    dataset_root = args.output_root / "dataset"
    public = dataset_root / "public"
    public.mkdir(parents=True, exist_ok=True)
    all_user_ids = base.user_ids + cold_users["user_id"].astype(str).tolist()
    pd.DataFrame({"user_id": all_user_ids}).to_parquet(public / "users.parquet", index=False)
    pd.read_parquet(args.base_dataset / "public/images.parquet").to_parquet(public / "images.parquet", index=False)
    image_link = public / "images"
    if not image_link.exists():
        image_link.symlink_to((args.base_dataset / "public/images").resolve(), target_is_directory=True)

    base_interactions = pd.read_parquet(args.base_dataset / "public/interactions.parquet")[["interaction_id", "user_id", "image_id"]]
    base_splits = pd.read_parquet(args.base_dataset / "public/splits.parquet")[["interaction_id", "split"]]
    cold_train = cold_edges[["interaction_id", "user_id", "image_id"]].copy()
    cold_train_splits = pd.DataFrame({"interaction_id": cold_train["interaction_id"], "split": "train"})
    next_id = int(max(base_interactions["interaction_id"].max(), cold_train["interaction_id"].max())) + 1
    placeholders = []
    placeholder_splits = []
    for user_id in cold_users["user_id"].astype(str):
        items = initial_items[user_id]
        for split, item in (("validation", items[0]), ("test", items[min(1, len(items) - 1)])):
            placeholders.append({"interaction_id": next_id, "user_id": user_id, "image_id": base.image_ids[int(item)]})
            placeholder_splits.append({"interaction_id": next_id, "split": split})
            next_id += 1
    interactions = pd.concat((base_interactions, cold_train, pd.DataFrame(placeholders)), ignore_index=True)
    splits = pd.concat((base_splits, cold_train_splits, pd.DataFrame(placeholder_splits)), ignore_index=True)
    interactions.to_parquet(public / "interactions.parquet", index=False)
    splits.to_parquet(public / "splits.parquet", index=False)

    base_reps = np.asarray(np.load(args.base_user_reps, mmap_mode="r"), dtype=np.float32)
    user_reps = np.empty((len(all_user_ids), 512), dtype=np.float32)
    user_reps[: base.num_users] = base_reps
    coherence = np.empty(len(all_user_ids), dtype=np.float32)
    base_coherence = args.base_user_reps.with_name(args.base_user_reps.stem + ".coherence.npy")
    coherence[: base.num_users] = np.asarray(np.load(base_coherence, mmap_mode="r"), dtype=np.float32)

    prefix = args.base_interest_cache / "adaptive_b"
    names = ("prototypes", "weights", "mask", "counts", "cv_scores", "auxiliary")
    suffix = {"prototypes": "float32", "weights": "float32", "mask": "bool", "counts": "int16", "cv_scores": "float32", "auxiliary": "float32"}
    base_interest = {
        name: np.asarray(np.load(prefix.with_name(prefix.name + f"_{name}.{suffix[name]}.npy"), mmap_mode="r"))
        for name in names
    }
    total_users = len(all_user_ids)
    combined = {
        "prototypes": np.zeros((total_users, 8, 512), dtype=np.float32),
        "weights": np.zeros((total_users, 8), dtype=np.float32),
        "mask": np.zeros((total_users, 8), dtype=bool),
        "counts": np.zeros(total_users, dtype=np.int16),
        "cv_scores": np.full((total_users, 2, 8), np.nan, dtype=np.float32),
        "auxiliary": np.zeros((total_users, 512), dtype=np.float32),
    }
    for name in names:
        combined[name][: base.num_users] = base_interest[name]

    history_rows = []
    balance_rows = []
    for cold_offset, user_id in enumerate(cold_users["user_id"].astype(str)):
        user_index = base.num_users + cold_offset
        items = initial_items[user_id]
        initial = catalog[items]
        generated = previous_features.get(user_id, [])
        history = normalize(np.concatenate((initial, np.stack(generated)), axis=0) if generated else initial)
        records = [
            {
                "user_id": user_id,
                "user_index": user_index,
                "source": "initial",
                "round": 0,
                "rank": rank,
                "image_id": base.image_ids[item],
                "path": str(base.image_paths[item]),
                "item_index": item,
            }
            for rank, item in enumerate(items.tolist(), 1)
        ]
        records.extend(
            {
                "user_id": user_id,
                "user_index": user_index,
                "source": "generated",
                "round": record["round"],
                "rank": rank,
                "image_id": "",
                "path": record["path"],
                "item_index": -1,
            }
            for rank, record in enumerate(previous_records.get(user_id, []), 1)
        )

        adaptive_result = _adaptive_user_interests(
            user_index,
            np.arange(len(history), dtype=np.int64),
            history,
            max_interests=8,
            folds=5,
            minimum_support=4,
            minimum_support_fraction=0.05,
            merge_cosine=0.90,
            retained_gain=0.75,
            seed=args.seed,
        )
        _, prototypes, weights, mask, cv_scores, count = adaptive_result
        archive_size = len(history)
        selected_indices = np.arange(archive_size, dtype=np.int64)
        assignments = np.zeros(archive_size, dtype=np.int64)
        before_counts = np.asarray([archive_size], dtype=np.int64)
        cap = archive_size
        if args.balance_history_max_ratio > 0.0 and count > 1:
            assignments = np.argmax(history @ prototypes[:count].T, axis=1)
            before_counts = np.bincount(assignments, minlength=count)
            smallest = int(before_counts.min())
            cap = max(smallest, int(np.floor(smallest * args.balance_history_max_ratio + 1e-8)))
            rng = np.random.default_rng(
                args.balance_seed + user_index * 1_000_003 + len(args.previous_round) * 10_007
            )
            selected_parts = []
            for theme_index in range(count):
                candidates = np.flatnonzero(assignments == theme_index)
                if len(candidates) > cap:
                    candidates = rng.choice(candidates, size=cap, replace=False)
                selected_parts.append(candidates)
            selected_indices = np.sort(np.concatenate(selected_parts)).astype(np.int64)
            if len(selected_indices) < archive_size:
                history = history[selected_indices]
                records = [records[index] for index in selected_indices.tolist()]
                adaptive_result = _adaptive_user_interests(
                    user_index,
                    np.arange(len(history), dtype=np.int64),
                    history,
                    max_interests=8,
                    folds=5,
                    minimum_support=4,
                    minimum_support_fraction=0.05,
                    merge_cosine=0.90,
                    retained_gain=0.75,
                    seed=args.seed,
                )
                _, prototypes, weights, mask, cv_scores, count = adaptive_result

        selected_assignment_counts = np.bincount(
            assignments[selected_indices], minlength=len(before_counts)
        )
        for theme_index, (before, after) in enumerate(
            zip(before_counts.tolist(), selected_assignment_counts.tolist(), strict=True)
        ):
            balance_rows.append(
                {
                    "user_id": user_id,
                    "user_index": user_index,
                    "previous_rounds": len(args.previous_round),
                    "archive_size": archive_size,
                    "selected_history_size": len(history),
                    "theme_index": theme_index,
                    "theme_count_before": before,
                    "theme_count_after": after,
                    "theme_cap": cap,
                    "max_ratio": args.balance_history_max_ratio,
                }
            )

        raw_mean = history.mean(axis=0)
        coherence[user_index] = np.linalg.norm(raw_mean)
        user_reps[user_index] = normalize(raw_mean[None])[0]
        combined["prototypes"][user_index] = prototypes
        combined["weights"][user_index] = weights
        combined["mask"][user_index] = mask
        combined["counts"][user_index] = count
        combined["cv_scores"][user_index] = cv_scores
        combined["auxiliary"][user_index] = user_reps[user_index]
        for archive_index, (record, selected_index) in enumerate(
            zip(records, selected_indices.tolist(), strict=True), 1
        ):
            record["input_rank"] = archive_index
            record["archive_index"] = selected_index
            record["balance_theme"] = int(assignments[selected_index])
            history_rows.append(record)

    user_cache = args.output_root / "user_reps.float32.npy"
    atomic_npy(user_cache, user_reps)
    atomic_npy(user_cache.with_name(user_cache.stem + ".coherence.npy"), coherence)
    user_cache.with_suffix(".json").write_text(
        json.dumps(
            {
                "complete": True,
                "users": total_users,
                "dimension": 512,
                "image_id_order_sha256": id_order_sha256(base.image_ids),
                "source_split": "train",
                "aggregation": "seen-user cache plus cold-user current-history normalized mean",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    interest_dir = args.output_root / "interests"
    interest_prefix = interest_dir / "adaptive_b"
    for name in names:
        atomic_npy(interest_prefix.with_name(interest_prefix.name + f"_{name}.{suffix[name]}.npy"), combined[name])
    metadata = {
        "complete": True,
        "variant": "adaptive_b",
        "users": total_users,
        "max_interests": 8,
        "folds": 5,
        "minimum_support": 4,
        "minimum_support_fraction": 0.05,
        "merge_cosine": 0.9,
        "retained_gain": 0.75,
        "seed": args.seed,
        "image_id_order_sha256": id_order_sha256(base.image_ids),
        "source_split": "train",
        "public_only": True,
        "selection_rule": "smallest K retaining the configured fraction of attainable heldout-cosine gain",
        "pruning_rule": "remove unsupported/similar center and refine remaining centers (v2)",
    }
    interest_prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    history_frame = pd.DataFrame(history_rows)
    balance_frame = pd.DataFrame(balance_rows)
    history_frame.to_parquet(args.output_root / "cold_history.parquet", index=False)
    balance_frame.to_parquet(args.output_root / "history_balance.parquet", index=False)
    cohort = {
        "selection": "100 held-out CIGR cold-start users in canonical user_id order",
        "user_indices": list(range(base.num_users, total_users)),
        "user_ids": cold_users["user_id"].astype(str).tolist(),
    }
    (args.output_root / "cold_users.json").write_text(json.dumps(cohort, indent=2) + "\n", encoding="utf-8")
    summary = {
        "users": len(cold_users),
        "base_users": base.num_users,
        "history_images_per_user_min": int(history_frame.groupby("user_id").size().min()),
        "history_images_per_user_max": int(history_frame.groupby("user_id").size().max()),
        "archive_images_per_user_min": int(balance_frame.groupby("user_id")["archive_size"].first().min()),
        "archive_images_per_user_max": int(balance_frame.groupby("user_id")["archive_size"].first().max()),
        "history_balance_max_ratio": args.balance_history_max_ratio,
        "history_images_removed": int(
            balance_frame.groupby("user_id")[["archive_size", "selected_history_size"]]
            .first()
            .eval("archive_size - selected_history_size")
            .sum()
        ),
        "users_trimmed": int(
            (
                balance_frame.groupby("user_id")["archive_size"].first()
                > balance_frame.groupby("user_id")["selected_history_size"].first()
            ).sum()
        ),
        "previous_rounds": len(args.previous_round),
        "cold_interest_count_distribution": {
            str(k): int((combined["counts"][base.num_users:] == k).sum()) for k in range(1, 9)
        },
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
