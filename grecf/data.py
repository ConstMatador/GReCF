from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PublicDataset:
    root: Path
    user_ids: list[str]
    image_ids: list[str]
    image_paths: list[Path]
    train: list[np.ndarray]
    validation: list[np.ndarray]
    test: list[np.ndarray]

    @property
    def num_users(self) -> int:
        return len(self.user_ids)

    @property
    def num_items(self) -> int:
        return len(self.image_ids)


def _group(edges: pd.DataFrame, split: str, num_users: int) -> list[np.ndarray]:
    result = [np.empty(0, dtype=np.int64) for _ in range(num_users)]
    rows = edges.loc[edges["split"] == split, ["user_index", "item_index"]]
    for user, values in rows.groupby("user_index", sort=False)["item_index"]:
        result[int(user)] = values.to_numpy(dtype=np.int64, copy=True)
    return result


def load_public_dataset(root: str | Path) -> PublicDataset:
    root = Path(root).expanduser().resolve()
    public = root / "public"
    users = pd.read_parquet(public / "users.parquet").sort_values("user_id").reset_index(drop=True)
    images = pd.read_parquet(public / "images.parquet").sort_values("image_id").reset_index(drop=True)
    interactions = pd.read_parquet(public / "interactions.parquet")
    splits = pd.read_parquet(public / "splits.parquet")
    edges = interactions.merge(splits, on="interaction_id", how="inner", validate="one_to_one")
    if len(edges) != len(interactions):
        raise ValueError("interactions and splits are not one-to-one")

    user_ids = users["user_id"].astype(str).tolist()
    image_ids = images["image_id"].astype(str).tolist()
    user_map = {value: index for index, value in enumerate(user_ids)}
    item_map = {value: index for index, value in enumerate(image_ids)}
    edges["user_index"] = edges["user_id"].map(user_map).astype(np.int64)
    edges["item_index"] = edges["image_id"].map(item_map).astype(np.int64)
    train = _group(edges, "train", len(user_ids))
    validation = _group(edges, "validation", len(user_ids))
    test = _group(edges, "test", len(user_ids))
    if any(len(items) == 0 for items in train + validation + test):
        raise ValueError("every user must have train, validation, and test interactions")
    paths = [public / str(value) for value in images["image_path"].tolist()]
    return PublicDataset(root, user_ids, image_ids, paths, train, validation, test)


def id_order_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_cache(path: str | Path, rows: int, order_sha256: str, shape_tail: tuple[int, ...]) -> np.ndarray:
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("complete") is not True:
        raise RuntimeError(f"cache is incomplete: {path}")
    if int(metadata["rows"]) != rows or metadata["image_id_order_sha256"] != order_sha256:
        raise ValueError(f"cache does not match the public dataset: {path}")
    array = np.load(path, mmap_mode="r")
    if array.shape != (rows, *shape_tail):
        raise ValueError(f"unexpected cache shape {array.shape}, expected {(rows, *shape_tail)}")
    return array


def sample_history(
    items: np.ndarray,
    max_history: int,
    rng: np.random.Generator,
    exclude: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pool = items if exclude is None else items[items != int(exclude)]
    if len(pool) == 0:
        raise ValueError("history is empty after target exclusion")
    size = min(max_history, len(pool))
    chosen = rng.choice(pool, size=size, replace=False)
    indices = np.zeros(max_history, dtype=np.int64)
    mask = np.zeros(max_history, dtype=bool)
    indices[:size] = chosen
    mask[:size] = True
    return indices, mask


def sample_batch(
    dataset: PublicDataset,
    rng: np.random.Generator,
    batch_users: int,
    max_history: int,
) -> dict[str, np.ndarray]:
    users = rng.choice(dataset.num_users, size=batch_users, replace=False).astype(np.int64)
    targets = np.asarray([rng.choice(dataset.train[int(user)]) for user in users], dtype=np.int64)
    wrong_users = np.empty(batch_users, dtype=np.int64)
    for row, (user, target) in enumerate(zip(users.tolist(), targets.tolist())):
        while True:
            candidate = int(rng.integers(dataset.num_users))
            if candidate != user and target not in dataset.train[candidate]:
                wrong_users[row] = candidate
                break

    history_indices, history_masks = [], []
    wrong_indices, wrong_masks = [], []
    for user, target, wrong_user in zip(users.tolist(), targets.tolist(), wrong_users.tolist()):
        idx, mask = sample_history(dataset.train[user], max_history, rng, exclude=target)
        wrong_idx, wrong_mask = sample_history(dataset.train[wrong_user], max_history, rng)
        history_indices.append(idx)
        history_masks.append(mask)
        wrong_indices.append(wrong_idx)
        wrong_masks.append(wrong_mask)
    return {
        "users": users,
        "targets": targets,
        "wrong_users": wrong_users,
        "history_indices": np.stack(history_indices),
        "history_masks": np.stack(history_masks),
        "wrong_history_indices": np.stack(wrong_indices),
        "wrong_history_masks": np.stack(wrong_masks),
    }

