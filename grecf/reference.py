from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from grecf.data import PublicDataset, id_order_sha256


def cluster_ids(image_ids: list[str]) -> np.ndarray:
    """Return stable integer IDs for the public Cxxxxxx image clusters."""
    names = [image_id.split("_", 1)[0] for image_id in image_ids]
    _, inverse = np.unique(np.asarray(names), return_inverse=True)
    return inverse.astype(np.int32, copy=False)


def build_cluster_balanced_user_representations(
    dataset: PublicDataset,
    image_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average images within a public cluster, then average clusters equally."""
    item_clusters = cluster_ids(dataset.image_ids)
    representations = np.empty((dataset.num_users, image_features.shape[1]), dtype=np.float32)
    raw_coherence = np.empty(dataset.num_users, dtype=np.float32)
    for user, items in enumerate(dataset.train):
        features = np.asarray(image_features[items], dtype=np.float32)
        raw_mean = features.mean(axis=0)
        raw_coherence[user] = np.linalg.norm(raw_mean)
        groups = item_clusters[items]
        unique_groups, inverse = np.unique(groups, return_inverse=True)
        sums = np.zeros((len(unique_groups), features.shape[1]), dtype=np.float32)
        np.add.at(sums, inverse, features)
        counts = np.bincount(inverse, minlength=len(unique_groups)).astype(np.float32)
        centroids = sums / counts[:, None]
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8)
        profile = centroids.mean(axis=0)
        representations[user] = profile / max(float(np.linalg.norm(profile)), 1e-8)
    return representations, raw_coherence


def build_item_average_user_representations(
    dataset: PublicDataset,
    image_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average every train-history item equally, then L2-normalize per user."""
    representations = np.empty((dataset.num_users, image_features.shape[1]), dtype=np.float32)
    raw_coherence = np.empty(dataset.num_users, dtype=np.float32)
    for user, items in enumerate(dataset.train):
        raw_mean = np.asarray(image_features[items], dtype=np.float32).mean(axis=0)
        norm = max(float(np.linalg.norm(raw_mean)), 1e-8)
        raw_coherence[user] = norm
        representations[user] = raw_mean / norm
    return representations, raw_coherence


def load_or_build_user_representation_cache(
    path: str | Path,
    dataset: PublicDataset,
    image_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    metadata_path = path.with_suffix(".json")
    order_sha = id_order_sha256(dataset.image_ids)
    if path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("complete") is True
            and metadata.get("image_id_order_sha256") == order_sha
            and int(metadata.get("users", -1)) == dataset.num_users
        ):
            values = np.load(path, mmap_mode="r")
            coherence = np.load(path.with_name(path.stem + ".coherence.npy"), mmap_mode="r")
            if values.shape == (dataset.num_users, 512) and coherence.shape == (dataset.num_users,):
                return values, coherence

    values, coherence = build_cluster_balanced_user_representations(dataset, image_features)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, path)
    coherence_path = path.with_name(path.stem + ".coherence.npy")
    temporary_coherence = coherence_path.with_suffix(coherence_path.suffix + ".tmp")
    with temporary_coherence.open("wb") as handle:
        np.save(handle, coherence)
    os.replace(temporary_coherence, coherence_path)
    metadata = {
        "complete": True,
        "users": dataset.num_users,
        "dimension": int(values.shape[1]),
        "dtype": str(values.dtype),
        "image_id_order_sha256": order_sha,
        "source_split": "train",
        "aggregation": "normalized image mean within public Cxxxxxx cluster, then equal normalized cluster mean",
        "public_only": True,
    }
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary_metadata, metadata_path)
    return np.load(path, mmap_mode="r"), np.load(coherence_path, mmap_mode="r")


def load_or_build_item_average_user_representation_cache(
    path: str | Path,
    dataset: PublicDataset,
    image_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    metadata_path = path.with_suffix(".json")
    order_sha = id_order_sha256(dataset.image_ids)
    if path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("complete") is True
            and metadata.get("representation") == "normalized_train_item_average"
            and metadata.get("image_id_order_sha256") == order_sha
            and int(metadata.get("users", -1)) == dataset.num_users
        ):
            values = np.load(path, mmap_mode="r")
            coherence = np.load(path.with_name(path.stem + ".coherence.npy"), mmap_mode="r")
            if values.shape == (dataset.num_users, 512) and coherence.shape == (dataset.num_users,):
                return values, coherence

    values, coherence = build_item_average_user_representations(dataset, image_features)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, path)
    coherence_path = path.with_name(path.stem + ".coherence.npy")
    temporary_coherence = coherence_path.with_suffix(coherence_path.suffix + ".tmp")
    with temporary_coherence.open("wb") as handle:
        np.save(handle, coherence)
    os.replace(temporary_coherence, coherence_path)
    metadata = {
        "complete": True,
        "users": dataset.num_users,
        "dimension": int(values.shape[1]),
        "representation": "normalized_train_item_average",
        "source_split": "train",
        "image_id_order_sha256": order_sha,
    }
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_metadata, metadata_path)
    return np.load(path, mmap_mode="r"), np.load(coherence_path, mmap_mode="r")
