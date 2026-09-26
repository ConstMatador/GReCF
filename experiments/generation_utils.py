"""Shared utilities for generation and diagnostic metadata."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def top_mean(feature: np.ndarray, features: np.ndarray, indices: np.ndarray, k: int = 5) -> float:
    if len(indices) == 0:
        return float("nan")
    scores = np.asarray(features[indices], dtype=np.float32) @ feature
    count = min(k, len(scores))
    return float(np.partition(scores, len(scores) - count)[-count:].mean())


def make_popularity_matched(
    target_items: np.ndarray,
    item_counts: np.ndarray,
    excluded: set[int],
    rng: np.random.Generator,
) -> np.ndarray:
    quantiles = np.quantile(item_counts, np.linspace(0.0, 1.0, 11))
    bins = np.clip(np.digitize(item_counts, quantiles[1:-1], right=True), 0, 9)
    selected: list[int] = []
    for value in target_items.tolist():
        candidates = np.flatnonzero(bins == bins[int(value)])
        for _ in range(20):
            item = int(rng.choice(candidates))
            if item not in excluded:
                selected.append(item)
                break
    return np.asarray(selected, dtype=np.int64)


def make_contact_sheet(
    path: Path,
    dataset,
    user: int,
    history_items: np.ndarray,
    heldout_items: np.ndarray,
    generated: dict[str, list[Path]],
) -> None:
    thumb = 180
    label_width = 130
    columns = max(4, max(len(paths) for paths in generated.values()))
    rows: list[tuple[str, list[Path]]] = [
        ("train likes", [dataset.image_paths[int(item)] for item in history_items[:columns]]),
        ("heldout likes", [dataset.image_paths[int(item)] for item in heldout_items[:columns]]),
        *generated.items(),
    ]
    sheet = Image.new("RGB", (label_width + columns * thumb, len(rows) * thumb), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (label, paths) in enumerate(rows):
        draw.text((8, row * thumb + 8), label, fill="black")
        for column, image_path in enumerate(paths[:columns]):
            with Image.open(image_path) as source:
                image = ImageOps.fit(source.convert("RGB"), (thumb, thumb), Image.Resampling.LANCZOS)
            sheet.paste(image, (label_width + column * thumb, row * thumb))
    draw.text((8, len(rows) * thumb - 20), dataset.user_ids[user], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def build_item_users(dataset) -> list[list[int]]:
    users: list[list[int]] = [[] for _ in range(dataset.num_items)]
    for user, items in enumerate(dataset.train):
        for item in items.tolist():
            users[int(item)].append(user)
    return users


def nearest_user_knn(dataset, item_users: list[list[int]], user: int) -> tuple[int, int, float]:
    overlaps: dict[int, int] = {}
    own = dataset.train[user]
    for item in own.tolist():
        for candidate in item_users[int(item)]:
            if candidate != user:
                overlaps[candidate] = overlaps.get(candidate, 0) + 1
    if not overlaps:
        raise RuntimeError(f"user {user} has no positive-overlap UserKNN neighbor")
    neighbor, overlap = max(
        overlaps.items(),
        key=lambda pair: (pair[1] / (len(own) + len(dataset.train[pair[0]]) - pair[1]), pair[1]),
    )
    jaccard = overlap / (len(own) + len(dataset.train[neighbor]) - overlap)
    return int(neighbor), int(overlap), float(jaccard)
