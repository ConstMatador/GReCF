from __future__ import annotations

import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from grecf.data import PublicDataset, id_order_sha256
from grecf.reference import cluster_ids


def _normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def spherical_kmeans(values: np.ndarray, clusters: int = 4, iterations: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic spherical K-means used independently inside each user."""
    values = _normalize(np.asarray(values, dtype=np.float32))
    clusters = min(clusters, len(values))
    mean = _normalize(values.mean(axis=0, keepdims=True))[0]
    first = int(np.argmax(values @ mean))
    selected = [first]
    best = values @ values[first]
    for _ in range(1, clusters):
        candidate = int(np.argmin(best))
        selected.append(candidate)
        best = np.maximum(best, values @ values[candidate])
    centers = values[selected].copy()
    assignments = np.zeros(len(values), dtype=np.int64)
    for _ in range(iterations):
        assignments = np.argmax(values @ centers.T, axis=1)
        for index in range(clusters):
            members = values[assignments == index]
            if len(members):
                centers[index] = _normalize(members.mean(axis=0, keepdims=True))[0]
    counts = np.bincount(assignments, minlength=clusters).astype(np.float32)
    order = np.argsort(-counts, kind="stable")
    centers, counts = centers[order], counts[order]
    if clusters < 4:
        centers = np.concatenate((centers, np.repeat(centers[-1:], 4 - clusters, axis=0)))
        counts = np.concatenate((counts, np.zeros(4 - clusters, dtype=np.float32)))
    return centers.astype(np.float32), counts.astype(np.float32)


def _spherical_kmeans_unpadded(
    values: np.ndarray, clusters: int, iterations: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    values = _normalize(np.asarray(values, dtype=np.float32))
    clusters = min(int(clusters), len(values))
    mean = _normalize(values.mean(axis=0, keepdims=True))[0]
    selected = [int(np.argmax(values @ mean))]
    nearest = values @ values[selected[0]]
    for _ in range(1, clusters):
        selected.append(int(np.argmin(nearest)))
        nearest = np.maximum(nearest, values @ values[selected[-1]])
    centers = values[selected].copy()
    assignments = np.zeros(len(values), dtype=np.int64)
    for _ in range(iterations):
        assignments = np.argmax(values @ centers.T, axis=1)
        for index in range(clusters):
            members = values[assignments == index]
            if len(members):
                centers[index] = _normalize(members.mean(axis=0, keepdims=True))[0]
    assignments = np.argmax(values @ centers.T, axis=1)
    counts = np.bincount(assignments, minlength=clusters).astype(np.float32)
    order = np.argsort(-counts, kind="stable")
    return centers[order].astype(np.float32), counts[order].astype(np.float32)


def _refine_spherical_centers(
    values: np.ndarray, centers: np.ndarray, iterations: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    values = _normalize(np.asarray(values, dtype=np.float32))
    centers = _normalize(np.asarray(centers, dtype=np.float32).copy())
    for _ in range(iterations):
        assignments = np.argmax(values @ centers.T, axis=1)
        for index in range(len(centers)):
            members = values[assignments == index]
            if len(members):
                centers[index] = _normalize(members.mean(axis=0, keepdims=True))[0]
    assignments = np.argmax(values @ centers.T, axis=1)
    counts = np.bincount(assignments, minlength=len(centers)).astype(np.float32)
    order = np.argsort(-counts, kind="stable")
    return centers[order].astype(np.float32), counts[order].astype(np.float32)


def _adaptive_user_interests(
    user: int,
    items: np.ndarray,
    image_features: np.ndarray,
    max_interests: int,
    folds: int,
    minimum_support: int,
    minimum_support_fraction: float,
    merge_cosine: float,
    retained_gain: float,
    seed: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    values = _normalize(np.asarray(image_features[items], dtype=np.float32))
    max_k = min(max_interests, max(1, len(values) // minimum_support))
    effective_folds = min(int(folds), len(values))
    rng = np.random.default_rng(seed + user * 1_000_003)
    fold_ids = np.empty(len(values), dtype=np.int64)
    fold_ids[rng.permutation(len(values))] = np.arange(len(values)) % max(effective_folds, 1)
    cv_means = np.full(max_interests, np.nan, dtype=np.float32)
    cv_errors = np.full(max_interests, np.nan, dtype=np.float32)
    if len(values) == 1:
        cv_means[0] = 1.0
        cv_errors[0] = 0.0
        selected_k = 1
    else:
        for clusters in range(1, max_k + 1):
            heldout_scores: list[np.ndarray] = []
            for fold in range(effective_folds):
                train = values[fold_ids != fold]
                heldout = values[fold_ids == fold]
                centers, _ = _spherical_kmeans_unpadded(train, clusters)
                heldout_scores.append((heldout @ centers.T).max(axis=1))
            scores = np.concatenate(heldout_scores).astype(np.float64)
            cv_means[clusters - 1] = float(scores.mean())
            cv_errors[clusters - 1] = float(
                scores.std(ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0.0
            )

        best = int(np.nanargmax(cv_means[:max_k]))
        baseline = float(cv_means[0])
        predictive_gain_threshold = baseline + retained_gain * (float(cv_means[best]) - baseline)
        selected_k = next(
            clusters
            for clusters in range(1, max_k + 1)
            if cv_means[clusters - 1] >= predictive_gain_threshold
        )
    required = max(minimum_support, int(np.ceil(minimum_support_fraction * len(values))))
    centers, counts = _spherical_kmeans_unpadded(values, selected_k)
    while selected_k > 1:
        pairwise = centers @ centers.T
        np.fill_diagonal(pairwise, -np.inf)
        if bool((counts < required).any()):
            remove = int(np.argmin(counts))
            centers = np.delete(centers, remove, axis=0)
        elif float(pairwise.max()) >= merge_cosine:
            left, right = np.unravel_index(int(np.argmax(pairwise)), pairwise.shape)
            centers[left] = _normalize(
                (counts[left] * centers[left] + counts[right] * centers[right])[None]
            )[0]
            centers = np.delete(centers, right, axis=0)
        else:
            break
        selected_k -= 1
        centers, counts = _refine_spherical_centers(values, centers)

    prototypes = np.zeros((max_interests, values.shape[1]), dtype=np.float32)
    weights = np.zeros(max_interests, dtype=np.float32)
    mask = np.zeros(max_interests, dtype=bool)
    prototypes[:selected_k] = centers
    weights[:selected_k] = counts / max(float(counts.sum()), 1.0)
    mask[:selected_k] = True
    return user, prototypes, weights, mask, np.stack((cv_means, cv_errors)), selected_k


_ADAPTIVE_WORKER_CONTEXT: tuple | None = None


def _adaptive_interest_worker(user: int):
    if _ADAPTIVE_WORKER_CONTEXT is None:
        raise RuntimeError("adaptive interest worker context was not initialized")
    train, image_features, parameters = _ADAPTIVE_WORKER_CONTEXT
    return _adaptive_user_interests(user, train[user], image_features, **parameters)


def build_adaptive_interest_cache(
    dataset: PublicDataset,
    image_features: np.ndarray,
    max_interests: int = 8,
    folds: int = 5,
    minimum_support: int = 5,
    minimum_support_fraction: float = 0.05,
    merge_cosine: float = 0.90,
    retained_gain: float = 0.75,
    seed: int = 20260726,
    workers: int | None = None,
    source_split: str = "train",
) -> dict[str, np.ndarray]:
    """Build Adaptive-B interests from one public split with per-user cross-validated K."""
    if source_split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported adaptive-interest source split: {source_split}")
    source_interactions = getattr(dataset, source_split)
    dimension = image_features.shape[1]
    prototypes = np.zeros((dataset.num_users, max_interests, dimension), dtype=np.float32)
    weights = np.zeros((dataset.num_users, max_interests), dtype=np.float32)
    mask = np.zeros((dataset.num_users, max_interests), dtype=bool)
    cv_scores = np.full((dataset.num_users, 2, max_interests), np.nan, dtype=np.float32)
    counts = np.zeros(dataset.num_users, dtype=np.int16)
    auxiliary = np.empty((dataset.num_users, dimension), dtype=np.float32)
    _, item_clusters = _catalog_cluster_centroids(dataset.image_ids, image_features)
    for user, items in enumerate(source_interactions):
        features = np.asarray(image_features[items], dtype=np.float32)
        group_ids = item_clusters[items]
        unique, inverse = np.unique(group_ids, return_inverse=True)
        group_sums = np.zeros((len(unique), dimension), dtype=np.float32)
        np.add.at(group_sums, inverse, features)
        group_counts = np.bincount(inverse, minlength=len(unique)).astype(np.float32)
        auxiliary[user] = _normalize(_normalize(group_sums / group_counts[:, None]).mean(axis=0, keepdims=True))[0]

    worker_count = workers or min(32, os.cpu_count() or 1)
    global _ADAPTIVE_WORKER_CONTEXT
    _ADAPTIVE_WORKER_CONTEXT = (
        source_interactions,
        image_features,
        {
            "max_interests": max_interests, "folds": folds,
            "minimum_support": minimum_support,
            "minimum_support_fraction": minimum_support_fraction,
            "merge_cosine": merge_cosine, "retained_gain": retained_gain, "seed": seed,
        },
    )
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context("fork")) as executor:
        for completed, result in enumerate(
            executor.map(_adaptive_interest_worker, range(dataset.num_users), chunksize=8), start=1
        ):
            user, user_prototypes, user_weights, user_mask, user_cv, selected_k = result
            prototypes[user] = user_prototypes
            weights[user] = user_weights
            mask[user] = user_mask
            cv_scores[user] = user_cv
            counts[user] = selected_k
            if completed % 500 == 0:
                print(f"adaptive-interest cache: {completed}/{dataset.num_users} users", flush=True)
    _ADAPTIVE_WORKER_CONTEXT = None
    return {
        "prototypes": prototypes,
        "weights": weights,
        "mask": mask,
        "counts": counts,
        "cv_scores": cv_scores,
        "auxiliary": auxiliary,
    }


def _catalog_cluster_centroids(image_ids: list[str], image_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = cluster_ids(image_ids)
    count = int(ids.max()) + 1
    sums = np.zeros((count, image_features.shape[1]), dtype=np.float32)
    np.add.at(sums, ids, np.asarray(image_features, dtype=np.float32))
    sizes = np.bincount(ids, minlength=count).astype(np.float32)
    return _normalize(sums / sizes[:, None]), ids


def build_interest_cache(
    dataset: PublicDataset,
    image_features: np.ndarray,
    variant: str,
) -> dict[str, np.ndarray]:
    if variant not in {"b", "d"}:
        raise ValueError(f"interest cache is only defined for b/d, got {variant}")
    prototypes = np.empty((dataset.num_users, 4, image_features.shape[1]), dtype=np.float32)
    weights = np.empty((dataset.num_users, 4), dtype=np.float32)
    auxiliary = np.empty((dataset.num_users, image_features.shape[1]), dtype=np.float32)
    catalog_centers, item_clusters = _catalog_cluster_centroids(dataset.image_ids, image_features)
    for user, items in enumerate(dataset.train):
        features = np.asarray(image_features[items], dtype=np.float32)
        if variant == "b":
            source = features
            group_ids = item_clusters[items]
            unique, inverse = np.unique(group_ids, return_inverse=True)
            group_sums = np.zeros((len(unique), features.shape[1]), dtype=np.float32)
            np.add.at(group_sums, inverse, features)
            group_counts = np.bincount(inverse, minlength=len(unique)).astype(np.float32)
            balanced = _normalize(group_sums / group_counts[:, None])
            auxiliary[user] = _normalize(balanced.mean(axis=0, keepdims=True))[0]
        else:
            unique = np.unique(item_clusters[items])
            source = catalog_centers[unique]
            residual = features - catalog_centers[item_clusters[items]]
            auxiliary[user] = _normalize(residual.mean(axis=0, keepdims=True))[0]
        prototypes[user], weights[user] = spherical_kmeans(source)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0)
    return {"prototypes": prototypes, "weights": weights, "auxiliary": auxiliary}


def load_or_build_interest_cache(
    cache_dir: str | Path,
    dataset: PublicDataset,
    image_features: np.ndarray,
    variant: str,
) -> dict[str, np.ndarray]:
    cache_dir = Path(cache_dir)
    prefix = cache_dir / f"v2_{variant}"
    metadata_path = prefix.with_suffix(".json")
    paths = {name: prefix.with_name(prefix.name + f"_{name}.float32.npy") for name in ("prototypes", "weights", "auxiliary")}
    order_sha = id_order_sha256(dataset.image_ids)
    valid = metadata_path.is_file() and all(path.is_file() for path in paths.values())
    if valid:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        valid = metadata.get("complete") is True and metadata.get("image_id_order_sha256") == order_sha
    if not valid:
        values = build_interest_cache(dataset, image_features, variant)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for name, path in paths.items():
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("wb") as handle:
                np.save(handle, values[name])
            os.replace(temporary, path)
        payload = {
            "complete": True,
            "variant": variant,
            "users": dataset.num_users,
            "interests": 4,
            "image_id_order_sha256": order_sha,
            "source_split": "train",
            "public_only": True,
        }
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, metadata_path)
    return {name: np.load(path, mmap_mode="r") for name, path in paths.items()}


def load_or_build_adaptive_interest_cache(
    cache_dir: str | Path,
    dataset: PublicDataset,
    image_features: np.ndarray,
    max_interests: int = 8,
    folds: int = 5,
    minimum_support: int = 5,
    minimum_support_fraction: float = 0.05,
    merge_cosine: float = 0.90,
    retained_gain: float = 0.75,
    seed: int = 20260726,
    source_split: str = "train",
) -> dict[str, np.ndarray]:
    cache_dir = Path(cache_dir)
    prefix = cache_dir / "adaptive_b"
    metadata_path = prefix.with_suffix(".json")
    names = ("prototypes", "weights", "mask", "counts", "cv_scores", "auxiliary")
    suffixes = {
        "prototypes": "float32", "weights": "float32", "mask": "bool",
        "counts": "int16", "cv_scores": "float32", "auxiliary": "float32",
    }
    paths = {
        name: prefix.with_name(prefix.name + f"_{name}.{suffixes[name]}.npy")
        for name in names
    }
    order_sha = id_order_sha256(dataset.image_ids)
    expected = {
        "complete": True,
        "variant": "adaptive_b",
        "users": dataset.num_users,
        "max_interests": max_interests,
        "folds": folds,
        "minimum_support": minimum_support,
        "minimum_support_fraction": minimum_support_fraction,
        "merge_cosine": merge_cosine,
        "retained_gain": retained_gain,
        "seed": seed,
        "image_id_order_sha256": order_sha,
        "source_split": source_split,
        "public_only": True,
        "selection_rule": "smallest K retaining the configured fraction of attainable heldout-cosine gain",
        "pruning_rule": "remove unsupported/similar center and refine remaining centers (v2)",
    }
    valid = metadata_path.is_file() and all(path.is_file() for path in paths.values())
    if valid:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        valid = all(metadata.get(key) == value for key, value in expected.items())
    if not valid:
        values = build_adaptive_interest_cache(
            dataset=dataset,
            image_features=image_features,
            max_interests=max_interests,
            folds=folds,
            minimum_support=minimum_support,
            minimum_support_fraction=minimum_support_fraction,
            merge_cosine=merge_cosine, retained_gain=retained_gain,
            seed=seed,
            workers=int(os.environ.get("GRECF_INTEREST_WORKERS", "0")) or None,
            source_split=source_split,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        for name, path in paths.items():
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("wb") as handle:
                np.save(handle, values[name])
            os.replace(temporary, path)
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(expected, indent=2), encoding="utf-8")
        os.replace(temporary, metadata_path)
    return {name: np.load(path, mmap_mode="r") for name, path in paths.items()}


def _unique_rows(user_ids: torch.Tensor) -> torch.Tensor:
    keep: list[int] = []
    seen: set[int] = set()
    for index, user in enumerate(user_ids.detach().cpu().tolist()):
        if int(user) not in seen:
            seen.add(int(user))
            keep.append(index)
    return torch.as_tensor(keep, device=user_ids.device, dtype=torch.long)


def _contrastive(predicted: torch.Tensor, targets: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    logits = F.normalize(predicted.float(), dim=-1) @ F.normalize(targets.float(), dim=-1).T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return F.cross_entropy(logits, labels), (logits.argmax(dim=-1) == labels).float().mean()


class SplitInterestAdapter(nn.Module):
    """Adapter with separately allocated global and routed-interest tokens."""

    def __init__(
        self,
        base_context: torch.Tensor,
        auxiliary_center: torch.Tensor,
        prototype_center: torch.Tensor,
        auxiliary_tokens: int = 16,
        interest_tokens: int = 48,
        hidden_dim: int = 1024,
        residual_scale: float = 2.0,
        residual_start: int = 0,
    ) -> None:
        super().__init__()
        if auxiliary_tokens < 0 or interest_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if auxiliary_tokens + interest_tokens == 0:
            raise ValueError("at least one conditioning token block is required")
        if residual_start < 0:
            raise ValueError("residual_start must be non-negative")
        if residual_start + auxiliary_tokens + interest_tokens > base_context.shape[1]:
            raise ValueError("adapter token allocation exceeds base context length")
        self.auxiliary_tokens = auxiliary_tokens
        self.interest_tokens = interest_tokens
        self.residual_start = residual_start
        self.residual_scale = residual_scale
        self.register_buffer("base_context", base_context.detach().float().clone())
        self.register_buffer("auxiliary_center", auxiliary_center.detach().float().clone())
        self.register_buffer("prototype_center", prototype_center.detach().float().clone())
        self.auxiliary_mlp = self._mlp(hidden_dim, auxiliary_tokens) if auxiliary_tokens else None
        self.interest_mlp = self._mlp(hidden_dim, interest_tokens) if interest_tokens else None
        self.auxiliary_head = nn.Linear(768, 512) if auxiliary_tokens else None
        self.interest_head = nn.Linear(768, 512) if interest_tokens else None

    @staticmethod
    def _mlp(hidden_dim: int, tokens: int) -> nn.Sequential:
        layers = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, tokens * 768),
        )
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        return layers

    def _center(self, values: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        return F.normalize(values.float() - center, dim=-1)

    def forward(
        self, auxiliary: torch.Tensor, interest: torch.Tensor, multiplier: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        auxiliary_input = self._center(auxiliary, self.auxiliary_center)
        interest_input = self._center(interest, self.prototype_center)
        residual_blocks = []
        if self.auxiliary_mlp is not None:
            residual_blocks.append(self.auxiliary_mlp(auxiliary_input).reshape(-1, self.auxiliary_tokens, 768))
        if self.interest_mlp is not None:
            residual_blocks.append(self.interest_mlp(interest_input).reshape(-1, self.interest_tokens, 768))
        residual = torch.cat(residual_blocks, dim=1)
        context = self.base_context.expand(len(auxiliary), -1, -1).clone()
        start = self.residual_start
        end = start + len(residual[0])
        context[:, start:end] += self.residual_scale * multiplier * residual
        return context, residual

    def user_contrastive_loss(
        self,
        residual: torch.Tensor,
        auxiliary: torch.Tensor,
        interest: torch.Tensor,
        user_ids: torch.Tensor,
        temperature: float = 0.07,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keep = _unique_rows(user_ids)
        losses = []
        accuracies = []
        if self.auxiliary_head is not None:
            auxiliary_prediction = self.auxiliary_head(
                residual[keep, : self.auxiliary_tokens].mean(dim=1)
            )
            loss, accuracy = _contrastive(
                auxiliary_prediction, self._center(auxiliary[keep], self.auxiliary_center), temperature
            )
            losses.append(loss)
            accuracies.append(accuracy)
        if self.interest_head is not None:
            interest_prediction = self.interest_head(
                residual[keep, self.auxiliary_tokens :].mean(dim=1)
            )
            loss, accuracy = _contrastive(
                interest_prediction, self._center(interest[keep], self.prototype_center), temperature
            )
            losses.append(loss)
            accuracies.append(accuracy)
        return torch.stack(losses).mean(), torch.stack(accuracies).mean()


class FullContextWeightedSumAdapter(nn.Module):
    """Map auxiliary and routed-interest vectors to full 77-slot residuals and fuse by weighted sum.

    Unlike :class:`SplitInterestAdapter`, this adapter does not allocate disjoint token
    ranges to the two sources.  Both sources produce a complete SD text context
    residual of shape ``[B, 77, 768]``.  The final residual added to the frozen
    CLIP text context is a learnable softmax-weighted sum:

    ``context = base_context + residual_scale * multiplier * (w_aux * aux + w_int * interest)``.
    """

    def __init__(
        self,
        base_context: torch.Tensor,
        auxiliary_center: torch.Tensor,
        prototype_center: torch.Tensor,
        context_tokens: int = 77,
        hidden_dim: int = 1024,
        residual_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        if context_tokens > base_context.shape[1]:
            raise ValueError("context_tokens cannot exceed base context length")
        self.context_tokens = int(context_tokens)
        self.auxiliary_tokens = int(context_tokens)
        self.interest_tokens = int(context_tokens)
        self.residual_scale = residual_scale
        self.register_buffer("base_context", base_context.detach().float().clone())
        self.register_buffer("auxiliary_center", auxiliary_center.detach().float().clone())
        self.register_buffer("prototype_center", prototype_center.detach().float().clone())
        self.auxiliary_mlp = self._mlp(hidden_dim, self.context_tokens)
        self.interest_mlp = self._mlp(hidden_dim, self.context_tokens)
        self.fusion_logits = nn.Parameter(torch.zeros(2))
        self.auxiliary_head = nn.Linear(768, 512)
        self.interest_head = nn.Linear(768, 512)

    @staticmethod
    def _mlp(hidden_dim: int, tokens: int) -> nn.Sequential:
        layers = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, tokens * 768),
        )
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        return layers

    def _center(self, values: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        return F.normalize(values.float() - center, dim=-1)

    def fusion_weights(self) -> torch.Tensor:
        return torch.softmax(self.fusion_logits.float(), dim=0)

    def forward(
        self,
        auxiliary: torch.Tensor,
        interest: torch.Tensor,
        multiplier: float = 1.0,
        ablation: str = "full",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ablation == "full":
            auxiliary_input = self._center(auxiliary, self.auxiliary_center)
            interest_input = self._center(interest, self.prototype_center)
            auxiliary_residual = self.auxiliary_mlp(auxiliary_input).reshape(-1, self.context_tokens, 768)
            interest_residual = self.interest_mlp(interest_input).reshape(-1, self.context_tokens, 768)
            weights = self.fusion_weights().to(device=auxiliary_residual.device, dtype=auxiliary_residual.dtype)
            fused = weights[0] * auxiliary_residual + weights[1] * interest_residual
        elif ablation == "user_only":
            auxiliary_input = self._center(auxiliary, self.auxiliary_center)
            auxiliary_residual = self.auxiliary_mlp(auxiliary_input).reshape(-1, self.context_tokens, 768)
            interest_residual = torch.zeros_like(auxiliary_residual)
            fused = auxiliary_residual
        elif ablation == "interest_only":
            interest_input = self._center(interest, self.prototype_center)
            interest_residual = self.interest_mlp(interest_input).reshape(-1, self.context_tokens, 768)
            auxiliary_residual = torch.zeros_like(interest_residual)
            fused = interest_residual
        else:
            raise ValueError(f"unknown full-context ablation: {ablation}")
        context = self.base_context.expand(len(auxiliary), -1, -1).clone()
        context[:, : self.context_tokens] += self.residual_scale * multiplier * fused
        residual = torch.cat((auxiliary_residual, interest_residual), dim=1)
        return context, residual

    def user_contrastive_loss(
        self,
        residual: torch.Tensor,
        auxiliary: torch.Tensor,
        interest: torch.Tensor,
        user_ids: torch.Tensor,
        temperature: float = 0.07,
        ablation: str = "full",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keep = _unique_rows(user_ids)
        losses = []
        accuracies = []
        if ablation in ("full", "user_only"):
            auxiliary_prediction = self.auxiliary_head(residual[keep, : self.context_tokens].mean(dim=1))
            auxiliary_loss, auxiliary_accuracy = _contrastive(
                auxiliary_prediction, self._center(auxiliary[keep], self.auxiliary_center), temperature
            )
            losses.append(auxiliary_loss)
            accuracies.append(auxiliary_accuracy)
        if ablation in ("full", "interest_only"):
            interest_prediction = self.interest_head(residual[keep, self.context_tokens :].mean(dim=1))
            interest_loss, interest_accuracy = _contrastive(
                interest_prediction, self._center(interest[keep], self.prototype_center), temperature
            )
            losses.append(interest_loss)
            accuracies.append(interest_accuracy)
        if not losses:
            raise ValueError(f"unknown full-context ablation: {ablation}")
        return torch.stack(losses).mean(), torch.stack(accuracies).mean()


class PreferenceIPDeltaAdapter(nn.Module):
    """Keep base prompt text context and add user/theme preference tokens.

    The SD text condition remains the frozen CLIP text hidden states from the base
    prompt.  The auxiliary user center and the routed interest prototype are mapped
    into separate image-condition token sets.  The routed interest is encoded as a
    local direction relative to the user center, ``delta = interest - auxiliary``.
    """

    def __init__(
        self,
        base_context: torch.Tensor,
        auxiliary_center: torch.Tensor,
        prototype_center: torch.Tensor,
        user_tokens: int = 8,
        delta_tokens: int = 8,
        hidden_dim: int = 1024,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if user_tokens <= 0 or delta_tokens <= 0:
            raise ValueError("user_tokens and delta_tokens must be positive")
        self.user_tokens = int(user_tokens)
        self.delta_tokens = int(delta_tokens)
        self.residual_scale = float(residual_scale)
        self.register_buffer("base_context", base_context.detach().float().clone())
        self.register_buffer("auxiliary_center", auxiliary_center.detach().float().clone())
        self.register_buffer("prototype_center", prototype_center.detach().float().clone())
        self.user_mlp = self._mlp(512, hidden_dim, self.user_tokens)
        self.delta_mlp = self._mlp(512, hidden_dim, self.delta_tokens)
        # Initial sigmoid ~= 0.88.  The MLP output is zero-initialized, so the
        # adapter starts from the base SD behavior while still allowing strong
        # learned conditioning once tokens become non-zero.
        self.user_gate_logit = nn.Parameter(torch.tensor(2.0))
        self.delta_gate_logit = nn.Parameter(torch.tensor(2.0))
        self.user_head = nn.Linear(768, 512)
        self.delta_head = nn.Linear(768, 512)

    @staticmethod
    def _mlp(input_dim: int, hidden_dim: int, tokens: int) -> nn.Sequential:
        layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, tokens * 768),
        )
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        return layers

    def _center(self, values: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        return F.normalize(values.float() - center, dim=-1)

    def _delta(self, auxiliary: torch.Tensor, interest: torch.Tensor) -> torch.Tensor:
        del self  # kept as an instance method for symmetry with _center.
        return F.normalize(interest.float() - auxiliary.float(), dim=-1)

    def gates(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.sigmoid(self.user_gate_logit), torch.sigmoid(self.delta_gate_logit)

    def forward(
        self,
        auxiliary: torch.Tensor,
        interest: torch.Tensor,
        multiplier: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        user_input = self._center(auxiliary, self.auxiliary_center)
        delta_input = F.normalize(interest.float() - auxiliary.float(), dim=-1)
        user_condition = self.user_mlp(user_input).reshape(-1, self.user_tokens, 768)
        delta_condition = self.delta_mlp(delta_input).reshape(-1, self.delta_tokens, 768)
        context = self.base_context.expand(len(auxiliary), -1, -1).clone()
        user_gate, delta_gate = self.gates()
        scale = torch.as_tensor(float(multiplier) * self.residual_scale, device=auxiliary.device)
        residual = torch.cat((user_condition, delta_condition), dim=1)
        attention_kwargs = {
            "preference_user_tokens": user_condition,
            "preference_delta_tokens": delta_condition,
            "preference_user_scale": scale * user_gate,
            "preference_delta_scale": scale * delta_gate,
        }
        return context, residual, attention_kwargs

    def zero_attention_kwargs(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        user_gate, delta_gate = self.gates()
        scale = torch.as_tensor(self.residual_scale, device=device, dtype=dtype)
        return {
            "preference_user_tokens": torch.zeros(batch_size, self.user_tokens, 768, device=device, dtype=dtype),
            "preference_delta_tokens": torch.zeros(batch_size, self.delta_tokens, 768, device=device, dtype=dtype),
            "preference_user_scale": scale * user_gate.to(device=device, dtype=dtype),
            "preference_delta_scale": scale * delta_gate.to(device=device, dtype=dtype),
        }

    def user_contrastive_loss(
        self,
        residual: torch.Tensor,
        auxiliary: torch.Tensor,
        interest: torch.Tensor,
        user_ids: torch.Tensor,
        temperature: float = 0.07,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keep = _unique_rows(user_ids)
        user_prediction = self.user_head(residual[keep, : self.user_tokens].mean(dim=1))
        delta_prediction = self.delta_head(residual[keep, self.user_tokens :].mean(dim=1))
        user_loss, user_accuracy = _contrastive(
            user_prediction, self._center(auxiliary[keep], self.auxiliary_center), temperature
        )
        delta_target = F.normalize(interest[keep].float() - auxiliary[keep].float(), dim=-1)
        delta_loss, delta_accuracy = _contrastive(delta_prediction, delta_target, temperature)
        return (user_loss + delta_loss) / 2.0, (user_accuracy + delta_accuracy) / 2.0


class SplitInterestCFAdapter(nn.Module):
    """Split adapter with an additional LightGCN collaborative-filtering user token block."""

    def __init__(
        self,
        base_context: torch.Tensor,
        auxiliary_center: torch.Tensor,
        prototype_center: torch.Tensor,
        cf_center: torch.Tensor,
        auxiliary_tokens: int = 16,
        interest_tokens: int = 32,
        cf_tokens: int = 16,
        hidden_dim: int = 1024,
        cf_dim: int | None = None,
        residual_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if auxiliary_tokens + interest_tokens + cf_tokens > base_context.shape[1]:
            raise ValueError("adapter token allocation exceeds base context length")
        self.auxiliary_tokens = auxiliary_tokens
        self.interest_tokens = interest_tokens
        self.cf_tokens = cf_tokens
        self.residual_scale = residual_scale
        self.register_buffer("base_context", base_context.detach().float().clone())
        self.register_buffer("auxiliary_center", auxiliary_center.detach().float().clone())
        self.register_buffer("prototype_center", prototype_center.detach().float().clone())
        self.register_buffer("cf_center", cf_center.detach().float().clone())
        cf_dim = int(cf_dim or cf_center.shape[-1])
        self.auxiliary_mlp = self._mlp(512, hidden_dim, auxiliary_tokens)
        self.interest_mlp = self._mlp(512, hidden_dim, interest_tokens)
        self.cf_mlp = self._mlp(cf_dim, hidden_dim, cf_tokens)
        self.auxiliary_head = nn.Linear(768, 512)
        self.interest_head = nn.Linear(768, 512)
        self.cf_head = nn.Linear(768, cf_dim)

    @staticmethod
    def _mlp(input_dim: int, hidden_dim: int, tokens: int) -> nn.Sequential:
        layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, tokens * 768),
        )
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        return layers

    def _center(self, values: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        return F.normalize(values.float() - center, dim=-1)

    def forward(
        self,
        auxiliary: torch.Tensor,
        interest: torch.Tensor,
        cf: torch.Tensor,
        multiplier: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        auxiliary_input = self._center(auxiliary, self.auxiliary_center)
        interest_input = self._center(interest, self.prototype_center)
        cf_input = self._center(cf, self.cf_center)
        auxiliary_residual = self.auxiliary_mlp(auxiliary_input).reshape(-1, self.auxiliary_tokens, 768)
        interest_residual = self.interest_mlp(interest_input).reshape(-1, self.interest_tokens, 768)
        cf_residual = self.cf_mlp(cf_input).reshape(-1, self.cf_tokens, 768)
        residual = torch.cat((auxiliary_residual, interest_residual, cf_residual), dim=1)
        context = self.base_context.expand(len(auxiliary), -1, -1).clone()
        context[:, : len(residual[0])] += self.residual_scale * multiplier * residual
        return context, residual

    def user_contrastive_loss(
        self,
        residual: torch.Tensor,
        auxiliary: torch.Tensor,
        interest: torch.Tensor,
        cf: torch.Tensor,
        user_ids: torch.Tensor,
        temperature: float = 0.07,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keep = _unique_rows(user_ids)
        interest_start = self.auxiliary_tokens
        cf_start = self.auxiliary_tokens + self.interest_tokens
        auxiliary_prediction = self.auxiliary_head(residual[keep, : self.auxiliary_tokens].mean(dim=1))
        interest_prediction = self.interest_head(residual[keep, interest_start:cf_start].mean(dim=1))
        cf_prediction = self.cf_head(residual[keep, cf_start:].mean(dim=1))
        auxiliary_loss, auxiliary_accuracy = _contrastive(
            auxiliary_prediction, self._center(auxiliary[keep], self.auxiliary_center), temperature
        )
        interest_loss, interest_accuracy = _contrastive(
            interest_prediction, self._center(interest[keep], self.prototype_center), temperature
        )
        cf_loss, cf_accuracy = _contrastive(
            cf_prediction, self._center(cf[keep], self.cf_center), temperature
        )
        return (
            (auxiliary_loss + interest_loss + cf_loss) / 3.0,
            (auxiliary_accuracy + interest_accuracy + cf_accuracy) / 3.0,
        )


class ResamplerBlock(nn.Module):
    def __init__(self, dimension: int = 768, heads: int = 8) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dimension)
        self.source_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(dimension)
        self.ff = nn.Sequential(nn.Linear(dimension, dimension * 4), nn.GELU(), nn.Linear(dimension * 4, dimension))

    def forward(self, query: torch.Tensor, source: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.query_norm(query)
        attended, _ = self.attention(
            normalized, self.source_norm(source), self.source_norm(source), key_padding_mask=padding_mask, need_weights=False
        )
        query = query + attended
        return query + self.ff(self.ff_norm(query))


class HistoryResamplerAdapter(nn.Module):
    """Map a target-excluded local history set to SD conditioning tokens."""

    def __init__(
        self,
        base_context: torch.Tensor,
        representation_center: torch.Tensor,
        token_count: int = 64,
        residual_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.token_count = token_count
        self.residual_scale = residual_scale
        self.register_buffer("base_context", base_context.detach().float().clone())
        self.register_buffer("representation_center", representation_center.detach().float().clone())
        self.source_projection = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 768))
        self.queries = nn.Parameter(torch.randn(token_count, 768) * 0.02)
        self.blocks = nn.ModuleList([ResamplerBlock(), ResamplerBlock()])
        self.output_norm = nn.LayerNorm(768)
        self.output_projection = nn.Linear(768, 768)
        self.profile_head = nn.Linear(768, 512)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self, support: torch.Tensor, support_mask: torch.Tensor, multiplier: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.source_projection(support.float())
        query = self.queries[None].expand(len(support), -1, -1)
        padding_mask = ~support_mask.bool()
        for block in self.blocks:
            query = block(query, source, padding_mask)
        residual = self.output_projection(self.output_norm(query))
        context = self.base_context.expand(len(support), -1, -1).clone()
        context[:, : self.token_count] += self.residual_scale * multiplier * residual
        return context, residual

    def user_contrastive_loss(
        self, residual: torch.Tensor, user_representations: torch.Tensor, user_ids: torch.Tensor, temperature: float = 0.07
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keep = _unique_rows(user_ids)
        prediction = self.profile_head(residual[keep].mean(dim=1))
        targets = F.normalize(user_representations[keep].float() - self.representation_center, dim=-1)
        return _contrastive(prediction, targets, temperature)


class PrototypeSetPreferenceIPDeltaAdapter(nn.Module):
    """Encode a variable-size set of user interest prototypes into user tokens.

    Compared with :class:`PreferenceIPDeltaAdapter`, the user/global branch is
    no longer produced from a single cluster-balanced mean vector.  Instead, a
    padded set of adaptive interest prototypes is encoded with attention masks
    and resampled into a fixed number of user preference tokens. The theme
    branch encodes the current interest as a direction relative to the user
    center.
    """

    def __init__(
        self,
        base_context: torch.Tensor,
        auxiliary_center: torch.Tensor,
        prototype_center: torch.Tensor,
        user_tokens: int = 8,
        delta_tokens: int = 8,
        hidden_dim: int = 1024,
        residual_scale: float = 1.0,
        set_layers: int = 2,
        query_layers: int = 2,
        heads: int = 8,
    ) -> None:
        super().__init__()
        if user_tokens <= 0 or delta_tokens <= 0:
            raise ValueError("user_tokens and delta_tokens must be positive")
        self.user_tokens = int(user_tokens)
        self.delta_tokens = int(delta_tokens)
        self.residual_scale = float(residual_scale)
        self.register_buffer("base_context", base_context.detach().float().clone())
        self.register_buffer("auxiliary_center", auxiliary_center.detach().float().clone())
        self.register_buffer("prototype_center", prototype_center.detach().float().clone())
        self.prototype_projection = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 768))
        self.weight_projection = nn.Sequential(nn.Linear(1, 768), nn.Tanh())
        self.source_blocks = nn.ModuleList([ResamplerBlock(768, heads) for _ in range(int(set_layers))])
        self.queries = nn.Parameter(torch.randn(self.user_tokens, 768) * 0.02)
        self.query_blocks = nn.ModuleList([ResamplerBlock(768, heads) for _ in range(int(query_layers))])
        self.user_norm = nn.LayerNorm(768)
        self.user_projection = nn.Linear(768, 768)
        self.delta_mlp = self._mlp(512, hidden_dim, self.delta_tokens)
        self.user_gate_logit = nn.Parameter(torch.tensor(2.0))
        self.delta_gate_logit = nn.Parameter(torch.tensor(2.0))
        self.user_head = nn.Linear(768, 512)
        self.delta_head = nn.Linear(768, 512)
        nn.init.zeros_(self.user_projection.weight)
        nn.init.zeros_(self.user_projection.bias)

    @staticmethod
    def _mlp(input_dim: int, hidden_dim: int, tokens: int) -> nn.Sequential:
        layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, tokens * 768),
        )
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        return layers

    def gates(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.sigmoid(self.user_gate_logit), torch.sigmoid(self.delta_gate_logit)

    def _encode_user_set(
        self,
        prototypes: torch.Tensor,
        prototype_mask: torch.Tensor,
        prototype_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = prototype_mask.bool()
        if prototype_weights is None:
            weights = mask.float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            weights = prototype_weights.float().masked_fill(~mask, 0.0)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        source = self.prototype_projection(prototypes.float()) + self.weight_projection(weights[..., None])
        padding_mask = ~mask
        # MultiheadAttention cannot handle rows where every key is masked.  The
        # adaptive interest cache always has at least one active prototype, but
        # this guard makes the adapter robust to malformed inputs.
        all_masked = padding_mask.all(dim=1)
        if bool(all_masked.any()):
            padding_mask = padding_mask.clone()
            padding_mask[all_masked, 0] = False
        for block in self.source_blocks:
            source = block(source, source, padding_mask)
        query = self.queries[None].expand(len(prototypes), -1, -1)
        for block in self.query_blocks:
            query = block(query, source, padding_mask)
        return self.user_projection(self.user_norm(query))

    def forward(
        self,
        prototypes: torch.Tensor,
        prototype_mask: torch.Tensor,
        auxiliary: torch.Tensor,
        interest: torch.Tensor,
        prototype_weights: torch.Tensor | None = None,
        multiplier: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        user_condition = self._encode_user_set(prototypes, prototype_mask, prototype_weights)
        delta_input = F.normalize(interest.float() - auxiliary.float(), dim=-1)
        delta_condition = self.delta_mlp(delta_input).reshape(-1, self.delta_tokens, 768)
        context = self.base_context.expand(len(prototypes), -1, -1).clone()
        user_gate, delta_gate = self.gates()
        scale = torch.as_tensor(float(multiplier) * self.residual_scale, device=prototypes.device)
        residual = torch.cat((user_condition, delta_condition), dim=1)
        attention_kwargs = {
            "preference_user_tokens": user_condition,
            "preference_delta_tokens": delta_condition,
            "preference_user_scale": scale * user_gate,
            "preference_delta_scale": scale * delta_gate,
        }
        return context, residual, attention_kwargs

    def zero_attention_kwargs(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        user_gate, delta_gate = self.gates()
        scale = torch.as_tensor(self.residual_scale, device=device, dtype=dtype)
        return {
            "preference_user_tokens": torch.zeros(batch_size, self.user_tokens, 768, device=device, dtype=dtype),
            "preference_delta_tokens": torch.zeros(batch_size, self.delta_tokens, 768, device=device, dtype=dtype),
            "preference_user_scale": scale * user_gate.to(device=device, dtype=dtype),
            "preference_delta_scale": scale * delta_gate.to(device=device, dtype=dtype),
        }

    def user_contrastive_loss(
        self,
        residual: torch.Tensor,
        prototypes: torch.Tensor,
        prototype_mask: torch.Tensor,
        auxiliary: torch.Tensor,
        interest: torch.Tensor,
        user_ids: torch.Tensor,
        temperature: float = 0.07,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del prototypes, prototype_mask  # encoded in residual; targets remain the same public CLIP-space vectors.
        keep = _unique_rows(user_ids)
        user_prediction = self.user_head(residual[keep, : self.user_tokens].mean(dim=1))
        delta_prediction = self.delta_head(residual[keep, self.user_tokens :].mean(dim=1))
        user_target = F.normalize(auxiliary[keep].float() - self.auxiliary_center, dim=-1)
        delta_target = F.normalize(interest[keep].float() - auxiliary[keep].float(), dim=-1)
        user_loss, user_accuracy = _contrastive(user_prediction, user_target, temperature)
        delta_loss, delta_accuracy = _contrastive(delta_prediction, delta_target, temperature)
        return (user_loss + delta_loss) / 2.0, (user_accuracy + delta_accuracy) / 2.0


class RoutedPrototypeSetPreferenceIPDeltaAdapter(PrototypeSetPreferenceIPDeltaAdapter):
    """Adapter with explicit direct and expansion modes in the theme branch."""

    DIRECT_ROUTE = 0
    EXPANSION_ROUTE = 1

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.route_tokens = nn.Parameter(torch.zeros(2, self.delta_tokens, 768))
        with torch.no_grad():
            self.route_tokens[self.EXPANSION_ROUTE].normal_(mean=0.0, std=0.01)

    def forward(
        self,
        prototypes: torch.Tensor,
        prototype_mask: torch.Tensor,
        auxiliary: torch.Tensor,
        interest: torch.Tensor,
        prototype_weights: torch.Tensor | None = None,
        multiplier: float = 1.0,
        route_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        user_condition = self._encode_user_set(prototypes, prototype_mask, prototype_weights)
        delta_input = F.normalize(interest.float() - auxiliary.float(), dim=-1)
        delta_condition = self.delta_mlp(delta_input).reshape(-1, self.delta_tokens, 768)
        if route_ids is None:
            route_ids = torch.zeros(len(prototypes), device=prototypes.device, dtype=torch.long)
        route_ids = route_ids.to(device=prototypes.device, dtype=torch.long)
        if route_ids.shape != (len(prototypes),):
            raise ValueError(f"route_ids must have shape {(len(prototypes),)}, got {tuple(route_ids.shape)}")
        if bool(((route_ids < 0) | (route_ids > 1)).any()):
            raise ValueError("route_ids must contain only 0 (DIRECT) or 1 (EXPANSION)")
        delta_condition = delta_condition + self.route_tokens[route_ids]
        context = self.base_context.expand(len(prototypes), -1, -1).clone()
        user_gate, delta_gate = self.gates()
        scale = torch.as_tensor(float(multiplier) * self.residual_scale, device=prototypes.device)
        residual = torch.cat((user_condition, delta_condition), dim=1)
        attention_kwargs = {
            "preference_user_tokens": user_condition,
            "preference_delta_tokens": delta_condition,
            "preference_user_scale": scale * user_gate,
            "preference_delta_scale": scale * delta_gate,
        }
        return context, residual, attention_kwargs

    def route_summary(self) -> dict[str, float]:
        direct = self.route_tokens[self.DIRECT_ROUTE].float().flatten()
        expansion = self.route_tokens[self.EXPANSION_ROUTE].float().flatten()
        return {
            "route_direct_rms": float(direct.square().mean().sqrt().detach().cpu()),
            "route_expansion_rms": float(expansion.square().mean().sqrt().detach().cpu()),
            "route_cosine_similarity": float(
                F.cosine_similarity(direct[None], expansion[None]).item()
            ),
        }


class UnifiedThemeSetPreferenceIPAdapter(nn.Module):
    """GReCF adapter with independent masked user-theme tokens and no route."""

    def __init__(
        self,
        base_context: torch.Tensor,
        auxiliary_center: torch.Tensor,
        prototype_center: torch.Tensor,
        user_tokens: int = 8,
        delta_tokens: int = 8,
        hidden_dim: int = 1024,
        residual_scale: float = 1.0,
        use_theme_anchor: bool = True,
        user_representation_mode: str = "theme_set",
    ) -> None:
        super().__init__()
        self.user_tokens = int(user_tokens)
        self.delta_tokens = int(delta_tokens)
        self.residual_scale = float(residual_scale)
        self.use_theme_anchor = bool(use_theme_anchor)
        self.user_representation_mode = str(user_representation_mode)
        if self.user_representation_mode not in ("theme_set", "item_average"):
            raise ValueError(f"unknown user representation mode: {self.user_representation_mode}")
        if self.user_representation_mode == "item_average" and self.user_tokens != 1:
            raise ValueError("item_average user representation requires one user token")
        self.register_buffer("base_context", base_context.detach().float().clone())
        self.register_buffer("auxiliary_center", auxiliary_center.detach().float().clone())
        self.register_buffer("prototype_center", prototype_center.detach().float().clone())
        self.theme_projection = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 768))
        self.theme_mlp = self._mlp(512, hidden_dim, self.delta_tokens)
        self.user_gate_logit = nn.Parameter(torch.tensor(2.0))
        self.delta_gate_logit = nn.Parameter(torch.tensor(2.0))
        self.user_head = nn.Linear(768, 512)
        self.delta_head = nn.Linear(768, 512)

    @staticmethod
    def _mlp(input_dim: int, hidden_dim: int, tokens: int) -> nn.Sequential:
        layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, tokens * 768),
        )
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        return layers

    def gates(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.sigmoid(self.user_gate_logit), torch.sigmoid(self.delta_gate_logit)

    def forward(
        self,
        prototypes: torch.Tensor,
        prototype_mask: torch.Tensor,
        auxiliary: torch.Tensor,
        interest: torch.Tensor,
        prototype_weights: torch.Tensor | None = None,
        multiplier: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        del prototype_weights
        if self.user_representation_mode == "item_average":
            user_source = auxiliary.float().unsqueeze(1)
            user_mask = torch.ones((len(prototypes), 1), dtype=torch.bool, device=prototypes.device)
        else:
            user_source = prototypes.float()
            user_mask = prototype_mask.bool()
            if prototypes.shape[1] != self.user_tokens:
                raise ValueError(
                    f"expected {self.user_tokens} padded theme slots, got {tuple(prototypes.shape)}"
                )
        user_condition = self.theme_projection(user_source)
        user_condition = user_condition * user_mask[..., None].to(user_condition.dtype)
        if self.use_theme_anchor:
            theme_input = F.normalize(interest.float(), dim=-1)
            theme_condition = self.theme_mlp(theme_input).reshape(-1, self.delta_tokens, 768)
        else:
            theme_condition = user_condition.new_zeros(len(prototypes), self.delta_tokens, 768)
        context = self.base_context.expand(len(prototypes), -1, -1).clone()
        user_gate, theme_gate = self.gates()
        scale = torch.as_tensor(float(multiplier) * self.residual_scale, device=prototypes.device)
        residual = torch.cat((user_condition, theme_condition), dim=1)
        attention_kwargs = {
            "preference_user_tokens": user_condition,
            "preference_user_mask": user_mask,
            "preference_delta_tokens": theme_condition,
            "preference_user_scale": scale * user_gate,
            "preference_delta_scale": scale * theme_gate if self.use_theme_anchor else scale.new_zeros(()),
        }
        return context, residual, attention_kwargs

    def zero_attention_kwargs(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        user_gate, theme_gate = self.gates()
        scale = torch.as_tensor(self.residual_scale, device=device, dtype=dtype)
        return {
            "preference_user_tokens": torch.zeros(
                batch_size, self.user_tokens, 768, device=device, dtype=dtype
            ),
            "preference_user_mask": torch.ones(
                batch_size, self.user_tokens, device=device, dtype=torch.bool
            ),
            "preference_delta_tokens": torch.zeros(
                batch_size, self.delta_tokens, 768, device=device, dtype=dtype
            ),
            "preference_user_scale": scale * user_gate.to(device=device, dtype=dtype),
            "preference_delta_scale": scale * theme_gate.to(device=device, dtype=dtype),
        }

    def user_contrastive_loss(
        self,
        residual: torch.Tensor,
        prototypes: torch.Tensor,
        prototype_mask: torch.Tensor,
        auxiliary: torch.Tensor,
        interest: torch.Tensor,
        user_ids: torch.Tensor,
        temperature: float = 0.07,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del prototypes
        keep = _unique_rows(user_ids)
        mask = prototype_mask[keep].float()
        user_tokens = residual[keep, : self.user_tokens]
        pooled_user = (user_tokens * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        user_prediction = self.user_head(pooled_user)
        user_target = F.normalize(auxiliary[keep].float() - self.auxiliary_center, dim=-1)
        user_loss, user_accuracy = _contrastive(user_prediction, user_target, temperature)
        if not self.use_theme_anchor:
            return user_loss, user_accuracy
        theme_tokens = residual[keep, self.user_tokens :]
        theme_prediction = self.delta_head(theme_tokens.mean(dim=1))
        theme_target = F.normalize(interest[keep].float(), dim=-1)
        theme_loss, theme_accuracy = _contrastive(theme_prediction, theme_target, temperature)
        return (user_loss + theme_loss) / 2.0, (user_accuracy + theme_accuracy) / 2.0


def padded_train_histories(dataset: PublicDataset) -> tuple[np.ndarray, np.ndarray]:
    width = max(len(items) for items in dataset.train)
    indices = np.zeros((dataset.num_users, width), dtype=np.int64)
    mask = np.zeros((dataset.num_users, width), dtype=bool)
    for user, items in enumerate(dataset.train):
        indices[user, : len(items)] = items
        mask[user, : len(items)] = True
    return indices, mask


def local_support_batch(
    users: torch.Tensor,
    targets: torch.Tensor | None,
    histories: torch.Tensor,
    history_mask: torch.Tensor,
    catalog_features: torch.Tensor,
    support_size: int,
    generator: torch.Generator | None = None,
    anchor_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose an anchor then retrieve its local liked-image support, excluding targets."""
    item_indices = histories[users]
    valid = history_mask[users].clone()
    if targets is not None:
        valid &= item_indices != targets[:, None]
    random_values = torch.rand(valid.shape, device=valid.device, generator=generator)
    random_values.masked_fill_(~valid, -1)
    if anchor_offsets is None:
        anchor_positions = random_values.argmax(dim=1)
    else:
        # Deterministic offsets are used for validation and generation.
        ranks = torch.cumsum(valid.long(), dim=1) - 1
        wanted = anchor_offsets.remainder(valid.sum(dim=1).clamp_min(1))
        anchor_positions = ((ranks == wanted[:, None]) & valid).long().argmax(dim=1)
    anchor_items = item_indices.gather(1, anchor_positions[:, None]).squeeze(1)
    history_features = catalog_features[item_indices]
    anchor_features = catalog_features[anchor_items]
    similarity = torch.einsum("bhd,bd->bh", history_features.float(), anchor_features.float())
    similarity.masked_fill_(~valid, -torch.inf)
    count = min(support_size, similarity.shape[1])
    positions = similarity.topk(count, dim=1).indices
    support_items = item_indices.gather(1, positions)
    support = catalog_features[support_items]
    support_mask = valid.gather(1, positions)
    return support, support_mask, anchor_items
