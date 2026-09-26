#!/usr/bin/env python3
"""Build NaviGen three-level collaborative identifiers from public CLIP features."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from grecf.data import id_order_sha256, load_cache, load_public_dataset  # noqa: E402

from common import cid_string, write_json  # noqa: E402


DEFAULT_DATASET = Path("/path/to/datasets/CIGR")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=8192)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test item cap; changes the artifact contract")
    return parser.parse_args()


def nearest_codes(values: torch.Tensor, centroids: torch.Tensor, batch_size: int) -> torch.Tensor:
    result = torch.empty(len(values), dtype=torch.long, device="cpu")
    centroid_bias = -0.5 * centroids.square().sum(dim=1)
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size].to(device=centroids.device, dtype=torch.float32)
        result[start : start + len(batch)] = (batch @ centroids.T + centroid_bias).argmax(dim=1).cpu()
    return result


def fit_kmeans(
    values_cpu: torch.Tensor,
    clusters: int,
    iterations: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    if len(values_cpu) < clusters:
        raise ValueError(f"clusters={clusters} exceeds rows={len(values_cpu)}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randperm(len(values_cpu), generator=generator)[:clusters]
    centroids = values_cpu[initial].to(device=device, dtype=torch.float32)
    history: list[dict[str, float]] = []

    for iteration in range(iterations):
        sums = torch.zeros_like(centroids)
        counts = torch.zeros(clusters, dtype=torch.long, device=device)
        objective_sum = 0.0
        for start in range(0, len(values_cpu), batch_size):
            batch = values_cpu[start : start + batch_size].to(device=device, dtype=torch.float32)
            bias = -0.5 * centroids.square().sum(dim=1)
            codes = (batch @ centroids.T + bias).argmax(dim=1)
            chosen = centroids[codes]
            objective_sum += float((batch - chosen).square().sum().item())
            sums.index_add_(0, codes, batch)
            counts.add_(torch.bincount(codes, minlength=clusters))
        nonempty = counts > 0
        updated = centroids.clone()
        updated[nonempty] = sums[nonempty] / counts[nonempty, None]
        empty = torch.nonzero(~nonempty, as_tuple=False).flatten()
        if len(empty):
            replacements = torch.randint(len(values_cpu), (len(empty),), generator=generator)
            updated[empty] = values_cpu[replacements].to(device=device, dtype=torch.float32)
        shift = float((updated - centroids).square().mean().sqrt().item())
        centroids = updated
        row = {
            "iteration": float(iteration + 1),
            "mse": objective_sum / (len(values_cpu) * values_cpu.shape[1]),
            "centroid_shift_rms": shift,
            "empty_centroids": float(len(empty)),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    codes = nearest_codes(values_cpu.to(dtype=torch.float32), centroids, batch_size)
    return centroids.cpu(), codes, history


def main() -> int:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    if args.levels != 3:
        raise SystemExit("NaviGen requires exactly three residual levels")
    dataset = load_public_dataset(args.dataset_root)
    features = np.asarray(
        load_cache(
            args.clip_cache,
            rows=dataset.num_items,
            order_sha256=id_order_sha256(dataset.image_ids),
            shape_tail=(512,),
        ),
        dtype=np.float32,
    )
    if args.limit > 0:
        features = features[: args.limit]
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-8)
    residual = torch.from_numpy(features)
    assignments = np.empty((len(features), args.levels), dtype=np.int16)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    level_summaries = []
    device = torch.device(args.device)

    for level in range(args.levels):
        print(json.dumps({"stage": "fit_level", "level": level + 1, "rows": len(residual)}), flush=True)
        centroids, codes, history = fit_kmeans(
            residual,
            clusters=args.clusters,
            iterations=args.iterations,
            batch_size=args.batch_size,
            seed=args.seed + level * 1009,
            device=device,
        )
        assignments[:, level] = codes.numpy().astype(np.int16)
        residual = residual - centroids[codes]
        np.save(args.output_dir / f"codebook_level{level + 1}.float32.npy", centroids.numpy())
        level_summaries.append(
            {
                "level": level + 1,
                "final_kmeans_mse": history[-1]["mse"],
                "residual_mse_after_level": float(residual.square().mean().item()),
                "used_codes": int(torch.unique(codes).numel()),
                "history": history,
            }
        )

    np.save(args.output_dir / "cid_assignments.int16.npy", assignments)
    cid_path = args.output_dir / "catalog_cids.jsonl"
    with cid_path.open("w", encoding="utf-8") as handle:
        for index, codes in enumerate(assignments):
            handle.write(
                json.dumps(
                    {
                        "item_index": index,
                        "image_id": dataset.image_ids[index],
                        "codes": [int(value) for value in codes],
                        "sid": cid_string(codes),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    summary = {
        "method": "three-level residual k-means on L2-normalized public CLIP image features",
        "rows": len(features),
        "dimension": features.shape[1],
        "clusters_per_level": args.clusters,
        "levels": args.levels,
        "iterations": args.iterations,
        "seed": args.seed,
        "clip_cache": str(args.clip_cache),
        "public_only": True,
        "level_summaries": level_summaries,
    }
    write_json(args.output_dir / "cid_summary.json", summary)
    print(json.dumps({"stage": "done", "rows": len(features), "output": str(args.output_dir)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
