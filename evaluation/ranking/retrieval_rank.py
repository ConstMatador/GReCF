#!/usr/bin/env python3
"""Full-pool item-level Recall@K / NDCG@K for generative recommendation.

Protocol:
- Generated images are retrieval queries. The candidate pool is the full
  catalog minus the user's own train/validation positives — the same "full
  pool" convention as the traditional baselines (UserKNN/iALS/BPR) and the
  oracle ceiling, so results are directly comparable with them.
- A hit is an exact test-split image ID. Deliberately no topic-level
  relaxation: CIGR ships oracle topic labels, but real datasets do not, so the
  headline protocol must stay reproducible on ordinary interaction data.
- Each user contributes several queries (one per generated image). Per-query
  scores are averaged within the user first, then macro-averaged over users.
  The score denominator is min(K, |test_u|) so users with small test sets are
  not systematically deflated.
- Controls through the identical pipeline: (a) `random_queries_per_user`
  deterministic random catalog images per user, giving the empirical floor;
  (b) optionally the cluster-balanced user-representation vector, giving a
  "no generation" reference line.
- Unfiltered-pool variants (train/validation positives kept in the pool) are
  reported next to the main numbers as a diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grecf.data import id_order_sha256, load_cache, load_public_dataset  # noqa: E402

# LOG_DISCOUNTS[r] = 1/log2(r+2) is the discount of 0-based rank r, i.e. the
# document at position r+1 in the ranked list; supports K up to 500.
LOG_DISCOUNTS = 1.0 / np.log2(np.arange(2, 503))
IDCG_CUMULATIVE = np.cumsum(LOG_DISCOUNTS)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_method(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("--method expects NAME=DIR_WITH_TABLES")
    directory = Path(path)
    if not (directory / "tables/generation_metrics.csv").exists():
        raise argparse.ArgumentTypeError(f"{directory} has no tables/generation_metrics.csv")
    return name, directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=Path("/path/to/datasets/CIGR"))
    parser.add_argument(
        "--target-split",
        choices=("validation", "test"),
        default="test",
        help="Held-out relevance split. Use validation for model selection and test only for final reporting.",
    )
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        type=parse_method,
        help="NAME=DIR containing tables/generation_metrics.csv and tables/generated_clip_features.float32.npy",
    )
    parser.add_argument(
        "--user-rep-cache",
        type=Path,
        default=None,
        help="optional .npy [num_users, dim] cluster-balanced user vectors used as reference queries",
    )
    parser.add_argument("--random-per-user", type=int, default=10)
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50, 100])
    parser.add_argument(
        "--semantic-percentile",
        type=float,
        default=99.0,
        help="calibrated CLIP-similarity percentile of random catalog image pairs above which "
        "a retrieved image counts as semantically equivalent to a test image (hit-rule variant)",
    )
    parser.add_argument("--limit-users", type=int, default=0, help="smoke-test cap on the number of users")
    parser.add_argument("--device", default=None, help="defaults to cuda:0 when torch sees a GPU")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.methods = dict(args.method)
    args.ks = sorted(set(args.ks))
    return args


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def calibrate_semantic_threshold(catalog_normalized: np.ndarray, seed: int, percentile: float) -> float:
    """CLIP-similarity cutoff between random catalog image pairs at `percentile`.

    Two images whose similarity exceeds this are more alike than 99% (default)
    of random image pairs; using it as the hit criterion tolerates near-
    duplicate substitutions without touching any dataset metadata.
    """
    generator = np.random.default_rng(seed + 411)
    left = generator.integers(0, len(catalog_normalized), size=200_000)
    right = generator.integers(0, len(catalog_normalized), size=200_000)
    right = (right + 1 + (left == right)) % len(catalog_normalized)
    sims = (catalog_normalized[left] * catalog_normalized[right]).sum(axis=1)
    return float(np.percentile(sims, percentile))


def build_relevance_masks(
    catalog_normalized: np.ndarray,
    user_indices: list[int],
    test_lookup: dict[int, np.ndarray],
    threshold: float,
    device: str,
) -> dict[int, np.ndarray]:
    """Per-user boolean mask of catalog items semantically equivalent to any test positive."""
    import torch

    catalog = torch.from_numpy(catalog_normalized).to(device)
    masks: dict[int, np.ndarray] = {}
    for user_index in sorted(user_indices):
        test_items = test_lookup[user_index]
        features = torch.from_numpy(catalog_normalized[test_items]).to(device)
        related = (features @ catalog.T >= threshold).any(dim=0)
        related |= torch.isin(
            torch.arange(len(catalog), device=device),
            torch.from_numpy(test_items).to(device),
        )
        masks[user_index] = related.cpu().numpy()
    return masks


class QueryScorer:
    """Scores blocks of CLIP queries against the full catalog on one device.

    Produces four metric families per K:
      exact              — top-K item is a test image (exclusion applied)
      exact_unfiltered   — same but train/validation positives stay in the pool (diagnostic)
      semantic           — top-K item is a test image or its calibrated equivalent
      fused_exact/fused_semantic — queries are collapsed by element-wise max similarity
                           into one user-level query before ranking
    """

    def __init__(self, catalog_normalized: np.ndarray, ks: list[int], device: str) -> None:
        import torch

        self.torch = torch
        self.device = device
        self.catalog = torch.from_numpy(catalog_normalized).to(device)
        self.ks = ks
        self.k_max = max(ks)

    def _metrics_from_orderings(
        self,
        orderings: dict[str, "object"],
        relevant_masks: dict[str, "object"],
        base_sizes: dict[str, int],
        m_u: int,
    ) -> dict[str, object]:
        """Per-row HitRate/NDCG for each metric family.

        `base_sizes` gives the number of pool-relevant items per relevance rule;
        the exact rule has exactly m_u of them, so its recall equals hits/m_u,
        while the semantic rule credits an entire equivalent neighborhood and is
        therefore reported as HitRate (bounded in [0,1]) rather than recall.
        """
        torch = self.torch
        output: dict[str, object] = {}
        for family, indices in orderings.items():
            rows = indices.cpu().numpy()
            if family.startswith("fused"):
                rows = rows[:1]
            mask_key = family.replace("fused_", "").replace("_unfiltered", "")
            hits = relevant_masks[mask_key].cpu().numpy()[rows]
            rule_size = base_sizes[mask_key]
            for k in self.ks:
                discounts = LOG_DISCOUNTS[:k]
                idcg = IDCG_CUMULATIVE[min(k, rule_size) - 1]
                prefix = hits[:, :k].astype(np.float64)
                output[f"{family}_hitrate@{k}"] = (prefix.sum(axis=1) > 0).astype(np.float64)
                output[f"{family}_ndcg@{k}"] = (prefix * discounts).sum(axis=1) / idcg
                if mask_key == "exact":
                    output[f"{family}_recall@{k}"] = prefix.sum(axis=1) / m_u
        return output

    def score(
        self,
        queries_normalized: np.ndarray,
        excluded: np.ndarray,
        test_items: np.ndarray,
        relevant_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        torch = self.torch
        queries = torch.from_numpy(np.ascontiguousarray(queries_normalized)).to(self.device)
        excluded_tensor = (
            torch.from_numpy(excluded.astype(np.int64)).to(self.device) if len(excluded) else None
        )
        m_u = len(test_items)

        exact_mask = torch.zeros(self.catalog.shape[0], dtype=torch.bool, device=self.device)
        exact_mask[torch.from_numpy(test_items.astype(np.int64)).to(self.device)] = True
        semantic_full = relevant_mask.copy()
        if len(excluded):
            semantic_full[excluded] = False
        semantic_mask = torch.from_numpy(semantic_full).to(self.device)
        base_sizes = {"exact": m_u, "semantic": int(semantic_full.sum())}

        similarities = queries @ self.catalog.T
        fused = similarities.max(dim=0, keepdim=True).values
        both = torch.cat((similarities, fused), dim=0)

        _, open_top = both.topk(self.k_max, dim=1)
        if excluded_tensor is not None:
            both[:, excluded_tensor] = -float("inf")
        _, top = both.topk(self.k_max, dim=1)

        rows_output: dict[str, np.ndarray] = {
            "test_size": np.full(len(queries), m_u, dtype=np.int64),
            **self._metrics_from_orderings(
                {"exact": top[: len(queries)], "exact_unfiltered": open_top[: len(queries)],
                 "semantic": top[: len(queries)], "semantic_unfiltered": open_top[: len(queries)]},
                {"exact": exact_mask, "semantic": semantic_mask},
                base_sizes,
                m_u,
            ),
        }
        fused_metrics = self._metrics_from_orderings(
            {"fused_exact": top[len(queries):], "fused_semantic": top[len(queries):],
             "fused_exact_unfiltered": open_top[len(queries):]},
            {"exact": exact_mask, "semantic": semantic_mask},
            base_sizes,
            m_u,
        )
        # Broadcast the single fused row to every generated query so per-user
        # aggregation in the frame stays uniform.
        for key, value in fused_metrics.items():
            rows_output[key] = np.repeat(value, len(queries), axis=0)
        return rows_output


USER_REP_KIND = "user_rep"
GENERATED_KIND = "generated"
RANDOM_KIND = "random"


def score_queries(
    scorer: QueryScorer,
    kind: str,
    query_rows_by_user: dict[int, np.ndarray],
    source_features: np.ndarray,
    dataset,
    seen_lookup: dict[int, np.ndarray],
    test_lookup: dict[int, np.ndarray],
    relevance_masks: dict[int, np.ndarray],
    frame_records: list[dict[str, object]],
) -> None:
    for user_index in sorted(query_rows_by_user):
        rows = query_rows_by_user[user_index]
        user_id = dataset.user_ids[int(user_index)]
        metrics = scorer.score(
            source_features[rows],
            seen_lookup[int(user_index)],
            test_lookup[int(user_index)],
            relevance_masks[int(user_index)],
        )
        for local, feature_row in enumerate(rows):
            record: dict[str, object] = {
                "query_kind": kind,
                "user_id": user_id,
                "test_size": int(metrics["test_size"][local]),
                "excluded_count": int(len(seen_lookup[int(user_index)])),
            }
            for key, value in metrics.items():
                if key == "test_size":
                    continue
                record[key] = float(value[local])
            frame_records.append(record)


def collect_frame(records: list[dict[str, object]], method_name: str) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame.insert(0, "method", method_name)
    ordered = ["method", "query_kind", "user_id", "test_size", "excluded_count"]
    ordered += [column for column in sorted(frame.columns) if column not in ordered]
    return frame.reindex(columns=ordered)


def summarize(frame_all: pd.DataFrame, ks: list[int]) -> dict[str, dict[str, object]]:
    label_of = {GENERATED_KIND: "generated", RANDOM_KIND: "random_floor", USER_REP_KIND: "user_representation"}
    families = ("exact", "semantic", "fused_exact", "fused_semantic")
    diagnostics = {"exact": "exact_unfiltered", "semantic": "semantic_unfiltered",
                   "fused_exact": "fused_exact_unfiltered"}
    results: dict[str, dict[str, object]] = {}
    for name in sorted(frame_all["method"].unique()):
        block: dict[str, dict[str, object]] = {}
        for kind, label in label_of.items():
            selected = frame_all.loc[(frame_all["method"] == name) & (frame_all["query_kind"] == kind)]
            if selected.empty:
                continue
            entry: dict[str, object] = {
                "queries": int(len(selected)),
                "users": int(selected["user_id"].nunique()),
                "mean_test_size": float(selected["test_size"].mean()),
            }
            for family in families:
                entry[family] = {}
                for k in ks:
                    metrics_entry: dict[str, float] = {
                        f"hitrate@{k}": float(selected[f"{family}_hitrate@{k}"].mean()),
                        f"ndcg@{k}": float(selected[f"{family}_ndcg@{k}"].mean()),
                    }
                    if family == "exact":
                        metrics_entry[f"recall@{k}"] = float(selected[f"exact_recall@{k}"].mean())
                    entry[family][str(k)] = metrics_entry
            block[label] = entry
        floor = block.get("random_floor")
        generated = block.get("generated")
        if floor and generated:
            for family in families:
                lift: dict[str, float] = {}
                for k in ks:
                    denominator = floor[family][str(k)][f"hitrate@{k}"]
                    if denominator > 0:
                        lift[f"hitrate_lift_over_random@{k}"] = (
                            generated[family][str(k)][f"hitrate@{k}"] / denominator
                        )
                generated[family]["lift_over_random_floor"] = lift
        results[name] = block
    return results


def main() -> int:
    args = parse_args()
    import torch

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = load_public_dataset(args.dataset_root)
    order_sha = id_order_sha256(dataset.image_ids)
    clip_features = np.asarray(load_cache(args.clip_cache, dataset.num_items, order_sha, (512,)), dtype=np.float32)
    catalog_normalized = normalized_rows(clip_features)

    num_users = dataset.num_users
    seen_lookup: dict[int, np.ndarray] = {}
    target_lookup: dict[int, np.ndarray] = {}
    for user_index in range(num_users):
        if args.target_split == "test":
            seen_lookup[user_index] = np.unique(
                np.concatenate((dataset.train[user_index], dataset.validation[user_index]))
            )
            target_lookup[user_index] = np.asarray(dataset.test[user_index], dtype=np.int64)
        else:
            seen_lookup[user_index] = np.asarray(dataset.train[user_index], dtype=np.int64)
            target_lookup[user_index] = np.asarray(dataset.validation[user_index], dtype=np.int64)

    scorer = QueryScorer(catalog_normalized, args.ks, device)

    frames: list[pd.DataFrame] = []
    reference_cohort: set[int] | None = None
    random_plan: dict[int, np.ndarray] | None = None
    relevance_masks: dict[int, np.ndarray] | None = None

    for name, directory in args.methods.items():
        metrics_table = pd.read_csv(directory / "tables/generation_metrics.csv")
        features = np.load(directory / "tables/generated_clip_features.float32.npy")
        if len(metrics_table) != len(features):
            raise SystemExit(f"row/feature mismatch under {directory}: {len(metrics_table)} vs {len(features)}")
        user_indices = metrics_table["user_index"].to_numpy(dtype=np.int64)
        if user_indices.min() < 0 or user_indices.max() >= num_users:
            raise SystemExit(f"user_index out of range under {directory}")
        cohort = set(user_indices.tolist())
        if reference_cohort is None:
            reference_cohort = cohort
        elif cohort != reference_cohort:
            raise SystemExit(f"user cohorts differ between methods: {directory}")
        if args.limit_users > 0:
            keep = set(sorted(cohort)[: args.limit_users])
            mask = np.isin(user_indices, sorted(keep))
            metrics_table = metrics_table.loc[mask].reset_index(drop=True)
            features = features[mask]
            user_indices = user_indices[mask]

        if relevance_masks is None:
            threshold = calibrate_semantic_threshold(
                catalog_normalized, args.seed, args.semantic_percentile
            )
            relevance_masks = build_relevance_masks(
                catalog_normalized,
                sorted(set(user_indices.tolist())),
                target_lookup,
                threshold,
                device,
            )
            scorer.semantic_threshold = threshold

        query_rows_by_user: dict[int, np.ndarray] = {}
        for user_index in np.unique(user_indices):
            query_rows_by_user[int(user_index)] = np.flatnonzero(user_indices == user_index)

        records: list[dict[str, object]] = []
        score_queries(
            scorer, GENERATED_KIND, query_rows_by_user, features,
            dataset, seen_lookup, target_lookup, relevance_masks, records,
        )

        if random_plan is None:
            generator = np.random.default_rng(args.seed)
            random_plan = {
                int(user_index): generator.choice(dataset.num_items, size=args.random_per_user, replace=False)
                for user_index in sorted(reference_cohort if args.limit_users <= 0 else set(map(int, query_rows_by_user)))
            }
        score_queries(
            scorer, RANDOM_KIND, random_plan, catalog_normalized,
            dataset, seen_lookup, target_lookup, relevance_masks, records,
        )

        if args.user_rep_cache is not None:
            reps = normalized_rows(np.load(args.user_rep_cache))
            rep_plan = {int(user): np.asarray([user]) for user in query_rows_by_user}
            score_queries(
                scorer, USER_REP_KIND, rep_plan, reps,
                dataset, seen_lookup, target_lookup, relevance_masks, records,
            )

        frames.append(collect_frame(records, name))

    assert reference_cohort is not None
    frame_all = pd.concat(frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_all.to_csv(args.output_dir / "retrieval_rank_per_query.csv", index=False)

    payload = {
        "protocol": {
            "pool": (
                "full catalog minus the user's train/validation positives"
                if args.target_split == "test"
                else "full catalog minus the user's train positives"
            ),
            "target_split": args.target_split,
            "hit_rule_exact": f"retrieved item is a {args.target_split}-split image id",
            "hit_rule_semantic": (
                f"retrieved item is a {args.target_split} image or its CLIP similarity exceeds a threshold "
                "calibrated as the P{:.0f} of random catalog image pairs".format(args.semantic_percentile)
            ),
            "semantic_threshold": getattr(scorer, "semantic_threshold", None),
            "fused_query": "element-wise max similarity over the user's queries before ranking",
            "query": "normalized CLIP feature of one generated image",
            "aggregation": "mean over queries within a user, then macro-mean over users",
            "recall_denominator": f"|{args.target_split}_u|",
            "ks": args.ks,
            "unfiltered_columns": "diagnostic only",
            "random_control_queries_per_user": args.random_per_user,
            "random_seed": args.seed,
            "users_evaluated": len(reference_cohort),
            "device": str(device),
            "dataset_root": str(args.dataset_root),
            "clip_cache": str(args.clip_cache),
            "methods": {name: str(path) for name, path in args.methods.items()},
            "user_rep_reference": str(args.user_rep_cache) if args.user_rep_cache else None,
        },
        "results": summarize(frame_all, args.ks),
    }
    write_json(args.output_dir / "retrieval_rank_summary.json", payload)

    compact = json.dumps(payload["results"], ensure_ascii=False)
    print(json.dumps({
        "summary": str(args.output_dir / "retrieval_rank_summary.json"),
        "per_query_csv": str(args.output_dir / "retrieval_rank_per_query.csv"),
        "rows": int(len(frame_all)),
        "preview": compact[:400],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
