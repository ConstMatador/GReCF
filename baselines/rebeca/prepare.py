#!/usr/bin/env python3
"""Convert a public positive-feedback dataset to REBECA ratings (step 1).

Emits the ratings-table layout consumed by the upstream REBECA pipeline
(worker_id / score / imagePair), but keeps THIS project's train/validation/test
partitioning instead of the upstream samples-per-user resplit: the REBECA prior
trains on the user's train positives, uses validation positives for early
stopping, and never sees test positives — so its conditioning is ignorant of the
held-out items that our own evaluation will probe.

Scores follow the upstream binary-like rule (like = score >= 4), with public
train and validation positives assigned score 5.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/path/to/datasets/CIGR"))
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="storage-side directory for processed/ratings.csv")
    return parser.parse_args()


def main() -> int:
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from grecf.data import load_public_dataset

    args = parse_args()
    dataset = load_public_dataset(args.dataset_root)

    records: list[dict[str, object]] = []
    for user_index in range(dataset.num_users):
        worker_id = int(user_index)  # contiguous already; identity mapping
        split_of = [
            ("train", np.asarray(dataset.train[user_index], dtype=np.int64)),
            ("validation", np.asarray(dataset.validation[user_index], dtype=np.int64)),
        ]
        image_id_at = lambda item: str(dataset.image_ids[int(item)])  # noqa: E731
        for split, items in split_of:
            for item in np.unique(items):
                records.append({
                    "worker_id": worker_id,
                    "score": 5,  # upstream: like = score >= 4
                    "imagePair": image_id_at(item),
                    "image_index": int(item),
                    "split": split,
                })

    frame = pd.DataFrame.from_records(records, columns=["worker_id", "score", "imagePair", "image_index", "split"])
    output = args.output_dir / "processed"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "ratings.csv", index=False)
    pd.DataFrame({"worker_id": range(dataset.num_users)}).to_csv(output / "worker_id_mapping.csv", index=False)
    write_json(output / "conversion_stats.json", {
        "dataset_root": str(args.dataset_root),
        "rows": int(len(frame)),
        "positive_rows": int((frame["score"] == 5).sum()),
        "users": int(dataset.num_users),
        "items": int(dataset.num_items),
        "splits": frame.groupby("split")["worker_id"].count().to_dict(),
        "test_interactions_written": False,
        "note": "identity worker mapping; public train/validation partitions preserved",
    })
    print(f"written {len(frame)} rows -> {output/'ratings.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
