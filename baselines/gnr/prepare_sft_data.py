#!/usr/bin/env python3
"""GNR (Generate, Not Recommend; arXiv 2506.01704) step 1: build SFT windows.

Adaptation of the paper's next-item construction to the image-only
catalog: for each user, a sliding window over the chronological train
positives -- the previous k images are the multimodal history, the next
image is the generation target. Windows are evenly spaced per user so the
sample count is deterministic without shuffling. Splits come from
grecf.data.load_public_dataset (the project-wide split, never re-cut).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from grecf.data import load_public_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/path/to/datasets/CIGR"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--history-k", type=int, default=5)
    parser.add_argument("--samples-per-user", type=int, default=12)
    parser.add_argument("--image-root-old", default=None,
                        help="replace this path prefix in catalog image paths (IO cache)")
    parser.add_argument("--image-root-new", default=None)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_public_dataset(args.dataset_root)

    def remap(path: Path) -> str:
        path = str(path)
        if args.image_root_old and args.image_root_new and path.startswith(args.image_root_old):
            return args.image_root_new + path[len(args.image_root_old):]
        return path

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_windows = 0
    users_no_window = 0
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for user in range(dataset.num_users):
            train_items = list(dataset.train[user])
            k = args.history_k
            if len(train_items) <= k:
                users_no_window += 1
                continue
            # candidate target positions: index into train_items, each window
            # is train_items[pos-k : pos] -> train_items[pos]
            positions = np.arange(k, len(train_items))
            take = min(args.samples_per_user, len(positions))
            chosen = np.linspace(0, len(positions) - 1, take).round().astype(int)
            for pos_index in sorted(set(chosen.tolist())):
                pos = int(positions[pos_index])
                history = [remap(dataset.image_paths[item]) for item in train_items[pos - k : pos]]
                target = remap(dataset.image_paths[train_items[pos]])
                handle.write(json.dumps({
                    "user_index": int(user),
                    "user_id": str(dataset.user_ids[user]),
                    "history_paths": history,
                    "target_path": target,
                }, ensure_ascii=False) + "\n")
                n_windows += 1
    print(json.dumps({"windows": n_windows, "users_skipped_short_history": users_no_window,
                      "history_k": args.history_k, "samples_per_user": args.samples_per_user}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
