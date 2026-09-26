#!/usr/bin/env python3
"""Map a public dataset into the NaviGen stage-1/stage-2 parquet contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from grecf.data import load_public_dataset  # noqa: E402

from common import cid_string, iter_jsonl, write_json  # noqa: E402


DEFAULT_DATASET = Path("/path/to/datasets/CIGR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tids-jsonl", type=Path, required=True)
    parser.add_argument("--cid-assignments", type=Path, required=True)
    parser.add_argument("--selected-users-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-input-dir", type=Path, required=True)
    parser.add_argument("--max-history", type=int, default=80)
    parser.add_argument("--minimum-history", type=int, default=5)
    parser.add_argument("--cid-windows-per-user", type=int, default=12)
    parser.add_argument("--validation-teacher-users", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def item_split(image_id: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(image_id.encode("utf-8")).digest()[:8], "big") % 100
    if bucket < 98:
        return "train"
    if bucket == 98:
        return "valid"
    return "test"


def load_tids(path: Path, expected: int) -> list[list[str]]:
    result: list[list[str] | None] = [None] * expected
    for row in iter_jsonl(path):
        index = int(row["item_index"])
        tid = [str(value).strip() for value in row.get("tid", []) if str(value).strip()]
        if not 0 <= index < expected:
            raise ValueError(f"TID item_index out of range: {index}")
        if len(tid) < 2:
            raise ValueError(f"missing/invalid TID for item_index={index}")
        result[index] = tid[:10]
    missing = [index for index, tid in enumerate(result) if tid is None]
    if missing:
        raise ValueError(f"TID file incomplete: missing={len(missing)}, sample={missing[:10]}")
    return [tid for tid in result if tid is not None]


def selected_users(path: Path, dataset) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "user_indices" in payload:
        users = [int(value) for value in payload["user_indices"]]
    elif "all_user_indices" in payload:
        users = [int(value) for value in payload["all_user_indices"]]
    elif "user_ids" in payload:
        by_id = {value: index for index, value in enumerate(dataset.user_ids)}
        users = [by_id[str(value)] for value in payload["user_ids"]]
    else:
        raise ValueError(f"unsupported cohort file: {path}")
    if not users or len(users) != len(set(users)):
        raise ValueError(f"cohort must contain non-empty unique user indices, got {len(users)}")
    invalid = [user for user in users if not 0 <= user < dataset.num_users]
    if invalid:
        raise ValueError(f"cohort has invalid user indices: {invalid[:10]}")
    return users


def history(values: np.ndarray, target_position: int, max_history: int) -> list[int]:
    start = max(0, target_position - max_history)
    return [int(value) for value in values[start:target_position]]


def cid_sequence(items: list[int], cids: list[str]) -> list[str]:
    return [cids[item] for item in items]


def tid_sequence(items: list[int], tids: list[list[str]]) -> list[list[str]]:
    return [tids[item] for item in items]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def mapping_frames(dataset, cids: list[str], tids: list[list[str]]) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    groups: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for item, image_id in enumerate(dataset.image_ids):
        groups[item_split(image_id)].append(
            {"pid": image_id, "item_index": item, "sid": cids[item], "tid": tids[item]}
        )
    return {
        split: (
            pd.DataFrame(rows, columns=["pid", "item_index", "sid", "tid"]),
            pd.DataFrame(rows, columns=["pid", "item_index", "tid", "sid"]),
        )
        for split, rows in groups.items()
    }


def make_teacher_sample(
    dataset,
    user: int,
    hist_items: list[int],
    target: int,
    cids: list[str],
    tids: list[list[str]],
    split: str,
) -> dict[str, Any]:
    return {
        "sample_id": f"{split}:{dataset.user_ids[user]}",
        "split": split,
        "user_index": int(user),
        "user_id": str(dataset.user_ids[user]),
        "hist_items": hist_items,
        "hist_sid": cid_sequence(hist_items, cids),
        "history_tids": tid_sequence(hist_items, tids),
        "target_item": int(target),
        "target_image_id": dataset.image_ids[target],
        "target_sid": cids[target],
        "target_tid": tids[target],
    }


def main() -> int:
    args = parse_args()
    dataset = load_public_dataset(args.dataset_root)
    tids = load_tids(args.tids_jsonl, dataset.num_items)
    assignments = np.load(args.cid_assignments)
    if assignments.shape != (dataset.num_items, 3):
        raise ValueError(f"unexpected CID assignments shape: {assignments.shape}")
    cids = [cid_string(row) for row in assignments]
    cohort = selected_users(args.selected_users_json, dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.teacher_input_dir.mkdir(parents=True, exist_ok=True)

    catalog = pd.DataFrame(
        {
            "pid": dataset.image_ids,
            "item_index": np.arange(dataset.num_items, dtype=np.int64),
            "sid": cids,
            "tid": tids,
        }
    )
    catalog.to_parquet(args.output_dir / "pid2cid2tid.parquet", index=False)
    mapping_counts = {}
    for split, (cid2tid, tid2cid) in mapping_frames(dataset, cids, tids).items():
        cid2tid.to_parquet(args.output_dir / f"{split}_cid2tid.parquet", index=False)
        tid2cid.to_parquet(args.output_dir / f"{split}_tid2cid.parquet", index=False)
        mapping_counts[split] = len(cid2tid)

    train_cid_rows: list[dict[str, Any]] = []
    valid_cid_rows: list[dict[str, Any]] = []
    test_cid_rows: list[dict[str, Any]] = []
    train_teacher: list[dict[str, Any]] = []
    valid_teacher: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    valid_teacher_set = set(
        rng.choice(dataset.num_users, size=min(args.validation_teacher_users, dataset.num_users), replace=False).tolist()
    )

    for user in range(dataset.num_users):
        train_items = np.asarray(dataset.train[user], dtype=np.int64)
        positions = np.arange(args.minimum_history, len(train_items), dtype=np.int64)
        take = min(args.cid_windows_per_user, len(positions))
        selected_positions = (
            positions[np.linspace(0, len(positions) - 1, take).round().astype(int)] if take else np.empty(0, dtype=np.int64)
        )
        for position in sorted(set(selected_positions.tolist())):
            hist_items = history(train_items, int(position), args.max_history)
            target = int(train_items[position])
            train_cid_rows.append(
                {
                    "sample_id": f"train:{dataset.user_ids[user]}:{position}",
                    "user_index": user,
                    "hist_sid": cid_sequence(hist_items, cids),
                    "target_sid": cids[target],
                }
            )

        teacher_target = int(train_items[-1])
        teacher_history = history(train_items, len(train_items) - 1, args.max_history)
        train_teacher.append(
            make_teacher_sample(dataset, user, teacher_history, teacher_target, cids, tids, "train")
        )

        valid_target = int(dataset.validation[user][0])
        valid_hist = [int(value) for value in train_items[-args.max_history :]]
        valid_cid_rows.append(
            {
                "sample_id": f"valid:{dataset.user_ids[user]}",
                "user_index": user,
                "hist_sid": cid_sequence(valid_hist, cids),
                "target_sid": cids[valid_target],
            }
        )
        if user in valid_teacher_set:
            valid_teacher.append(
                make_teacher_sample(dataset, user, valid_hist, valid_target, cids, tids, "valid")
            )

    for user in cohort:
        train_items = np.asarray(dataset.train[user], dtype=np.int64)
        hist_items = [int(value) for value in train_items[-args.max_history :]]
        target = int(dataset.test[user][0])
        test_cid_rows.append(
            {
                "sample_id": f"test:{dataset.user_ids[user]}",
                "user_index": user,
                "hist_sid": cid_sequence(hist_items, cids),
                "target_sid": cids[target],
            }
        )

    pd.DataFrame(train_cid_rows).to_parquet(args.output_dir / "train_cid2cid.parquet", index=False)
    pd.DataFrame(valid_cid_rows).to_parquet(args.output_dir / "valid_cid2cid.parquet", index=False)
    pd.DataFrame(test_cid_rows).to_parquet(args.output_dir / "test_cid2cid.parquet", index=False)
    write_jsonl(args.teacher_input_dir / "train_teacher_input.jsonl", train_teacher)
    write_jsonl(args.teacher_input_dir / "valid_teacher_input.jsonl", valid_teacher)

    test_ins_rows = []
    for user in cohort:
        train_items = np.asarray(dataset.train[user], dtype=np.int64)
        hist_items = [int(value) for value in train_items[-args.max_history :]]
        target = int(dataset.test[user][0])
        test_ins_rows.append(
            {
                "sample_id": f"test:{dataset.user_ids[user]}",
                "user_index": user,
                "user_id": dataset.user_ids[user],
                "hist_sid": cid_sequence(hist_items, cids),
                "target_tid": tids[target],
                "target_ins": "ground-truth instruction withheld; field is not included in the inference prompt",
                "reasoning": "",
            }
        )
    pd.DataFrame(test_ins_rows).to_parquet(args.output_dir / "test_cid2ins.parquet", index=False)

    summary = {
        "dataset_root": str(args.dataset_root),
        "public_only": True,
        "users": dataset.num_users,
        "items": dataset.num_items,
        "mapping_rows": mapping_counts,
        "train_cid2cid_rows": len(train_cid_rows),
        "valid_cid2cid_rows": len(valid_cid_rows),
        "test_cid2cid_rows": len(test_cid_rows),
        "train_teacher_rows": len(train_teacher),
        "valid_teacher_rows": len(valid_teacher),
        "test_cid2ins_rows": len(test_ins_rows),
        "max_history": args.max_history,
        "cid_windows_per_user": args.cid_windows_per_user,
        "test_interactions_used_for_training": False,
        "selected_users_json": str(args.selected_users_json),
    }
    write_json(args.output_dir / "prepare_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
