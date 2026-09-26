#!/usr/bin/env python3
"""Select and validate the canonical merged NaviGen CID2INS prediction file."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def instruction_from_target_tid(target_tid: object) -> str:
    """Create a deterministic visual prompt when the small-data model omits target_ins."""
    if isinstance(target_tid, str):
        terms = [part.strip() for part in re.split(r"[,;|\n]", target_tid)]
    elif isinstance(target_tid, (list, tuple)):
        terms = [str(part).strip() for part in target_tid]
    else:
        terms = []
    terms = [re.sub(r"\s+", " ", term).strip(" .,;:\"'[]{}()") for term in terms]
    terms = list(dict.fromkeys(term for term in terms if term))
    if not terms:
        return ""

    subject = ", ".join(terms[:3])
    attributes = ", ".join(terms[3:])
    if attributes:
        return (
            f"A detailed photographic image featuring {subject}. "
            f"Incorporate the visual attributes {attributes}. "
            "Use a clear subject, natural lighting, balanced composition, realistic detail, "
            "and a visually coherent scene."
        )
    return (
        f"A detailed photographic image featuring {subject}. "
        "Use natural lighting, balanced composition, realistic detail, and a visually coherent scene."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--expected-users", type=int, default=1000)
    parser.add_argument(
        "--fallback-target-tid",
        action="store_true",
        help="Build target_ins deterministically from target_tid when the model omits it.",
    )
    args = parser.parse_args()
    candidates = [
        path
        for path in args.prediction_dir.glob("*cid2ins*predictions*.jsonl")
        if ".rank" not in path.name and path != args.output_jsonl
    ]
    if not candidates:
        raise SystemExit(f"no merged CID2INS predictions under {args.prediction_dir}")
    source = max(candidates, key=lambda path: (path.stat().st_size, path.stat().st_mtime_ns))
    rows = []
    users = set()
    recovered = 0
    with source.open(encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            source_row = row.get("source_row") or {}
            prediction = row.get("prediction") or {}
            user = source_row.get("user_index")
            instruction = str(prediction.get("target_ins", "")).strip()
            if not instruction and args.fallback_target_tid:
                instruction = instruction_from_target_tid(prediction.get("target_tid"))
                if instruction:
                    prediction = dict(prediction)
                    prediction["target_ins"] = instruction
                    row["prediction"] = prediction
                    row["target_ins_recovery"] = "deterministic_target_tid_template"
                    recovered += 1
            if user is None or not instruction:
                raise ValueError(f"invalid prediction at {source}:{line_number}")
            users.add(int(user))
            rows.append(row)
    if len(rows) != args.expected_users or len(users) != args.expected_users:
        raise ValueError(f"prediction contract mismatch: rows={len(rows)}, users={len(users)}")
    rows.sort(key=lambda row: int((row.get("source_row") or {})["user_index"]))
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output_jsonl)
    print(
        json.dumps(
            {
                "source": str(source),
                "output": str(args.output_jsonl),
                "rows": len(rows),
                "target_ins_recovered": recovered,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
