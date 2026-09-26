#!/usr/bin/env python3
"""Compare source attribution across judges, datasets, and approaches."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


LABELS = ("direct_history_theme", "true_cf_expansion", "other_or_uncertain")
DISPLAY = {
    "direct_history_theme": "History",
    "true_cf_expansion": "CF",
    "other_or_uncertain": "Other",
}


def parse_spec(value: str) -> tuple[str, str, str, Path]:
    metadata, separator, path = value.partition("=")
    fields = metadata.split("::")
    if not separator or len(fields) != 3 or not all(fields) or not path:
        raise argparse.ArgumentTypeError(
            "group must use DATASET::APPROACH::JUDGE=/path/to/judge.jsonl"
        )
    return fields[0], fields[1], fields[2], Path(path)


def load_labels(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = (row.get("judge") or {}).get("label")
            if label in LABELS:
                rows[str(row["task_id"])] = str(label)
    return rows


def rates(labels: dict[str, str], task_ids: set[str]) -> dict[str, float]:
    counts = Counter(labels[task_id] for task_id in task_ids)
    return {DISPLAY[label]: counts[label] / len(task_ids) for label in LABELS}


def agreement(first: dict[str, str], second: dict[str, str], task_ids: set[str]) -> dict[str, float]:
    total = len(task_ids)
    observed = sum(first[task] == second[task] for task in task_ids) / total
    first_counts = Counter(first[task] for task in task_ids)
    second_counts = Counter(second[task] for task in task_ids)
    chance = sum(first_counts[label] * second_counts[label] for label in LABELS) / (total * total)
    kappa = (observed - chance) / (1.0 - chance) if chance < 1.0 else 1.0
    return {"agreement": observed, "cohen_kappa": kappa}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", type=parse_spec, required=True)
    parser.add_argument("--reference-judge", default="Qwen3-VL-32B")
    parser.add_argument("--expected-groups", type=int, default=21)
    parser.add_argument("--expected-judges", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for dataset, approach, judge, path in args.group:
        target = grouped.setdefault((dataset, approach), {})
        if judge in target:
            raise ValueError(f"duplicate result for {dataset}/{approach}/{judge}")
        target[judge] = load_labels(path)

    group_report: dict[str, object] = {}
    distributions: dict[tuple[str, str, str], dict[str, float]] = {}
    for (dataset, approach), judges in sorted(grouped.items()):
        if args.reference_judge not in judges:
            raise ValueError(f"missing reference judge for {dataset}/{approach}")
        common = set.intersection(*(set(rows) for rows in judges.values()))
        if not common:
            raise ValueError(f"no common tasks for {dataset}/{approach}")
        judge_rates = {judge: rates(rows, common) for judge, rows in judges.items()}
        for judge, values in judge_rates.items():
            distributions[(dataset, approach, judge)] = values
        pairwise = {
            f"{left}__{right}": agreement(left_rows, right_rows, common)
            for (left, left_rows), (right, right_rows) in combinations(judges.items(), 2)
        }
        group_report[f"{dataset}::{approach}"] = {
            "tasks": len(common),
            "distributions": judge_rates,
            "pairwise": pairwise,
        }

    judges = sorted({judge for _, _, judge in distributions})
    if len(grouped) != args.expected_groups or len(judges) != args.expected_judges:
        raise ValueError(
            f"expected {args.expected_groups} groups and {args.expected_judges} judges, "
            f"found {len(grouped)} and {len(judges)}"
        )
    macro = {
        judge: {
            label: float(np.mean([values[label] for (dataset, approach, name), values in distributions.items() if name == judge]))
            for label in DISPLAY.values()
        }
        for judge in judges
    }

    correlations: dict[str, object] = {}
    datasets = sorted({dataset for dataset, _ in grouped})
    for dataset in datasets:
        approaches = sorted(approach for name, approach in grouped if name == dataset)
        reference = args.reference_judge
        correlations[dataset] = {}
        for judge in judges:
            if judge == reference:
                continue
            correlations[dataset][judge] = {
                label: float(
                    spearmanr(
                        [distributions[(dataset, approach, reference)][label] for approach in approaches],
                        [distributions[(dataset, approach, judge)][label] for approach in approaches],
                    ).statistic
                )
                for label in DISPLAY.values()
            }

    report = {
        "reference_judge": args.reference_judge,
        "groups": group_report,
        "macro_distributions": macro,
        "spearman_by_dataset": correlations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
