from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def iter_jsonl(path: Path, *, skip_invalid: bool = False) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                if skip_invalid:
                    continue
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                if skip_invalid:
                    continue
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def completed_indices(path: Path, key: str) -> set[int]:
    # Resume after interrupted writes by retaining every complete record and
    # regenerating only the truncated or missing records.
    return {int(row[key]) for row in iter_jsonl(path, skip_invalid=True) if key in row}


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1].strip()
    candidates = [cleaned]
    match = JSON_OBJECT_RE.search(cleaned)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize_tid(value: Any, maximum: int = 10) -> list[str]:
    if isinstance(value, str):
        values = re.split(r"[,;|\n]", value)
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = []
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        term = re.sub(r"\s+", " ", str(raw)).strip(" \t\r\n.,;:\"'[]{}()")
        if not term or len(term) > 64:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(term)
        if len(result) >= maximum:
            break
    return result


def cid_string(codes: Iterable[int]) -> str:
    values = [int(value) for value in codes]
    if len(values) != 3:
        raise ValueError(f"NaviGen CID requires exactly three levels, got {values}")
    return (
        "<|cid_begin|>"
        f"<s_a_{values[0]}><s_b_{values[1]}><s_c_{values[2]}>"
        "<|cid_end|>"
    )


def cached_image_path(dataset_path: Path, cache_root: Path | None) -> Path:
    if cache_root is None:
        return dataset_path
    parts = dataset_path.parts
    try:
        image_pos = parts.index("images")
    except ValueError:
        return dataset_path
    candidate = cache_root.joinpath(*parts[image_pos + 1 :])
    return candidate if candidate.is_file() else dataset_path
