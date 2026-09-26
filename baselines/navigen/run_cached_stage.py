#!/usr/bin/env python3
"""Run an official NaviGen SFT script while reusing validated packed data."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-script", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    spec = importlib.util.spec_from_file_location("navigen_official_sft", known.official_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {known.official_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module._build_packed_tokenized_jsonl

    def cached_packing(*args, **kwargs):
        output = kwargs.get("output_jsonl")
        if output is None and len(args) >= 2:
            output = args[1]
        output = Path(output)
        if output.is_file() and output.stat().st_size > 0:
            print(f"[safe_packing] reuse validated cache: {output}", flush=True)
            return output
        return original(*args, **kwargs)

    module._build_packed_tokenized_jsonl = cached_packing
    sys.argv = [str(known.official_script), *remaining]
    result = module.main()
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
