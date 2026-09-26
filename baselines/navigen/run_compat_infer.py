#!/usr/bin/env python3
"""Run official NaviGen inference with a PEFT compatibility fix."""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-script", type=Path, required=True)
    args, remaining = parser.parse_known_args()

    from peft.utils import save_and_load

    original_maybe_shard = save_and_load._maybe_shard_state_dict_for_tp

    def maybe_shard_without_missing_transformers_symbol(model, state_dict, adapter_name):
        has_tp_plan = any(
            getattr(module, "_hf_tp_plan", None) is not None
            and getattr(module, "_hf_device_mesh", None) is not None
            for module in model.modules()
        )
        if not has_tp_plan:
            return None
        return original_maybe_shard(model, state_dict, adapter_name)

    save_and_load._maybe_shard_state_dict_for_tp = maybe_shard_without_missing_transformers_symbol
    sys.argv = [str(args.official_script), *remaining]
    runpy.run_path(str(args.official_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
