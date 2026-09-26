#!/usr/bin/env python3
"""Launch the official NaviGen GRPO trainer with compatibility fixes."""
from __future__ import annotations

import argparse
import importlib.util
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", help="Path to the official NaviGen GRPO entry point")
    parser.add_argument(
        "--grpo-max-steps",
        type=int,
        default=600,
        help="Optimization budget forwarded to GRPOConfig.",
    )
    launcher_args, official_args = parser.parse_known_args()

    import unsloth  # noqa: F401
    from transformers import Trainer
    import trl.trainer.grpo_trainer as grpo_module

    if not hasattr(grpo_module, "Trainer"):
        grpo_module.Trainer = Trainer
    if not hasattr(grpo_module, "truncate_with_protected_tokens"):
        def truncate_with_protected_tokens(input_ids, attention_mask, max_length, protected_tokens):
            del protected_tokens
            if input_ids.shape[1] <= max_length:
                return input_ids, attention_mask
            return input_ids[:, -max_length:], attention_mask[:, -max_length:]

        grpo_module.truncate_with_protected_tokens = truncate_with_protected_tokens

    script = launcher_args.script
    sys.argv = [script, *official_args]
    spec = importlib.util.spec_from_file_location("navigen_official_grpo", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official GRPO script: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    original_build_grpo_config = module.build_grpo_config

    def build_grpo_config_with_explicit_budget(**kwargs):
        kwargs.setdefault("max_steps", launcher_args.grpo_max_steps)
        return original_build_grpo_config(**kwargs)

    module.build_grpo_config = build_grpo_config_with_explicit_budget
    trainer_cls = getattr(module, "ConstrainedCidGRPOTrainer", None)
    if trainer_cls is None:
        raise RuntimeError("official GRPO trainer class was not defined")

    original_init = trainer_cls.__init__

    def init_without_multimodal_tokens(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.image_token_id = None
        self.vision_start_token_id = None
        self.vision_end_token_id = None
        self.image_token = None

    trainer_cls.__init__ = init_without_multimodal_tokens
    result = module.main()
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
