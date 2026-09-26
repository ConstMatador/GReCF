#!/usr/bin/env python3
"""Add Hugging Face assistant-generation spans to a Qwen3 chat template."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


ASSISTANT_BRANCH = '{%- elif message.role == "assistant" %}'
TOOL_BRANCH = '{%- elif message.role == "tool" %}'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    template = str(tokenizer.chat_template or "")
    changed = False
    if "{% generation %}" not in template and "{%- generation %}" not in template:
        if template.count(ASSISTANT_BRANCH) != 1 or template.count(TOOL_BRANCH) != 1:
            raise ValueError("unexpected Qwen3 chat template; cannot safely add generation spans")
        template = template.replace(ASSISTANT_BRANCH, ASSISTANT_BRANCH + "\n        {%- generation %}", 1)
        template = template.replace(TOOL_BRANCH, "        {%- endgeneration %}\n    " + TOOL_BRANCH, 1)
        tokenizer.chat_template = template
        tokenizer.save_pretrained(args.model_dir)
        changed = True

    probe = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
        {"role": "assistant", "content": '{"target_cid":"<|cid_begin|><s_a_1><s_b_2><s_c_3><|cid_end|>"}'},
    ]
    rendered = tokenizer.apply_chat_template(
        probe,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True,
    )
    assistant_tokens = int(sum(rendered.get("assistant_masks") or []))
    if assistant_tokens <= 0:
        raise ValueError("assistant generation mask is still empty after template patch")
    result = {
        "model_dir": str(args.model_dir),
        "changed": changed,
        "assistant_tokens_in_probe": assistant_tokens,
        "tokenizer_size": len(tokenizer),
    }
    (args.model_dir / "navigen_chat_template_patch.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

