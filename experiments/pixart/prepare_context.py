#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import T5EncoderModel, T5Tokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="a high quality photograph")
    parser.add_argument("--max-length", type=int, default=104)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.model / "tokenizer", local_files_only=True)
    encoder = T5EncoderModel.from_pretrained(
        args.model / "text_encoder", torch_dtype=torch.bfloat16, local_files_only=True
    ).to(args.device).eval()
    tokens = tokenizer(
        args.prompt,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        context = encoder(
            input_ids=tokens.input_ids.to(args.device),
            attention_mask=tokens.attention_mask.to(args.device),
        ).last_hidden_state
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prompt": args.prompt,
            "max_length": args.max_length,
            "prompt_embeds": context.cpu().to(torch.float16),
            "prompt_attention_mask": tokens.attention_mask.bool(),
        },
        args.output,
    )
    print({"output": str(args.output), "shape": list(context.shape), "prompt": args.prompt}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
