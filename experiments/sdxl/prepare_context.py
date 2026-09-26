#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


def encode(tokenizer, encoder, prompt: str, device: str, max_length: int = 77):
    tokens = tokenizer(prompt, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = encoder(input_ids=tokens.input_ids.to(device), attention_mask=tokens.attention_mask.to(device), output_hidden_states=True)
    hidden = out.hidden_states[-2].float()
    pooled = getattr(out, "text_embeds", None)
    return hidden, (pooled.float() if pooled is not None else hidden[:, 0]) , tokens.attention_mask.bool()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="a high quality photograph")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    tokenizer = CLIPTokenizer.from_pretrained(args.model / "tokenizer", local_files_only=True)
    tokenizer2 = CLIPTokenizer.from_pretrained(args.model / "tokenizer_2", local_files_only=True)
    encoder = CLIPTextModel.from_pretrained(args.model / "text_encoder", torch_dtype=torch.bfloat16, local_files_only=True).to(args.device).eval()
    encoder2 = CLIPTextModelWithProjection.from_pretrained(args.model / "text_encoder_2", torch_dtype=torch.bfloat16, local_files_only=True).to(args.device).eval()
    for module in (encoder, encoder2):
        module.requires_grad_(False)
    h1, _, m1 = encode(tokenizer, encoder, args.prompt, args.device)
    h2, p2, m2 = encode(tokenizer2, encoder2, args.prompt, args.device)
    n1, _, nm1 = encode(tokenizer, encoder, "", args.device)
    n2, np2, nm2 = encode(tokenizer2, encoder2, "", args.device)
    payload = {
        "prompt": args.prompt,
        "prompt_embeds": torch.cat((h1, h2), dim=-1).cpu().to(torch.float16),
        "prompt_attention_mask": (m1 & m2).cpu(),
        "pooled_prompt_embeds": p2.cpu().to(torch.float16),
        "negative_prompt_embeds": torch.cat((n1, n2), dim=-1).cpu().to(torch.float16),
        "negative_prompt_attention_mask": (nm1 & nm2).cpu(),
        "negative_pooled_prompt_embeds": np2.cpu().to(torch.float16),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print({"output": str(args.output), "prompt_shape": list(payload["prompt_embeds"].shape), "pooled_shape": list(payload["pooled_prompt_embeds"].shape)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
