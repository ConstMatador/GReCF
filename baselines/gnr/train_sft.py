#!/usr/bin/env python3
"""GNR step 2: SFT Janus-Pro-1B for personalized next-item image generation.

Follows arXiv 2506.01704: input is the user's image history plus an
instruction; the target is the VQ token sequence of the real next item and
the loss is cross-entropy on the image tokens only. The understanding vision
encoder (SigLIP) and the VQ codec stay frozen; language model, aligners and
gen head are fine-tuned. Runs DDP via torchrun.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset

INSTRUCTION = (
    "\nBased on the user's interaction history above, "
    "generate the image of the next item this user would like."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--janus-root", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=Path("/path/to/models/Janus-Pro-1B"))
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


class SftWindows(Dataset):
    def __init__(self, path: Path):
        self.rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def build_conversation(row: dict) -> list[dict]:
    content = "<image_placeholder>" * len(row["history_paths"]) + INSTRUCTION
    return [
        {"role": "User", "content": content, "images": row["history_paths"]},
        {"role": "Assistant", "content": ""},
    ]


def collate(batch: list[dict], processor: VLChatProcessor, image_size: int):
    images = [[Image.open(p).convert("RGB").resize((image_size, image_size))
               for p in row["history_paths"]] for row in batch]
    prepare_list = [
        processor.process_one(conversations=build_conversation(row), images=imgs)
        for row, imgs in zip(batch, images)
    ]
    prepare = processor.batchify(prepare_list)
    targets = torch.stack([
        torch.from_numpy(np.asarray(Image.open(row["target_path"]).convert("RGB")
                                    .resize((image_size, image_size))))
        for row in batch
    ])
    targets = targets.to(torch.float32).permute(0, 3, 1, 2) / 127.5 - 1.0
    return prepare, targets


def encode_target_codes(model: MultiModalityCausalLM, targets: torch.Tensor):
    # VQ-16: quant [b, 8, 24, 24] -> codes [b, 576]
    quant, _, info = model.gen_vision_model.encode(targets.to(next(model.parameters()).dtype))
    codes = info[-1].view(targets.shape[0], -1)
    return codes


def prepare_inputs_embeds(model: MultiModalityCausalLM, prepare):
    """Janus prepare_inputs_embeds with an explicit dtype align between the
    fp32 word embeddings and the autocast (bf16) vision branch."""
    from einops import rearrange
    device = next(model.parameters()).device
    input_ids = prepare.input_ids.to(device).clone()
    input_ids[input_ids < 0] = 0
    inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
    pixel_values = prepare.pixel_values.to(device)
    bs, n = pixel_values.shape[0:2]
    images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
    images_embeds = model.aligner(model.vision_model(images))
    images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
    emb_mask = rearrange(prepare.images_emb_mask.to(device), "b n t -> b (n t)")
    images_embeds = images_embeds.to(inputs_embeds.dtype)
    inputs_embeds[prepare.images_seq_mask.to(device)] = images_embeds[emb_mask]
    return inputs_embeds


def compute_loss(model: MultiModalityCausalLM, prepare, targets: torch.Tensor, pad_id: int):
    device = targets.device

    with torch.no_grad():
        codes = encode_target_codes(model, targets)

    und_embeds = prepare_inputs_embeds(model, prepare)
    bsz, seq_len, dim = und_embeds.shape
    n_codes = codes.shape[1]
    # append the first n_codes-1 gen token embeds after the understanding part
    gen_embeds = model.prepare_gen_img_embeds(codes[:, :-1])
    full_embeds = torch.cat([und_embeds, gen_embeds], dim=1)
    hidden = model.language_model.model(inputs_embeds=full_embeds).last_hidden_state
    logits = model.gen_head(hidden[:, seq_len - 1 :, :])
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(bsz * n_codes, -1).float(), codes.reshape(bsz * n_codes))
    return loss


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.janus_root))
    from janus.models import MultiModalityCausalLM, VLChatProcessor

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    is_main = rank == 0

    torch.manual_seed(args.seed + rank)

    processor = VLChatProcessor.from_pretrained(args.model)
    model = MultiModalityCausalLM.from_pretrained(args.model, trust_remote_code=True).to(device)
    model.vision_model.requires_grad_(False)
    model.gen_vision_model.requires_grad_(False)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if is_main:
        n_tr = sum(p.numel() for p in trainable)
        print(json.dumps({"trainable_params_m": round(n_tr / 1e6, 1),
                          "world_size": world}), flush=True)

    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    train_set = SftWindows(args.train_jsonl)
    sampler = (torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True, seed=args.seed)
               if world > 1 else None)
    loader = DataLoader(train_set, batch_size=args.micro_batch_size,
                        sampler=sampler, shuffle=sampler is None, num_workers=4,
                        collate_fn=lambda b: collate(b, processor, args.image_size),
                        drop_last=True, pin_memory=True)

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    if is_main:
        print(json.dumps({"windows": len(train_set), "batches": len(loader),
                          "steps_per_epoch": steps_per_epoch, "total_steps": total_steps}), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"

    step = 0
    micro = 0
    running = 0.0
    t0 = time.time()
    done = False
    while not done:
        if sampler is not None:
            sampler.set_epoch(step)
        for prepare, targets in loader:
            targets = targets.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = compute_loss(model, prepare, targets, processor.pad_id)
            (loss / args.grad_accum).backward()
            running += loss.item()
            micro += 1
            if micro % args.grad_accum != 0:
                continue
            if world > 1:
                for parameter in trainable:
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                        parameter.grad.div_(world)
            lr = lr_at(step)
            for group in optim.param_groups:
                group["lr"] = lr
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optim.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if is_main and step % args.log_every == 0:
                elapsed = time.time() - t0
                record = {"step": step, "total_steps": total_steps,
                          "loss": round(running / (args.log_every * args.grad_accum), 4),
                          "lr": lr, "sec_per_step": round(elapsed / args.log_every, 2),
                          "eta_min": round(elapsed / args.log_every * (total_steps - step) / 60, 1)}
                running = 0.0
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            if step >= total_steps:
                done = True
                break

    if is_main:
        ckpt_dir = args.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = model
        state = {k: v.to(torch.bfloat16) for k, v in model_to_save.state_dict().items()}
        torch.save(state, ckpt_dir / "sft_full_bf16.pth")
        (ckpt_dir / "train_meta.json").write_text(json.dumps({
            "base_model": str(args.model), "steps": step, "lr": args.lr,
            "epochs": args.epochs, "micro_batch": args.micro_batch_size,
            "grad_accum": args.grad_accum, "world_size": world,
            "windows": len(train_set), "seed": args.seed,
            "instruction": INSTRUCTION,
        }, indent=2) + "\n", encoding="utf-8")
        print("SFT_DONE", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
