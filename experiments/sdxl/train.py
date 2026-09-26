#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel
from torch.nn.parallel import DistributedDataParallel

from grecf.data import id_order_sha256, load_cache, load_public_dataset
from grecf.preference_ip_adapter import (
    install_preference_ip_trainable_kv_processors,
    load_preference_ip_processor_state_dict,
    preference_ip_processor_state_dict,
    preference_ip_trainable_parameters,
)
from grecf.preference_ip_adapter import preference_ip_processor_gate_summary
from grecf.sdxl import SDXLPreferenceAdapter


def load_interests(path: Path) -> dict[str, np.ndarray]:
    return {
        "prototypes": np.load(path / "adaptive_b_prototypes.float32.npy", mmap_mode="r"),
        "weights": np.load(path / "adaptive_b_weights.float32.npy", mmap_mode="r"),
        "mask": np.load(path / "adaptive_b_mask.bool.npy", mmap_mode="r"),
        "auxiliary": np.load(path / "adaptive_b_auxiliary.float32.npy", mmap_mode="r"),
    }


def build_inputs(
    users, targets, prototypes, masks, weights, auxiliary, catalog, probability,
    anchor_selection, routing, teacher_probability, generator,
):
    candidates = prototypes[users]
    active = masks[users]
    scores = torch.einsum("bkd,bd->bk", candidates.float(), catalog[targets].float()).masked_fill(~active, -torch.inf)
    target_chosen = scores.argmax(dim=1)
    sampling_weights = weights[users].float().masked_fill(~active, 0.0)
    sampling_weights = sampling_weights / sampling_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    sampled = torch.multinomial(sampling_weights, 1, generator=generator).squeeze(1)
    if routing == "target":
        chosen = target_chosen
    elif routing == "evidence":
        chosen = sampled
    else:
        use_target = torch.rand(len(users), generator=generator, device=users.device) < teacher_probability
        chosen = torch.where(use_target, target_chosen, sampled)
    encoder_mask = active.clone()
    user_auxiliary = auxiliary[users].clone()
    row = torch.arange(len(users), device=users.device)
    context_mask = active & ~F.one_hot(target_chosen, num_classes=candidates.shape[1]).bool()
    can_mask = context_mask.any(dim=1)
    use_masked = (torch.rand(len(users), generator=generator, device=users.device) < probability) & can_mask
    if bool(use_masked.any()):
        if anchor_selection == "nearest":
            anchors = scores.masked_fill(~context_mask, -torch.inf).argmax(dim=1)
        else:
            random_scores = torch.rand(context_mask.shape, generator=generator, device=users.device).masked_fill(~context_mask, -torch.inf)
            anchors = random_scores.argmax(dim=1)
        encoder_mask = torch.where(use_masked[:, None], context_mask, encoder_mask)
        context_weights = weights[users].float().masked_fill(~context_mask, 0.0)
        context_auxiliary = torch.einsum("bk,bkd->bd", context_weights, candidates.float())
        context_auxiliary = F.normalize(context_auxiliary / context_weights.sum(dim=1, keepdim=True).clamp_min(1e-8), dim=-1)
        user_auxiliary = torch.where(use_masked[:, None], context_auxiliary, user_auxiliary)
        chosen = torch.where(use_masked, anchors, chosen)
    return candidates, encoder_mask, user_auxiliary, candidates[row, chosen], int(use_masked.sum().item())


def combine_attention_kwargs(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: (
            torch.cat((first[key], second[key]), dim=0)
            if key.endswith("_tokens") or key.endswith("_mask")
            else first[key]
        )
        for key in first
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--interest-cache-dir", type=Path, required=True)
    parser.add_argument("--base-context", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-users", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--masked-interest-prob", type=float, default=0.5)
    parser.add_argument("--masked-interest-prob-start", type=float, default=None)
    parser.add_argument("--masked-anchor-selection", choices=("random", "nearest"), default="random")
    parser.add_argument("--interest-routing", choices=("target", "mixed", "evidence"), default="target")
    parser.add_argument("--interest-teacher-forcing-prob", type=float, default=1.0)
    parser.add_argument("--interest-teacher-forcing-prob-start", type=float, default=None)
    parser.add_argument("--condition-margin-weight", type=float, default=0.0)
    parser.add_argument("--condition-margin", type=float, default=0.01)
    parser.add_argument("--user-ce-weight", type=float, default=0.05)
    parser.add_argument("--residual-reg-weight", type=float, default=1e-7)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if not 0.0 <= args.interest_teacher_forcing_prob <= 1.0:
        raise ValueError("--interest-teacher-forcing-prob must be in [0, 1]")
    if args.interest_teacher_forcing_prob_start is not None and not 0.0 <= args.interest_teacher_forcing_prob_start <= 1.0:
        raise ValueError("--interest-teacher-forcing-prob-start must be in [0, 1]")
    if args.interest_teacher_forcing_prob_start is not None and args.interest_routing != "mixed":
        raise ValueError("teacher-forcing scheduling requires --interest-routing=mixed")
    if args.condition_margin_weight < 0.0 or args.condition_margin < 0.0:
        raise ValueError("condition-margin values must be non-negative")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        rank = dist.get_rank() if dist.is_initialized() else 0
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        if not dist.is_initialized():
            dist.init_process_group("nccl", device_id=device)
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world, device = 0, 1, torch.device("cuda:0")
        torch.cuda.set_device(device)
    main_rank = rank == 0
    torch.manual_seed(args.seed + rank * 1_000_003)
    np.random.seed((args.seed + rank) % (2**32))
    dtype = torch.bfloat16

    dataset = load_public_dataset(args.dataset_root)
    order_sha = id_order_sha256(dataset.image_ids)
    clip = load_cache(args.clip_cache, dataset.num_items, order_sha, (512,))
    latents = load_cache(args.latent_cache, dataset.num_items, order_sha, (4, 64, 64))
    interest = load_interests(args.interest_cache_dir)
    state = torch.load(args.base_context, map_location="cpu", weights_only=True)
    catalog = torch.from_numpy(np.asarray(clip, dtype=np.float32)).to(device, dtype=dtype)
    prototypes = torch.from_numpy(np.asarray(interest["prototypes"], dtype=np.float32)).to(device)
    weights = torch.from_numpy(np.asarray(interest["weights"], dtype=np.float32)).to(device)
    masks = torch.from_numpy(np.asarray(interest["mask"], dtype=bool)).to(device)
    auxiliary = torch.from_numpy(np.asarray(interest["auxiliary"], dtype=np.float32)).to(device)
    adapter_base = SDXLPreferenceAdapter(
        state["prompt_embeds"], state["prompt_attention_mask"],
        auxiliary.mean(dim=0, keepdim=True),
    ).to(device)
    initial = None
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        adapter_base.load_state_dict(initial["adapter"], strict=True)
    adapter = DistributedDataParallel(adapter_base, device_ids=[local_rank], find_unused_parameters=True) if distributed else adapter_base
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet", torch_dtype=dtype, local_files_only=True).to(device).eval()
    unet.requires_grad_(False)
    install_preference_ip_trainable_kv_processors(unet, layer_gates=True)
    if initial is not None:
        load_preference_ip_processor_state_dict(unet, initial["preference_ip_processor_state"])
    unet_trainable = preference_ip_trainable_parameters(unet)
    unet.enable_gradient_checkpointing()
    unet_base = unet
    unet = DistributedDataParallel(unet, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True) if distributed else unet
    optimizer = torch.optim.AdamW(list(adapter_base.parameters()) + list(unet_trainable), lr=args.lr, weight_decay=1e-5)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", prediction_type="epsilon")
    edge_users = np.repeat(np.arange(dataset.num_users, dtype=np.int32), [len(x) for x in dataset.train])
    edge_items = np.concatenate(dataset.train).astype(np.int32, copy=False)
    global_batch = args.batch_users * world
    steps_per_epoch = math.ceil(len(edge_users) / global_batch)
    planned_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        planned_steps = min(planned_steps, args.max_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train.jsonl"
    if main_rank and log_path.exists():
        log_path.unlink()
    if distributed:
        dist.barrier(device_ids=[device.index])
    generator = torch.Generator(device=device).manual_seed(args.seed + rank * 10_007)
    time_ids = torch.tensor([512, 512, 0, 0, 512, 512], dtype=dtype, device=device).view(1, 6)
    started = time.time()
    step = 0
    for epoch in range(args.epochs):
        order = np.random.default_rng(args.seed + epoch).permutation(len(edge_users))
        padded = steps_per_epoch * global_batch
        if padded > len(order):
            order = np.concatenate((order, order[: padded - len(order)]))
        for batch_start in range(0, len(order), global_batch):
            if step >= planned_steps:
                break
            global_indices = order[batch_start : batch_start + global_batch]
            local_indices = global_indices[rank * args.batch_users : (rank + 1) * args.batch_users]
            if len(local_indices) != args.batch_users:
                raise RuntimeError(
                    f"rank {rank} received {len(local_indices)} samples at global batch "
                    f"offset {batch_start}; expected {args.batch_users}"
                )
            step += 1
            users = torch.from_numpy(edge_users[local_indices].astype(np.int64)).to(device)
            targets = torch.from_numpy(edge_items[local_indices].astype(np.int64)).to(device)
            clean = torch.from_numpy(np.asarray(latents[edge_items[local_indices]], dtype=np.float32)).to(device, dtype=dtype)
            mask_probability = args.masked_interest_prob
            schedule_fraction = (step - 1) / max(planned_steps - 1, 1)
            if args.masked_interest_prob_start is not None and planned_steps > 1:
                mask_probability = args.masked_interest_prob_start + schedule_fraction * (
                    args.masked_interest_prob - args.masked_interest_prob_start
                )
            teacher_probability = args.interest_teacher_forcing_prob
            if args.interest_teacher_forcing_prob_start is not None:
                teacher_probability = args.interest_teacher_forcing_prob_start + schedule_fraction * (
                    args.interest_teacher_forcing_prob - args.interest_teacher_forcing_prob_start
                )
            candidates, encoder_mask, user_aux, selected_interest, masked_count = build_inputs(
                users, targets, prototypes, masks, weights, auxiliary, catalog,
                mask_probability, args.masked_anchor_selection, args.interest_routing,
                teacher_probability, generator,
            )
            wrong_inputs = None
            if args.condition_margin_weight > 0.0:
                wrong_offsets = torch.randint(1, dataset.num_users, (len(users),), generator=generator, device=device)
                wrong_users = (users + wrong_offsets) % dataset.num_users
                wrong_inputs = build_inputs(
                    wrong_users, targets, prototypes, masks, weights, auxiliary, catalog,
                    mask_probability, args.masked_anchor_selection, args.interest_routing,
                    teacher_probability, generator,
                )[:4]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                context, residual, attention_kwargs = adapter(candidates, encoder_mask, user_aux, selected_interest)
                noise = torch.randn(clean.shape, generator=generator, device=device, dtype=dtype)
                timesteps = torch.randint(20, 981, (len(users),), generator=generator, device=device)
                noisy = noise_scheduler.add_noise(clean, noise, timesteps)
                added = {"text_embeds": state["pooled_prompt_embeds"].expand(len(users), -1).to(device=device, dtype=dtype), "time_ids": time_ids.expand(len(users), -1)}
                if wrong_inputs is not None:
                    wrong_context, _, wrong_attention = adapter(*wrong_inputs)
                    added_pair = {
                        "text_embeds": state["pooled_prompt_embeds"].expand(2 * len(users), -1).to(device=device, dtype=dtype),
                        "time_ids": time_ids.expand(2 * len(users), -1),
                    }
                    predicted_pair = unet(
                        torch.cat((noisy, noisy)), torch.cat((timesteps, timesteps)),
                        encoder_hidden_states=torch.cat((context, wrong_context)).to(dtype),
                        encoder_attention_mask=state["prompt_attention_mask"].expand(2 * len(users), -1).to(device),
                        added_cond_kwargs=added_pair,
                        cross_attention_kwargs=combine_attention_kwargs(attention_kwargs, wrong_attention),
                    ).sample
                    predicted, wrong_predicted = predicted_pair.chunk(2)
                    correct_error = (predicted.float() - noise.float()).square().flatten(1).mean(1)
                    wrong_error = (wrong_predicted.float() - noise.float()).square().flatten(1).mean(1)
                    condition_margin_loss = F.relu(args.condition_margin + correct_error - wrong_error).mean()
                    condition_win_rate = (correct_error < wrong_error).float().mean()
                else:
                    predicted = unet(noisy, timesteps, encoder_hidden_states=context.to(dtype), encoder_attention_mask=state["prompt_attention_mask"].expand(len(users), -1).to(device), added_cond_kwargs=added, cross_attention_kwargs=attention_kwargs).sample
                    condition_margin_loss = torch.zeros((), device=device)
                    condition_win_rate = torch.zeros((), device=device)
                mse = F.mse_loss(predicted.float(), noise.float())
                user_ce, user_accuracy = adapter_base.user_contrastive_loss(residual, encoder_mask, user_aux, selected_interest, users)
                loss = mse + args.user_ce_weight * user_ce + args.condition_margin_weight * condition_margin_loss + args.residual_reg_weight * residual.float().square().mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(list(adapter_base.parameters()) + list(unet_trainable), 1.0)
            optimizer.step()
            if main_rank and (step == 1 or step % args.log_every == 0 or step == planned_steps):
                row = {"step": step, "max_steps": planned_steps, "epoch": epoch + 1, "loss": float(loss), "mse": float(mse), "user_ce": float(user_ce), "user_accuracy": float(user_accuracy), "condition_margin_loss": float(condition_margin_loss), "condition_win_rate": float(condition_win_rate), "masked_fraction": masked_count / max(len(users), 1), "configured_mask_probability": mask_probability, "configured_teacher_forcing_probability": teacher_probability, "grad_norm": float(grad_norm), "seen_interactions": min(len(edge_users) * args.epochs, step * global_batch), "elapsed_seconds": round(time.time() - started, 2)}
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if main_rank and (step % args.checkpoint_every == 0 or step == planned_steps):
                payload = {"adapter": adapter_base.state_dict(), "preference_ip_processor_state": preference_ip_processor_state_dict(unet_base), "step": step, "config": vars(args), "training": {"world_size": world, "batch_users_per_gpu": args.batch_users, "seen_interactions": step * global_batch, "full_train_interactions": len(edge_users), "processed_interactions_including_final_padding": planned_steps * global_batch, "epochs": args.epochs, "full_coverage": True, "attention_route": "extra_preference_ip_kv_with_layer_gates", "preference_gate_summary": preference_ip_processor_gate_summary(unet_base), "elapsed_seconds": time.time() - started}}
                torch.save(payload, args.output_dir / f"step_{step:06d}.pt")
                torch.save(payload, args.output_dir / "last.pt")
        if distributed:
            dist.barrier(device_ids=[device.index])
        if step >= planned_steps:
            break
    if main_rank:
        write_json(args.output_dir / "training_summary.json", {"world_size": world, "batch_users_per_gpu": args.batch_users, "seen_interactions": planned_steps * global_batch, "full_train_interactions": len(edge_users), "processed_interactions_including_final_padding": planned_steps * global_batch, "epochs": args.epochs, "steps": planned_steps, "full_coverage": True, "elapsed_seconds": time.time() - started})
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
