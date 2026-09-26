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
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, PixArtTransformer2DModel
from torch.nn.parallel import DistributedDataParallel

from grecf.data import id_order_sha256, load_cache, load_public_dataset
from grecf.pixart import PixArtPrefixAdapter


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--interest-cache-dir", type=Path, required=True)
    parser.add_argument("--base-context", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional adapter checkpoint to continue training from.",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--sampling-mode",
        choices=("user_balanced", "all_interactions"),
        default="all_interactions",
        help=(
            "user_balanced samples one interaction per user per round; all_interactions "
            "visits every train edge once per epoch before any final-batch padding."
        ),
    )
    parser.add_argument("--batch-users", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--user-ce-weight", type=float, default=0.05)
    parser.add_argument("--residual-reg-weight", type=float, default=1e-7)
    parser.add_argument("--masked-interest-prob", type=float, default=0.5)
    parser.add_argument("--masked-interest-prob-start", type=float, default=None)
    parser.add_argument("--masked-anchor-selection", choices=("random", "nearest"), default="random")
    parser.add_argument("--interest-routing", choices=("target", "mixed", "evidence"), default="target")
    parser.add_argument("--interest-teacher-forcing-prob", type=float, default=1.0)
    parser.add_argument("--interest-teacher-forcing-prob-start", type=float, default=None)
    parser.add_argument("--condition-margin-weight", type=float, default=0.0)
    parser.add_argument("--condition-margin", type=float, default=0.01)
    parser.add_argument("--timestep-min", type=int, default=20)
    parser.add_argument("--timestep-max", type=int, default=980)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def load_interest_cache(path: Path) -> dict[str, np.ndarray]:
    return {
        "prototypes": np.load(path / "adaptive_b_prototypes.float32.npy", mmap_mode="r"),
        "weights": np.load(path / "adaptive_b_weights.float32.npy", mmap_mode="r"),
        "mask": np.load(path / "adaptive_b_mask.bool.npy", mmap_mode="r"),
        "auxiliary": np.load(path / "adaptive_b_auxiliary.float32.npy", mmap_mode="r"),
    }


def build_inputs(
    users: torch.Tensor,
    targets: torch.Tensor,
    prototypes: torch.Tensor,
    masks: torch.Tensor,
    weights: torch.Tensor,
    auxiliary: torch.Tensor,
    catalog: torch.Tensor,
    mask_probability: float,
    anchor_selection: str,
    routing: str,
    teacher_probability: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    candidates = prototypes[users]
    active = masks[users]
    target_features = catalog[targets].float()
    scores = torch.einsum("bkd,bd->bk", candidates.float(), target_features).masked_fill(~active, -torch.inf)
    target_chosen = scores.argmax(dim=1)
    sampling_weights = weights[users].float().masked_fill(~active, 0.0)
    sampling_weights = sampling_weights / sampling_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    sampled = torch.multinomial(sampling_weights, 1, generator=generator).squeeze(1)
    if routing == "target":
        chosen = target_chosen.clone()
    elif routing == "evidence":
        chosen = sampled
    else:
        use_target = torch.rand(len(users), generator=generator, device=users.device) < teacher_probability
        chosen = torch.where(use_target, target_chosen, sampled)
    encoder_mask = active.clone()
    user_auxiliary = auxiliary[users].clone()
    row = torch.arange(len(users), device=users.device)
    target_one_hot = F.one_hot(target_chosen, num_classes=candidates.shape[1]).bool()
    context_mask = active & ~target_one_hot
    can_mask = context_mask.any(dim=1)
    use_masked = (
        torch.rand(len(users), generator=generator, device=users.device) < float(mask_probability)
    ) & can_mask
    if bool(use_masked.any()):
        if anchor_selection == "nearest":
            anchors = scores.masked_fill(~context_mask, -torch.inf).argmax(dim=1)
        else:
            random_scores = torch.rand(context_mask.shape, generator=generator, device=users.device)
            random_scores.masked_fill_(~context_mask, -torch.inf)
            anchors = random_scores.argmax(dim=1)
        chosen = torch.where(use_masked, anchors, chosen)
        encoder_mask = torch.where(use_masked[:, None], context_mask, encoder_mask)
        context_weights = weights[users].float().masked_fill(~context_mask, 0.0)
        context_auxiliary = torch.einsum("bk,bkd->bd", context_weights, candidates.float())
        context_auxiliary = F.normalize(
            context_auxiliary / context_weights.sum(dim=1, keepdim=True).clamp_min(1e-8), dim=-1
        )
        user_auxiliary = torch.where(use_masked[:, None], context_auxiliary, user_auxiliary)
    interests = candidates[row, chosen]
    return candidates, encoder_mask, user_auxiliary, interests, int(use_masked.sum().item())


def main() -> int:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.sampling_mode == "user_balanced" and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive with user_balanced sampling")
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
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        local_rank = rank = 0
        world = 1
        torch.cuda.set_device(0)
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0
    torch.manual_seed(args.seed + rank * 1_000_003)
    np.random.seed((args.seed + rank) % (2**32))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    dtype = torch.bfloat16

    dataset = load_public_dataset(args.dataset_root)
    order_sha = id_order_sha256(dataset.image_ids)
    clip = load_cache(args.clip_cache, dataset.num_items, order_sha, (512,))
    latents = load_cache(args.latent_cache, dataset.num_items, order_sha, (4, 64, 64))
    interests = load_interest_cache(args.interest_cache_dir)
    context_state = torch.load(args.base_context, map_location="cpu", weights_only=True)

    catalog = torch.from_numpy(np.asarray(clip, dtype=np.float32)).to(device=device, dtype=dtype)
    prototypes = torch.from_numpy(np.asarray(interests["prototypes"], dtype=np.float32)).to(device)
    masks = torch.from_numpy(np.asarray(interests["mask"], dtype=bool)).to(device)
    weights = torch.from_numpy(np.asarray(interests["weights"], dtype=np.float32)).to(device)
    auxiliary = torch.from_numpy(np.asarray(interests["auxiliary"], dtype=np.float32)).to(device)

    adapter_base = PixArtPrefixAdapter(
        context_state["prompt_embeds"].float(),
        context_state["prompt_attention_mask"].bool(),
        auxiliary_center=auxiliary.mean(dim=0, keepdim=True),
    ).to(device)
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if "adapter" not in initial:
            raise ValueError(f"checkpoint has no adapter state: {args.init_checkpoint}")
        adapter_base.load_state_dict(initial["adapter"], strict=True)
    adapter = DistributedDataParallel(adapter_base, device_ids=[local_rank]) if distributed else adapter_base

    transformer = PixArtTransformer2DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=dtype, local_files_only=True
    ).to(device).eval()
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    scheduler_config = DPMSolverMultistepScheduler.load_config(
        args.model, subfolder="scheduler", local_files_only=True
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=int(scheduler_config["num_train_timesteps"]),
        beta_start=float(scheduler_config["beta_start"]),
        beta_end=float(scheduler_config["beta_end"]),
        beta_schedule=str(scheduler_config["beta_schedule"]),
        prediction_type="epsilon",
    )
    trainable_adapter_parameters = [
        parameter for parameter in adapter_base.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_adapter_parameters, lr=args.lr, weight_decay=1e-5)

    edge_users = np.repeat(
        np.arange(dataset.num_users, dtype=np.int32), [len(items) for items in dataset.train]
    )
    edge_items = np.concatenate(dataset.train).astype(np.int32, copy=False)
    global_batch = args.batch_users * world
    if args.sampling_mode == "all_interactions":
        steps_per_epoch = math.ceil(len(edge_users) / global_batch)
        full_steps = steps_per_epoch * args.epochs
        planned_steps = full_steps if args.max_steps <= 0 else min(args.max_steps, full_steps)
        epochs: list[np.ndarray] = []
        for epoch in range(args.epochs):
            epoch_order = np.random.default_rng(args.seed + epoch).permutation(len(edge_users))
            padded = steps_per_epoch * global_batch
            if padded > len(epoch_order):
                epoch_order = np.concatenate((epoch_order, epoch_order[: padded - len(epoch_order)]))
            epochs.append(epoch_order)
        flat_order = np.concatenate(epochs)[: planned_steps * global_batch]
    else:
        planned_steps = args.max_steps
        required = planned_steps * global_batch
        user_offsets = np.concatenate(([0], np.cumsum([len(items) for items in dataset.train])))
        rounds: list[np.ndarray] = []
        for epoch in range(math.ceil(required / dataset.num_users)):
            rng = np.random.default_rng(args.seed + epoch)
            selected_edges = np.asarray(
                [rng.integers(user_offsets[user], user_offsets[user + 1]) for user in range(dataset.num_users)],
                dtype=np.int64,
            )
            rounds.append(rng.permutation(selected_edges))
        flat_order = np.concatenate(rounds)[:required]
        full_steps = planned_steps
    order = flat_order.reshape(planned_steps, world, args.batch_users)
    generator = torch.Generator(device=device).manual_seed(args.seed + rank * 10_007)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train.jsonl"
    if is_main and log_path.exists():
        log_path.unlink()
    if distributed:
        dist.barrier(device_ids=[local_rank])
    started = time.time()

    for step in range(1, planned_steps + 1):
        local = order[step - 1, rank]
        users_np = edge_users[local]
        items_np = edge_items[local]
        users = torch.from_numpy(users_np.astype(np.int64)).to(device)
        targets = torch.from_numpy(items_np.astype(np.int64)).to(device)
        clean = torch.from_numpy(np.asarray(latents[items_np], dtype=np.float32)).to(device, dtype=dtype)
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
        inputs = build_inputs(
            users, targets, prototypes, masks, weights, auxiliary, catalog,
            mask_probability, args.masked_anchor_selection, args.interest_routing,
            teacher_probability, generator,
        )
        candidates, encoder_mask, user_auxiliary, selected_interest, masked_count = inputs
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
            context, context_mask, residual = adapter(
                candidates, encoder_mask, user_auxiliary, selected_interest
            )
            noise = torch.randn(clean.shape, generator=generator, device=device, dtype=dtype)
            timesteps = torch.randint(
                args.timestep_min, args.timestep_max + 1, (len(users),), generator=generator, device=device
            )
            noisy = noise_scheduler.add_noise(clean, noise, timesteps)
            if wrong_inputs is not None:
                wrong_context, wrong_context_mask, _ = adapter(*wrong_inputs)
                prediction_pair = transformer(
                    hidden_states=torch.cat((noisy, noisy)),
                    encoder_hidden_states=torch.cat((context, wrong_context)).to(dtype),
                    encoder_attention_mask=torch.cat((context_mask, wrong_context_mask)),
                    timestep=torch.cat((timesteps, timesteps)),
                ).sample
                if prediction_pair.shape[1] == clean.shape[1] * 2:
                    prediction_pair = prediction_pair.chunk(2, dim=1)[0]
                prediction, wrong_prediction = prediction_pair.chunk(2)
                correct_error = (prediction.float() - noise.float()).square().flatten(1).mean(1)
                wrong_error = (wrong_prediction.float() - noise.float()).square().flatten(1).mean(1)
                condition_margin_loss = F.relu(args.condition_margin + correct_error - wrong_error).mean()
                condition_win_rate = (correct_error < wrong_error).float().mean()
            else:
                prediction = transformer(
                    hidden_states=noisy,
                    encoder_hidden_states=context.to(dtype),
                    encoder_attention_mask=context_mask,
                    timestep=timesteps,
                ).sample
                if prediction.shape[1] == clean.shape[1] * 2:
                    prediction = prediction.chunk(2, dim=1)[0]
                condition_margin_loss = torch.zeros((), device=device)
                condition_win_rate = torch.zeros((), device=device)
            mse = F.mse_loss(prediction.float(), noise.float())
            user_ce, user_accuracy = adapter_base.user_contrastive_loss(
                residual, encoder_mask, user_auxiliary, selected_interest, users
            )
            residual_reg = residual.float().square().mean()
            loss = mse + args.user_ce_weight * user_ce + args.condition_margin_weight * condition_margin_loss + args.residual_reg_weight * residual_reg
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_adapter_parameters, args.grad_clip)
        optimizer.step()

        if is_main and (step == 1 or step % args.log_every == 0):
            row = {
                "step": step,
                "max_steps": planned_steps,
                "loss": float(loss.item()),
                "mse": float(mse.item()),
                "user_ce": float(user_ce.item()),
                "user_accuracy": float(user_accuracy.item()),
                "condition_margin_loss": float(condition_margin_loss.item()),
                "condition_win_rate": float(condition_win_rate.item()),
                "masked_fraction": masked_count / max(len(users), 1),
                "configured_mask_probability": mask_probability,
                "configured_teacher_forcing_probability": teacher_probability,
                "grad_norm": float(grad_norm),
                "elapsed_seconds": round(time.time() - started, 2),
                "gpu_memory_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if is_main and (step % args.checkpoint_every == 0 or step == planned_steps):
            checkpoint = {
                "adapter": adapter_base.state_dict(),
                "step": step,
                "config": vars(args),
                "training": {
                    "world_size": world,
                    "batch_users_per_gpu": args.batch_users,
                    "seen_interactions": step * global_batch,
                    "full_train_interactions": len(edge_users),
                    "processed_interactions_including_final_padding": planned_steps * global_batch,
                    "epochs": args.epochs,
                    "sampling_mode": args.sampling_mode,
                    "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
                    "seed": args.seed,
                    "full_coverage": bool(
                        args.sampling_mode == "all_interactions" and planned_steps == full_steps
                    ),
                    "elapsed_seconds": time.time() - started,
                },
            }
            torch.save(checkpoint, args.output_dir / f"step_{step:06d}.pt")
            torch.save(checkpoint, args.output_dir / "last.pt")
    if distributed:
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()
    if is_main:
        write_json(args.output_dir / "training_summary.json", checkpoint["training"] | {"steps": planned_steps})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
