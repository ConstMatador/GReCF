#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from grecf.data import id_order_sha256, load_cache, load_public_dataset
from grecf.multi_interest import (
    FullContextWeightedSumAdapter,
    HistoryResamplerAdapter,
    PreferenceIPDeltaAdapter,
    PrototypeSetPreferenceIPDeltaAdapter,
    RoutedPrototypeSetPreferenceIPDeltaAdapter,
    UnifiedThemeSetPreferenceIPAdapter,
    SplitInterestAdapter,
    SplitInterestCFAdapter,
    load_or_build_adaptive_interest_cache,
    load_or_build_interest_cache,
    local_support_batch,
    padded_train_histories,
)
from grecf.preference_ip_adapter import (
    install_preference_ip_processors,
    install_preference_ip_trainable_kv_processors,
    load_preference_ip_processor_state_dict,
    preference_ip_processor_gate_summary,
    preference_ip_processor_state_dict,
    preference_ip_trainable_parameters,
)
from grecf.reference import (
    load_or_build_item_average_user_representation_cache,
    load_or_build_user_representation_cache,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def all_train_edges(dataset) -> tuple[np.ndarray, np.ndarray]:
    users = np.repeat(np.arange(dataset.num_users, dtype=np.int32), [len(x) for x in dataset.train])
    items = np.concatenate(dataset.train).astype(np.int32, copy=False)
    return users, items


def distributed_epoch_indices(count: int, batch_size: int, world_size: int, rank: int, seed: int) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(count)
    global_batch = batch_size * world_size
    padded = math.ceil(count / global_batch) * global_batch
    if padded > count:
        order = np.concatenate((order, order[: padded - count]))
    return order.reshape(-1, world_size, batch_size)[:, rank, :]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train GReCF with a frozen SD1.5 backbone.")
    parser.add_argument("--variant", default="b", choices=("b", "d", "e"))
    parser.add_argument(
        "--adaptive-interests", action=argparse.BooleanOptionalAction, default=True,
        help="Represent each user with an adaptive multi-interest theme set.",
    )
    parser.add_argument("--max-interests", type=int, default=8)
    parser.add_argument("--interest-cv-folds", type=int, default=5)
    parser.add_argument("--interest-min-support", type=int, default=5)
    parser.add_argument("--interest-min-support-fraction", type=float, default=0.05)
    parser.add_argument("--interest-merge-cosine", type=float, default=0.90)
    parser.add_argument("--interest-retained-gain", type=float, default=0.75)
    parser.add_argument(
        "--interest-routing",
        choices=("target", "mixed", "evidence"),
        default="target",
        help=(
            "How to choose the per-sample interest prototype. target uses the target image nearest "
            "prototype; mixed combines target routing with user-history weighted sampling; evidence "
            "always samples from user-history weights without looking at the target image."
        ),
    )
    parser.add_argument(
        "--interest-teacher-forcing-prob",
        type=float,
        default=1.0,
        help="Teacher-forced target-route probability used when --interest-routing=mixed.",
    )
    parser.add_argument(
        "--interest-teacher-forcing-prob-start",
        type=float,
        default=None,
        help="Optional starting probability, linearly scheduled to --interest-teacher-forcing-prob.",
    )
    parser.add_argument(
        "--masked-interest-prob",
        type=float,
        default=0.5,
        help=(
            "Masked-interest training probability. With this probability, "
            "the target-routed interest is removed from the user profile, the auxiliary/user "
            "condition is recomputed from the remaining interests, and the model is asked to "
            "reconstruct the masked target image from the nearest remaining anchor interest."
        ),
    )
    parser.add_argument(
        "--masked-anchor-selection",
        choices=("nearest", "random", "mixed"),
        default="random",
        help="How to choose the visible context interest when --masked-interest-prob > 0.",
    )
    parser.add_argument(
        "--masked-random-anchor-prob",
        type=float,
        default=0.25,
        help="For mixed anchor selection, probability of choosing a random rather than nearest visible anchor.",
    )
    parser.add_argument(
        "--masked-interest-prob-start",
        type=float,
        default=None,
        help="Optional starting mask probability; linearly ramps to --masked-interest-prob over max steps.",
    )
    parser.add_argument(
        "--training-path-mode",
        choices=(
            "single",
            "dual_leave_target_subset",
            "dual_leave_target_all_anchors",
        ),
        default="single",
        help=(
            "single keeps the original one-path objective. dual_leave_target_subset trains every "
            "batch with both a direct reconstruction path and a leave-target-interest-out relation "
            "path. In the relation path, the target-routed prototype is always hidden and a random "
            "subset of the remaining prototypes is used as the visible few-shot/relation context. "
            "dual_leave_target_all_anchors also always trains the direct path, then removes the "
            "target prototype from the virtual-user set and exhaustively routes every remaining "
            "prototype through the interest branch."
        ),
    )
    parser.add_argument(
        "--relation-loss-weight",
        type=float,
        default=0.5,
        help="Loss weight for the relation/few-shot path when --training-path-mode=dual_leave_target_subset.",
    )
    parser.add_argument(
        "--relation-context-keep-prob",
        type=float,
        default=0.5,
        help=(
            "Probability of keeping each non-target active prototype in the relation path. "
            "If all visible prototypes are dropped, the nearest non-target anchor is restored."
        ),
    )
    parser.add_argument(
        "--cf-token",
        action="store_true",
        help="Add a LightGCN collaborative-filtering user embedding token block; valid with --variant b.",
    )
    parser.add_argument(
        "--cf-user-cache",
        type=Path,
        default=None,
        help="Path to LightGCN user embeddings (.npy) used when --cf-token is enabled.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("/path/to/datasets/CIGR"))
    parser.add_argument("--sd15-model", type=Path, default=Path("/path/to/models/stable-diffusion-v1-5"))
    parser.add_argument("--clip-cache", type=Path, default=Path("/path/to/evaluation-cache/cache/public_clip_clean_features.float16.npy"))
    parser.add_argument("--latent-cache", type=Path, default=Path("cache/public_sd15_latents.float16.npy"))
    parser.add_argument("--user-rep-cache", type=Path, default=Path("cache/v2_cluster_balanced_user_reps.float32.npy"))
    parser.add_argument(
        "--item-average-cache",
        type=Path,
        default=Path("cache/train_item_average_user_reps.float32.npy"),
        help="Train-history item-average cache used by the Item Average ablation.",
    )
    parser.add_argument("--interest-cache-dir", type=Path, default=Path("cache/v2_multinterest"))
    parser.add_argument("--results-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-users", type=int, default=192)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Initialize adapter and preference-attention weights from a compatible checkpoint; optimizer is reset.",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--residual-scale", type=float, default=2.0)
    parser.add_argument(
        "--adapter-mode",
        choices=(
            "split",
            "full77_weighted_sum",
            "preference_ip_delta",
            "preference_ip_delta_kv",
            "preference_ip_delta_kv_laygate",
            "preference_ip_proto_set_delta_kv_laygate",
            "preference_ip_proto_set_route_delta_kv_laygate",
            "preference_ip_unified_theme_set_kv_laygate",
        ),
        default="preference_ip_unified_theme_set_kv_laygate",
        help=(
            "split is the disjoint-token residual adapter; full77_weighted_sum maps both "
            "auxiliary and interest vectors to full 77-slot residual contexts and fuses them "
            "with learnable softmax weights; preference_ip_delta keeps the base text context "
            "and injects user/delta tokens through a parameter-free IP-Adapter-style extra attention "
            "branch; preference_ip_delta_kv further gives that branch trainable per-layer K/V projections; "
            "preference_ip_delta_kv_laygate also learns per-UNet-layer user/delta gates; "
            "preference_ip_proto_set_delta_kv_laygate replaces the single mean user vector with "
            "a masked prototype-set encoder while keeping the delta branch; "
            "preference_ip_proto_set_route_delta_kv_laygate adds trainable DIRECT/EXPANSION "
            "route tokens to that delta branch; preference_ip_unified_theme_set_kv_laygate "
            "keeps themes as independent masked tokens and uses one unified absolute-theme branch."
        ),
    )
    parser.add_argument("--preference-user-tokens", type=int, default=8)
    parser.add_argument("--preference-delta-tokens", type=int, default=8)
    parser.add_argument("--context-tokens", type=int, default=77)
    parser.add_argument(
        "--full77-train-source",
        choices=("full", "user_only", "interest_only"),
        default="full",
        help=(
            "Only for --adapter-mode full77_weighted_sum. "
            "full trains the original weighted sum; user_only trains and conditions with only "
            "the auxiliary/user-overall full-context residual; interest_only trains and conditions "
            "with only the routed-interest full-context residual."
        ),
    )
    parser.add_argument("--auxiliary-tokens", type=int, default=16)
    parser.add_argument("--interest-tokens", type=int, default=48)
    parser.add_argument("--base-prompt", default="a high quality photograph")
    parser.add_argument(
        "--base-context-mode",
        choices=("prompt", "zero", "prompt_prefix"),
        default="prompt",
        help="Use full prompt context, zero context, or keep only the first --prompt-token-slots prompt slots and reserve the rest for learned tokens.",
    )
    parser.add_argument("--prompt-token-slots", type=int, default=13)
    parser.add_argument("--support-size", type=int, default=16)
    parser.add_argument("--timestep-min", type=int, default=20)
    parser.add_argument("--timestep-max", type=int, default=980)
    parser.add_argument(
        "--min-snr-gamma",
        type=float,
        default=0.0,
        help="Apply epsilon-prediction Min-SNR loss weighting when positive; zero preserves plain MSE.",
    )
    parser.add_argument(
        "--condition-margin-weight",
        type=float,
        default=0.0,
        help="Weight for a training-only hinge that makes correct-user denoising beat a wrong user.",
    )
    parser.add_argument(
        "--condition-margin",
        type=float,
        default=0.01,
        help="Per-sample denoising-error margin used by --condition-margin-weight.",
    )
    parser.add_argument("--user-ce-weight", type=float, default=0.05)
    parser.add_argument(
        "--user-cf-contrast-weight",
        type=float,
        default=0.0,
        help=(
            "Collaborative user-space contrastive weight for the prototype-set adapter. "
            "Direct path uses full-user similarity; masked relation path treats the visible remaining "
            "prototypes as a virtual partial user and retrieves target-supporting positives."
        ),
    )
    parser.add_argument("--user-cf-temperature", type=float, default=0.07)
    parser.add_argument("--user-cf-negatives", type=int, default=31)
    parser.add_argument(
        "--user-cf-positive-skip",
        type=int,
        default=4,
        help="Skip the nearest users when selecting direct-path positives, to avoid only pulling exact duplicates.",
    )
    parser.add_argument(
        "--user-cf-positive-pool",
        type=int,
        default=32,
        help="Top candidate pool used for CF positives before selection.",
    )
    parser.add_argument("--user-cf-target-weight", type=float, default=0.45)
    parser.add_argument("--residual-reg-weight", type=float, default=1e-7)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--validation-every", type=int, default=250)
    parser.add_argument("--validation-users", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument(
        "--checkpoint-every-epoch",
        action="store_true",
        help="Validate and save epoch_XX.pt at every epoch boundary.",
    )
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ablation",
        choices=("full", "item_average", "no_theme_anchor", "no_mask"),
        default="full",
        help="Controlled GReCF representation/conditioning ablation.",
    )
    args = parser.parse_args()
    if args.ablation != "full" and args.adapter_mode != "preference_ip_unified_theme_set_kv_laygate":
        parser.error("--ablation is only supported by the GReCF unified theme-set adapter")
    if args.ablation == "item_average":
        args.preference_user_tokens = 1
    if args.ablation == "no_mask":
        args.masked_interest_prob = 0.0
        args.masked_interest_prob_start = None
    if args.adaptive_interests and args.variant != "b":
        parser.error("--adaptive-interests is only valid with --variant b")
    if args.cf_token and args.variant != "b":
        parser.error("--cf-token is only valid with --variant b")
    if args.cf_token and args.cf_user_cache is None:
        parser.error("--cf-token requires --cf-user-cache")
    if not 0.0 < args.interest_retained_gain <= 1.0:
        parser.error("--interest-retained-gain must be in (0, 1]")
    if not 0.0 <= args.interest_teacher_forcing_prob <= 1.0:
        parser.error("--interest-teacher-forcing-prob must be in [0, 1]")
    if (
        args.interest_teacher_forcing_prob_start is not None
        and not 0.0 <= args.interest_teacher_forcing_prob_start <= 1.0
    ):
        parser.error("--interest-teacher-forcing-prob-start must be in [0, 1]")
    if args.interest_teacher_forcing_prob_start is not None and args.interest_routing != "mixed":
        parser.error("--interest-teacher-forcing-prob-start requires --interest-routing=mixed")
    if not 0.0 <= args.masked_interest_prob <= 1.0:
        parser.error("--masked-interest-prob must be in [0, 1]")
    if args.masked_interest_prob_start is not None and not 0.0 <= args.masked_interest_prob_start <= 1.0:
        parser.error("--masked-interest-prob-start must be in [0, 1]")
    if not 0.0 <= args.masked_random_anchor_prob <= 1.0:
        parser.error("--masked-random-anchor-prob must be in [0, 1]")
    if args.relation_loss_weight < 0.0:
        parser.error("--relation-loss-weight must be non-negative")
    if args.min_snr_gamma < 0.0:
        parser.error("--min-snr-gamma must be non-negative")
    if args.condition_margin_weight < 0.0 or args.condition_margin < 0.0:
        parser.error("--condition-margin-weight and --condition-margin must be non-negative")
    if args.condition_margin_weight > 0.0 and args.training_path_mode != "single":
        parser.error("--condition-margin-weight currently requires --training-path-mode=single")
    if (
        args.condition_margin_weight > 0.0
        and args.adapter_mode != "preference_ip_unified_theme_set_kv_laygate"
    ):
        parser.error("--condition-margin-weight currently targets the GReCF unified theme-set adapter")
    if args.user_cf_contrast_weight < 0.0:
        parser.error("--user-cf-contrast-weight must be non-negative")
    # Keep this allowlist identical to cf_user_contrastive_loss() below: that
    # function silently returns zero for any other adapter mode.
    if args.user_cf_contrast_weight > 0.0 and args.adapter_mode not in (
        "preference_ip_proto_set_delta_kv_laygate",
        "preference_ip_proto_set_route_delta_kv_laygate",
    ):
        parser.error("--user-cf-contrast-weight requires the prototype-set adapter")
    if args.user_cf_temperature <= 0.0:
        parser.error("--user-cf-temperature must be positive")
    if args.user_cf_negatives <= 0:
        parser.error("--user-cf-negatives must be positive")
    if args.user_cf_positive_pool <= 0:
        parser.error("--user-cf-positive-pool must be positive")
    if not 0.0 <= args.user_cf_target_weight <= 1.0:
        parser.error("--user-cf-target-weight must be in [0, 1]")
    if not 0.0 < args.relation_context_keep_prob <= 1.0:
        parser.error("--relation-context-keep-prob must be in (0, 1]")
    if args.masked_interest_prob > 0.0 and args.variant not in "bd":
        parser.error("--masked-interest-prob is valid only with multi-interest variants b/d")
    if args.training_path_mode != "single" and args.variant not in "bd":
        parser.error("dual training-path modes are valid only with multi-interest variants b/d")
    if args.training_path_mode != "single" and args.adapter_mode not in (
        "preference_ip_proto_set_delta_kv_laygate",
        "preference_ip_proto_set_route_delta_kv_laygate",
    ):
        parser.error("dual training-path modes require the prototype-set adapter")
    if args.auxiliary_tokens < 0 or args.interest_tokens < 0:
        parser.error("--auxiliary-tokens and --interest-tokens must be non-negative")
    if args.adapter_mode in ("full77_weighted_sum", "preference_ip_delta", "preference_ip_delta_kv", "preference_ip_delta_kv_laygate", "preference_ip_proto_set_delta_kv_laygate", "preference_ip_proto_set_route_delta_kv_laygate", "preference_ip_unified_theme_set_kv_laygate") and args.variant not in "bd":
        parser.error("--adapter-mode full77_weighted_sum/preference_ip_delta/preference_ip_delta_kv/preference_ip_delta_kv_laygate/preference_ip_proto_set_delta_kv_laygate is valid only with --variant b/d")
    if args.adapter_mode == "full77_weighted_sum" and args.cf_token:
        parser.error("--adapter-mode full77_weighted_sum is not compatible with --cf-token")
    if args.adapter_mode in ("preference_ip_delta", "preference_ip_delta_kv", "preference_ip_delta_kv_laygate", "preference_ip_proto_set_delta_kv_laygate", "preference_ip_proto_set_route_delta_kv_laygate", "preference_ip_unified_theme_set_kv_laygate") and args.cf_token:
        parser.error("--adapter-mode preference_ip_delta/preference_ip_delta_kv/preference_ip_delta_kv_laygate/preference_ip_proto_set_delta_kv_laygate is not compatible with --cf-token")
    if args.full77_train_source != "full" and args.adapter_mode != "full77_weighted_sum":
        parser.error("--full77-train-source can only be changed with --adapter-mode full77_weighted_sum")
    if args.context_tokens <= 0 or args.context_tokens > 77:
        parser.error("--context-tokens must be in [1, 77]")
    if args.variant in "bd" and not args.cf_token and args.adapter_mode == "split" and args.auxiliary_tokens + args.interest_tokens == 0:
        parser.error("the multi-interest adapter requires at least one conditioning token")
    residual_start = args.prompt_token_slots if args.base_context_mode == "prompt_prefix" else 0
    if args.prompt_token_slots < 0 or args.prompt_token_slots > 77:
        parser.error("--prompt-token-slots must be in [0, 77]")
    if args.variant in "bd" and not args.cf_token and args.adapter_mode == "split" and residual_start + args.auxiliary_tokens + args.interest_tokens > 77:
        parser.error("token allocation cannot exceed the SD text context length of 77")
    if args.run_name is None:
        args.run_name = "grecf_sd15"

    from diffusers import DDPMScheduler, PNDMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1
        device = torch.device(args.device)
    is_main = rank == 0
    output_dir = args.results_dir / args.run_name
    if is_main:
        for child in ("checkpoints", "logs", "tables", "reports"):
            (output_dir / child).mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "config.json", {**vars(args), "world_size": world_size})
        log_path = output_dir / "logs" / "train.jsonl"
        if log_path.exists():
            log_path.unlink()
    if distributed:
        dist.barrier(device_ids=[device.index])

    set_seed(args.seed + rank * 1_000_003)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    dtype = torch.bfloat16

    dataset = load_public_dataset(args.dataset_root)
    order_sha = id_order_sha256(dataset.image_ids)
    clip_features = load_cache(args.clip_cache, dataset.num_items, order_sha, (512,))
    latents = load_cache(args.latent_cache, dataset.num_items, order_sha, (4, 64, 64))
    if is_main:
        user_reps, coherence = load_or_build_user_representation_cache(args.user_rep_cache, dataset, clip_features)
        item_average_reps = (
            load_or_build_item_average_user_representation_cache(args.item_average_cache, dataset, clip_features)[0]
            if args.ablation == "item_average"
            else None
        )
        if args.adaptive_interests:
            interest_cache = load_or_build_adaptive_interest_cache(
                args.interest_cache_dir, dataset, clip_features,
                max_interests=args.max_interests, folds=args.interest_cv_folds,
                minimum_support=args.interest_min_support,
                minimum_support_fraction=args.interest_min_support_fraction,
                merge_cosine=args.interest_merge_cosine,
                retained_gain=args.interest_retained_gain, seed=args.seed,
            )
        else:
            interest_cache = load_or_build_interest_cache(args.interest_cache_dir, dataset, clip_features, args.variant) if args.variant in "bd" else None
    if distributed:
        dist.barrier(device_ids=[device.index])
    if not is_main:
        user_reps, coherence = load_or_build_user_representation_cache(args.user_rep_cache, dataset, clip_features)
        item_average_reps = (
            load_or_build_item_average_user_representation_cache(args.item_average_cache, dataset, clip_features)[0]
            if args.ablation == "item_average"
            else None
        )
        if args.adaptive_interests:
            interest_cache = load_or_build_adaptive_interest_cache(
                args.interest_cache_dir, dataset, clip_features,
                max_interests=args.max_interests, folds=args.interest_cv_folds,
                minimum_support=args.interest_min_support,
                minimum_support_fraction=args.interest_min_support_fraction,
                merge_cosine=args.interest_merge_cosine,
                retained_gain=args.interest_retained_gain, seed=args.seed,
            )
        else:
            interest_cache = load_or_build_interest_cache(args.interest_cache_dir, dataset, clip_features, args.variant) if args.variant in "bd" else None

    tokenizer = CLIPTokenizer.from_pretrained(args.sd15_model, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(
        args.sd15_model, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True
    ).to(device).eval()
    text_encoder.requires_grad_(False)
    prompt_ids = tokenizer(args.base_prompt, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        base_context = text_encoder(prompt_ids).last_hidden_state.float()
    if args.base_context_mode == "zero":
        base_context = torch.zeros_like(base_context)
    elif args.base_context_mode == "prompt_prefix":
        base_context = base_context.clone()
        base_context[:, args.prompt_token_slots :] = 0.0
    del text_encoder
    torch.cuda.empty_cache()

    unet = UNet2DConditionModel.from_pretrained(
        args.sd15_model, subfolder="unet", torch_dtype=dtype, local_files_only=True
    ).to(device).eval()
    unet.requires_grad_(False)
    trainable_preference_kv = args.adapter_mode in (
        "preference_ip_delta_kv",
        "preference_ip_delta_kv_laygate",
        "preference_ip_proto_set_delta_kv_laygate",
        "preference_ip_proto_set_route_delta_kv_laygate",
        "preference_ip_unified_theme_set_kv_laygate",
    )
    if args.adapter_mode == "preference_ip_delta":
        install_preference_ip_processors(unet)
    elif trainable_preference_kv:
        install_preference_ip_trainable_kv_processors(
            unet,
            layer_gates=args.adapter_mode in (
                "preference_ip_delta_kv_laygate",
                "preference_ip_proto_set_delta_kv_laygate",
                "preference_ip_proto_set_route_delta_kv_laygate",
                "preference_ip_unified_theme_set_kv_laygate",
            ),
        )
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    unet_base = unet
    unet_trainable_params = preference_ip_trainable_parameters(unet_base) if trainable_preference_kv else []
    unet = (
        DistributedDataParallel(
            unet_base,
            device_ids=[device.index],
            broadcast_buffers=False,
        )
        if distributed and trainable_preference_kv
        else unet_base
    )
    scheduler_config = PNDMScheduler.load_config(args.sd15_model, subfolder="scheduler", local_files_only=True)
    noise_scheduler = DDPMScheduler.from_config(scheduler_config)
    snr_alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)

    user_rep_tensor = torch.from_numpy(np.asarray(user_reps, dtype=np.float32)).to(device)
    if args.cf_token:
        assert args.cf_user_cache is not None
        cf_user_embeddings_np = np.load(args.cf_user_cache)
        if cf_user_embeddings_np.shape[0] != dataset.num_users:
            raise ValueError(
                f"CF user cache has {cf_user_embeddings_np.shape[0]} users, expected {dataset.num_users}"
            )
        cf_user_tensor = torch.from_numpy(np.asarray(cf_user_embeddings_np, dtype=np.float32)).to(device)
    else:
        cf_user_tensor = None
    catalog_tensor = torch.from_numpy(np.asarray(clip_features, dtype=np.float32)).to(device=device, dtype=dtype)
    if args.variant in "bd":
        assert interest_cache is not None
        auxiliary_np = (
            np.asarray(item_average_reps, dtype=np.float32)
            if args.ablation == "item_average"
            else np.asarray(interest_cache["auxiliary"], dtype=np.float32)
        )
        auxiliary_tensor = torch.from_numpy(auxiliary_np).to(device)
        prototype_tensor = torch.from_numpy(np.asarray(interest_cache["prototypes"], dtype=np.float32)).to(device)
        interest_weight_tensor = torch.from_numpy(np.asarray(interest_cache["weights"], dtype=np.float32)).to(device)
        if args.adaptive_interests:
            interest_mask_tensor = torch.from_numpy(np.asarray(interest_cache["mask"], dtype=bool)).to(device)
            active_prototypes = prototype_tensor[interest_mask_tensor]
        else:
            interest_mask_tensor = torch.ones(prototype_tensor.shape[:2], dtype=torch.bool, device=device)
            active_prototypes = prototype_tensor.reshape(-1, prototype_tensor.shape[-1])
        auxiliary_target_tensor = None
        prototype_flat_tensor = None
        prototype_flat_user_tensor = None
        prototype_flat_mask_tensor = None
        if args.adapter_mode == "full77_weighted_sum":
            model_base = FullContextWeightedSumAdapter(
                base_context,
                auxiliary_center=auxiliary_tensor.mean(dim=0, keepdim=True),
                prototype_center=active_prototypes.mean(dim=0, keepdim=True),
                context_tokens=args.context_tokens,
                residual_scale=args.residual_scale,
            ).to(device)
            if args.full77_train_source == "user_only":
                model_base.interest_mlp.requires_grad_(False)
                model_base.interest_head.requires_grad_(False)
                model_base.fusion_logits.requires_grad_(False)
            elif args.full77_train_source == "interest_only":
                model_base.auxiliary_mlp.requires_grad_(False)
                model_base.auxiliary_head.requires_grad_(False)
                model_base.fusion_logits.requires_grad_(False)
        elif args.adapter_mode in ("preference_ip_delta", "preference_ip_delta_kv", "preference_ip_delta_kv_laygate"):
            model_base = PreferenceIPDeltaAdapter(
                base_context,
                auxiliary_center=auxiliary_tensor.mean(dim=0, keepdim=True),
                prototype_center=active_prototypes.mean(dim=0, keepdim=True),
                user_tokens=args.preference_user_tokens,
                delta_tokens=args.preference_delta_tokens,
                residual_scale=args.residual_scale,
            ).to(device)
        elif args.adapter_mode == "preference_ip_unified_theme_set_kv_laygate":
            model_base = UnifiedThemeSetPreferenceIPAdapter(
                base_context,
                auxiliary_center=auxiliary_tensor.mean(dim=0, keepdim=True),
                prototype_center=active_prototypes.mean(dim=0, keepdim=True),
                user_tokens=args.preference_user_tokens,
                delta_tokens=args.preference_delta_tokens,
                residual_scale=args.residual_scale,
                use_theme_anchor=args.ablation != "no_theme_anchor",
                user_representation_mode="item_average" if args.ablation == "item_average" else "theme_set",
            ).to(device)
            if args.ablation == "no_theme_anchor":
                model_base.theme_mlp.requires_grad_(False)
                model_base.delta_head.requires_grad_(False)
                model_base.delta_gate_logit.requires_grad_(False)
        elif args.adapter_mode in (
            "preference_ip_proto_set_delta_kv_laygate",
            "preference_ip_proto_set_route_delta_kv_laygate",
        ):
            adapter_class = (
                RoutedPrototypeSetPreferenceIPDeltaAdapter
                if args.adapter_mode == "preference_ip_proto_set_route_delta_kv_laygate"
                else PrototypeSetPreferenceIPDeltaAdapter
            )
            model_base = adapter_class(
                base_context,
                auxiliary_center=auxiliary_tensor.mean(dim=0, keepdim=True),
                prototype_center=active_prototypes.mean(dim=0, keepdim=True),
                user_tokens=args.preference_user_tokens,
                delta_tokens=args.preference_delta_tokens,
                residual_scale=args.residual_scale,
            ).to(device)
            if args.user_cf_contrast_weight > 0.0:
                auxiliary_target_tensor = F.normalize(
                    auxiliary_tensor.float() - model_base.auxiliary_center.to(device).float(), dim=-1
                )
                prototype_flat_tensor = F.normalize(prototype_tensor.float(), dim=-1).reshape(-1, prototype_tensor.shape[-1])
                prototype_flat_user_tensor = (
                    torch.arange(prototype_tensor.shape[0], device=device)[:, None]
                    .expand(-1, prototype_tensor.shape[1])
                    .reshape(-1)
                )
                prototype_flat_mask_tensor = interest_mask_tensor.reshape(-1)
        elif args.cf_token:
            assert cf_user_tensor is not None
            model_base = SplitInterestCFAdapter(
                base_context,
                auxiliary_center=auxiliary_tensor.mean(dim=0, keepdim=True),
                prototype_center=active_prototypes.mean(dim=0, keepdim=True),
                cf_center=cf_user_tensor.mean(dim=0, keepdim=True),
                residual_scale=args.residual_scale,
            ).to(device)
        else:
            model_base = SplitInterestAdapter(
                base_context,
                auxiliary_center=auxiliary_tensor.mean(dim=0, keepdim=True),
                prototype_center=active_prototypes.mean(dim=0, keepdim=True),
                auxiliary_tokens=args.auxiliary_tokens,
                interest_tokens=args.interest_tokens,
                residual_scale=args.residual_scale,
                residual_start=residual_start,
            ).to(device)
        history_tensor = history_mask_tensor = None
    else:
        histories, history_mask = padded_train_histories(dataset)
        history_tensor = torch.from_numpy(histories).to(device)
        history_mask_tensor = torch.from_numpy(history_mask).to(device)
        model_base = HistoryResamplerAdapter(
            base_context, representation_center=user_rep_tensor.mean(dim=0, keepdim=True), residual_scale=args.residual_scale
        ).to(device)
        auxiliary_tensor = prototype_tensor = interest_mask_tensor = interest_weight_tensor = None
    if args.init_checkpoint is not None:
        init_payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        init_config = init_payload.get("config") or {}
        if init_config.get("variant") != args.variant:
            raise ValueError(
                f"init checkpoint variant {init_config.get('variant')} does not match {args.variant}"
            )
        if init_config.get("adapter_mode") != args.adapter_mode:
            raise ValueError(
                f"init checkpoint adapter {init_config.get('adapter_mode')} does not match {args.adapter_mode}"
            )
        model_base.load_state_dict(init_payload["model"], strict=True)
        if trainable_preference_kv:
            load_preference_ip_processor_state_dict(
                unet_base, init_payload["preference_ip_processor_state"]
            )
        del init_payload

    model = (
        DistributedDataParallel(
            model_base,
            device_ids=[device.index],
            broadcast_buffers=False,
        )
        if distributed
        else model_base
    )
    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in list(model_base.parameters()) + list(unet_trainable_params)
            if parameter.requires_grad
        ),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    edge_users, edge_items = all_train_edges(dataset)
    effective_global_batch = args.batch_users * world_size
    steps_per_epoch = math.ceil(len(edge_users) / effective_global_batch)
    planned_steps = min(steps_per_epoch * args.epochs, args.max_steps) if args.max_steps > 0 else steps_per_epoch * args.epochs
    validation_rng = np.random.default_rng(args.seed + 29)
    validation_users = np.sort(validation_rng.choice(dataset.num_users, size=args.validation_users, replace=False)).astype(np.int64)
    validation_targets = np.asarray([validation_rng.choice(dataset.validation[int(user)]) for user in validation_users], dtype=np.int64)
    wrong_validation_users = np.roll(validation_users, 1)
    if is_main:
        write_json(output_dir / "tables" / "validation_users.json", {
            "user_indices": validation_users.tolist(), "user_ids": [dataset.user_ids[int(user)] for user in validation_users]
        })

    torch_generator = torch.Generator(device=device).manual_seed(args.seed + 31 + rank * 1_000_003)
    current_mask_prob = float(
        args.masked_interest_prob_start
        if args.masked_interest_prob_start is not None
        else args.masked_interest_prob
    )
    current_teacher_forcing_prob = float(
        args.interest_teacher_forcing_prob_start
        if args.interest_teacher_forcing_prob_start is not None
        else args.interest_teacher_forcing_prob
    )

    def clean_latents(items: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(latents[items], dtype=np.float32)).to(device, dtype=dtype, non_blocking=True)

    def deterministic_uniform(users: torch.Tensor, offsets: torch.Tensor, salt: int) -> torch.Tensor:
        hashed = (
            users.long() * 1_103_515_245
            + offsets.long() * 12_345
            + int(args.seed)
            + salt
        ).remainder(1_000_003)
        return (hashed.float() + 0.5) / 1_000_003.0

    def weighted_interest_sample(
        users: torch.Tensor, deterministic_offset: int | None = None
    ) -> torch.Tensor:
        assert interest_weight_tensor is not None and interest_mask_tensor is not None
        probabilities = interest_weight_tensor[users].float().masked_fill(~interest_mask_tensor[users], 0.0)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
        if deterministic_offset is None:
            return torch.multinomial(probabilities, 1, generator=torch_generator).squeeze(1)
        offsets = torch.arange(len(users), device=device) + int(deterministic_offset)
        thresholds = deterministic_uniform(users, offsets, salt=7919)
        cumulative = probabilities.cumsum(dim=1)
        chosen = (thresholds[:, None] > cumulative).sum(dim=1)
        return chosen.clamp(max=probabilities.shape[1] - 1)

    def split_inputs(
        users: torch.Tensor,
        targets: torch.Tensor,
        deterministic_offset: int | None = None,
        enable_masked_interest: bool = False,
        force_masked_interest: bool = False,
    ):
        assert auxiliary_tensor is not None and prototype_tensor is not None and interest_mask_tensor is not None
        candidates = prototype_tensor[users]
        active_mask = interest_mask_tensor[users]
        target_features = catalog_tensor[targets].float()
        scores = torch.einsum("bkd,bd->bk", candidates, target_features)
        scores = scores.masked_fill(~active_mask, -torch.inf)
        target_chosen = scores.argmax(dim=1)
        if args.interest_routing == "target":
            chosen = target_chosen
        else:
            sampled = weighted_interest_sample(users, deterministic_offset)
            if args.interest_routing == "evidence":
                chosen = sampled
            else:
                if deterministic_offset is None:
                    use_target = torch.rand(
                        len(users), generator=torch_generator, device=device
                    ) < current_teacher_forcing_prob
                else:
                    offsets = torch.arange(len(users), device=device) + int(deterministic_offset)
                    use_target = (
                        deterministic_uniform(users, offsets, salt=104_729)
                        < current_teacher_forcing_prob
                    )
                chosen = torch.where(use_target, target_chosen, sampled)
        auxiliary = auxiliary_tensor[users]
        encoder_mask = active_mask
        masked_count = 0
        maskable_count = 0
        if enable_masked_interest and (current_mask_prob > 0.0 or force_masked_interest):
            row_index = torch.arange(len(users), device=device)
            target_one_hot = F.one_hot(target_chosen, num_classes=candidates.shape[1]).to(torch.bool)
            context_mask = active_mask & ~target_one_hot
            context_weights = interest_weight_tensor[users].float().masked_fill(~context_mask, 0.0)
            context_weight_sum = context_weights.sum(dim=1, keepdim=True)
            can_mask = context_weight_sum.squeeze(1) > 1e-8
            maskable_count = int(can_mask.detach().sum().cpu())
            if force_masked_interest:
                use_masked = torch.ones(len(users), dtype=torch.bool, device=device)
            else:
                if deterministic_offset is None:
                    use_masked = torch.rand(
                        len(users), generator=torch_generator, device=device
                    ) < current_mask_prob
                else:
                    offsets = torch.arange(len(users), device=device) + int(deterministic_offset)
                    use_masked = deterministic_uniform(users, offsets, salt=193_939) < current_mask_prob
            use_masked = use_masked & can_mask
            if bool(use_masked.any()):
                target_proto = candidates[row_index, target_chosen].float()
                relation_mask = context_mask
                if force_masked_interest and args.relation_context_keep_prob < 1.0:
                    keep_random = (
                        torch.rand(
                            context_mask.shape,
                            generator=torch_generator,
                            device=device,
                        )
                        < float(args.relation_context_keep_prob)
                    )
                    relation_mask = context_mask & keep_random
                    empty_relation = relation_mask.sum(dim=1) == 0
                    if bool(empty_relation.any()):
                        fallback_scores = torch.einsum("bkd,bd->bk", candidates.float(), target_proto)
                        fallback_scores = fallback_scores.masked_fill(~context_mask, -torch.inf)
                        fallback = fallback_scores.argmax(dim=1)
                        relation_mask = relation_mask.clone()
                        relation_mask[empty_relation, fallback[empty_relation]] = True
                    relation_mask = torch.where(use_masked[:, None], relation_mask, context_mask)
                random_anchor_scores = torch.rand(
                    relation_mask.shape,
                    generator=torch_generator,
                    device=device,
                ).masked_fill(~relation_mask, -torch.inf)
                nearest_anchor_scores = torch.einsum("bkd,bd->bk", candidates.float(), target_proto)
                nearest_anchor_scores = nearest_anchor_scores.masked_fill(~relation_mask, -torch.inf)
                if args.masked_anchor_selection == "random":
                    anchor_scores = random_anchor_scores
                elif args.masked_anchor_selection == "mixed":
                    use_random_anchor = torch.rand(
                        len(users), generator=torch_generator, device=device
                    ) < float(args.masked_random_anchor_prob)
                    anchor_scores = torch.where(
                        use_random_anchor[:, None], random_anchor_scores, nearest_anchor_scores
                    )
                else:
                    anchor_scores = nearest_anchor_scores
                anchor_chosen = anchor_scores.argmax(dim=1)
                relation_weights = interest_weight_tensor[users].float().masked_fill(~relation_mask, 0.0)
                relation_weight_sum = relation_weights.sum(dim=1, keepdim=True)
                context_auxiliary = torch.einsum("bk,bkd->bd", relation_weights, candidates.float())
                context_auxiliary = F.normalize(context_auxiliary / relation_weight_sum.clamp_min(1e-8), dim=-1)
                chosen = torch.where(use_masked, anchor_chosen, chosen)
                auxiliary = torch.where(use_masked[:, None], context_auxiliary.to(auxiliary.dtype), auxiliary)
                encoder_mask = torch.where(use_masked[:, None], relation_mask, encoder_mask)
            masked_count = int(use_masked.detach().sum().cpu())
        split_inputs.last_masked_count = masked_count
        split_inputs.last_maskable_count = maskable_count
        interests = candidates[torch.arange(len(users), device=device), chosen]
        if args.adapter_mode in (
            "preference_ip_proto_set_delta_kv_laygate",
            "preference_ip_proto_set_route_delta_kv_laygate",
            "preference_ip_unified_theme_set_kv_laygate",
        ):
            return candidates, encoder_mask, auxiliary, interests
        if args.cf_token:
            assert cf_user_tensor is not None
            return auxiliary, interests, cf_user_tensor[users]
        return auxiliary, interests

    def model_inputs(
        users: torch.Tensor,
        targets: torch.Tensor,
        deterministic_offset: int | None = None,
        enable_masked_interest: bool = False,
        force_masked_interest: bool = False,
    ):
        if args.variant in "bd":
            return split_inputs(users, targets, deterministic_offset, enable_masked_interest, force_masked_interest)
        assert history_tensor is not None and history_mask_tensor is not None
        offsets = None if deterministic_offset is None else torch.arange(len(users), device=device) + deterministic_offset
        support, mask, _ = local_support_batch(
            users, targets, history_tensor, history_mask_tensor, catalog_tensor, args.support_size,
            generator=torch_generator if offsets is None else None, anchor_offsets=offsets,
        )
        return support, mask

    def exhaustive_relation_inputs(
        users: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...] | None, torch.Tensor, torch.Tensor]:
        """Build one leave-target-out relation instance for every remaining prototype.

        The target prototype is used only to decide which set element to hide.  It is
        never used to select a relation anchor: every visible prototype is routed once.
        The returned owner indices map expanded relation rows back to their original
        direct-path samples so relation losses can be normalized per target sample.
        """
        assert prototype_tensor is not None and interest_mask_tensor is not None
        assert interest_weight_tensor is not None
        candidates = prototype_tensor[users]
        active_mask = interest_mask_tensor[users]
        target_features = catalog_tensor[targets].float()
        target_scores = torch.einsum("bkd,bd->bk", candidates.float(), target_features)
        target_scores = target_scores.masked_fill(~active_mask, -torch.inf)
        target_chosen = target_scores.argmax(dim=1)
        target_one_hot = F.one_hot(target_chosen, num_classes=candidates.shape[1]).bool()
        visible_mask = active_mask & ~target_one_hot
        visible_counts = visible_mask.sum(dim=1)
        owners, anchors = torch.where(visible_mask)
        if owners.numel() == 0:
            return None, owners, visible_counts

        visible_weights = interest_weight_tensor[users].float().masked_fill(~visible_mask, 0.0)
        visible_weight_sum = visible_weights.sum(dim=1, keepdim=True)
        virtual_user = torch.einsum("bk,bkd->bd", visible_weights, candidates.float())
        virtual_user = F.normalize(virtual_user / visible_weight_sum.clamp_min(1e-8), dim=-1)
        relation_inputs = (
            candidates[owners],
            visible_mask[owners],
            virtual_user[owners],
            candidates[owners, anchors],
        )
        return relation_inputs, owners, visible_counts

    def auxiliary_loss(residual: torch.Tensor, inputs, users: torch.Tensor):
        if args.variant in "bd":
            if args.cf_token:
                return model_base.user_contrastive_loss(residual, inputs[0], inputs[1], inputs[2], users)
            if args.adapter_mode == "full77_weighted_sum":
                return model_base.user_contrastive_loss(
                    residual, inputs[0], inputs[1], users, ablation=args.full77_train_source
                )
            if args.adapter_mode in (
                "preference_ip_proto_set_delta_kv_laygate",
                "preference_ip_proto_set_route_delta_kv_laygate",
                "preference_ip_unified_theme_set_kv_laygate",
            ):
                return model_base.user_contrastive_loss(residual, inputs[0], inputs[1], inputs[2], inputs[3], users)
            if args.adapter_mode in ("preference_ip_delta", "preference_ip_delta_kv", "preference_ip_delta_kv_laygate"):
                return model_base.user_contrastive_loss(residual, inputs[0], inputs[1], users)
        return model_base.user_contrastive_loss(residual, user_rep_tensor[users], users)

    def cf_user_contrastive_loss(
        residual: torch.Tensor,
        inputs,
        users: torch.Tensor,
        targets: torch.Tensor,
        *,
        relation_path: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if args.user_cf_contrast_weight <= 0.0:
            zero = torch.zeros((), device=device)
            return zero, zero
        if args.adapter_mode not in (
            "preference_ip_proto_set_delta_kv_laygate",
            "preference_ip_proto_set_route_delta_kv_laygate",
        ):
            zero = torch.zeros((), device=device)
            return zero, zero
        assert auxiliary_target_tensor is not None
        assert prototype_flat_tensor is not None and prototype_flat_mask_tensor is not None
        assert prototype_tensor is not None and interest_mask_tensor is not None and auxiliary_tensor is not None
        if len(users) <= 1:
            zero = torch.zeros((), device=device)
            return zero, zero

        user_token_count = int(getattr(model_base, "user_tokens", args.preference_user_tokens))
        anchor = model_base.user_head(residual[:, :user_token_count].mean(dim=1))
        anchor = F.normalize(anchor.float(), dim=-1)

        # inputs for prototype-set adapter are:
        # candidates, encoder_mask, auxiliary, interests
        query_auxiliary = F.normalize(inputs[2].float(), dim=-1)
        all_auxiliary = F.normalize(auxiliary_tensor.float(), dim=-1)
        user_scores = query_auxiliary @ all_auxiliary.T

        if relation_path:
            target_features = F.normalize(catalog_tensor[targets].float(), dim=-1)
            target_proto_scores = target_features @ prototype_flat_tensor.T
            target_proto_scores = target_proto_scores.masked_fill(~prototype_flat_mask_tensor[None], -torch.inf)
            target_user_scores = target_proto_scores.view(len(users), prototype_tensor.shape[0], prototype_tensor.shape[1]).max(dim=2).values
            target_weight = float(args.user_cf_target_weight)
            scores = (1.0 - target_weight) * user_scores + target_weight * target_user_scores
        else:
            scores = user_scores

        row = torch.arange(len(users), device=device)
        scores = scores.clone()
        scores[row, users] = -torch.inf

        positive_k = min(
            int(args.user_cf_positive_skip) + int(args.user_cf_positive_pool),
            max(1, scores.shape[1] - 1),
        )
        positive_ranked = torch.topk(scores, k=positive_k, dim=1).indices
        if positive_ranked.shape[1] > int(args.user_cf_positive_skip):
            positive_pool = positive_ranked[:, int(args.user_cf_positive_skip) :]
        else:
            positive_pool = positive_ranked
        if positive_pool.shape[1] > 1:
            choice = torch.randint(
                positive_pool.shape[1],
                (len(users),),
                generator=torch_generator,
                device=device,
            )
            positives = positive_pool[row, choice]
        else:
            positives = positive_pool[:, 0]

        negative_scores = scores.clone()
        negative_scores[row, users] = torch.inf
        negative_count = min(int(args.user_cf_negatives), max(1, negative_scores.shape[1] - 1))
        negatives = torch.topk(-negative_scores, k=negative_count, dim=1).indices

        positive_targets = auxiliary_target_tensor[positives]
        negative_targets = auxiliary_target_tensor[negatives]
        candidate_targets = torch.cat((positive_targets[:, None, :], negative_targets), dim=1)
        logits = torch.einsum("bd,bnd->bn", anchor, F.normalize(candidate_targets.float(), dim=-1))
        logits = logits / float(args.user_cf_temperature)
        labels = torch.zeros(len(users), dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        return loss, accuracy

    def combine_attention_kwargs(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: (
                torch.cat((first[key], second[key]), dim=0)
                if key.endswith("_tokens") or key.endswith("_mask")
                else first[key]
            )
            for key in first
        }

    def denoising_error(
        prediction: torch.Tensor,
        target_noise: torch.Tensor,
    ) -> torch.Tensor:
        return (prediction.float() - target_noise.float()).square().flatten(1).mean(dim=1)

    def min_snr_weights(timesteps: torch.Tensor) -> torch.Tensor:
        if args.min_snr_gamma <= 0.0:
            return torch.ones(len(timesteps), device=device, dtype=torch.float32)
        alpha = snr_alphas_cumprod[timesteps]
        snr = alpha / (1.0 - alpha).clamp_min(1e-8)
        return snr.clamp(max=float(args.min_snr_gamma)) / snr.clamp_min(1e-8)

    def denoising_mse(
        prediction: torch.Tensor,
        target_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return (denoising_error(prediction, target_noise) * min_snr_weights(timesteps)).mean()

    @torch.no_grad()
    def validate() -> dict[str, float]:
        model.eval()
        rows = {key: [] for key in ("mse", "wrong_mse", "correct_better")}
        generator = torch.Generator(device=device).manual_seed(args.seed + 100_000)
        for start in range(0, len(validation_users), args.batch_users):
            users = torch.from_numpy(validation_users[start : start + args.batch_users]).to(device)
            targets = torch.from_numpy(validation_targets[start : start + args.batch_users]).to(device)
            wrong_users = torch.from_numpy(wrong_validation_users[start : start + args.batch_users]).to(device)
            clean = clean_latents(validation_targets[start : start + args.batch_users])
            correct_inputs = model_inputs(users, targets, deterministic_offset=start)
            wrong_inputs = model_inputs(wrong_users, targets, deterministic_offset=start + 17)
            with torch.autocast("cuda", dtype=dtype):
                if args.adapter_mode == "full77_weighted_sum":
                    correct_context, _ = model(*correct_inputs, ablation=args.full77_train_source)
                    wrong_context, _ = model(*wrong_inputs, ablation=args.full77_train_source)
                    attention_kwargs = None
                elif args.adapter_mode in ("preference_ip_delta", "preference_ip_delta_kv", "preference_ip_delta_kv_laygate", "preference_ip_proto_set_delta_kv_laygate", "preference_ip_proto_set_route_delta_kv_laygate", "preference_ip_unified_theme_set_kv_laygate"):
                    correct_context, _, correct_attention = model(*correct_inputs)
                    wrong_context, _, wrong_attention = model(*wrong_inputs)
                    attention_kwargs = combine_attention_kwargs(correct_attention, wrong_attention)
                else:
                    correct_context, _ = model(*correct_inputs)
                    wrong_context, _ = model(*wrong_inputs)
                    attention_kwargs = None
                noise = torch.randn(clean.shape, generator=generator, device=device, dtype=dtype)
                timesteps = torch.randint(args.timestep_min, args.timestep_max + 1, (len(users),), generator=generator, device=device)
                noisy = noise_scheduler.add_noise(clean, noise, timesteps)
                prediction = unet(
                    torch.cat((noisy, noisy)), torch.cat((timesteps, timesteps)),
                    encoder_hidden_states=torch.cat((correct_context, wrong_context)).to(dtype),
                    cross_attention_kwargs=attention_kwargs,
                ).sample
                correct_error = (prediction[: len(users)].float() - noise.float()).square().flatten(1).mean(1)
                wrong_error = (prediction[len(users) :].float() - noise.float()).square().flatten(1).mean(1)
            rows["mse"].extend(correct_error.cpu().tolist())
            rows["wrong_mse"].extend(wrong_error.cpu().tolist())
            rows["correct_better"].extend((correct_error < wrong_error).float().cpu().tolist())
        return {f"validation_{key}": float(np.mean(value)) for key, value in rows.items()}

    best_score, best_step = float("inf"), 0
    global_step = 0
    last_validation: dict[str, float] = {}
    started = time.time()
    stop = False
    for epoch in range(args.epochs):
        epoch_indices = distributed_epoch_indices(len(edge_users), args.batch_users, world_size, rank, args.seed + epoch)
        for local_indices in epoch_indices:
            global_step += 1
            if global_step > planned_steps:
                stop = True
                break
            schedule_progress = (global_step - 1) / max(planned_steps - 1, 1)
            if args.masked_interest_prob_start is not None:
                current_mask_prob = float(args.masked_interest_prob_start) + schedule_progress * (
                    float(args.masked_interest_prob) - float(args.masked_interest_prob_start)
                )
            if args.interest_teacher_forcing_prob_start is not None:
                current_teacher_forcing_prob = float(
                    args.interest_teacher_forcing_prob_start
                ) + schedule_progress * (
                    float(args.interest_teacher_forcing_prob)
                    - float(args.interest_teacher_forcing_prob_start)
                )
            users_np, items_np = edge_users[local_indices], edge_items[local_indices]
            users = torch.from_numpy(users_np.astype(np.int64, copy=False)).to(device)
            targets = torch.from_numpy(items_np.astype(np.int64, copy=False)).to(device)
            clean = clean_latents(items_np)
            relation_owners = None
            relation_counts = None
            relation_instances = 0
            if args.training_path_mode == "dual_leave_target_subset":
                direct_inputs = model_inputs(users, targets, enable_masked_interest=False)
                relation_inputs = model_inputs(
                    users,
                    targets,
                    enable_masked_interest=True,
                    force_masked_interest=True,
                )
                inputs = direct_inputs
                masked_samples = int(getattr(split_inputs, "last_masked_count", 0))
                maskable_samples = int(getattr(split_inputs, "last_maskable_count", 0))
                relation_instances = int(masked_samples)
            elif args.training_path_mode == "dual_leave_target_all_anchors":
                direct_inputs = model_inputs(users, targets, enable_masked_interest=False)
                relation_inputs, relation_owners, relation_counts = exhaustive_relation_inputs(
                    users, targets
                )
                inputs = direct_inputs
                masked_samples = int((relation_counts > 0).sum().detach().cpu())
                maskable_samples = masked_samples
                relation_instances = int(relation_owners.numel())
            else:
                inputs = model_inputs(users, targets, enable_masked_interest=True)
                direct_inputs = relation_inputs = None
                masked_samples = int(getattr(split_inputs, "last_masked_count", 0))
                maskable_samples = int(getattr(split_inputs, "last_maskable_count", 0))
            wrong_train_inputs = None
            if args.condition_margin_weight > 0.0:
                wrong_offsets = torch.randint(
                    1,
                    dataset.num_users,
                    (len(users),),
                    generator=torch_generator,
                    device=device,
                )
                wrong_train_users = (users + wrong_offsets) % dataset.num_users
                wrong_train_inputs = model_inputs(
                    wrong_train_users,
                    targets,
                    enable_masked_interest=True,
                )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            condition_margin_loss = torch.zeros((), device=device)
            condition_win_rate = torch.zeros((), device=device)
            with torch.autocast("cuda", dtype=dtype):
                if args.training_path_mode == "dual_leave_target_all_anchors":
                    assert direct_inputs is not None
                    if args.adapter_mode == "preference_ip_proto_set_route_delta_kv_laygate":
                        direct_routes = torch.full(
                            (len(users),),
                            RoutedPrototypeSetPreferenceIPDeltaAdapter.DIRECT_ROUTE,
                            device=device,
                            dtype=torch.long,
                        )
                        direct_context, direct_residual, direct_attention = model(
                            *direct_inputs, route_ids=direct_routes
                        )
                    else:
                        direct_context, direct_residual, direct_attention = model(*direct_inputs)
                    noise = torch.randn(clean.shape, generator=torch_generator, device=device, dtype=dtype)
                    timesteps = torch.randint(
                        args.timestep_min,
                        args.timestep_max + 1,
                        (len(users),),
                        generator=torch_generator,
                        device=device,
                    )
                    noisy = noise_scheduler.add_noise(clean, noise, timesteps)
                    if relation_inputs is not None:
                        assert relation_owners is not None and relation_counts is not None
                        if args.adapter_mode == "preference_ip_proto_set_route_delta_kv_laygate":
                            expansion_routes = torch.full(
                                (len(relation_owners),),
                                RoutedPrototypeSetPreferenceIPDeltaAdapter.EXPANSION_ROUTE,
                                device=device,
                                dtype=torch.long,
                            )
                            relation_context, relation_residual, relation_attention = model(
                                *relation_inputs, route_ids=expansion_routes
                            )
                        else:
                            relation_context, relation_residual, relation_attention = model(*relation_inputs)
                        context = torch.cat((direct_context, relation_context), dim=0)
                        residual = torch.cat((direct_residual, relation_residual), dim=0)
                        attention_kwargs = combine_attention_kwargs(direct_attention, relation_attention)
                        relation_noisy = noisy[relation_owners]
                        relation_timesteps = timesteps[relation_owners]
                        predicted = unet(
                            torch.cat((noisy, relation_noisy), dim=0),
                            torch.cat((timesteps, relation_timesteps), dim=0),
                            encoder_hidden_states=context.to(dtype),
                            cross_attention_kwargs=attention_kwargs,
                        ).sample
                        direct_predicted = predicted[: len(users)]
                        relation_predicted = predicted[len(users) :]
                        relation_noise = noise[relation_owners]
                        relation_error = denoising_error(relation_predicted, relation_noise)
                        relation_error = relation_error * min_snr_weights(relation_timesteps)
                        relation_weights = relation_counts[relation_owners].float().reciprocal()
                        mse_relation = (relation_error * relation_weights).sum() / max(masked_samples, 1)
                    else:
                        context = direct_context
                        residual = direct_residual
                        attention_kwargs = direct_attention
                        direct_predicted = unet(
                            noisy,
                            timesteps,
                            encoder_hidden_states=context.to(dtype),
                            cross_attention_kwargs=attention_kwargs,
                        ).sample
                        mse_relation = torch.zeros((), device=device)
                    mse_direct = denoising_mse(direct_predicted, noise, timesteps)
                    mse = mse_direct + float(args.relation_loss_weight) * mse_relation
                    # The virtual relation rows are not trained to recover the original
                    # user identity; only the full-history direct path keeps this loss.
                    user_ce, user_accuracy = auxiliary_loss(direct_residual, direct_inputs, users)
                    user_cf, user_cf_accuracy = cf_user_contrastive_loss(
                        direct_residual, direct_inputs, users, targets, relation_path=False
                    )
                    residual_reg = residual.float().square().mean()
                elif args.training_path_mode == "dual_leave_target_subset":
                    assert direct_inputs is not None and relation_inputs is not None
                    if args.adapter_mode == "full77_weighted_sum":
                        direct_context, direct_residual = model(*direct_inputs, ablation=args.full77_train_source)
                        relation_context, relation_residual = model(*relation_inputs, ablation=args.full77_train_source)
                        attention_kwargs = None
                    elif args.adapter_mode in ("preference_ip_delta", "preference_ip_delta_kv", "preference_ip_delta_kv_laygate", "preference_ip_proto_set_delta_kv_laygate", "preference_ip_proto_set_route_delta_kv_laygate", "preference_ip_unified_theme_set_kv_laygate"):
                        direct_context, direct_residual, direct_attention = model(*direct_inputs)
                        relation_context, relation_residual, relation_attention = model(*relation_inputs)
                        attention_kwargs = combine_attention_kwargs(direct_attention, relation_attention)
                    else:
                        direct_context, direct_residual = model(*direct_inputs)
                        relation_context, relation_residual = model(*relation_inputs)
                        attention_kwargs = None
                    context = torch.cat((direct_context, relation_context), dim=0)
                    residual = torch.cat((direct_residual, relation_residual), dim=0)
                    noise = torch.randn(clean.shape, generator=torch_generator, device=device, dtype=dtype)
                    timesteps = torch.randint(args.timestep_min, args.timestep_max + 1, (len(users),), generator=torch_generator, device=device)
                    noisy = noise_scheduler.add_noise(clean, noise, timesteps)
                    predicted = unet(
                        torch.cat((noisy, noisy), dim=0),
                        torch.cat((timesteps, timesteps), dim=0),
                        encoder_hidden_states=context.to(dtype),
                        cross_attention_kwargs=attention_kwargs,
                    ).sample
                    direct_predicted, relation_predicted = predicted[: len(users)], predicted[len(users) :]
                    mse_direct = denoising_mse(direct_predicted, noise, timesteps)
                    mse_relation = denoising_mse(relation_predicted, noise, timesteps)
                    mse = mse_direct + float(args.relation_loss_weight) * mse_relation
                    direct_user_ce, direct_user_accuracy = auxiliary_loss(direct_residual, direct_inputs, users)
                    relation_user_ce, relation_user_accuracy = auxiliary_loss(relation_residual, relation_inputs, users)
                    user_ce = direct_user_ce + float(args.relation_loss_weight) * relation_user_ce
                    user_accuracy = 0.5 * (direct_user_accuracy + relation_user_accuracy)
                    direct_user_cf, direct_user_cf_accuracy = cf_user_contrastive_loss(
                        direct_residual, direct_inputs, users, targets, relation_path=False
                    )
                    relation_user_cf, relation_user_cf_accuracy = cf_user_contrastive_loss(
                        relation_residual, relation_inputs, users, targets, relation_path=True
                    )
                    user_cf = direct_user_cf + float(args.relation_loss_weight) * relation_user_cf
                    user_cf_accuracy = 0.5 * (direct_user_cf_accuracy + relation_user_cf_accuracy)
                    residual_reg = residual.float().square().mean()
                else:
                    if args.adapter_mode == "full77_weighted_sum":
                        context, residual = model(*inputs, ablation=args.full77_train_source)
                        attention_kwargs = None
                    elif args.adapter_mode in ("preference_ip_delta", "preference_ip_delta_kv", "preference_ip_delta_kv_laygate", "preference_ip_proto_set_delta_kv_laygate", "preference_ip_proto_set_route_delta_kv_laygate", "preference_ip_unified_theme_set_kv_laygate"):
                        context, residual, attention_kwargs = model(*inputs)
                    else:
                        context, residual = model(*inputs)
                        attention_kwargs = None
                    noise = torch.randn(clean.shape, generator=torch_generator, device=device, dtype=dtype)
                    timesteps = torch.randint(args.timestep_min, args.timestep_max + 1, (len(users),), generator=torch_generator, device=device)
                    noisy = noise_scheduler.add_noise(clean, noise, timesteps)
                    if wrong_train_inputs is not None:
                        wrong_context, _, wrong_attention_kwargs = model(*wrong_train_inputs)
                        predicted_pair = unet(
                            torch.cat((noisy, noisy), dim=0),
                            torch.cat((timesteps, timesteps), dim=0),
                            encoder_hidden_states=torch.cat((context, wrong_context), dim=0).to(dtype),
                            cross_attention_kwargs=combine_attention_kwargs(
                                attention_kwargs, wrong_attention_kwargs
                            ),
                        ).sample
                        predicted, wrong_predicted = predicted_pair.chunk(2, dim=0)
                        correct_error = denoising_error(predicted, noise)
                        wrong_error = denoising_error(wrong_predicted, noise)
                        condition_margin_loss = F.relu(
                            float(args.condition_margin) + correct_error - wrong_error
                        ).mean()
                        condition_win_rate = (correct_error < wrong_error).float().mean()
                    else:
                        predicted = unet(
                            noisy,
                            timesteps,
                            encoder_hidden_states=context.to(dtype),
                            cross_attention_kwargs=attention_kwargs,
                        ).sample
                    mse = denoising_mse(predicted, noise, timesteps)
                    mse_direct = mse
                    mse_relation = torch.zeros((), device=device)
                    user_ce, user_accuracy = auxiliary_loss(residual, inputs, users)
                    user_cf, user_cf_accuracy = cf_user_contrastive_loss(
                        residual, inputs, users, targets, relation_path=bool(args.masked_interest_prob > 0.0)
                    )
                    residual_reg = residual.float().square().mean()
                loss = (
                    mse
                    + args.user_ce_weight * user_ce
                    + args.user_cf_contrast_weight * user_cf
                    + args.condition_margin_weight * condition_margin_loss
                    + args.residual_reg_weight * residual_reg
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in list(model_base.parameters()) + list(unet_trainable_params) if parameter.requires_grad],
                args.grad_clip,
            )
            optimizer.step()
            if is_main and (global_step == 1 or global_step % args.log_every == 0):
                row = {
                    "stage": "train", "variant": args.variant, "epoch": epoch + 1, "step": global_step,
                    "planned_steps": planned_steps,
                    "seen_interactions": min(len(edge_users) * args.epochs, global_step * effective_global_batch),
                    "loss": float(loss.item()), "mse": float(mse.item()), "user_ce": float(user_ce.item()),
                    "user_cf": float(user_cf.item()),
                    "condition_margin_loss": float(condition_margin_loss.item()),
                    "condition_win_rate": float(condition_win_rate.item()),
                    "mse_direct": float(mse_direct.item()),
                    "mse_relation": float(mse_relation.item()),
                    "user_accuracy": float(user_accuracy.item()),
                    "user_cf_accuracy": float(user_cf_accuracy.item()),
                    "residual_reg": float(residual_reg.item()),
                    "residual_rms": float(residual.detach().float().square().mean().sqrt().item()),
                    "grad_norm": float(grad_norm.item()), "elapsed_seconds": round(time.time() - started, 3),
                    "gpu_memory_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
                    "masked_interest_samples": masked_samples,
                    "masked_interest_fraction": float(masked_samples / max(1, len(users))),
                    "configured_mask_probability": current_mask_prob,
                    "configured_teacher_forcing_probability": current_teacher_forcing_prob,
                    "maskable_interest_samples": maskable_samples,
                    "relation_instances": relation_instances,
                    "relation_instances_per_maskable_sample": float(
                        relation_instances / max(1, maskable_samples)
                    ),
                }
                append_jsonl(output_dir / "logs" / "train.jsonl", row)
                print(json.dumps(row), flush=True)
            epoch_boundary = global_step % steps_per_epoch == 0
            if (
                global_step % args.validation_every == 0
                or global_step == planned_steps
                or (args.checkpoint_every_epoch and epoch_boundary)
            ):
                validation = {"stage": "validation", "step": global_step, **validate()}
                last_validation = {key: float(value) for key, value in validation.items() if key.startswith("validation_")}
                if is_main:
                    append_jsonl(output_dir / "logs" / "train.jsonl", validation)
                    print(json.dumps(validation), flush=True)
                    score = validation["validation_mse"] - 0.05 * (validation["validation_wrong_mse"] - validation["validation_mse"])
                    payload = {
                        "model": model_base.state_dict(), "step": global_step, "epoch": epoch + 1,
                        "config": vars(args), "validation": validation, "image_id_order_sha256": order_sha,
                        "aggregation": (
                            "v3_final_adaptive_b" if args.adaptive_interests
                            else f"v2_{args.variant}_multi_interest"
                        ),
                    }
                    if trainable_preference_kv:
                        payload["preference_ip_processor_state"] = preference_ip_processor_state_dict(unet_base)
                    torch.save(payload, output_dir / "checkpoints" / "last.pt")
                    if score < best_score:
                        best_score, best_step = score, global_step
                        torch.save(payload, output_dir / "checkpoints" / "best.pt")
                    if global_step % args.checkpoint_every == 0:
                        torch.save(payload, output_dir / "checkpoints" / f"step_{global_step:06d}.pt")
                    if args.checkpoint_every_epoch and epoch_boundary:
                        torch.save(payload, output_dir / "checkpoints" / f"epoch_{epoch + 1:02d}.pt")
                if distributed:
                    dist.barrier(device_ids=[device.index])
        if stop:
            break

    if is_main:
        summary = {
            "variant": args.variant, "full_train_interactions": len(edge_users), "epochs": args.epochs,
            "processed_interactions_including_final_padding": planned_steps * effective_global_batch,
            "planned_steps": planned_steps, "best_step": best_step, "best_selection_score": best_score,
            "trainable_parameters": (
                sum(parameter.numel() for parameter in model_base.parameters() if parameter.requires_grad)
                + sum(parameter.numel() for parameter in unet_trainable_params if parameter.requires_grad)
            ),
            "trainable_adapter_parameters": sum(parameter.numel() for parameter in model_base.parameters() if parameter.requires_grad),
            "trainable_preference_kv_parameters": sum(parameter.numel() for parameter in unet_trainable_params if parameter.requires_grad),
            "sd15_parameters_updated": 0, "world_size": world_size, "batch_users_per_gpu": args.batch_users,
            "effective_batch_users": effective_global_batch, "elapsed_seconds": time.time() - started,
            "mean_profile_coherence": float(np.asarray(coherence).mean()),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "cf_token": bool(args.cf_token),
            "adapter_mode": args.adapter_mode,
            "ablation": args.ablation,
            "item_average_cache": str(args.item_average_cache) if args.ablation == "item_average" else None,
            "full77_train_source": args.full77_train_source,
            "context_tokens": int(args.context_tokens),
            "auxiliary_tokens": int(args.auxiliary_tokens),
            "interest_tokens": int(args.interest_tokens),
            "preference_user_tokens": int(args.preference_user_tokens),
            "preference_delta_tokens": int(args.preference_delta_tokens),
            "base_context_mode": args.base_context_mode,
            "prompt_token_slots": int(args.prompt_token_slots),
            "residual_start": int(residual_start),
            "masked_interest_prob": float(args.masked_interest_prob),
            "masked_anchor_selection": args.masked_anchor_selection,
            "interest_routing": args.interest_routing,
            "interest_teacher_forcing_prob": float(args.interest_teacher_forcing_prob),
            "interest_teacher_forcing_prob_start": args.interest_teacher_forcing_prob_start,
            "timestep_min": int(args.timestep_min),
            "timestep_max": int(args.timestep_max),
            "min_snr_gamma": float(args.min_snr_gamma),
            "condition_margin_weight": float(args.condition_margin_weight),
            "condition_margin": float(args.condition_margin),
            "training_path_mode": args.training_path_mode,
            "relation_loss_weight": float(args.relation_loss_weight),
            "relation_context_keep_prob": float(args.relation_context_keep_prob),
            "user_cf_contrast_weight": float(args.user_cf_contrast_weight),
            "user_cf_temperature": float(args.user_cf_temperature),
            "user_cf_negatives": int(args.user_cf_negatives),
            "user_cf_positive_skip": int(args.user_cf_positive_skip),
            "user_cf_positive_pool": int(args.user_cf_positive_pool),
            "user_cf_target_weight": float(args.user_cf_target_weight),
            "masked_interest_training": (
                "dual_direct_plus_leave_target_out_all_remaining_anchors"
                if args.training_path_mode == "dual_leave_target_all_anchors"
                else "dual_direct_plus_leave_target_out_subset_relation"
                if args.training_path_mode == "dual_leave_target_subset"
                else f"context_user_plus_{args.masked_anchor_selection}_visible_anchor_reconstructs_masked_target"
                if args.masked_interest_prob > 0.0
                else "disabled"
            ),
        }
        if args.cf_token:
            assert cf_user_tensor is not None and args.cf_user_cache is not None
            summary["cf_user_cache"] = str(args.cf_user_cache)
            summary["cf_embedding_dim"] = int(cf_user_tensor.shape[1])
        if args.adaptive_interests:
            assert interest_mask_tensor is not None
            interest_counts = interest_mask_tensor.detach().sum(dim=1).cpu().numpy().astype(np.int64)
            summary["adaptive_interest_count_distribution"] = {
                str(value): int((interest_counts == value).sum())
                for value in range(1, args.max_interests + 1)
            }
            summary["adaptive_interest_count_mean"] = float(interest_counts.mean())
        if hasattr(model_base, "fusion_weights"):
            weights = model_base.fusion_weights().detach().cpu().tolist()
            summary["fusion_weight_auxiliary"] = float(weights[0])
            summary["fusion_weight_interest"] = float(weights[1])
        if hasattr(model_base, "gates"):
            user_gate, delta_gate = model_base.gates()
            summary["preference_user_gate"] = float(user_gate.detach().cpu())
            summary["preference_delta_gate"] = float(delta_gate.detach().cpu())
        if hasattr(model_base, "route_summary"):
            summary.update(model_base.route_summary())
        summary.update(preference_ip_processor_gate_summary(unet_base))
        summary.update(last_validation)
        write_json(output_dir / "tables" / "training_summary.json", summary)
        print(json.dumps({"stage": "done", **summary}), flush=True)
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
