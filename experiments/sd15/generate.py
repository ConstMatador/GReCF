#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from grecf.data import id_order_sha256, load_cache, load_public_dataset
from grecf.image_embedder import DifferentiableCLIPImageEmbedder
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
)
from grecf.reference import (
    load_or_build_item_average_user_representation_cache,
    load_or_build_user_representation_cache,
)
from experiments.generation_utils import (
    build_item_users,
    make_contact_sheet,
    make_popularity_matched,
    nearest_user_knn,
    top_mean,
    write_json,
)


def farthest_anchor_offsets(features: np.ndarray, count: int) -> list[int]:
    features = np.asarray(features, dtype=np.float32)
    mean = features.mean(axis=0)
    mean /= max(float(np.linalg.norm(mean)), 1e-8)
    selected = [int(np.argmax(features @ mean))]
    nearest = features @ features[selected[0]]
    while len(selected) < min(count, len(features)):
        candidate = int(np.argmin(nearest))
        selected.append(candidate)
        nearest = np.maximum(nearest, features @ features[candidate])
    return [selected[index % len(selected)] for index in range(count)]


def weighted_interest_schedule(weights: np.ndarray, mask: np.ndarray, count: int) -> list[int]:
    active = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(active) == 0:
        raise ValueError("adaptive interest schedule has no active interests")
    probabilities = np.asarray(weights[active], dtype=np.float64)
    probabilities /= probabilities.sum()
    allocation = np.zeros(len(active), dtype=np.int64)
    if count >= len(active):
        allocation += 1
        remaining = count - len(active)
    else:
        remaining = count
    raw = probabilities * remaining
    allocation += np.floor(raw).astype(np.int64)
    missing = count - int(allocation.sum())
    if missing:
        order = np.argsort(-(raw - np.floor(raw)), kind="stable")
        allocation[order[:missing]] += 1
    schedule: list[int] = []
    while len(schedule) < count:
        for local_index, interest in enumerate(active):
            if allocation[local_index] > 0:
                schedule.append(int(interest))
                allocation[local_index] -= 1
    return schedule


def _normalize_np(values: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(norms, eps)


def _softmax_np(values: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    temperature = max(float(temperature), 1e-6)
    shifted = (values - np.max(values)) / temperature
    weights = np.exp(shifted)
    return (weights / np.maximum(weights.sum(), 1e-12)).astype(np.float32)


def v6_fused_user(
    auxiliary_all: np.ndarray,
    normalized_auxiliary_all: np.ndarray,
    user: int,
    topk: int,
    scale: float,
    temperature: float,
    selection: str = "nearest",
    prototype_all: np.ndarray | None = None,
    mask_all: np.ndarray | None = None,
    bridge_candidate_topk: int = 100,
) -> np.ndarray:
    if scale <= 0.0 or topk <= 0:
        return np.asarray(auxiliary_all[user], dtype=np.float32)
    sims = normalized_auxiliary_all @ normalized_auxiliary_all[user]
    sims = np.asarray(sims, dtype=np.float32)
    sims[user] = -np.inf
    k = min(int(topk), len(sims) - 1)
    if k <= 0:
        return np.asarray(auxiliary_all[user], dtype=np.float32)
    if selection == "bridge":
        if prototype_all is None or mask_all is None:
            raise ValueError("bridge user selection requires prototype_all and mask_all")
        candidate_k = min(max(int(bridge_candidate_topk), k), len(sims) - 1)
        candidates = np.argpartition(-sims, kth=candidate_k - 1)[:candidate_k]
        candidates = candidates[np.argsort(-sims[candidates], kind="stable")]
        target_active = np.flatnonzero(np.asarray(mask_all[user], dtype=bool))
        if len(target_active) == 0:
            indices = candidates[:k]
            weights = _softmax_np(sims[indices], temperature)
        else:
            target = _normalize_np(np.asarray(prototype_all[user, target_active], dtype=np.float32), axis=1)
            bridge_scores = []
            for candidate in candidates:
                active = np.flatnonzero(np.asarray(mask_all[int(candidate)], dtype=bool))
                if len(active) == 0:
                    bridge_scores.append(-np.inf)
                    continue
                candidate_prototypes = _normalize_np(
                    np.asarray(prototype_all[int(candidate), active], dtype=np.float32),
                    axis=1,
                )
                per_interest_match = (candidate_prototypes @ target.T).max(axis=1)
                overlap = float(per_interest_match.max())
                novelty = float(1.0 - per_interest_match.min())
                # A bridge user should be globally similar, share at least one interest,
                # and still contain at least one relatively non-overlapping interest.
                score = max(float(sims[int(candidate)]), 0.0) * max(overlap, 0.0) * max(novelty, 0.0)
                bridge_scores.append(score)
            bridge_scores = np.asarray(bridge_scores, dtype=np.float32)
            if not np.isfinite(bridge_scores).any() or float(np.nanmax(bridge_scores)) <= 0.0:
                indices = candidates[:k]
                weights = _softmax_np(sims[indices], temperature)
            else:
                order = np.argsort(-bridge_scores, kind="stable")
                indices = candidates[order[:k]]
                weights = _softmax_np(bridge_scores[order[:k]], temperature)
    else:
        indices = np.argpartition(-sims, kth=k - 1)[:k]
        indices = indices[np.argsort(-sims[indices], kind="stable")]
        weights = _softmax_np(sims[indices], temperature)
    neighbor_summary = (weights[:, None] * np.asarray(auxiliary_all[indices], dtype=np.float32)).sum(axis=0)
    fused = (1.0 - float(scale)) * np.asarray(auxiliary_all[user], dtype=np.float32) + float(scale) * neighbor_summary
    return _normalize_np(fused)


def v6_fused_delta(
    auxiliary_all: np.ndarray,
    prototype_all: np.ndarray,
    mask_all: np.ndarray,
    user: int,
    interest_index: int,
    scale: float,
    temperature: float,
) -> np.ndarray:
    auxiliary = np.asarray(auxiliary_all[user], dtype=np.float32)
    current = _normalize_np(np.asarray(prototype_all[user, interest_index], dtype=np.float32) - auxiliary)
    if scale <= 0.0:
        return current
    active = np.flatnonzero(np.asarray(mask_all[user], dtype=bool))
    active = active[active != int(interest_index)]
    if len(active) == 0:
        return current
    other = _normalize_np(np.asarray(prototype_all[user, active], dtype=np.float32) - auxiliary, axis=1)
    sims = other @ current
    weights = _softmax_np(sims, temperature)
    mixed = (weights[:, None] * other).sum(axis=0)
    fused = (1.0 - float(scale)) * current + float(scale) * mixed
    return _normalize_np(fused)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate images while cycling adaptive interest themes.")
    parser.add_argument("--variant", default="b", choices=("b", "d", "e"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/path/to/datasets/CIGR"))
    parser.add_argument("--sd15-model", type=Path, default=Path("/path/to/models/stable-diffusion-v1-5"))
    parser.add_argument("--clip-model", type=Path, default=Path("/path/to/models/clip-vit-base-patch32"))
    parser.add_argument("--clip-cache", type=Path, default=Path("/path/to/evaluation-cache/cache/public_clip_clean_features.float16.npy"))
    parser.add_argument("--user-rep-cache", type=Path, default=Path("cache/v2_cluster_balanced_user_reps.float32.npy"))
    parser.add_argument("--interest-cache-dir", type=Path, default=Path("cache/v2_multinterest"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--images-per-user", type=int, default=10)
    parser.add_argument(
        "--interest-schedule-mode",
        choices=("weighted", "two_per_interest"),
        default="weighted",
        help=(
            "For adaptive multi-interest generation, weighted keeps the historical fixed "
            "--images-per-user schedule. two_per_interest generates exactly two samples "
            "for each active interest prototype of every user."
        ),
    )
    parser.add_argument("--history-examples", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--condition-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--route-mode",
        choices=("direct", "expansion", "bernoulli"),
        default="direct",
        help="Optional direct/expansion route selection for routed adapters.",
    )
    parser.add_argument("--expansion-prob", type=float, default=0.5)
    parser.add_argument("--route-seed", type=int, default=20260825)
    parser.add_argument("--skip-contact-sheets", action="store_true")
    parser.add_argument("--skip-nearest-catalog", action="store_true")
    parser.add_argument(
        "--base-context-mode",
        choices=("prompt", "zero"),
        default="prompt",
        help="Use the checkpoint prompt context, or zero it so conditioning comes only from learned residual tokens.",
    )
    parser.add_argument(
        "--full77-ablation",
        choices=("full", "user_only", "interest_only"),
        default="full",
        help=(
            "Only for full77_weighted_sum checkpoints: keep the full weighted residual, "
            "or keep only the auxiliary/user-overall residual, or only the interest residual. "
            "The base prompt context is still retained unless --base-context-mode=zero."
        ),
    )
    parser.add_argument(
        "--selected-users-json",
        type=Path,
        required=True,
        help="Evaluation users, provided as user_indices, all_user_indices, or user_ids.",
    )
    parser.add_argument("--method-name", default="GReCF")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--generate-variants",
        nargs="+",
        choices=("global", "personalized", "wrong_user"),
        default=("personalized",),
        help="Generated variants to save; the paper evaluates personalized outputs.",
    )
    parser.add_argument(
        "--v6-fusion-mode",
        choices=("no_fusion", "user_fusion", "interest_fusion", "dual_fusion"),
        default="no_fusion",
        help=(
            "Optional generation-time preference composition. Training is unchanged; this only modifies "
            "the auxiliary/user vector and interest delta before they are projected into SD."
        ),
    )
    parser.add_argument("--v6-neighbor-topk", type=int, default=10)
    parser.add_argument("--v6-neighbor-selection", choices=("nearest", "bridge"), default="nearest")
    parser.add_argument("--v6-bridge-candidate-topk", type=int, default=100)
    parser.add_argument("--v6-neighbor-scale", type=float, default=0.35)
    parser.add_argument("--v6-neighbor-temperature", type=float, default=0.05)
    parser.add_argument("--v6-interest-scale", type=float, default=0.35)
    parser.add_argument("--v6-interest-temperature", type=float, default=0.10)
    args = parser.parse_args()
    if not 0.0 <= args.expansion_prob <= 1.0:
        parser.error("--expansion-prob must be in [0, 1]")

    from diffusers import AutoencoderKL, DDIMScheduler, PNDMScheduler, UNet2DConditionModel
    from transformers import CLIPImageProcessor, CLIPModel, CLIPTextModel, CLIPTokenizer

    for child in ("images", "contact_sheets", "tables", "reports"):
        (args.output_dir / child).mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    device, dtype = torch.device(args.device), torch.bfloat16
    dataset = load_public_dataset(args.dataset_root)
    order_sha = id_order_sha256(dataset.image_ids)
    clip_features = load_cache(args.clip_cache, dataset.num_items, order_sha, (512,))
    user_reps, coherence = load_or_build_user_representation_cache(args.user_rep_cache, dataset, clip_features)
    selection_payload = json.loads(args.selected_users_json.read_text(encoding="utf-8"))
    if "user_indices" in selection_payload:
        all_selected_users = [int(value) for value in selection_payload["user_indices"]]
    elif "all_user_indices" in selection_payload:
        all_selected_users = [int(value) for value in selection_payload["all_user_indices"]]
    elif "user_ids" in selection_payload:
        index_by_id = {str(user_id): index for index, user_id in enumerate(dataset.user_ids)}
        all_selected_users = [index_by_id[str(value)] for value in selection_payload["user_ids"]]
    else:
        raise ValueError("selected user JSON must contain user_indices, all_user_indices, or user_ids")
    if not all_selected_users or len(set(all_selected_users)) != len(all_selected_users):
        raise ValueError("selected users must be non-empty and unique")
    if min(all_selected_users) < 0 or max(all_selected_users) >= dataset.num_users:
        raise ValueError("selected users contain an out-of-range index")
    selection_description = selection_payload.get("selection", "predefined evaluation users")
    selected_users = all_selected_users[args.shard_index :: args.num_shards]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config, state = checkpoint["config"], checkpoint["model"]
    if config["variant"] != args.variant:
        raise ValueError(f"checkpoint variant {config['variant']} does not match --variant {args.variant}")
    adaptive_interests = bool(config.get("adaptive_interests", False))
    cf_token = bool(config.get("cf_token", False))
    adapter_mode = str(config.get("adapter_mode", "split"))
    ablation = str(config.get("ablation", "full"))
    if adapter_mode not in (
        "preference_ip_proto_set_route_delta_kv_laygate",
    ) and args.route_mode != "direct":
        raise ValueError("non-direct route modes require a routed adapter checkpoint")
    if cf_token:
        cf_user_cache = Path(config["cf_user_cache"])
        cf_user_embeddings = np.load(cf_user_cache)
        if cf_user_embeddings.shape[0] != dataset.num_users:
            raise ValueError(
                f"CF user cache has {cf_user_embeddings.shape[0]} users, expected {dataset.num_users}"
            )
        cf_user_tensor = torch.from_numpy(np.asarray(cf_user_embeddings, dtype=np.float32)).to(device)
    else:
        cf_user_tensor = None
    if args.variant in "bd":
        if adaptive_interests:
            interest_cache = load_or_build_adaptive_interest_cache(
                args.interest_cache_dir, dataset, clip_features,
                max_interests=int(config["max_interests"]),
                folds=int(config["interest_cv_folds"]),
                minimum_support=int(config["interest_min_support"]),
                minimum_support_fraction=float(config["interest_min_support_fraction"]),
                merge_cosine=float(config["interest_merge_cosine"]),
                retained_gain=float(config["interest_retained_gain"]),
                seed=int(config["seed"]),
            )
        else:
            interest_cache = load_or_build_interest_cache(args.interest_cache_dir, dataset, clip_features, args.variant)
        if adapter_mode == "full77_weighted_sum":
            model = FullContextWeightedSumAdapter(
                state["base_context"], state["auxiliary_center"], state["prototype_center"],
                context_tokens=int(config.get("context_tokens", 77)),
                residual_scale=float(config["residual_scale"]),
            ).to(device).eval()
        elif adapter_mode in ("preference_ip_delta", "preference_ip_delta_kv", "preference_ip_delta_kv_laygate"):
            model = PreferenceIPDeltaAdapter(
                state["base_context"], state["auxiliary_center"], state["prototype_center"],
                user_tokens=int(config.get("preference_user_tokens", 8)),
                delta_tokens=int(config.get("preference_delta_tokens", 8)),
                residual_scale=float(config["residual_scale"]),
            ).to(device).eval()
        elif adapter_mode == "preference_ip_unified_theme_set_kv_laygate":
            model = UnifiedThemeSetPreferenceIPAdapter(
                state["base_context"], state["auxiliary_center"], state["prototype_center"],
                user_tokens=int(config.get("preference_user_tokens", 8)),
                delta_tokens=int(config.get("preference_delta_tokens", 8)),
                residual_scale=float(config["residual_scale"]),
                use_theme_anchor=ablation != "no_theme_anchor",
                user_representation_mode="item_average" if ablation == "item_average" else "theme_set",
            ).to(device).eval()
        elif adapter_mode in (
            "preference_ip_proto_set_delta_kv_laygate",
            "preference_ip_proto_set_route_delta_kv_laygate",
        ):
            adapter_class = (
                RoutedPrototypeSetPreferenceIPDeltaAdapter
                if adapter_mode == "preference_ip_proto_set_route_delta_kv_laygate"
                else PrototypeSetPreferenceIPDeltaAdapter
            )
            model = adapter_class(
                state["base_context"], state["auxiliary_center"], state["prototype_center"],
                user_tokens=int(config.get("preference_user_tokens", 8)),
                delta_tokens=int(config.get("preference_delta_tokens", 8)),
                residual_scale=float(config["residual_scale"]),
            ).to(device).eval()
        elif cf_token:
            model = SplitInterestCFAdapter(
                state["base_context"], state["auxiliary_center"], state["prototype_center"], state["cf_center"],
                residual_scale=float(config["residual_scale"]),
            ).to(device).eval()
        else:
            model = SplitInterestAdapter(
                state["base_context"], state["auxiliary_center"], state["prototype_center"],
                auxiliary_tokens=int(config.get("auxiliary_tokens", 16)),
                interest_tokens=int(config.get("interest_tokens", 48)),
                residual_scale=float(config["residual_scale"]),
                residual_start=int(config.get("residual_start", 0)),
            ).to(device).eval()
        histories = history_masks = catalog_tensor = None
        auxiliary_all_np = np.asarray(interest_cache["auxiliary"], dtype=np.float32)
        prototype_all_np = np.asarray(interest_cache["prototypes"], dtype=np.float32)
        weight_all_np = np.asarray(interest_cache["weights"], dtype=np.float32)
        if adaptive_interests and "mask" in interest_cache:
            mask_all_np = np.asarray(interest_cache["mask"], dtype=bool)
        else:
            mask_all_np = np.ones(prototype_all_np.shape[:2], dtype=bool)
        if ablation == "item_average":
            item_average_cache = Path(config["item_average_cache"])
            item_average_np, _ = load_or_build_item_average_user_representation_cache(
                item_average_cache, dataset, clip_features
            )
            auxiliary_all_np = np.asarray(item_average_np, dtype=np.float32)
        normalized_auxiliary_all_np = _normalize_np(auxiliary_all_np, axis=1)
    else:
        interest_cache = None
        auxiliary_all_np = prototype_all_np = weight_all_np = mask_all_np = normalized_auxiliary_all_np = None
        histories_np, masks_np = padded_train_histories(dataset)
        histories = torch.from_numpy(histories_np).to(device)
        history_masks = torch.from_numpy(masks_np).to(device)
        catalog_tensor = torch.from_numpy(np.asarray(clip_features, dtype=np.float32)).to(device=device, dtype=dtype)
        model = HistoryResamplerAdapter(
            state["base_context"], state["representation_center"], residual_scale=float(config["residual_scale"])
        ).to(device).eval()
    model.load_state_dict(state)
    if args.base_context_mode == "zero":
        model.base_context.zero_()

    tokenizer = CLIPTokenizer.from_pretrained(args.sd15_model, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(
        args.sd15_model, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True
    ).to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(
        args.sd15_model, subfolder="unet", torch_dtype=dtype, local_files_only=True
    ).to(device).eval()
    if adapter_mode == "preference_ip_delta":
        install_preference_ip_processors(unet)
    elif adapter_mode in ("preference_ip_delta_kv", "preference_ip_delta_kv_laygate", "preference_ip_proto_set_delta_kv_laygate", "preference_ip_proto_set_route_delta_kv_laygate", "preference_ip_unified_theme_set_kv_laygate"):
        install_preference_ip_trainable_kv_processors(
            unet,
            layer_gates=adapter_mode in ("preference_ip_delta_kv_laygate", "preference_ip_proto_set_delta_kv_laygate", "preference_ip_proto_set_route_delta_kv_laygate", "preference_ip_unified_theme_set_kv_laygate"),
        )
        load_preference_ip_processor_state_dict(unet, checkpoint["preference_ip_processor_state"])
    vae = AutoencoderKL.from_pretrained(
        args.sd15_model, subfolder="vae", torch_dtype=dtype, local_files_only=True
    ).to(device).eval()
    clip_processor = CLIPImageProcessor.from_pretrained(args.clip_model, local_files_only=True)
    clip_model = CLIPModel.from_pretrained(args.clip_model, torch_dtype=dtype, local_files_only=True).to(device).eval()
    for frozen in (text_encoder, unet, vae, clip_model, model):
        frozen.requires_grad_(False)
    scheduler = DDIMScheduler.from_config(
        PNDMScheduler.load_config(args.sd15_model, subfolder="scheduler", local_files_only=True)
    )
    empty_ids = tokenizer("", padding="max_length", max_length=77, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        empty_context = text_encoder(empty_ids).last_hidden_state
    image_embedder = DifferentiableCLIPImageEmbedder(
        clip_model, int(clip_processor.crop_size["height"]), clip_processor.image_mean, clip_processor.image_std
    ).to(device)

    item_counts = np.zeros(dataset.num_items, dtype=np.int64)
    for items in dataset.train:
        np.add.at(item_counts, items, 1)
    item_users = build_item_users(dataset)
    rows: list[dict[str, object]] = []
    generated_features: list[np.ndarray] = []
    base_context = model.base_context.to(device=device, dtype=dtype)

    def cat_attention_kwargs(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            key: (
                torch.cat([part[key] for part in parts], dim=0)
                if key.endswith("_tokens") or key.endswith("_mask")
                # For scalar gates/scales, preserve the conditional branch scale
                # when concatenating [unconditional, conditional] CFG kwargs.
                else parts[-1][key]
            )
            for key in parts[0]
        }

    def compose_v6_condition(target_user: int, target_interest_index: int) -> tuple[np.ndarray, np.ndarray]:
        if args.variant not in "bd":
            raise ValueError("preference fusion is only defined for multi-interest variants B/D")
        assert auxiliary_all_np is not None
        assert prototype_all_np is not None
        assert mask_all_np is not None
        assert normalized_auxiliary_all_np is not None
        base_auxiliary = np.asarray(auxiliary_all_np[target_user], dtype=np.float32)
        base_prototype = np.asarray(prototype_all_np[target_user, target_interest_index], dtype=np.float32)
        base_delta = _normalize_np(base_prototype - base_auxiliary)
        if args.v6_fusion_mode in ("user_fusion", "dual_fusion"):
            auxiliary = v6_fused_user(
                auxiliary_all_np,
                normalized_auxiliary_all_np,
                target_user,
                args.v6_neighbor_topk,
                args.v6_neighbor_scale,
                args.v6_neighbor_temperature,
                selection=args.v6_neighbor_selection,
                prototype_all=prototype_all_np,
                mask_all=mask_all_np,
                bridge_candidate_topk=args.v6_bridge_candidate_topk,
            )
        else:
            auxiliary = base_auxiliary
        if args.v6_fusion_mode in ("interest_fusion", "dual_fusion"):
            delta = v6_fused_delta(
                auxiliary_all_np,
                prototype_all_np,
                mask_all_np,
                target_user,
                target_interest_index,
                args.v6_interest_scale,
                args.v6_interest_temperature,
            )
            interest = auxiliary + delta
        elif args.v6_fusion_mode in ("user_fusion", "dual_fusion"):
            interest = auxiliary + base_delta
        else:
            interest = base_prototype
        return np.asarray(auxiliary, dtype=np.float32), np.asarray(interest, dtype=np.float32)

    for user in selected_users:
        rng = np.random.default_rng(args.seed + 71 + user * 1_000_003)
        similarities = np.asarray(user_reps, dtype=np.float32) @ np.asarray(user_reps[user], dtype=np.float32)
        similarities[user] = -np.inf
        neighbor, neighbor_overlap, neighbor_jaccard = nearest_user_knn(dataset, item_users, user)
        random_pool = rng.choice(dataset.num_users, size=min(2000, dataset.num_users), replace=False)
        random_pool = random_pool[random_pool != user]
        wrong_user = int(random_pool[np.argmin(similarities[random_pool])])
        own_set = set(dataset.train[user].tolist())
        neighbor_only = np.asarray([item for item in dataset.train[neighbor] if int(item) not in own_set], dtype=np.int64)
        matched_random = make_popularity_matched(neighbor_only, item_counts, own_set | set(neighbor_only.tolist()), rng)
        user_image_count = int(args.images_per_user)
        if adaptive_interests and args.interest_schedule_mode == "two_per_interest":
            assert mask_all_np is not None
            user_image_count = int(2 * int(mask_all_np[user].sum()))
        if args.variant == "e":
            own_anchors = farthest_anchor_offsets(clip_features[dataset.train[user]], user_image_count)
            wrong_anchors = farthest_anchor_offsets(clip_features[dataset.train[wrong_user]], user_image_count)
        else:
            own_anchors = wrong_anchors = list(range(user_image_count))
        if adaptive_interests:
            assert mask_all_np is not None and weight_all_np is not None
            if args.interest_schedule_mode == "two_per_interest":
                own_active = np.flatnonzero(mask_all_np[user])
                own_interest_schedule = [int(index) for index in own_active for _ in range(2)]
                wrong_interest_schedule = weighted_interest_schedule(
                    weight_all_np[wrong_user], mask_all_np[wrong_user], user_image_count
                )
            else:
                own_interest_schedule = weighted_interest_schedule(
                    weight_all_np[user], mask_all_np[user], user_image_count
                )
                wrong_interest_schedule = weighted_interest_schedule(
                    weight_all_np[wrong_user], mask_all_np[wrong_user], user_image_count
                )
        else:
            own_interest_schedule = wrong_interest_schedule = []
        requested_variants = tuple(args.generate_variants)
        generated_paths: dict[str, list[Path]] = {
            variant.replace("_", " "): [] for variant in requested_variants
        }

        for seed_offset in range(user_image_count):
            if args.route_mode == "direct":
                route_name = "direct"
                route_id = RoutedPrototypeSetPreferenceIPDeltaAdapter.DIRECT_ROUTE
            elif args.route_mode == "expansion":
                route_name = "expansion"
                route_id = RoutedPrototypeSetPreferenceIPDeltaAdapter.EXPANSION_ROUTE
            else:
                route_rng = np.random.default_rng(
                    int(args.route_seed) + int(user) * 1_000_003 + int(seed_offset) * 97_409
                )
                is_expansion = bool(route_rng.random() < float(args.expansion_prob))
                route_name = "expansion" if is_expansion else "direct"
                route_id = (
                    RoutedPrototypeSetPreferenceIPDeltaAdapter.EXPANSION_ROUTE
                    if is_expansion
                    else RoutedPrototypeSetPreferenceIPDeltaAdapter.DIRECT_ROUTE
                )
            if args.variant in "bd":
                assert interest_cache is not None
                interest_index = own_interest_schedule[seed_offset] if adaptive_interests else seed_offset % 4
                wrong_interest_index = (
                    wrong_interest_schedule[seed_offset] if adaptive_interests else interest_index
                )
                own_auxiliary_np, own_interest_np = compose_v6_condition(user, interest_index)
                wrong_auxiliary_np, wrong_interest_np = compose_v6_condition(wrong_user, wrong_interest_index)
                auxiliary = torch.from_numpy(
                    np.stack([own_auxiliary_np, wrong_auxiliary_np]).astype(np.float32)
                ).to(device)
                interests = torch.from_numpy(
                    np.stack([own_interest_np, wrong_interest_np]).astype(np.float32)
                ).to(device)
                if cf_token:
                    assert cf_user_tensor is not None
                    cf = cf_user_tensor[torch.tensor([user, wrong_user], device=device)]
                    model_args = (auxiliary, interests, cf)
                elif adapter_mode in (
                    "preference_ip_proto_set_delta_kv_laygate",
                    "preference_ip_proto_set_route_delta_kv_laygate",
                    "preference_ip_unified_theme_set_kv_laygate",
                ):
                    prototypes = torch.from_numpy(
                        np.stack([prototype_all_np[user], prototype_all_np[wrong_user]]).astype(np.float32)
                    ).to(device)
                    prototype_mask = torch.from_numpy(
                        np.stack([mask_all_np[user], mask_all_np[wrong_user]]).astype(bool)
                    ).to(device)
                    model_args = (prototypes, prototype_mask, auxiliary, interests)
                else:
                    model_args = (auxiliary, interests)
                anchor_item = wrong_anchor_item = -1
            else:
                assert histories is not None and history_masks is not None and catalog_tensor is not None
                pair_users = torch.tensor([user, wrong_user], device=device)
                offsets = torch.tensor([own_anchors[seed_offset], wrong_anchors[seed_offset]], device=device)
                support, support_mask, anchor_items = local_support_batch(
                    pair_users, None, histories, history_masks, catalog_tensor,
                    int(config["support_size"]), anchor_offsets=offsets,
                )
                model_args = (support, support_mask)
                anchor_item, wrong_anchor_item = [int(value) for value in anchor_items.tolist()]
                interest_index = seed_offset
                wrong_interest_index = seed_offset
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                if adapter_mode == "full77_weighted_sum":
                    conditions, _ = model(
                        *model_args,
                        multiplier=args.condition_multiplier,
                        ablation=args.full77_ablation,
                    )
                    attention_pair = None
                elif adapter_mode in ("preference_ip_delta", "preference_ip_delta_kv", "preference_ip_delta_kv_laygate", "preference_ip_proto_set_delta_kv_laygate", "preference_ip_proto_set_route_delta_kv_laygate", "preference_ip_unified_theme_set_kv_laygate"):
                    if args.full77_ablation != "full":
                        raise ValueError("--full77-ablation is only supported for full77_weighted_sum checkpoints")
                    if adapter_mode in (
                        "preference_ip_proto_set_route_delta_kv_laygate",
                    ):
                        pair_route_ids = torch.full(
                            (len(model_args[0]),), route_id, device=device, dtype=torch.long
                        )
                        conditions, _, attention_pair = model(
                            *model_args,
                            multiplier=args.condition_multiplier,
                            route_ids=pair_route_ids,
                        )
                    else:
                        conditions, _, attention_pair = model(
                            *model_args,
                            multiplier=args.condition_multiplier,
                        )
                else:
                    if args.full77_ablation != "full":
                        raise ValueError("--full77-ablation is only supported for full77_weighted_sum checkpoints")
                    conditions, _ = model(*model_args, multiplier=args.condition_multiplier)
                    attention_pair = None
                user_context, wrong_context = conditions[0:1].to(dtype), conditions[1:2].to(dtype)
                context_by_variant = {
                    "global": base_context,
                    "personalized": user_context,
                    "wrong_user": wrong_context,
                }
                conditional_context = torch.cat(
                    [context_by_variant[variant] for variant in requested_variants], dim=0
                )
                if adapter_mode in ("preference_ip_delta", "preference_ip_delta_kv", "preference_ip_delta_kv_laygate", "preference_ip_proto_set_delta_kv_laygate", "preference_ip_proto_set_route_delta_kv_laygate", "preference_ip_unified_theme_set_kv_laygate"):
                    assert attention_pair is not None
                    zero_attention = model.zero_attention_kwargs(1, device, dtype)
                    attention_by_variant = {
                        "global": zero_attention,
                        "personalized": {
                            key: (
                                value[0:1].bool()
                                if key.endswith("_mask")
                                else value[0:1].to(dtype)
                                if key.endswith("_tokens")
                                else value.to(dtype)
                            )
                            for key, value in attention_pair.items()
                        },
                        "wrong_user": {
                            key: (
                                value[1:2].bool()
                                if key.endswith("_mask")
                                else value[1:2].to(dtype)
                                if key.endswith("_tokens")
                                else value.to(dtype)
                            )
                            for key, value in attention_pair.items()
                        },
                    }
                    conditional_attention = cat_attention_kwargs([
                        attention_by_variant[variant] for variant in requested_variants
                    ])
                    unconditional_attention = model.zero_attention_kwargs(variant_count := len(requested_variants), device, dtype)
                    guidance_attention = cat_attention_kwargs([unconditional_attention, conditional_attention])
                else:
                    guidance_attention = None

            seed = args.seed + user * 10_000 + seed_offset
            generator = torch.Generator(device=device).manual_seed(seed)
            scheduler.set_timesteps(args.steps, device=device)
            initial = torch.randn((1, 4, 64, 64), generator=generator, device=device, dtype=dtype)
            variant_count = len(requested_variants)
            latents = initial.repeat(variant_count, 1, 1, 1) * scheduler.init_noise_sigma
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                for timestep in scheduler.timesteps:
                    scaled = scheduler.scale_model_input(latents, timestep)
                    prediction = unet(
                        torch.cat((scaled, scaled)), timestep.expand(2 * variant_count),
                        encoder_hidden_states=torch.cat((
                            empty_context.expand(variant_count, -1, -1), conditional_context
                        )),
                        cross_attention_kwargs=guidance_attention,
                    ).sample
                    unconditional, conditional = prediction[:variant_count], prediction[variant_count:]
                    latents = scheduler.step(
                        unconditional + args.guidance_scale * (conditional - unconditional), timestep, latents
                    ).prev_sample
                decoded = vae.decode(latents / float(vae.config.scaling_factor)).sample
                embeddings = image_embedder(decoded).float().cpu().numpy()

            for variant_index, generated_variant in enumerate(requested_variants):
                image = decoded[variant_index].float().cpu().add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                image_path = args.output_dir / "images" / f"{dataset.user_ids[user]}_seed{seed_offset}_{generated_variant}.png"
                Image.fromarray(image).save(image_path)
                feature = embeddings[variant_index]
                generated_features.append(feature)
                train_scores = clip_features[dataset.train[user]] @ feature
                test_scores = clip_features[dataset.test[user]] @ feature
                if args.variant in "bd" and adaptive_interests:
                    assert prototype_all_np is not None and mask_all_np is not None
                    active_topic_indices = np.flatnonzero(np.asarray(mask_all_np[user], dtype=bool))
                    active_prototypes = np.asarray(prototype_all_np[user, active_topic_indices], dtype=np.float32)
                    active_prototypes = active_prototypes / np.maximum(
                        np.linalg.norm(active_prototypes, axis=1, keepdims=True), 1e-8
                    )
                    topic_scores = active_prototypes @ feature
                    assigned_local = int(np.argmax(topic_scores))
                    assigned_interest_index = int(active_topic_indices[assigned_local])
                    assigned_interest_similarity = float(topic_scores[assigned_local])
                else:
                    assigned_interest_index = -1
                    assigned_interest_similarity = float("nan")
                rows.append({
                    "method": args.method_name or f"v2-{args.variant.upper()}", "user_index": user, "user_id": dataset.user_ids[user],
                    "neighbor_index": neighbor, "neighbor_id": dataset.user_ids[neighbor],
                    "neighbor_train_overlap": neighbor_overlap, "neighbor_train_jaccard": neighbor_jaccard,
                    "wrong_user_index": wrong_user, "wrong_user_id": dataset.user_ids[wrong_user],
                    "seed": seed, "seed_offset": seed_offset, "interest_index": interest_index,
                    "route": route_name,
                    "route_id": route_id,
                    "route_mode": args.route_mode,
                    "expansion_prob": float(args.expansion_prob),
                    "wrong_interest_index": wrong_interest_index,
                    "interest_count": int(mask_all_np[user].sum()) if adaptive_interests else 4 if args.variant in "bd" else args.images_per_user,
                    "wrong_interest_count": int(mask_all_np[wrong_user].sum()) if adaptive_interests else 4 if args.variant in "bd" else args.images_per_user,
                    "interest_weight": float(weight_all_np[user, interest_index]) if args.variant in "bd" else float("nan"),
                    "anchor_item_index": anchor_item if generated_variant == "personalized" else wrong_anchor_item if generated_variant == "wrong_user" else -1,
                    "anchor_image_id": dataset.image_ids[anchor_item] if generated_variant == "personalized" and anchor_item >= 0 else dataset.image_ids[wrong_anchor_item] if generated_variant == "wrong_user" and wrong_anchor_item >= 0 else "",
                    "variant": generated_variant, "image_path": str(image_path),
                    "train_maxsim": float(train_scores.max()) if len(train_scores) else float("nan"),
                    "test_maxsim": float(test_scores.max()) if len(test_scores) else float("nan"),
                    "assigned_train_topic_index": assigned_interest_index,
                    "assigned_train_topic_similarity": assigned_interest_similarity,
                    "v6_fusion_mode": args.v6_fusion_mode,
                    "v6_neighbor_selection": args.v6_neighbor_selection,
                    "v6_neighbor_topk": args.v6_neighbor_topk,
                    "v6_bridge_candidate_topk": args.v6_bridge_candidate_topk,
                    "v6_neighbor_scale": args.v6_neighbor_scale,
                    "v6_interest_scale": args.v6_interest_scale,
                    "train_top5": top_mean(feature, clip_features, dataset.train[user]),
                    "validation_top5": top_mean(feature, clip_features, dataset.validation[user]),
                    "test_top5": top_mean(feature, clip_features, dataset.test[user]),
                    "neighbor_only_top5": top_mean(feature, clip_features, neighbor_only),
                    "popularity_matched_top5": top_mean(feature, clip_features, matched_random),
                })
                generated_paths[generated_variant.replace("_", " ")].append(image_path)

        if not args.skip_contact_sheets:
            history_count = min(args.history_examples, len(dataset.train[user]))
            history_items = rng.choice(dataset.train[user], size=history_count, replace=False)
            heldout = np.concatenate((dataset.validation[user], dataset.test[user]))
            make_contact_sheet(
                args.output_dir / "contact_sheets" / f"{dataset.user_ids[user]}.png", dataset, user,
                history_items, heldout, generated_paths,
            )

    if args.skip_nearest_catalog:
        for row in rows:
            row["nearest_catalog_index"] = -1
            row["nearest_catalog_image_id"] = ""
            row["nearest_catalog_similarity"] = float("nan")
    else:
        all_generated = torch.from_numpy(np.stack(generated_features)).to(device)
        best_scores = torch.full((len(rows),), -torch.inf, device=device)
        best_indices = torch.zeros(len(rows), dtype=torch.long, device=device)
        for start in range(0, dataset.num_items, 10_000):
            catalog = torch.from_numpy(np.asarray(clip_features[start : start + 10_000], dtype=np.float32)).to(device)
            values, indices = (all_generated @ catalog.T).max(dim=1)
            update = values > best_scores
            best_scores[update], best_indices[update] = values[update], indices[update] + start
        for row, item, score in zip(rows, best_indices.cpu().tolist(), best_scores.cpu().tolist()):
            row["nearest_catalog_index"] = item
            row["nearest_catalog_image_id"] = dataset.image_ids[item]
            row["nearest_catalog_similarity"] = score
    suffix = "" if args.num_shards == 1 else f"_shard{args.shard_index}"
    with (args.output_dir / "tables" / f"generation_metrics{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if generated_features:
        np.save(
            args.output_dir / "tables" / f"generated_clip_features{suffix}.float32.npy",
            np.stack(generated_features).astype(np.float32),
        )
    write_json(args.output_dir / "tables" / f"selected_users{suffix}.json", {
        "selection": selection_description, "all_user_indices": all_selected_users,
        "user_indices": selected_users, "user_ids": [dataset.user_ids[user] for user in selected_users],
        "checkpoint": str(args.checkpoint), "checkpoint_step": checkpoint["step"], "variant": args.variant,
        "images_per_user": args.images_per_user, "guidance_scale": args.guidance_scale,
        "interest_schedule_mode": args.interest_schedule_mode,
        "condition_multiplier": args.condition_multiplier,
        "skip_contact_sheets": args.skip_contact_sheets,
        "skip_nearest_catalog": args.skip_nearest_catalog,
        "base_context_mode": args.base_context_mode,
        "full77_ablation": args.full77_ablation,
        "generate_variants": list(requested_variants),
        "checkpoint_base_context_mode": config.get("base_context_mode", "prompt"),
        "prompt_token_slots": int(config.get("prompt_token_slots", 0)),
        "residual_start": int(config.get("residual_start", 0)),
        "adapter_mode": adapter_mode,
        "context_tokens": int(config.get("context_tokens", 0)),
        "preference_user_tokens": int(config.get("preference_user_tokens", 0)),
        "preference_delta_tokens": int(config.get("preference_delta_tokens", 0)),
        "adaptive_interests": adaptive_interests,
        "ablation": ablation,
        "cf_token": cf_token,
        "auxiliary_tokens": int(config.get("auxiliary_tokens", 16)),
        "interest_tokens": int(config.get("interest_tokens", 48)),
        "v6_fusion_mode": args.v6_fusion_mode,
        "v6_neighbor_selection": args.v6_neighbor_selection,
        "v6_neighbor_topk": args.v6_neighbor_topk,
        "v6_bridge_candidate_topk": args.v6_bridge_candidate_topk,
        "v6_neighbor_scale": args.v6_neighbor_scale,
        "v6_neighbor_temperature": args.v6_neighbor_temperature,
        "v6_interest_scale": args.v6_interest_scale,
        "v6_interest_temperature": args.v6_interest_temperature,
        "route_mode": args.route_mode,
        "expansion_prob": float(args.expansion_prob),
        "route_seed": int(args.route_seed),
    })
    print(json.dumps({"stage": "done", "variant": args.variant, "users": len(selected_users), "images": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
