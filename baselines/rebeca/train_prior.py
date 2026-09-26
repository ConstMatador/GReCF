#!/usr/bin/env python3
"""Train the REBECA user-conditioned diffusion prior (adapter step 2).

Imports the upstream training loop (`train_diffusion_prior`) and model
(`RebecaDiffusionPrior`) unchanged; only the data
loading is adapted: rows come from the prepare.py ratings.csv and each row's
target embedding is looked up in the encode_catalog.py matrix by image_index.

Normalization follows upstream train_priors.py: L2-normalize then scale by
sqrt(D), clamp to [-5, 5], divide by 5.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--ratings-csv", type=Path, required=True, help="prepare.py output")
    parser.add_argument("--catalog-features", type=Path, required=True,
                        help="encode_catalog.py output .pt [num_items, D] float16")
    parser.add_argument("--output-dir", type=Path, required=True, help="storage run dir for weights/config")
    parser.add_argument("--img-embed-dim", type=int, default=1024)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=2048,
                        help="kept for CLI compatibility; RebecaDiffusionPrior uses hidden_dim")
    parser.add_argument("--samples-per-user", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--beta-schedule", default="squaredcos_cap_v2",
                        choices=["squaredcos_cap_v2", "laplace", "linear"])
    parser.add_argument("--objective", default="sample", choices=["noise-pred", "epsilon", "sample", "v_prediction"])
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, str(args.upstream_root))
    sys.path.insert(0, str(REPO_ROOT))

    import random

    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from torch.utils.data import DataLoader

    # Upstream modules (import after sys.path setup).
    from Datasets import EmbeddingsDataset, RecommenderUserSampler  # noqa: E402
    from prior_models import RebecaDiffusionPrior  # noqa: E402
    from train_priors import train_diffusion_prior  # noqa: E402
    from utils import save_rebeca_config, set_seeds  # noqa: E402

    set_seeds(args.seed)

    ratings = pd.read_csv(args.ratings_csv)
    train_df = ratings.loc[ratings["split"] == "train"].reset_index(drop=True)
    val_df = ratings.loc[ratings["split"] == "validation"].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise SystemExit("train/validation partitions missing in ratings.csv")

    catalog = torch.load(args.catalog_features, weights_only=True).float()
    dim_per_row = catalog.shape[1]
    if args.img_embed_dim != dim_per_row:
        print(f"note: --img-embed-dim {args.img_embed_dim} ignored, using stored feature dim {dim_per_row}")
        args.img_embed_dim = int(dim_per_row)

    def expanded(rows: pd.DataFrame) -> torch.Tensor:
        indices = torch.from_numpy(rows["image_index"].to_numpy(dtype=np.int64))
        targets = catalog[indices]
        norms = targets.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scaled = targets / norms * math.sqrt(targets.shape[-1])
        return torch.clamp(scaled, -5.0, 5.0) / 5.0

    num_users = int(ratings["worker_id"].max()) + 1  # global ids so null token id == num_users

    train_dataset = EmbeddingsDataset(
        pd.DataFrame({"worker_id": train_df["worker_id"], "score": train_df["score"],
                      "imagePair": train_df["imagePair"]}),
        image_embeddings=expanded(train_df),
    )
    val_dataset = EmbeddingsDataset(
        pd.DataFrame({"worker_id": val_df["worker_id"], "score": val_df["score"],
                      "imagePair": val_df["imagePair"]}),
        image_embeddings=expanded(val_df),
    )
    unique_users = train_df["worker_id"].unique()
    worker_frame = pd.DataFrame({"worker_id": train_df["worker_id"]})
    train_sampler = RecommenderUserSampler(
        worker_frame,
        num_users=len(unique_users),
        samples_per_user=args.samples_per_user,
    )
    # The prior is tiny, so per-step time is data-loading bound; worker
    # processes parallelize batch assembly without touching training numerics
    # (the user sampler still runs in the main process, order unchanged).
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=max(2, args.num_workers // 2), pin_memory=True)

    model = RebecaDiffusionPrior(
        img_embed_dim=args.img_embed_dim,
        num_users=num_users,
        num_tokens=args.num_tokens,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        score_classes=2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps,
                                    beta_schedule=args.beta_schedule)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.5)

    weights_dir = args.output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    savepath = str(weights_dir / "prior.pth")

    weights_file_meta = {
        "rows_per_epoch": len(train_sampler),
        "batches_per_epoch": math.ceil(len(train_sampler) / args.batch_size),
    }
    print(f"users={num_users} trainable_params={sum(p.numel() for p in model.parameters()):,} {weights_file_meta}")

    train_loss, val_loss = train_diffusion_prior(
        model=model,
        noise_scheduler=noise_scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        num_unique_users=num_users,
        objective=args.objective if args.objective != "noise-pred" else "epsilon",
        device=str(device),
        num_epochs=args.epochs,
        patience=args.patience,
        savepath=savepath,
        return_losses=True,
        verbose=True,
    )

    save_rebeca_config(
        model_class="RebecaDiffusionPrior",
        model_kwargs={
            "img_embed_dim": args.img_embed_dim,
            "num_users": num_users,
            "num_tokens": args.num_tokens,
            "hidden_dim": args.hidden_dim,
            "n_heads": args.n_heads,
            "num_layers": args.num_layers,
            "score_classes": 2,
        },
        weights_file="prior.pth",
        output_dir=str(weights_dir),
        diffusion_model_id="/path/to/models/stable-diffusion-v1-5",
        ip_adapter={"dir": "/path/to/models/IP-Adapter",
                    "subfolder": "models", "weight_name": "ip-adapter_sd15.safetensors"},
        num_train_timesteps=args.num_train_timesteps,
        num_users=num_users,
        img_embed_dim=args.img_embed_dim,
    )
    write_json(weights_dir / "training_meta.json", {
        **weights_file_meta,
        "beta_schedule": args.beta_schedule,
        "objective": args.objective,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "samples_per_user": args.samples_per_user,
        "epochs_requested": args.epochs,
        "patience": args.patience,
        "final_train_loss": train_loss,
        "best_val_loss": val_loss,
        "seed": args.seed,
        "positives_only": bool((ratings["score"] < 4).sum() == 0),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
