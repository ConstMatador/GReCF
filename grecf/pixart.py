from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _unique_rows(ids: torch.Tensor) -> torch.Tensor:
    seen: set[int] = set()
    keep: list[int] = []
    for index, value in enumerate(ids.detach().cpu().tolist()):
        if value not in seen:
            seen.add(value)
            keep.append(index)
    return torch.tensor(keep, device=ids.device, dtype=torch.long)


def _contrastive(
    predictions: torch.Tensor, targets: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = F.normalize(predictions.float(), dim=-1)
    targets = F.normalize(targets.float(), dim=-1)
    logits = predictions @ targets.T / float(temperature)
    labels = torch.arange(len(logits), device=logits.device)
    return F.cross_entropy(logits, labels), (logits.argmax(dim=1) == labels).float().mean()


class PixArtPrefixAdapter(nn.Module):
    """GReCF adaptive theme-set conditioning as PixArt caption-prefix tokens."""

    def __init__(
        self,
        base_context: torch.Tensor,
        base_mask: torch.Tensor,
        auxiliary_center: torch.Tensor,
        user_tokens: int = 8,
        delta_tokens: int = 8,
        hidden_dim: int = 1024,
        caption_dim: int = 4096,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.user_tokens = int(user_tokens)
        self.delta_tokens = int(delta_tokens)
        self.caption_dim = int(caption_dim)
        self.residual_scale = float(residual_scale)
        self.register_buffer("base_context", base_context.detach().float().clone())
        self.register_buffer("base_mask", base_mask.detach().bool().clone())
        self.register_buffer("auxiliary_center", auxiliary_center.detach().float().clone())
        self.user_projection = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, caption_dim))
        self.theme_mlp = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.delta_tokens * caption_dim),
        )
        nn.init.zeros_(self.theme_mlp[-1].weight)
        nn.init.zeros_(self.theme_mlp[-1].bias)
        self.user_gate_logit = nn.Parameter(torch.tensor(2.0))
        self.delta_gate_logit = nn.Parameter(torch.tensor(2.0))
        self.user_head = nn.Linear(caption_dim, 512)
        self.delta_head = nn.Linear(caption_dim, 512)
        self.user_head.requires_grad_(False)
        self.delta_head.requires_grad_(False)

    def gates(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.sigmoid(self.user_gate_logit), torch.sigmoid(self.delta_gate_logit)

    def forward(
        self,
        prototypes: torch.Tensor,
        prototype_mask: torch.Tensor,
        auxiliary: torch.Tensor,
        interest: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del auxiliary
        user_mask = prototype_mask.bool()
        if prototypes.shape[1] != self.user_tokens:
            raise ValueError(f"expected {self.user_tokens} prototype slots, got {prototypes.shape[1]}")
        user = self.user_projection(prototypes.float())
        user = user * user_mask[..., None].to(user.dtype)
        theme = self.theme_mlp(F.normalize(interest.float(), dim=-1)).reshape(
            -1, self.delta_tokens, self.caption_dim
        )
        user_gate, delta_gate = self.gates()
        user = user * (self.residual_scale * user_gate)
        theme = theme * (self.residual_scale * delta_gate)
        base = self.base_context.expand(len(prototypes), -1, -1)
        base_mask = self.base_mask.expand(len(prototypes), -1)
        mask = torch.cat(
            (
                base_mask,
                user_mask,
                torch.ones(len(prototypes), self.delta_tokens, device=prototypes.device, dtype=torch.bool),
            ),
            dim=1,
        )
        residual = torch.cat((user, theme), dim=1)
        return torch.cat((base, residual), dim=1), mask, residual

    def unconditional(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.base_context.expand(batch_size, -1, -1)
        zeros = torch.zeros(
            batch_size,
            self.user_tokens + self.delta_tokens,
            self.caption_dim,
            device=base.device,
            dtype=base.dtype,
        )
        mask = torch.cat(
            (
                self.base_mask.expand(batch_size, -1),
                torch.zeros(
                    batch_size,
                    self.user_tokens + self.delta_tokens,
                    device=base.device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )
        return torch.cat((base, zeros), dim=1), mask

    def user_contrastive_loss(
        self,
        residual: torch.Tensor,
        prototype_mask: torch.Tensor,
        auxiliary: torch.Tensor,
        interest: torch.Tensor,
        user_ids: torch.Tensor,
        temperature: float = 0.07,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keep = _unique_rows(user_ids)
        mask = prototype_mask[keep].float()
        user = residual[keep, : self.user_tokens]
        pooled_user = (user * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        theme = residual[keep, self.user_tokens :].mean(dim=1)
        user_target = F.normalize(auxiliary[keep].float() - self.auxiliary_center, dim=-1)
        theme_target = F.normalize(interest[keep].float(), dim=-1)
        user_loss, user_accuracy = _contrastive(self.user_head(pooled_user), user_target, temperature)
        theme_loss, theme_accuracy = _contrastive(self.delta_head(theme), theme_target, temperature)
        return (user_loss + theme_loss) / 2, (user_accuracy + theme_accuracy) / 2
