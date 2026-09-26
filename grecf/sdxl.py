from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDXLPrefixAdapter(nn.Module):
    """GReCF user/theme prefix plus pooled conditioning for SDXL's UNet."""

    def __init__(
        self,
        base_context: torch.Tensor,
        base_mask: torch.Tensor,
        base_pooled: torch.Tensor,
        negative_context: torch.Tensor,
        negative_mask: torch.Tensor,
        negative_pooled: torch.Tensor,
        auxiliary_center: torch.Tensor,
        user_tokens: int = 8,
        delta_tokens: int = 8,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.user_tokens = int(user_tokens)
        self.delta_tokens = int(delta_tokens)
        self.register_buffer("base_context", base_context.detach().float().clone())
        self.register_buffer("base_mask", base_mask.detach().bool().clone())
        self.register_buffer("base_pooled", base_pooled.detach().float().clone())
        self.register_buffer("negative_context", negative_context.detach().float().clone())
        self.register_buffer("negative_mask", negative_mask.detach().bool().clone())
        self.register_buffer("negative_pooled", negative_pooled.detach().float().clone())
        self.register_buffer("auxiliary_center", auxiliary_center.detach().float().clone())
        self.user_projection = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 2048))
        self.theme_mlp = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, self.delta_tokens * 2048),
        )
        self.user_pooled = nn.Linear(512, 1280)
        self.theme_pooled = nn.Linear(512, 1280)
        self.user_head = nn.Linear(2048, 512)
        self.theme_head = nn.Linear(2048, 512)
        nn.init.zeros_(self.theme_mlp[-1].weight)
        nn.init.zeros_(self.theme_mlp[-1].bias)
        nn.init.zeros_(self.user_pooled.weight)
        nn.init.zeros_(self.user_pooled.bias)
        nn.init.zeros_(self.theme_pooled.weight)
        nn.init.zeros_(self.theme_pooled.bias)
        self.user_gate_logit = nn.Parameter(torch.tensor(2.0))
        self.delta_gate_logit = nn.Parameter(torch.tensor(2.0))
        self.pooled_gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(
        self,
        prototypes: torch.Tensor,
        prototype_mask: torch.Tensor,
        auxiliary: torch.Tensor,
        interest: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if prototypes.shape[1] != self.user_tokens:
            raise ValueError(f"expected {self.user_tokens} prototype slots, got {prototypes.shape[1]}")
        mask = prototype_mask.bool()
        user = self.user_projection(prototypes.float()) * mask[..., None].to(torch.float32)
        theme = self.theme_mlp(torch.nn.functional.normalize(interest.float(), dim=-1)).reshape(
            -1, self.delta_tokens, 2048
        )
        user_gate = torch.sigmoid(self.user_gate_logit)
        delta_gate = torch.sigmoid(self.delta_gate_logit)
        pooled_gate = torch.sigmoid(self.pooled_gate_logit)
        user = user * user_gate
        theme = theme * delta_gate
        base = self.base_context.expand(len(prototypes), -1, -1)
        base_mask = self.base_mask.expand(len(prototypes), -1)
        context = torch.cat((base, user, theme), dim=1)
        context_mask = torch.cat((base_mask, mask, torch.ones_like(mask)), dim=1)
        pooled_delta = self.user_pooled((auxiliary.float() - self.auxiliary_center))
        pooled_delta = pooled_delta + self.theme_pooled(interest.float())
        pooled = self.base_pooled.expand(len(prototypes), -1) + pooled_gate * pooled_delta
        residual = torch.cat((user, theme), dim=1)
        return context, context_mask, pooled, residual

    def unconditional(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.negative_context.expand(batch_size, -1, -1)
        mask = self.negative_mask.expand(batch_size, -1)
        pooled = self.negative_pooled.expand(batch_size, -1)
        zeros = torch.zeros(
            batch_size, self.user_tokens + self.delta_tokens, 2048,
            device=context.device, dtype=context.dtype,
        )
        context = torch.cat((context, zeros), dim=1)
        mask = torch.cat((mask, torch.zeros(batch_size, self.user_tokens + self.delta_tokens, device=context.device, dtype=torch.bool)), dim=1)
        return context, mask, pooled

    def user_contrastive_loss(
        self,
        residual: torch.Tensor,
        prototype_mask: torch.Tensor,
        auxiliary: torch.Tensor,
        interest: torch.Tensor,
        user_ids: torch.Tensor,
        temperature: float = 0.07,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Select one row per user while keeping batch positions, not raw user IDs.
        keep_list: list[int] = []
        seen: set[int] = set()
        for position, user_id in enumerate(user_ids.detach().cpu().tolist()):
            if int(user_id) not in seen:
                seen.add(int(user_id))
                keep_list.append(position)
        keep = torch.tensor(keep_list, device=user_ids.device, dtype=torch.long)
        mask = prototype_mask[keep].float()
        user = residual[keep, : self.user_tokens]
        pooled_user = (user * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        theme = residual[keep, self.user_tokens :].mean(dim=1)
        user_target = F.normalize((auxiliary[keep] - self.auxiliary_center).float(), dim=-1)
        theme_target = F.normalize(interest[keep].float(), dim=-1)
        def loss(pred, target):
            logits = F.normalize(pred.float(), dim=-1) @ target.T / temperature
            labels = torch.arange(len(logits), device=logits.device)
            return F.cross_entropy(logits, labels), (logits.argmax(dim=1) == labels).float().mean()
        user_loss, user_acc = loss(self.user_head(pooled_user), user_target)
        theme_loss, theme_acc = loss(self.theme_head(theme), theme_target)
        return (user_loss + theme_loss) / 2, (user_acc + theme_acc) / 2


class SDXLPreferenceAdapter(nn.Module):
    """SDXL version of the SD1.5 GReCF extra-attention conditioning route."""

    def __init__(self, base_context, base_mask, auxiliary_center, user_tokens=8, delta_tokens=8, hidden_dim=1024):
        super().__init__()
        self.user_tokens = int(user_tokens)
        self.delta_tokens = int(delta_tokens)
        self.register_buffer("base_context", base_context.detach().float().clone())
        self.register_buffer("base_mask", base_mask.detach().bool().clone())
        self.register_buffer("auxiliary_center", auxiliary_center.detach().float().clone())
        self.user_projection = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 2048))
        self.theme_mlp = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, self.delta_tokens * 2048),
        )
        nn.init.zeros_(self.theme_mlp[-1].weight)
        nn.init.zeros_(self.theme_mlp[-1].bias)
        self.user_gate_logit = nn.Parameter(torch.tensor(2.0))
        self.delta_gate_logit = nn.Parameter(torch.tensor(2.0))
        self.user_head = nn.Linear(2048, 512)
        self.delta_head = nn.Linear(2048, 512)
        # These projection heads define the auxiliary identity loss target and
        # remain fixed; DDP only synchronizes the actual conditioning adapter.
        self.user_head.requires_grad_(False)
        self.delta_head.requires_grad_(False)

    def gates(self):
        return torch.sigmoid(self.user_gate_logit), torch.sigmoid(self.delta_gate_logit)

    def forward(self, prototypes, prototype_mask, auxiliary, interest, multiplier=1.0):
        del auxiliary
        user_mask = prototype_mask.bool()
        user = self.user_projection(prototypes.float()) * user_mask[..., None].to(torch.float32)
        theme = self.theme_mlp(F.normalize(interest.float(), dim=-1)).reshape(-1, self.delta_tokens, 2048)
        user_gate, delta_gate = self.gates()
        scale = torch.as_tensor(float(multiplier), device=prototypes.device, dtype=user.dtype)
        context = self.base_context.expand(len(prototypes), -1, -1)
        residual = torch.cat((user, theme), dim=1)
        kwargs = {
            "preference_user_tokens": user,
            "preference_user_mask": user_mask,
            "preference_delta_tokens": theme,
            "preference_user_scale": scale * user_gate,
            "preference_delta_scale": scale * delta_gate,
        }
        return context, residual, kwargs

    def zero_attention_kwargs(self, batch_size, device, dtype):
        user_gate, delta_gate = self.gates()
        scale = torch.as_tensor(1.0, device=device, dtype=dtype)
        return {
            "preference_user_tokens": torch.zeros(batch_size, self.user_tokens, 2048, device=device, dtype=dtype),
            "preference_user_mask": torch.ones(batch_size, self.user_tokens, device=device, dtype=torch.bool),
            "preference_delta_tokens": torch.zeros(batch_size, self.delta_tokens, 2048, device=device, dtype=dtype),
            "preference_user_scale": scale * user_gate.to(device=device, dtype=dtype),
            "preference_delta_scale": scale * delta_gate.to(device=device, dtype=dtype),
        }

    def user_contrastive_loss(self, residual, prototype_mask, auxiliary, interest, user_ids, temperature=0.07):
        keep_list, seen = [], set()
        for position, user_id in enumerate(user_ids.detach().cpu().tolist()):
            if int(user_id) not in seen:
                seen.add(int(user_id)); keep_list.append(position)
        keep = torch.tensor(keep_list, device=user_ids.device, dtype=torch.long)
        mask = prototype_mask[keep].float()
        user = residual[keep, : self.user_tokens]
        pooled_user = (user * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        theme = residual[keep, self.user_tokens :].mean(dim=1)
        user_target = F.normalize((auxiliary[keep] - self.auxiliary_center).float(), dim=-1)
        theme_target = F.normalize(interest[keep].float(), dim=-1)
        def loss(pred, target):
            logits = F.normalize(pred.float(), dim=-1) @ target.T / temperature
            labels = torch.arange(len(logits), device=logits.device)
            return F.cross_entropy(logits, labels), (logits.argmax(dim=1) == labels).float().mean()
        user_loss, user_acc = loss(self.user_head(pooled_user), user_target)
        theme_loss, theme_acc = loss(self.delta_head(theme), theme_target)
        return (user_loss + theme_loss) / 2, (user_acc + theme_acc) / 2
