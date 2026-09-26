from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableCLIPImageEmbedder(nn.Module):
    def __init__(self, clip_model: nn.Module, image_size: int, mean: list[float], std: list[float]) -> None:
        super().__init__()
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection
        self.image_size = image_size
        self.model_dtype = next(clip_model.parameters()).dtype
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, decoded: torch.Tensor) -> torch.Tensor:
        pixels = decoded.float().add(1.0).mul(0.5).clamp(0.0, 1.0)
        pixels = F.interpolate(
            pixels,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        pixels = (pixels - self.mean) / self.std
        pooled = self.vision_model(pixel_values=pixels.to(self.model_dtype)).pooler_output
        projected = self.visual_projection(pooled)
        return F.normalize(projected.float(), dim=-1)
