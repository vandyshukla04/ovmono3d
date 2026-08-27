"""Frozen DINOv2 ViT-L/14 backbone -> one stride-14 feature map (spec 2.1).  [M3]

Taps blocks {12,18,24} (1-indexed, vendored convention), projects each to d=256 with a 1x1,
fuses by top-down sum (deepest first). Backbone is FROZEN and kept in eval mode permanently
(train() never unfreezes it). A7 swaps this module for DINOv3/16 -- nothing downstream changes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .dinov2 import build_vit_large


class TADETRBackbone(nn.Module):
    def __init__(self, d_model: int = 256, blocks=(12, 18, 24), checkpoint_path: str = ""):
        super().__init__()
        self.blocks = tuple(blocks)
        self.vit = build_vit_large(patch_size=14,
                                   checkpoint_path=checkpoint_path or None)
        for p in self.vit.parameters():
            p.requires_grad_(False)
        self.vit.eval()
        self.proj = nn.ModuleList(nn.Conv2d(1024, d_model, 1) for _ in self.blocks)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vit.eval()                      # frozen forever; BN/LN stay in eval statistics
        return self

    @torch.no_grad()
    def _features(self, x: torch.Tensor):
        return self.vit.get_intermediate_layers(x, n=self.blocks, reshape=True,
                                                return_class_token=False, norm=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._features(x)
        out = None
        for f, proj in zip(reversed(feats), reversed(list(self.proj))):
            p = proj(f.to(next(proj.parameters()).dtype))
            out = p if out is None else out + p
        return out                           # (B, d_model, H/14, W/14)
