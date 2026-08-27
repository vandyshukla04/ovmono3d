"""Herd latent scale module (spec 2.5).  [M3; active from A4]

Per (image, species): attention-pool the queries argmax-assigned to that species with a learned
per-species probe vector -> (mu, logvar) -> reparameterized lscale sample (mu at eval).
dims_i = exp(lscale_{s_i}) * exp(dhat_i) * D_bar_{s_i}.

WildBox gauge caveat (recorded in the plan): labels are per-segment scale-normalized, so lscale
absorbs the residual segment gauge, not metric scale; the 1/sqrt(N) shrinkage prediction is a
within-gauge claim. D_bar/Sigma come from LABEL-SANE train stats (buffers, never learned).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HerdScale(nn.Module):
    def __init__(self, d: int, n_classes: int, dims_median: torch.Tensor,
                 prior_std: float = 0.15):
        super().__init__()
        self.n_classes = n_classes
        self.prior_std = prior_std
        self.probe = nn.Parameter(torch.randn(n_classes, d) * 0.02)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)            # init: lscale = 0, logvar = 0
        self.register_buffer("dims_median", dims_median)   # (C,3) label-gauge class medians

    def forward(self, hs: torch.Tensor, logits: torch.Tensor, dhat: torch.Tensor,
                enabled: bool, sample: bool):
        """hs (B,Q,d), logits (B,Q,C+1), dhat (B,Q,3) ->
        dims (B,Q,3), lscale (B,C), mu (B,C), logvar (B,C), has_species (B,C) bool."""
        B, Q, _ = hs.shape
        C = self.n_classes
        cls = logits[..., :C].argmax(-1)                       # (B,Q) straight-through assignment
        onehot = torch.zeros(B, Q, C, device=hs.device, dtype=hs.dtype)
        onehot.scatter_(2, cls[..., None], 1.0)
        att = torch.einsum("bqd,cd->bqc", hs, self.probe)
        att = att.masked_fill(onehot == 0, -1e4).softmax(dim=1)     # pool over queries per class
        g = torch.einsum("bqc,bqd->bcd", att, hs)
        mu, logvar = self.mlp(g).unbind(-1)                    # (B,C) each
        has = onehot.sum(1) > 0
        if enabled:
            if sample:
                lscale = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
            else:
                lscale = mu
        else:
            lscale = torch.zeros_like(mu)
        lscale = lscale * has.to(lscale.dtype)
        per_q_ls = lscale.gather(1, cls.clamp(0, C - 1))       # (B,Q)... gather over class dim
        base = self.dims_median[cls]                           # (B,Q,3)
        dims = torch.exp(per_q_ls[..., None] + dhat) * base
        return {"dims": dims, "lscale": lscale, "mu": mu, "logvar": logvar,
                "has_species": has, "query_class": cls}
