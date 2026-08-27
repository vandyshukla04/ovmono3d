"""TA-DETR per-query heads (spec 2.3, amended: yaw -> axis x sign; box2d head ADDED).  [M3]

Depth is NOT a head -- it is composed in detector.py by ray-terrain intersection. The A5 control's
direct z head lives here behind its flag.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_in, d_h, d_out, n_layers=3):
        super().__init__()
        dims = [d_in] + [d_h] * (n_layers - 1) + [d_out]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x) if i == len(self.layers) - 1 else F.relu(l(x))
        return x


class TADETRHeads(nn.Module):
    def __init__(self, d: int, n_classes: int, dh_bounds=(-0.05, 0.15),
                 center_offset_bound: float = 0.1):
        super().__init__()
        self.n_classes = n_classes
        self.dh_lo, self.dh_hi = dh_bounds
        self.off_bound = center_offset_bound
        self.cls = nn.Linear(d, n_classes + 1)
        self.box = MLP(d, d, 4)
        self.contact = MLP(d, d, 2)
        self.contact_sigma = nn.Linear(d, 1)
        # sigma starts ~30 px (softplus(30) ~ 30): at init contact errors are ~300 px, and an
        # O(1)-px sigma would make the NLL term explode (the run-2 frozen-training lesson).
        nn.init.zeros_(self.contact_sigma.weight)
        nn.init.constant_(self.contact_sigma.bias, 30.0)
        self.dh = nn.Linear(d, 1)
        self.center_offset = MLP(d, d, 3)
        self.dim_residual = MLP(d, d, 3)
        self.axis = nn.Linear(d, 2)             # unnormalized (sin 2psi, cos 2psi)
        self.sign = nn.Linear(d, 1)             # logit: heading = axis + pi
        self.z_direct = nn.Linear(d, 1)         # A5 control only
        # zero-init the residual-style heads so the init model IS geometric_lift (preflight P8)
        for lin in (self.dh, self.center_offset.layers[-1], self.dim_residual.layers[-1],
                    self.z_direct):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        # axis must NOT be zero-init: the criterion normalizes its output and F.normalize has a
        # singular gradient at 0 (measured: 5e12 max grad). Tiny normal init, cubercnn convention.
        nn.init.normal_(self.axis.weight, std=0.001)
        nn.init.zeros_(self.axis.bias)
        nn.init.zeros_(self.sign.weight)
        nn.init.zeros_(self.sign.bias)

    def forward(self, hs: torch.Tensor) -> dict:
        """hs (B,Q,d) -> dict of raw + activated head outputs."""
        dh_mid = (self.dh_hi + self.dh_lo) / 2
        dh_amp = (self.dh_hi - self.dh_lo) / 2
        return {
            "logits": self.cls(hs),
            "boxes": self.box(hs).sigmoid(),
            "contact": self.contact(hs).sigmoid(),
            "contact_sigma": F.softplus(self.contact_sigma(hs)[..., 0]) + 0.5,
            "dh_frac": dh_mid + dh_amp * torch.tanh(self.dh(hs)[..., 0]),
            "center_offset": self.off_bound * torch.tanh(self.center_offset(hs)),
            "dim_residual": self.dim_residual(hs),
            "axis": self.axis(hs),
            "sign_logit": self.sign(hs)[..., 0],
            "z_direct": self.z_direct(hs)[..., 0],
        }
