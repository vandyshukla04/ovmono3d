"""TerrainField: differentiable torch view of one terrain cache.  [M2]

Wraps the npz contract (tadetr/data/terrain_cache.py) as torch tensors and exposes
bilinear height / variance lookups and central-difference surface normals, all
differentiable. The bilinear lookup matches tadetr.data.terrain_cache.sample_height
bit-for-bit at fp64 (preflight-asserted).

Frames: world -> tangent is  x_t = R_grid @ (x_w - ctr); heights are the tangent z.
"""
from __future__ import annotations

import numpy as np
import torch


class TerrainField:
    def __init__(self, cache: dict, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.G = cache["H_grid"].shape[0]
        t = lambda x: torch.as_tensor(np.asarray(x), device=device, dtype=dtype)
        self.H = t(cache["H_grid"])
        self.H_var = t(cache["H_var"])
        self.R_grid = t(cache["R_grid"])          # (3,3) rows e1,e2,n
        self.ctr = t(cache["ctr"])
        self.origin = t(cache["grid_origin"])
        self.scale = t(cache["grid_scale"])
        self.plane = t(cache["plane"])            # [n, d], n.x + d = 0
        self.s_seg = float(cache["s_seg"])
        self.n_points = torch.as_tensor(np.asarray(cache["n_points"]), device=device)
        self.dtype = dtype
        self.device = device
        # terrain dropout (training-time, edge-tier robustness): when True, the field degrades to
        # its own RANSAC plane -- which is EXACTLY h = 0 in the tangent frame (the frame is built
        # on that plane), with the plane normal everywhere. Toggled per batch by the train loop.
        self.plane_only = False

    def world_to_tangent(self, p_w: torch.Tensor) -> torch.Tensor:
        """(...,3) world -> (...,3) tangent (a, b, h)."""
        return (p_w - self.ctr) @ self.R_grid.T

    def _bilinear(self, grid: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        G = self.G
        ga = ((a - self.origin[0]) / self.scale - 0.5).clamp(0, G - 1 - 1e-6)
        gb = ((b - self.origin[1]) / self.scale - 0.5).clamp(0, G - 1 - 1e-6)
        i0 = ga.floor().long()
        j0 = gb.floor().long()
        i1 = (i0 + 1).clamp(max=G - 1)
        j1 = (j0 + 1).clamp(max=G - 1)
        fa = ga - i0.to(ga.dtype)
        fb = gb - j0.to(gb.dtype)
        return ((grid[i0, j0] * (1 - fa) + grid[i1, j0] * fa) * (1 - fb)
                + (grid[i0, j1] * (1 - fa) + grid[i1, j1] * fa) * fb)

    def height(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.plane_only:
            return torch.zeros_like(a)
        return self._bilinear(self.H, a, b)

    def var(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._bilinear(self.H_var, a, b)

    def normal_world(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """(...,) tangent coords -> (...,3) unit surface normal in WORLD coordinates,
        from central differences of the height field (surface h = f(a,b):
        normal ~ n - f_a e1 - f_b e2)."""
        if self.plane_only:
            return self.R_grid[2].expand(a.shape + (3,))
        eps = self.scale
        fa = (self.height(a + eps, b) - self.height(a - eps, b)) / (2 * eps)
        fb = (self.height(a, b + eps) - self.height(a, b - eps)) / (2 * eps)
        e1, e2, n = self.R_grid[0], self.R_grid[1], self.R_grid[2]
        v = n - fa[..., None] * e1 - fb[..., None] * e2
        return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
