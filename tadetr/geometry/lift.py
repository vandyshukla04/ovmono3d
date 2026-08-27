"""geometric_lift: contact pixel -> depth by ray-terrain intersection.  [M2 = ablation A0;
reused verbatim as the in-network intersection layer from M3 on. Spec section 1.]

Works in the LABEL gauge end to end:
  ray:    label-space pixel (u,v) + label K  ->  cam direction K^-1 [u,v,1]  ->  world direction
          R^T d (scale never rotates), origin = camera center from the cache extrinsic
  march:  in VGGT world coordinates against the TerrainField (+ optional dh surface offset,
          species table for A0, learned height_residual from A2 on)
  output: z_label = s_seg * (E @ p_world)_z   -- the gauge bridge, verified exact.

sigma_z (spec 1.4): (z / f) * sigma_px / sin(theta_g), sigma_px default 2.0.
Everything batched torch, differentiable, no .item() / numpy in the forward path.
"""
from __future__ import annotations

import torch

from .heightfield import TerrainField
from .intersect import ray_terrain_intersect

SIGMA_PX_DEFAULT = 2.0


def geometric_lift(contact_uv: torch.Tensor, K: torch.Tensor, extrinsic: torch.Tensor,
                   field: TerrainField, cam_height: torch.Tensor,
                   dh: torch.Tensor | None = None,
                   sigma_px: torch.Tensor | None = None) -> dict:
    """contact_uv (B,2) full-res label-space pixels; K (B,3,3) label-space intrinsics;
    extrinsic (B,3,4) world->cam for each instance's frame; cam_height (B,) camera height above
    terrain (world units, from the cache); dh (B,) optional surface offset in WORLD units.

    Returns: p_world (B,3), z_label (B,), t (B,), sin_theta_g (B,), sigma_z (B,),
             fallback (B,) bool, d_world (B,3), o_world (B,3).
    """
    B = contact_uv.shape[0]
    ones = torch.ones(B, 1, dtype=contact_uv.dtype, device=contact_uv.device)
    pix = torch.cat([contact_uv, ones], dim=1)                     # (B,3)
    d_cam = torch.linalg.solve(K, pix[..., None])[..., 0]          # K^-1 [u,v,1]
    R = extrinsic[:, :, :3]
    t = extrinsic[:, :, 3]
    d_world = (R.transpose(1, 2) @ d_cam[..., None])[..., 0]
    d_world = d_world / d_world.norm(dim=1, keepdim=True).clamp_min(1e-12)
    o_world = (R.transpose(1, 2) @ (-t)[..., None])[..., 0]        # camera center

    hit = ray_terrain_intersect(field, o_world, d_world, cam_height, dh=dh)
    p_world = hit["p_world"]
    p_cam = (R @ p_world[..., None])[..., 0] + t
    z_label = field.s_seg * p_cam[:, 2]

    f_px = 0.5 * (K[:, 0, 0] + K[:, 1, 1])
    sp = sigma_px if sigma_px is not None else torch.full_like(z_label, SIGMA_PX_DEFAULT)
    sigma_z = (z_label / f_px) * sp / hit["sin_theta_g"].clamp_min(0.05)

    return {"p_world": p_world, "z_label": z_label, "t": hit["t"],
            "sin_theta_g": hit["sin_theta_g"], "sigma_z": sigma_z,
            "fallback": hit["fallback"], "d_world": d_world, "o_world": o_world}
