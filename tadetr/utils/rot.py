"""Rotation / orientation utilities for TA-DETR.  [M3]

Conventions (all measured/verified in this project -- do not "fix"):
  - Camera frame is OpenCV (x right, y down, z forward). GT `R_cam` columns are the box's local
    axes in camera coordinates; `up = -R_cam[:, 1]` (verified 100.0% on 7,460; +R[:,1] is the
    arrow-inversion bug); the LENGTH axis (X-extent, dims=[W,H,L]) is column 0, valid mod pi only
    (PCA signs arbitrary: 10.9% frame flips -- never use column 0's sign as heading supervision).
  - The allocentric ground basis at an animal position p_cam with up u:
        r = normalize(horizontal component of p_cam)  (viewing ray, flattened to the ground plane)
        s = u x r
    alpha = atan2(d.s, d.r) for a ground direction d; the AXIS angle psi is the same construction
    from the box length axis, defined mod pi (doubled-angle supervision: (sin 2psi, cos 2psi)).
"""
from __future__ import annotations

import torch


def normalize(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return v / v.norm(dim=dim, keepdim=True).clamp_min(1e-9)


def ground_basis(p_cam: torch.Tensor, up_cam: torch.Tensor):
    """(...,3),(...,3) -> (r, s): the allocentric basis. Raises no error near-nadir; the caller
    owns that case (0% nadir measured in this data)."""
    u = normalize(up_cam)
    r = p_cam - (p_cam * u).sum(-1, keepdim=True) * u
    r = normalize(r)
    s = torch.cross(u, r, dim=-1)
    return r, s


def axis_angle_mod_pi(axis_cam: torch.Tensor, p_cam: torch.Tensor, up_cam: torch.Tensor):
    """Box length axis (mod-pi valid) -> psi in the allocentric basis, plus the doubled-angle
    embedding (sin 2psi, cos 2psi) that is invariant to the PCA sign."""
    u = normalize(up_cam)
    a = axis_cam - (axis_cam * u).sum(-1, keepdim=True) * u
    a = normalize(a)
    r, s = ground_basis(p_cam, u)
    psi = torch.atan2((a * s).sum(-1), (a * r).sum(-1))
    return psi, torch.stack([torch.sin(2 * psi), torch.cos(2 * psi)], dim=-1)


def compose_pose(up_cam: torch.Tensor, psi: torch.Tensor, p_cam: torch.Tensor,
                 sign_flip: torch.Tensor | None = None) -> torch.Tensor:
    """Build R_cam (...,3,3) from surface up + axis angle psi (+ optional pi flip from the sign
    head). Columns: [heading axis, -up, heading x -up] -- length on col 0, height on col 1,
    right-handed."""
    u = normalize(up_cam)
    r, s = ground_basis(p_cam, u)
    ang = psi if sign_flip is None else psi + torch.pi * sign_flip
    d = torch.cos(ang)[..., None] * r + torch.sin(ang)[..., None] * s
    c0 = normalize(d)
    c1 = -u
    c2 = torch.cross(c0, c1, dim=-1)
    return torch.stack([c0, c1, c2], dim=-1)


def alpha_from_pose(R_cam: torch.Tensor, p_cam: torch.Tensor) -> torch.Tensor:
    """Allocentric alpha of the (signed) heading axis R_cam[:,0] -- the quantity
    grade_orientation.py consumes."""
    u = -R_cam[..., 1]
    r, s = ground_basis(p_cam, u)
    d = R_cam[..., 0]
    d = normalize(d - (d * normalize(u)).sum(-1, keepdim=True) * normalize(u))
    return torch.atan2((d * s).sum(-1), (d * r).sum(-1))


def wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))
