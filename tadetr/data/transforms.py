"""Geometric augmentation for TA-DETR — the intrinsics rule is ABSOLUTE.  [M3]

THE BRIDGE (read before editing): the terrain and extrinsics describe the REAL camera. A flipped or
cropped image is NOT a valid rigid camera (a mirror has det -1), so we never fake K/extrinsics.
Instead every sample carries an affine bridge `view` mapping VIEW pixel coords (the tensor the
network sees, W_in x H_in) back to FULL-RES ORIGINAL coords:

    u_full = ox + sx * (W_in - 1 - u_view   if flip else   u_view)
    v_full = oy + sy * v_view

2D supervision (boxes, contact targets) lives in VIEW coords; the detector maps predicted contact
through the bridge before ray-casting with the ORIGINAL K/extrinsic/terrain; orientation angles
flip sign with the mirror (alpha -> -alpha, psi -> -psi: the measured B1 rule). The sign BIT
cos(alpha - psi) is flip-invariant (both flip together) -- do not "fix" it in the flip path.

Crop convention: RandomResizedCrop picks a scale in [rrc_scale_min, 1] of the full frame (aspect
matched to the output), always inside the image; then resize to (W_in, H_in).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class ViewBridge:
    sx: float          # full-res units per view pixel, x
    sy: float
    ox: float          # full-res coords of the view origin
    oy: float
    flip: bool
    w_in: int
    h_in: int

    def view_to_full(self, u, v):
        uu = (self.w_in - 1 - u) if self.flip else u
        return self.ox + self.sx * uu, self.oy + self.sy * v

    def full_to_view(self, uf, vf):
        u = (uf - self.ox) / self.sx
        v = (vf - self.oy) / self.sy
        if self.flip:
            u = self.w_in - 1 - u
        return u, v

    def as_tensor_args(self):
        return np.array([self.sx, self.sy, self.ox, self.oy, 1.0 if self.flip else 0.0],
                        np.float32)


def sample_bridge(full_w: int, full_h: int, w_in: int, h_in: int, *, training: bool,
                  hflip_p: float, rrc_p: float, rrc_scale_min: float,
                  rng: random.Random) -> ViewBridge:
    if training and rng.random() < rrc_p:
        s = rng.uniform(rrc_scale_min, 1.0)
        cw, ch = full_w * s, full_h * s
        ox = rng.uniform(0, full_w - cw)
        oy = rng.uniform(0, full_h - ch)
    else:
        cw, ch, ox, oy = full_w, full_h, 0.0, 0.0
    flip = training and rng.random() < hflip_p
    return ViewBridge(sx=cw / w_in, sy=ch / h_in, ox=ox, oy=oy, flip=flip,
                      w_in=w_in, h_in=h_in)


def boxes_full_to_view(boxes_xywh: np.ndarray, br: ViewBridge) -> np.ndarray:
    """(N,4) full-res XYWH -> view-coord XYWH (clipped to the view)."""
    x1 = (boxes_xywh[:, 0] - br.ox) / br.sx
    y1 = (boxes_xywh[:, 1] - br.oy) / br.sy
    w = boxes_xywh[:, 2] / br.sx
    h = boxes_xywh[:, 3] / br.sy
    if br.flip:
        x1 = br.w_in - 1 - (x1 + w)
    x2 = np.clip(x1 + w, 0, br.w_in)
    y2 = np.clip(y1 + h, 0, br.h_in)
    x1 = np.clip(x1, 0, br.w_in)
    y1 = np.clip(y1, 0, br.h_in)
    return np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)


def photometric(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Cheap jitter on a float32 HWC [0,1] image: brightness/contrast/saturation."""
    b = 1.0 + rng.uniform(-0.25, 0.25)
    c = 1.0 + rng.uniform(-0.25, 0.25)
    s = 1.0 + rng.uniform(-0.25, 0.25)
    mean = img.mean(axis=(0, 1), keepdims=True)
    gray = img.mean(axis=2, keepdims=True)
    out = (img - mean) * c + mean * b
    out = gray + (out - gray) * s
    return np.clip(out, 0.0, 1.0)
