"""Depth-map -> world-point unprojection, matching VGGT's own convention exactly.

Convention (verified against vggt/vggt/utils/geometry.py:15 `unproject_depth_map_to_point_map`,
the function that produced the stored point_cloud.ply files; preflight P13 asserts equality):
    pixel (u=col, v=row), depth d:   p_cam = d * K^-1 @ [u, v, 1]
    extrinsic E = [R | t] is world->cam (OpenCV):  p_cam = R @ p_world + t
    so                                p_world = R.T @ (p_cam - t)
No +0.5 pixel-center offset: VGGT uses integer pixel coordinates in its meshgrid.
"""
from __future__ import annotations

import numpy as np


def unproject_depth(depth: np.ndarray, K: np.ndarray, extrinsic: np.ndarray,
                    valid: np.ndarray | None = None) -> np.ndarray:
    """depth (H,W), K (3,3) in the SAME pixel space as depth, extrinsic (3,4) world->cam.
    Returns (N,3) world points for valid pixels (valid: bool (H,W), default depth>0)."""
    H, W = depth.shape
    if valid is None:
        valid = depth > 0
    v, u = np.nonzero(valid)
    d = depth[v, u]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    p_cam = np.stack([x, y, d], axis=1)
    R, t = extrinsic[:, :3], extrinsic[:, 3]
    return (p_cam - t) @ R          # == (R.T @ (p_cam - t).T).T


def camera_center(extrinsic: np.ndarray) -> np.ndarray:
    """World-frame camera center: -R.T @ t."""
    R, t = extrinsic[:, :3], extrinsic[:, 3]
    return -R.T @ t
