"""Terrain cache contract v1 — one npz per segment: datasets/tadetr/terrain/<name>.npz.

Everything the model/dataloader needs from the dense VGGT tree travels HERE (the 56 GB of
depth_maps.npz never leave /mnt/d; these caches are ~0.5-1 MB each and rsync to the cluster).

Keys (all float32 unless noted):
  H_grid      (G,G)    height along the terrain normal, in the plane-tangent frame. VGGT world is
                       the camera-0 frame (NOT gravity-aligned), so the grid lives in a frame built
                       from the RANSAC ground plane -- spec's "grid the XY plane" realized properly.
  H_var       (G,G)    per-cell height variance, inflated by 1/mean(cell confidence).
  n_points    (G,G) i32  points per cell BEFORE fill (0 => filled/fallback cell).
  grid_origin (2,)     tangent-frame (a,b) of cell (0,0) corner.
  grid_scale  ()       cell size (square cells; grid spans the max in-plane extent).
  R_grid      (3,3)    rows (e1, e2, n): world -> tangent frame. n signed toward the cameras
                       ("drone above ground"), cross-checked against the box-up consensus.
  ctr         (3,)     world reference point of the tangent frame: tangent coords of a world point
                       x are R_grid @ (x - ctr). Heights and grid_origin are all relative to it.
  plane       (4,)     RANSAC ground plane [n, d] in WORLD coords (n.x + d = 0 on the plane),
                       the fallback where n_points==0 / H_var high.
  s_seg       ()       THE GAUGE BRIDGE: label_center_cam = s_seg * (extrinsic @ center_world).
                       Verified exact (0.00% 3-axis median) 2026-08-26.
  extrinsics  (N,3,4)  per-frame world->cam (the detector jsons carry none).
  K_518       (N,3,3)  per-frame intrinsics in 518x294 depth-pixel space.
  cam_height  (N,)     per-frame camera height above local terrain (world units) -- the
                       ray-march bounds' "altitude".
  frame_names (N,) str join key to detector-json file_path basenames.
  meta        str      json: schema, thresholds, fill fraction, sources, s_seg cross-check, warnings.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
GRID_SIZE = 256


def write_terrain_cache(path: Path, *, H_grid, H_var, n_points, grid_origin, grid_scale,
                        R_grid, ctr, plane, s_seg, extrinsics, K_518, cam_height, frame_names,
                        meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        H_grid=np.asarray(H_grid, np.float32),
        H_var=np.asarray(H_var, np.float32),
        n_points=np.asarray(n_points, np.int32),
        grid_origin=np.asarray(grid_origin, np.float32),
        grid_scale=np.float32(grid_scale),
        R_grid=np.asarray(R_grid, np.float32),
        ctr=np.asarray(ctr, np.float64),
        plane=np.asarray(plane, np.float32),
        s_seg=np.float32(s_seg),
        extrinsics=np.asarray(extrinsics, np.float32),
        K_518=np.asarray(K_518, np.float32),
        cam_height=np.asarray(cam_height, np.float32),
        frame_names=np.asarray(frame_names),
        meta=json.dumps({"schema": SCHEMA_VERSION, **meta}),
    )


def load_terrain_cache(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files if k != "meta"}
    out["meta"] = json.loads(str(z["meta"]))
    if out["meta"]["schema"] != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema {out['meta']['schema']} != {SCHEMA_VERSION}")
    return out


def sample_height(cache: dict, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bilinear height lookup at tangent-frame coords (numpy, for offline tools; the training-time
    torch version lives in tadetr/geometry/heightfield.py)."""
    G = cache["H_grid"].shape[0]
    ga = np.clip((a - cache["grid_origin"][0]) / cache["grid_scale"] - 0.5, 0, G - 1 - 1e-6)
    gb = np.clip((b - cache["grid_origin"][1]) / cache["grid_scale"] - 0.5, 0, G - 1 - 1e-6)
    i0, j0 = np.floor(ga).astype(int), np.floor(gb).astype(int)
    i1, j1 = np.minimum(i0 + 1, G - 1), np.minimum(j0 + 1, G - 1)
    fa, fb = ga - i0, gb - j0
    H = cache["H_grid"]
    return ((H[i0, j0] * (1 - fa) + H[i1, j0] * fa) * (1 - fb)
            + (H[i0, j1] * (1 - fa) + H[i1, j1] * fa) * fb)
