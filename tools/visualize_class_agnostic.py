#!/usr/bin/env python
"""OVMono3D-style 3D detection visualizations, CPU-only.

Faithfully ports the paper-figure "front-view + novel-view" layout from
cubercnn/vis/vis.py:draw_scene_view to pure numpy + cv2, so it runs on
clusters where pytorch3d's CUDA rasterizer is unavailable.

For each sampled image, produces a 2x3 grid laid out as:

    +---------------+------------------+--------------------------+
    | GT 2D BOXES   | GT 3D WIREFRAMES | GT NOVEL VIEW (60° pitch)|
    +---------------+------------------+--------------------------+
    | PRED 2D BOXES | PRED 3D WIREFR.  | PRED NOVEL VIEW          |
    +---------------+------------------+--------------------------+

The novel view is the OVMono3D default: `R = euler2mat([π/3, 0, 0])`
(60° pitch tilted bird's-eye), scene auto-centered, auto-zoom to keep
everything in frame, optional ground-plane grid (same drawing code as
cubercnn's draw_scene_view, lines 469-571 ported to numpy).

Two files per sample:
    img_NNNNNN.jpg         -- ground grid ON  (paper-figure style)
    img_NNNNNN_nogrid.jpg  -- ground grid OFF (cleaner)

Per-instance colors are consistent across all three panels of a row,
so you can follow the same animal from 2D → 3D → novel view by color.

Usage:
    python tools/visualize_class_agnostic.py \\
        --preds <instances_predictions.pth> \\
        --gt    <WildBox_val.json> \\
        --out   <out_dir> \\
        --top-k 5 --every 100 --limit 40
"""
import argparse
import colorsys
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------
# Core geometry (mirrors cubercnn.util.euler2mat and verts construction)
# ---------------------------------------------------------------------

def euler2mat(euler):
    """Same as cubercnn.util.math_util.euler2mat."""
    rx = np.array([[1, 0, 0],
                   [0, math.cos(euler[0]), -math.sin(euler[0])],
                   [0, math.sin(euler[0]),  math.cos(euler[0])]])
    ry = np.array([[math.cos(euler[1]), 0, math.sin(euler[1])],
                   [0, 1, 0],
                   [-math.sin(euler[1]), 0, math.cos(euler[1])]])
    rz = np.array([[math.cos(euler[2]), -math.sin(euler[2]), 0],
                   [math.sin(euler[2]),  math.cos(euler[2]), 0],
                   [0, 0, 1]])
    return rz @ ry @ rx


def project(points3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    xs = points3d[:, 0] / points3d[:, 2]
    ys = points3d[:, 1] / points3d[:, 2]
    u = K[0, 0] * xs + K[0, 2]
    v = K[1, 1] * ys + K[1, 2]
    return np.stack([u, v], axis=1)


def cuboid_corners(center, dims_whl, R):
    """Omni3D corner ordering: L=X-ext, H=Y-ext, W=Z-ext."""
    W, H, L = float(dims_whl[0]), float(dims_whl[1]), float(dims_whl[2])
    local = np.array([
        [-L/2, -H/2, -W/2], [+L/2, -H/2, -W/2],
        [+L/2, +H/2, -W/2], [-L/2, +H/2, -W/2],
        [-L/2, -H/2, +W/2], [+L/2, -H/2, +W/2],
        [+L/2, +H/2, +W/2], [-L/2, +H/2, +W/2],
    ], dtype=np.float64)
    return (R @ local.T).T + np.asarray(center, dtype=np.float64)


# Edge list used by cubercnn's draw_3d_box_from_verts (vis.py:676)
BB3D_EDGES = [[0, 1], [1, 2], [2, 3], [3, 0],
              [1, 5], [5, 6], [6, 2], [4, 5],
              [4, 7], [6, 7], [0, 4], [3, 7]]


# ---------------------------------------------------------------------
# Cuboid wireframe drawing (CPU port of cubercnn.vis.vis.draw_3d_box_from_verts)
# ---------------------------------------------------------------------

def draw_3d_wireframe(im, K, verts3d, color, thickness,
                     zplane=0.05, eps=1e-4):
    """Same behavior as cubercnn.vis.vis.draw_3d_box_from_verts:
    clips edges that cross the camera plane so lines project sensibly
    even when a cuboid corner is at the camera's back."""
    K = np.asarray(K, dtype=np.float64)
    v = np.asarray(verts3d, dtype=np.float64)
    for (i, j) in BB3D_EDGES:
        v0 = v[i].copy()
        v1 = v[j].copy()
        z0, z1 = v0[-1], v1[-1]
        if z0 >= zplane or z1 >= zplane:
            s = (zplane - z0) / max((z1 - z0), eps)
            new_v = v0 + s * (v1 - v0)
            if z0 < zplane <= z1:
                v0 = new_v
            elif z1 < zplane <= z0:
                v1 = new_v
            p0 = (K @ v0) / max(v0[-1], eps)
            p1 = (K @ v1) / max(v1[-1], eps)
            cv2.line(im,
                     (int(p0[0]), int(p0[1])),
                     (int(p1[0]), int(p1[1])),
                     color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------
# Per-instance coloring (same instance → same color across panels)
# ---------------------------------------------------------------------

def _color_for(idx: int) -> Tuple[int, int, int]:
    hue = (idx * 30 + 10) % 360
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR


# ---------------------------------------------------------------------
# Banner header for each panel
# ---------------------------------------------------------------------

def _add_banner(img, text, color):
    banner_h = max(40, img.shape[0] // 20)
    banner = np.full((banner_h, img.shape[1], 3), 32, dtype=np.uint8)
    cv2.rectangle(banner, (0, banner_h - 3), (img.shape[1], banner_h),
                  color, -1)
    scale = banner_h / 55.0
    thick = max(2, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                  scale, thick)
    cv2.putText(banner, text,
                ((img.shape[1] - tw) // 2, (banner_h + th) // 2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
    return np.vstack([banner, img])


# ---------------------------------------------------------------------
# Front-view panels: 2D boxes only, 3D wireframes only
# ---------------------------------------------------------------------

def _draw_label(canvas, xy, text, color, font_scale=0.6):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                   font_scale, 2)
    x = max(0, int(xy[0]))
    y = max(int(xy[1]), th + 4)
    cv2.rectangle(canvas, (x, y - th - 4), (x + tw + 4, y + bl),
                  color, -1)
    cv2.putText(canvas, text, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 2)


def draw_2d_panel(base, instances, thickness):
    out = base.copy()
    font = max(0.5, out.shape[0] / 1000.0)
    for i, inst in enumerate(instances):
        color = _color_for(i)
        x1, y1, x2, y2 = inst["box2d_xyxy"]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                      color, thickness)
        _draw_label(out, (x1, y1 - 4), inst.get("label", f"#{i}"),
                    color, font_scale=font)
    return out


def draw_3d_front_panel(base, instances, K, thickness):
    out = base.copy()
    font = max(0.5, out.shape[0] / 1000.0)
    # Depth-sort: far first, so near ones end up on top.
    order = sorted(range(len(instances)),
                   key=lambda i: -(instances[i]["corners3d"].mean(0)[2]
                                   if instances[i]["corners3d"] is not None
                                   else -1e9))
    for i in order:
        inst = instances[i]
        c = inst["corners3d"]
        if c is None:
            continue
        color = _color_for(i)
        draw_3d_wireframe(out, K, c, color, thickness)
        # Label at top-front corner (idx 4 in our ordering)
        if c[4, 2] > 0:
            p = project(c[4:5], K)[0]
            _draw_label(out, (p[0], p[1]), inst.get("label", f"#{i}"),
                        color, font_scale=font)
    return out


# ---------------------------------------------------------------------
# Novel-view panel — ported from cubercnn.vis.vis.draw_scene_view (CPU)
# ---------------------------------------------------------------------

def novel_view_panel(K, im_shape, instances, out_size,
                     pitch_rad=math.pi / 3,
                     with_grid=True,
                     thickness=4,
                     bg=(245, 245, 245),
                     grid_color=(175, 175, 175)):
    """Render cuboid wireframes from a novel viewpoint rotated by
    pitch_rad about the scene center. Optionally draws a projected
    ground-plane grid. CPU-only, no pytorch3d.

    Replicates the geometry in cubercnn/vis/vis.py:draw_scene_view
    (mode='novel' path, lines 395-613) without the rasterized mesh render.
    """
    H, W = out_size, out_size  # square canvas, matches OVMono3D demo
    canvas = np.full((H, W, 3), bg, dtype=np.uint8)

    # Gather cuboid verts with valid 3D geometry
    vis_insts = [inst for inst in instances if inst["corners3d"] is not None]
    if not vis_insts:
        cv2.putText(canvas, "no 3D boxes to render", (20, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        return canvas

    all_verts = np.concatenate([i["corners3d"] for i in vis_insts], axis=0)

    # --- scene center (same convention as draw_scene_view) ---
    vmin = all_verts.min(axis=0)
    vmax = all_verts.max(axis=0)
    center = (vmin + vmax) / 2.0
    max_y = vmax[1]  # ground plane = max-y (Y points down in camera frame)

    # Rotation about scene center (60° pitch by default)
    view_R = euler2mat([pitch_rad, 0, 0])

    # Intrinsics for the novel-view canvas (scaled principal point)
    K_nv = np.array(K, dtype=np.float64).copy()
    K_nv[0, -1] *= W / im_shape[1]
    K_nv[1, -1] *= H / im_shape[0]

    # --- Auto-zoom logic (port of draw_scene_view lines 425-454) ---
    def rotate_verts(v):
        return (view_R @ (v - center).T).T

    base_rot = rotate_verts(all_verts)

    # Starting zoom_factor, shrink until some vertex leaves margin or
    # goes behind camera, then step back.
    zoom_factor = 100.0
    zoom_factor_in = zoom_factor
    margin = 0.01
    max_trials = 10000
    best_zoom = zoom_factor

    while max_trials > 0:
        zoom_factor_in *= 0.95
        verts = base_rot.copy()
        verts[:, -1] += center[-1] * zoom_factor_in
        if (verts[:, -1] < 0.25).any():
            break
        proj = (K_nv @ verts.T) / verts[:, -1]
        if (proj[:2, :] < W * margin).any():
            break
        if (proj[:2, :] > W * (1 - margin)).any():
            break
        best_zoom = zoom_factor_in
        max_trials -= 1

    zoom_out_bias = float(center[-1])
    zoom_factor = best_zoom

    def transform(v):
        r = rotate_verts(v)
        r[:, -1] += zoom_out_bias * zoom_factor
        return r

    # --- Ground-plane grid (port of draw_scene_view lines 469-571) ---
    if with_grid:
        # First-pass: huge grid to find what's visible
        min_x3d, _, min_z3d = vmin.tolist()
        max_x3d, _, max_z3d = vmax.tolist()
        span_x = max(max_x3d - min_x3d, 1e-3)
        span_z = max(max_z3d - min_z3d, 1e-3)
        x0 = round(min_x3d - span_x * 50)
        x1 = round(max_x3d + span_x * 50)
        z0 = round(min_z3d - span_z * 50)
        z1 = round(max_z3d + span_z * 50)
        # Stride the grid cheaply
        step = max(1.0, int(max(x1 - x0, z1 - z0) / 200))
        grid_xs = np.arange(x0, x1, step)
        grid_zs = np.arange(z0, z1, step)
        xs_mesh, zs_mesh = np.meshgrid(grid_xs, grid_zs)
        ys_mesh = np.ones_like(xs_mesh) * max_y
        pts = np.stack([xs_mesh, ys_mesh, zs_mesh], axis=-1).reshape(-1, 3)
        pts_t = transform(pts)
        pts_t[:, -1] = np.clip(pts_t[:, -1], 0.25, None)
        pts_2d = (K_nv @ pts_t.T).T
        pts_2d[:, :2] /= pts_2d[:, 2:]
        # Find the sensible bounds based on which grid points project in-canvas
        in_x = (pts_2d[:, 0] >= -50) & (pts_2d[:, 0] < W + 50) & (pts_2d[:, 2] > 0)
        in_z = (pts_2d[:, 1] >= -50) & (pts_2d[:, 1] < H + 50) & (pts_2d[:, 2] > 0)
        if in_x.any() and in_z.any():
            x3d_start = round(pts[:, 0][in_x].min() - 10)
            x3d_end = round(pts[:, 0][in_x].max() + 10)
            z3d_start = round(pts[:, 2][in_z].min() - 10)
            z3d_end = round(pts[:, 2][in_z].max() + 10)
            grid_xs = np.arange(x3d_start, x3d_end + 1)
            grid_zs = np.arange(z3d_start, z3d_end + 1)
            # Subsample if too dense
            max_lines = 60
            if len(grid_xs) > max_lines:
                grid_xs = grid_xs[::max(1, len(grid_xs) // max_lines)]
            if len(grid_zs) > max_lines:
                grid_zs = grid_zs[::max(1, len(grid_zs) // max_lines)]

            xs_mesh, zs_mesh = np.meshgrid(grid_xs, grid_zs)
            ys_mesh = np.ones_like(xs_mesh) * max_y
            pts = np.stack([xs_mesh, ys_mesh, zs_mesh], axis=-1)
            shape0 = pts.shape
            pts_flat = pts.reshape(-1, 3)
            pts_t = transform(pts_flat)
            pts_t[:, -1] = np.clip(pts_t[:, -1], 0.25, None)
            pts_2d = (K_nv @ pts_t.T).T
            pts_2d[:, :2] /= pts_2d[:, 2:]
            pts_2d = pts_2d.reshape(shape0[0], shape0[1], 3)

            # Draw horizontal (const-z) and vertical (const-x) grid lines
            for r in range(shape0[0] - 1):
                for c in range(shape0[1] - 1):
                    # horizontal segment: same row (r), cols c -> c+1
                    a = pts_2d[r, c]
                    b = pts_2d[r, c + 1]
                    if a[2] > 0 and b[2] > 0:
                        cv2.line(canvas,
                                 (int(a[0]), int(a[1])),
                                 (int(b[0]), int(b[1])),
                                 grid_color, 1, cv2.LINE_AA)
                    # vertical segment: same col (c), rows r -> r+1
                    a = pts_2d[r, c]
                    b = pts_2d[r + 1, c]
                    if a[2] > 0 and b[2] > 0:
                        cv2.line(canvas,
                                 (int(a[0]), int(a[1])),
                                 (int(b[0]), int(b[1])),
                                 grid_color, 1, cv2.LINE_AA)

    # --- Cuboids, depth-sorted (far first) ---
    vis_order = sorted(
        range(len(vis_insts)),
        key=lambda i: -vis_insts[i]["corners3d"].mean(0)[2]
    )
    font = max(0.5, H / 1000.0)
    for i in vis_order:
        inst = vis_insts[i]
        # pick the ORIGINAL index for color consistency
        original_idx = inst.get("_idx", i)
        color = _color_for(original_idx)
        verts_t = transform(inst["corners3d"])
        draw_3d_wireframe(canvas, K_nv, verts_t, color, thickness)
        # Label
        top_c = verts_t[4]
        if top_c[-1] > 0:
            p = (K_nv @ top_c) / top_c[-1]
            _draw_label(canvas, (p[0], p[1]),
                        inst.get("label", f"#{original_idx}"),
                        color, font_scale=font)

    # Small axis marker in lower-left
    cv2.putText(canvas,
                f"novel view  pitch={math.degrees(pitch_rad):.0f}°",
                (12, H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (90, 90, 90), 2)
    return canvas


# ---------------------------------------------------------------------
# Instance extraction
# ---------------------------------------------------------------------

def gt_instances(anns):
    out = []
    for ann in anns:
        try:
            x, y, w, h = ann["bbox"]
            d = {"box2d_xyxy": (x, y, x + w, y + h)}
        except Exception:
            continue
        corners3d = None
        if ann.get("valid3D", True) and not ann.get("behind_camera", False):
            try:
                c = np.array(ann["center_cam"], dtype=np.float64)
                dd = np.array(ann["dimensions"], dtype=np.float64)
                R = np.array(ann["R_cam"], dtype=np.float64)
                corners3d = cuboid_corners(c, dd, R)
            except Exception:
                pass
        d["corners3d"] = corners3d
        name = ann.get("category_name", "?")
        tid = ann.get("track_id")
        d["label"] = f"{name}#{tid}" if tid is not None else name
        out.append(d)
    for i, d in enumerate(out):
        d["_idx"] = i
    return out


def pred_instances(im_obj, top_k, score_min):
    if im_obj is None:
        return []
    insts = [inst for inst in im_obj.get("instances", [])
             if float(inst.get("score", 0)) >= score_min]
    insts.sort(key=lambda x: -float(x.get("score", 0)))
    insts = insts[:top_k]
    out = []
    for i, inst in enumerate(insts):
        try:
            x, y, w, h = inst["bbox"]
            d = {"box2d_xyxy": (x, y, x + w, y + h)}
        except Exception:
            continue
        corners3d = None
        if inst.get("bbox3D_cam") is not None:
            try:
                c = np.array(inst["bbox3D_cam"], dtype=np.float64)
                if c.shape == (8, 3):
                    corners3d = c
            except Exception:
                pass
        if corners3d is None and all(k in inst for k in ("center_cam", "dimensions", "pose")):
            try:
                c = np.array(inst["center_cam"], dtype=np.float64).reshape(-1)
                dd = np.array(inst["dimensions"], dtype=np.float64).reshape(-1)
                p = np.array(inst["pose"], dtype=np.float64)
                if p.size == 9:
                    p = p.reshape(3, 3)
                if c.size == 3 and dd.size == 3 and p.shape == (3, 3):
                    corners3d = cuboid_corners(c, dd, p)
            except Exception:
                pass
        d["corners3d"] = corners3d
        score = float(inst.get("score", 0))
        d["label"] = f"#{i} {score:.2f}"
        d["_idx"] = i
        out.append(d)
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--every", type=int, default=50)
    p.add_argument("--score-min", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--thickness-2d", type=int, default=3)
    p.add_argument("--thickness-3d", type=int, default=4)
    p.add_argument("--pitch-deg", type=float, default=60.0,
                   help="Novel view pitch in degrees (OVMono3D default: 60)")
    p.add_argument("--novel-size", type=int, default=720,
                   help="Novel view canvas side length in pixels")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from {args.preds}...")
    preds = torch.load(args.preds, weights_only=False)
    print(f"Loading GT from {args.gt}...")
    gt = json.load(open(args.gt))

    ann_by_img = {}
    for ann in gt["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)
    img_by_id = {im["id"]: im for im in gt["images"]}
    pred_by_img = {im["image_id"]: im for im in preds}

    GT_COLOR = (0, 200, 0)
    PRED_COLOR = (0, 0, 220)

    pitch_rad = math.radians(args.pitch_deg)

    saved = 0
    for i, img_id in enumerate(sorted(img_by_id.keys())):
        if i % args.every != 0:
            continue
        if args.limit and saved >= args.limit:
            break
        img_info = img_by_id[img_id]
        img_path = Path(img_info["file_path"])
        if not img_path.is_absolute():
            img_path = Path("datasets") / img_path
        if not img_path.exists():
            continue
        im_base = cv2.imread(str(img_path))
        if im_base is None:
            continue
        K = np.array(img_info["K"], dtype=np.float64)
        H, W = im_base.shape[:2]

        gt_insts = gt_instances(ann_by_img.get(img_id, []))
        pd_insts = pred_instances(pred_by_img.get(img_id), args.top_k,
                                  args.score_min)

        # ---- Front-view panels (no dependence on grid flag) ----
        gt_2d = draw_2d_panel(im_base, gt_insts, args.thickness_2d)
        gt_3d = draw_3d_front_panel(im_base, gt_insts, K, args.thickness_3d)
        pd_2d = draw_2d_panel(im_base, pd_insts, args.thickness_2d)
        pd_3d = draw_3d_front_panel(im_base, pd_insts, K, args.thickness_3d)

        # Make all front panels the same height (they already are).
        # Novel view is square (args.novel_size × args.novel_size). We pad it
        # to match the image height for a clean row assembly, or downscale.
        target_h = H
        nv_size = min(args.novel_size, target_h)

        def _novel(instances, with_grid):
            nv = novel_view_panel(K, (H, W), instances, nv_size,
                                  pitch_rad=pitch_rad, with_grid=with_grid,
                                  thickness=args.thickness_3d)
            # Pad to target_h if smaller
            if nv.shape[0] < target_h:
                pad = np.full((target_h - nv.shape[0], nv.shape[1], 3), 245,
                              dtype=np.uint8)
                nv = np.vstack([nv, pad])
            elif nv.shape[0] > target_h:
                nv = cv2.resize(nv, (int(nv.shape[1] * target_h / nv.shape[0]),
                                     target_h))
            return nv

        def _assemble(with_grid):
            gt_nv = _novel(gt_insts, with_grid)
            pd_nv = _novel(pd_insts, with_grid)

            # Banner titles
            g_label = "with ground grid" if with_grid else "no grid"
            gt_2d_b = _add_banner(gt_2d,
                                  f"GT  2D BOXES  ({len(gt_insts)})", GT_COLOR)
            gt_3d_b = _add_banner(gt_3d, "GT  3D CUBOIDS", GT_COLOR)
            gt_nv_b = _add_banner(
                gt_nv, f"GT  NOVEL VIEW  (pitch {args.pitch_deg:.0f}°, {g_label})",
                GT_COLOR)
            pd_2d_b = _add_banner(pd_2d,
                                  f"PRED  2D BOXES  (top {len(pd_insts)})",
                                  PRED_COLOR)
            pd_3d_b = _add_banner(pd_3d, "PRED  3D CUBOIDS", PRED_COLOR)
            pd_nv_b = _add_banner(
                pd_nv, f"PRED  NOVEL VIEW  ({g_label})",
                PRED_COLOR)

            def _row(panels):
                hmax = max(p.shape[0] for p in panels)
                panels = [
                    (np.vstack([p, np.full((hmax - p.shape[0], p.shape[1], 3),
                                           32, dtype=np.uint8)])
                     if p.shape[0] < hmax else p)
                    for p in panels]
                return np.hstack(panels)

            gt_row = _row([gt_2d_b, gt_3d_b, gt_nv_b])
            pd_row = _row([pd_2d_b, pd_3d_b, pd_nv_b])
            wmax = max(gt_row.shape[1], pd_row.shape[1])
            if gt_row.shape[1] < wmax:
                gt_row = np.hstack([gt_row, np.full(
                    (gt_row.shape[0], wmax - gt_row.shape[1], 3), 32, np.uint8)])
            if pd_row.shape[1] < wmax:
                pd_row = np.hstack([pd_row, np.full(
                    (pd_row.shape[0], wmax - pd_row.shape[1], 3), 32, np.uint8)])
            return np.vstack([gt_row, pd_row])

        img_with = _assemble(with_grid=True)
        img_no = _assemble(with_grid=False)

        cv2.imwrite(str(args.out / f"img_{img_id:06d}.jpg"), img_with)
        cv2.imwrite(str(args.out / f"img_{img_id:06d}_nogrid.jpg"), img_no)
        saved += 1

    print(f"\nWrote {saved} samples × 2 files = {saved * 2} images to {args.out}/")
    print(f"  img_NNNNNN.jpg         — 2x3 grid with ground grid in novel view")
    print(f"  img_NNNNNN_nogrid.jpg  — same 2x3 grid, novel view without grid")
    print(f"\nLayout (each file):")
    print(f"  GT  row:  [2D boxes]  [3D cuboids]  [novel view {args.pitch_deg:.0f}°]")
    print(f"  PRED row: [2D boxes]  [3D cuboids]  [novel view {args.pitch_deg:.0f}°]")
    print(f"\nPer-instance colors are consistent across all three panels of a row,")
    print(f"so you can follow the same animal from 2D → 3D → novel view by color.")


if __name__ == "__main__":
    main()
