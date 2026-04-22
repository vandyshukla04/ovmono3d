#!/usr/bin/env python
"""Paper-quality 3D detection visualizations following the Cube R-CNN /
OVMono3D convention.

For each sampled image, produces a 2x3 panel grid:

    +---------------------+---------------------+---------------------+
    | GT — 2D bounding    | GT — 3D cuboid      | GT — BEV (top-down) |
    | boxes on image      | wireframes on image | cuboid footprints   |
    +---------------------+---------------------+---------------------+
    | PRED — 2D bounding  | PRED — 3D cuboid    | PRED — BEV          |
    | boxes on image      | wireframes on image | cuboid footprints   |
    +---------------------+---------------------+---------------------+

Each panel has a titled banner at the top identifying what it shows.
Per-instance coloring: each GT track (or each prediction) gets a
distinct color consistently across its 2D box, 3D wireframe, and BEV
footprint, so the eye can follow the same object across the three views.

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
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------

def project(points3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    xs = points3d[:, 0] / points3d[:, 2]
    ys = points3d[:, 1] / points3d[:, 2]
    u = K[0, 0] * xs + K[0, 2]
    v = K[1, 1] * ys + K[1, 2]
    return np.stack([u, v], axis=1)


def cuboid_corners(center, dims_whl, R):
    W, H, L = float(dims_whl[0]), float(dims_whl[1]), float(dims_whl[2])
    local = np.array([
        [-L/2, -H/2, -W/2], [+L/2, -H/2, -W/2],
        [+L/2, +H/2, -W/2], [-L/2, +H/2, -W/2],
        [-L/2, -H/2, +W/2], [+L/2, -H/2, +W/2],
        [+L/2, +H/2, +W/2], [-L/2, +H/2, +W/2],
    ], dtype=np.float64)
    return (R @ local.T).T + np.asarray(center, dtype=np.float64)


# 12 edges of a cuboid given the corner ordering above
CUBOID_EDGES = [(0,1),(1,2),(2,3),(3,0),
                (4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7)]

# Corners 0-3 form the BOTTOM face (min Y), 4-7 form the TOP face.
# For BEV projection we take the bottom face (min-Y 4 corners).
BEV_BOTTOM_IDX = [0, 1, 2, 3]


# ---------------------------------------------------------------------
# Per-instance coloring
# ---------------------------------------------------------------------

def _color_for(idx: int) -> Tuple[int, int, int]:
    """Return a distinct BGR color for the given instance index by cycling
    through 12 hues at high saturation. Stable across panels so the same
    object gets the same color in 2D, 3D and BEV views."""
    hue = (idx * 30 + 10) % 360  # 30-degree steps
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR


# ---------------------------------------------------------------------
# Panel drawers
# ---------------------------------------------------------------------

def _font_scale(img_h: int) -> float:
    return max(0.5, img_h / 1000.0)


def _add_banner(img: np.ndarray, text: str, color: Tuple[int, int, int]) -> np.ndarray:
    banner_h = max(40, img.shape[0] // 20)
    banner = np.full((banner_h, img.shape[1], 3), 32, dtype=np.uint8)
    cv2.rectangle(banner, (0, banner_h - 3), (img.shape[1], banner_h),
                  color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = banner_h / 55.0
    thick = max(2, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(banner, text,
                ((img.shape[1] - tw) // 2, (banner_h + th) // 2 - 4),
                font, scale, color, thick)
    return np.vstack([banner, img])


def draw_2d_panel(base_img: np.ndarray,
                  instances: List[dict],
                  font_scale: float,
                  thickness_2d: int) -> np.ndarray:
    """2D bounding boxes only (no wireframes)."""
    out = base_img.copy()
    for i, inst in enumerate(instances):
        color = _color_for(i)
        x1, y1, x2, y2 = inst["box2d_xyxy"]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                      color, thickness_2d)
        label = inst.get("label", f"#{i}")
        # Background for text
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, 2)
        y_txt = max(int(y1) - 4, th + 4)
        cv2.rectangle(out,
                      (int(x1), y_txt - th - 4),
                      (int(x1) + tw + 4, y_txt + bl),
                      color, -1)
        cv2.putText(out, label, (int(x1) + 2, y_txt),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    return out


def draw_3d_panel(base_img: np.ndarray,
                  instances: List[dict],
                  K: np.ndarray,
                  thickness_3d: int,
                  font_scale: float) -> np.ndarray:
    """3D cuboid wireframes only (projected onto image)."""
    out = base_img.copy()
    for i, inst in enumerate(instances):
        corners3d = inst.get("corners3d")
        if corners3d is None:
            continue
        if not (corners3d[:, 2] > 0).all():
            continue
        color = _color_for(i)
        corners2d = project(corners3d, K)
        for a, b in CUBOID_EDGES:
            pa = tuple(int(v) for v in corners2d[a])
            pb = tuple(int(v) for v in corners2d[b])
            cv2.line(out, pa, pb, color, thickness_3d, cv2.LINE_AA)
        # Draw a small marker at the projection of the center
        center3d = corners3d.mean(axis=0)
        if center3d[2] > 0:
            cpt = project(center3d[None], K)[0]
            cv2.circle(out, (int(cpt[0]), int(cpt[1])), thickness_3d + 1,
                       color, -1)
        # Label at top-front corner (corner 4 in our ordering)
        lp = corners2d[4]
        label = inst.get("label", f"#{i}")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, 2)
        cv2.rectangle(out,
                      (int(lp[0]), int(lp[1]) - th - 4),
                      (int(lp[0]) + tw + 4, int(lp[1])),
                      color, -1)
        cv2.putText(out, label,
                    (int(lp[0]) + 2, int(lp[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    return out


def draw_bev_panel(instances: List[dict],
                   size: Tuple[int, int] = (720, 720),
                   margin: float = 0.15) -> np.ndarray:
    """Top-down bird's-eye view of the cuboid footprints. Camera is at the
    bottom-center pointing upward (+Z into the image). X is horizontal,
    Z depth-ish (vertical).
    """
    H, W = size
    canvas = np.full((H, W, 3), 240, dtype=np.uint8)  # light gray

    # Gather footprints (bottom face) in XZ
    foot_sets = []
    for inst in instances:
        c = inst.get("corners3d")
        if c is None:
            continue
        foot = c[BEV_BOTTOM_IDX][:, [0, 2]]  # (4, 2) in (x, z)
        foot_sets.append(foot)

    if not foot_sets:
        cv2.putText(canvas, "no 3D boxes", (20, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
        return canvas

    all_xz = np.concatenate(foot_sets, axis=0)
    x_min, x_max = all_xz[:, 0].min(), all_xz[:, 0].max()
    z_min, z_max = max(0.0, all_xz[:, 1].min()), all_xz[:, 1].max()
    # Widen so the plot has room & camera origin is visible
    dx = x_max - x_min
    dz = z_max - z_min
    span = max(dx, dz, 1e-6) * (1 + margin * 2)
    cx = (x_min + x_max) / 2
    cz = (z_min + z_max) / 2
    x_lo = cx - span / 2
    x_hi = cx + span / 2
    z_lo = max(0.0, cz - span / 2)
    z_hi = cz + span / 2
    if z_hi - z_lo < 1e-6:
        z_hi = z_lo + 1.0
    # z_lo can't be above camera (0); force include origin
    z_lo = min(z_lo, 0.0)

    def xz_to_px(pt):
        x, z = float(pt[0]), float(pt[1])
        u = (x - x_lo) / (x_hi - x_lo) * W
        v = H - (z - z_lo) / (z_hi - z_lo) * H
        return int(u), int(v)

    # Grid
    gridc = (200, 200, 200)
    # Vertical lines (x=const) at integer values
    x_step = max(1.0, round(span / 10.0))
    x_val = np.floor(x_lo / x_step) * x_step
    while x_val <= x_hi:
        u, _ = xz_to_px((x_val, z_lo))
        cv2.line(canvas, (u, 0), (u, H), gridc, 1)
        x_val += x_step
    z_val = np.floor(z_lo / x_step) * x_step
    while z_val <= z_hi:
        _, v = xz_to_px((0, z_val))
        cv2.line(canvas, (0, v), (W, v), gridc, 1)
        z_val += x_step

    # Camera icon at (0, 0), looking upward (+z)
    cam_u, cam_v = xz_to_px((0.0, 0.0))
    cam_tri = np.array([
        [cam_u, cam_v - 18],
        [cam_u - 12, cam_v + 8],
        [cam_u + 12, cam_v + 8],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [cam_tri], (50, 50, 50))
    cv2.putText(canvas, "cam", (cam_u + 14, cam_v + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    # Axis labels
    cv2.putText(canvas, "+X -->", (W - 90, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 2)
    cv2.putText(canvas, "+Z", (12, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 2)

    # Footprints, polygon per instance
    for i, inst in enumerate(instances):
        c = inst.get("corners3d")
        if c is None:
            continue
        color = _color_for(i)
        # Use the convex hull of all 8 corners in XZ to be robust to any
        # axis-convention quirks (some 3D heads emit cuboids where "bottom"
        # depends on sign of Y).
        xz = c[:, [0, 2]]
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(xz)
            poly = xz[hull.vertices]
        except Exception:
            poly = xz
        px_poly = np.array([xz_to_px(p) for p in poly], dtype=np.int32)
        # Fill with 30% alpha-like by drawing on a copy then blending
        fill = canvas.copy()
        cv2.fillPoly(fill, [px_poly], color)
        canvas = cv2.addWeighted(fill, 0.35, canvas, 0.65, 0)
        cv2.polylines(canvas, [px_poly], True, color, 3, cv2.LINE_AA)
        # Label at centroid
        cent = px_poly.mean(axis=0).astype(int)
        label = inst.get("label", f"#{i}")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.55, 2)
        cv2.rectangle(canvas,
                      (cent[0] - tw // 2 - 2, cent[1] - th // 2 - 2),
                      (cent[0] + tw // 2 + 2, cent[1] + th // 2 + 4),
                      color, -1)
        cv2.putText(canvas, label,
                    (cent[0] - tw // 2, cent[1] + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return canvas


# ---------------------------------------------------------------------
# Instance extraction
# ---------------------------------------------------------------------

def gt_instances(anns, img_w, img_h):
    out = []
    for i, ann in enumerate(anns):
        try:
            x, y, w, h = ann["bbox"]
            d = {"box2d_xyxy": (x, y, x + w, y + h)}
        except Exception:
            continue
        # 3D corners
        corners3d = None
        if ann.get("valid3D", True) and not ann.get("behind_camera", False):
            try:
                c = np.array(ann["center_cam"], dtype=np.float64)
                dd = np.array(ann["dimensions"], dtype=np.float64)
                R = np.array(ann["R_cam"], dtype=np.float64)
                corners3d = cuboid_corners(c, dd, R)
            except Exception:
                corners3d = None
        d["corners3d"] = corners3d
        name = ann.get("category_name", "?")
        tid = ann.get("track_id")
        d["label"] = f"{name}#{tid}" if tid is not None else name
        out.append(d)
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
        # 3D corners: prefer stored bbox3D_cam, else build from center/dims/pose
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
    p.add_argument("--thickness-3d", type=int, default=4,
                   help="Thicker lines for 3D wireframes (paper: 4-6 reads well)")
    p.add_argument("--bev-size", type=int, default=720)
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

    # Colors for banner bars
    GT_COLOR = (0, 200, 0)
    PRED_COLOR = (0, 0, 220)

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
        font = _font_scale(H)

        gt_insts = gt_instances(ann_by_img.get(img_id, []), W, H)
        pd_insts = pred_instances(pred_by_img.get(img_id), args.top_k,
                                  args.score_min)

        # GT row
        gt_2d = draw_2d_panel(im_base, gt_insts, font, args.thickness_2d)
        gt_3d = draw_3d_panel(im_base, gt_insts, K, args.thickness_3d, font)
        gt_bev = draw_bev_panel(gt_insts, size=(H, args.bev_size))
        gt_2d = _add_banner(gt_2d, f"GT  2D BOXES  ({len(gt_insts)} GT)", GT_COLOR)
        gt_3d = _add_banner(gt_3d, f"GT  3D CUBOIDS", GT_COLOR)
        gt_bev = _add_banner(gt_bev, f"GT  BIRD'S-EYE VIEW", GT_COLOR)

        # PRED row
        pd_2d = draw_2d_panel(im_base, pd_insts, font, args.thickness_2d)
        pd_3d = draw_3d_panel(im_base, pd_insts, K, args.thickness_3d, font)
        pd_bev = draw_bev_panel(pd_insts, size=(H, args.bev_size))
        pd_2d = _add_banner(pd_2d,
                            f"PRED  2D BOXES  (top {len(pd_insts)})",
                            PRED_COLOR)
        pd_3d = _add_banner(pd_3d, f"PRED  3D CUBOIDS", PRED_COLOR)
        pd_bev = _add_banner(pd_bev, f"PRED  BIRD'S-EYE VIEW", PRED_COLOR)

        # Compose 2 rows. Each row: [2D | 3D | BEV]. All panels must be
        # the same height (they are, thanks to matching image + banner).
        def _row(pans):
            hmax = max(p.shape[0] for p in pans)
            pans = [
                np.vstack([p, np.full((hmax - p.shape[0], p.shape[1], 3),
                                      32, dtype=np.uint8)])
                if p.shape[0] < hmax else p for p in pans]
            return np.hstack(pans)

        gt_row = _row([gt_2d, gt_3d, gt_bev])
        pd_row = _row([pd_2d, pd_3d, pd_bev])

        # Pad rows to same width if different
        wmax = max(gt_row.shape[1], pd_row.shape[1])
        def _pad_w(r):
            if r.shape[1] == wmax:
                return r
            return np.hstack([r, np.full(
                (r.shape[0], wmax - r.shape[1], 3), 32, dtype=np.uint8)])
        gt_row = _pad_w(gt_row)
        pd_row = _pad_w(pd_row)

        grid = np.vstack([gt_row, pd_row])
        cv2.imwrite(str(args.out / f"img_{img_id:06d}.jpg"), grid)
        saved += 1

    print(f"\nWrote {saved} samples to {args.out}/")
    print(f"  Each img_NNNNNN.jpg is a 2x3 grid:")
    print(f"    row 1: GT      — 2D boxes | 3D cuboids | BEV footprints")
    print(f"    row 2: PRED    — 2D boxes | 3D cuboids | BEV footprints")
    print(f"  Per-instance colors are consistent across the three views,")
    print(f"  so you can track a single animal from 2D -> 3D -> BEV by color.")


if __name__ == "__main__":
    main()
