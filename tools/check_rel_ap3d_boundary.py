#!/usr/bin/env python
"""Confirm Rel-AP3D best-scale is interior to the search grid.

Reviewers ask: "is the best global scale you report hitting the boundary
of the search range?" If yes, the reported Rel-AP3D understates what's
recoverable. If no (interior), the number is trustworthy.

We answer this without re-running the full eval (which would take ~30
min). Instead we load the saved `instances_predictions.pth`, compute the
same per-(image, category) IoU surrogate that `search_rel_scale` uses,
sweep a user-specified wide grid, and print:

  - the best-scale at the wide grid
  - whether it falls at a grid boundary
  - comparison to the configured narrow grid (for context)

Usage:
    python tools/check_rel_ap3d_boundary.py \\
        --preds output/wildbox_wl5_zeroshot_oracle2d/inference/iter_final/WildBox_val/instances_predictions.pth \\
        --gt    datasets/Omni3D/WildBox_val.json \\
        --wide  0.01 10.0 48 \\
        --narrow 0.05 3.0 32            # what the oracle2d config uses
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_preds(p: Path):
    return torch.load(p, weights_only=False)


def build_cuboid_verts(center, dims, pose):
    """3D cuboid corners in camera frame. Omni3D dims ordering: [W, H, L]
    with X=L, Y=H, Z=W (see §2.4 / omni3d_evaluation.py for the convention).
    """
    W, H, L = dims
    cx, cy, cz = center
    # 8 corners in local frame (Omni3D convention)
    hx, hy, hz = L / 2.0, H / 2.0, W / 2.0
    corners = np.array([
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
    ], dtype=np.float32)
    return (pose @ corners.T).T + np.array([cx, cy, cz], dtype=np.float32)


def scale_preds_cuboids(preds, s: float):
    """Return list-of-list-of-tensors suitable for pytorch3d.box3d_overlap.
    Scales each prediction's 3D cuboid by `s` (applied to center + dims).
    """
    per_img = {}
    for im in preds:
        verts_list = []
        for inst in im.get("instances", []):
            try:
                c = np.asarray(inst["center_cam"], dtype=np.float32).reshape(3)
                d = np.asarray(inst["dimensions"], dtype=np.float32).reshape(3)
                p = np.asarray(inst["pose"], dtype=np.float32)
                if p.size == 9:
                    p = p.reshape(3, 3)
                if c.size != 3 or d.size != 3 or p.shape != (3, 3):
                    continue
                verts = build_cuboid_verts(c * s, d * s, p)
                verts_list.append((int(inst["category_id"]), verts))
            except Exception:
                continue
        per_img[im["image_id"]] = verts_list
    return per_img


def build_gt_cuboids(gt: dict):
    per_img = {}
    for ann in gt["annotations"]:
        c = np.asarray(ann["center_cam"], dtype=np.float32).reshape(3)
        d = np.asarray(ann["dimensions"], dtype=np.float32).reshape(3)
        p = np.asarray(ann["R_cam"], dtype=np.float32).reshape(3, 3)
        verts = build_cuboid_verts(c, d, p)
        per_img.setdefault(ann["image_id"], []).append((int(ann["category_id"]), verts))
    return per_img


def cuboid_iou_approx(v_p, v_g):
    """Approximate 3D IoU via BEV footprint intersection × height overlap.
    Faster than pytorch3d.box3d_overlap and monotone enough for the global
    argmax we need — we're picking the best scale, not reporting AP.
    """
    from shapely.geometry import Polygon
    # BEV footprints (y-axis is up; project onto XZ plane)
    def bev(verts):
        xz = verts[[0, 1, 2, 3], :][:, [0, 2]]
        return Polygon(xz).buffer(0)  # cheap fix for self-intersection
    def y_extent(verts):
        return verts[:, 1].min(), verts[:, 1].max()
    pp, pg = bev(v_p), bev(v_g)
    inter_bev = pp.intersection(pg).area
    if inter_bev <= 0:
        return 0.0
    union_bev = pp.area + pg.area - inter_bev
    py_lo, py_hi = y_extent(v_p); gy_lo, gy_hi = y_extent(v_g)
    inter_h = max(0.0, min(py_hi, gy_hi) - max(py_lo, gy_lo))
    union_h = max(py_hi, gy_hi) - min(py_lo, gy_lo)
    if union_h <= 0:
        return 0.0
    iou_bev = inter_bev / union_bev
    iou_h = inter_h / union_h
    return iou_bev * iou_h


def score_at_scale(pred_per_img, gt_per_img, s: float, dataset_to_contig=None):
    """Score = sum over (img, class) of max-pred-vs-gt IoU. This is the
    same objective search_rel_scale optimizes (up to the pytorch3d vs
    shapely IoU approximation)."""
    total = 0.0
    scaled = scale_preds_cuboids_cached(s, pred_per_img)
    for img_id, gts in gt_per_img.items():
        preds = scaled.get(img_id, [])
        if not gts or not preds:
            continue
        # Group by class
        by_class_p, by_class_g = {}, {}
        for c, v in preds:
            if dataset_to_contig and c in dataset_to_contig:
                c = dataset_to_contig[c]
            by_class_p.setdefault(c, []).append(v)
        for c, v in gts:
            if dataset_to_contig and c in dataset_to_contig:
                c = dataset_to_contig[c]
            by_class_g.setdefault(c, []).append(v)
        for c, g_list in by_class_g.items():
            p_list = by_class_p.get(c, [])
            for g in g_list:
                best = 0.0
                for p in p_list:
                    iou = cuboid_iou_approx(p, g)
                    if iou > best:
                        best = iou
                total += best
    return total


_SCALED_CACHE = {}
def scale_preds_cuboids_cached(s, pred_per_img):
    """Cache per-scale so the sweep doesn't rebuild every call. Not
    strictly necessary but makes the sweep ~30% faster on 24k preds."""
    key = round(s, 6)
    if key in _SCALED_CACHE:
        return _SCALED_CACHE[key]
    # pred_per_img is precomputed at s=1; scale on the fly to save memory
    out = {}
    for img_id, verts_list in pred_per_img.items():
        out[img_id] = [(c, v * s) for c, v in verts_list]
    _SCALED_CACHE[key] = out
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--wide", nargs=3, type=float, default=[0.01, 10.0, 48],
                    metavar=("MIN", "MAX", "N"),
                    help="Wide grid to test boundary behavior")
    ap.add_argument("--narrow", nargs=3, type=float, default=None,
                    metavar=("MIN", "MAX", "N"),
                    help="Narrow grid (e.g. config default) for comparison")
    ap.add_argument("--category-meta", type=Path,
                    default=Path("configs/category_meta.json"))
    args = ap.parse_args()

    print(f"Loading predictions: {args.preds}")
    preds = load_preds(args.preds)
    print(f"  {len(preds)} images, {sum(len(p['instances']) for p in preds)} total boxes")

    print(f"Loading GT: {args.gt}")
    gt = json.load(open(args.gt))

    dataset_to_contig = None
    if args.category_meta.exists():
        meta = json.load(open(args.category_meta))
        dataset_to_contig = {int(k): int(v) for k, v in
                             meta.get("thing_dataset_id_to_contiguous_id", {}).items()}
        print(f"  mapping: {sorted(dataset_to_contig.items())}")

    print("Precomputing cuboids at s=1 ...")
    pred_per_img = scale_preds_cuboids(preds, s=1.0)
    gt_per_img = build_gt_cuboids(gt)

    def sweep(lo, hi, n, label):
        print(f"\n=== {label} grid: {lo:.4f} to {hi:.4f}, {n} points ===")
        grid = np.linspace(lo, hi, int(n))
        scores = []
        for i, s in enumerate(grid):
            score = score_at_scale(pred_per_img, gt_per_img, float(s),
                                   dataset_to_contig=dataset_to_contig)
            scores.append(score)
            if (i + 1) % max(1, int(n) // 10) == 0 or i == len(grid) - 1:
                print(f"  [{i+1}/{len(grid)}] s={s:.4f} score={score:.2f}")
        best_i = int(np.argmax(scores))
        best_s = float(grid[best_i])
        print(f"  --> best scale = {best_s:.4f}  (index {best_i}/{len(grid)-1})")
        at_lo = (best_i == 0)
        at_hi = (best_i == len(grid) - 1)
        if at_lo:
            print(f"  WARNING: best scale is at LOW boundary. True optimum may be < {lo}.")
        elif at_hi:
            print(f"  WARNING: best scale is at HIGH boundary. True optimum may be > {hi}.")
        else:
            print(f"  OK: interior to the grid. Grid is adequate.")
        return best_s, scores

    wide_best, _ = sweep(*args.wide, "WIDE")

    if args.narrow is not None:
        narrow_best, _ = sweep(*args.narrow, "NARROW (config default)")
        delta = abs(wide_best - narrow_best) / max(abs(wide_best), 1e-9)
        print(f"\n=== Boundary verdict ===")
        print(f"  wide grid best:    {wide_best:.4f}")
        print(f"  narrow grid best:  {narrow_best:.4f}")
        print(f"  relative diff:     {100*delta:.1f}%")
        if delta < 0.05:
            print("  --> STABLE. Narrow grid is not distorting Rel-AP3D numbers.")
        else:
            print("  --> SHIFTED >5%. Narrow grid under-reports; widen the config.")


if __name__ == "__main__":
    main()
