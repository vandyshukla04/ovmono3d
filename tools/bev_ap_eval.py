#!/usr/bin/env python
"""Bird's-Eye-View AP for aerial 3D detection.

Primary metric for the WildBox paper per the reviewer's advice:
    AP_BEV @ IoU 0.25 (and optionally 0.5)

BEV projection: drop the camera-Y axis (camera convention: Y points down for
a nadir-facing drone), keeping (X, Z) as the ground-plane footprint. This is
a pragmatic approximation for aerial drone data; a ground-plane aware
variant would require per-frame gravity estimation that we don't have here.

Rotated 2D IoU is computed with shapely, which makes this pytorch3d-free and
runnable on CPU-only installs.

Usage:
    python tools/bev_ap_eval.py \\
        --preds output/<run>/inference/iter_final/WildBox_val/instances_predictions.pth \\
        --gt    datasets/Omni3D/WildBox_val.json \\
        --out   output/<run>/bev_ap.json
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from shapely.geometry import Polygon
    from shapely.errors import GEOSException
except ImportError:
    print("ERROR: shapely not installed. Run: pip install shapely", flush=True)
    raise


# -----------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------

def cuboid_corners(center: np.ndarray, dims_whl: np.ndarray,
                   R: np.ndarray) -> np.ndarray:
    """(3,) center + (3,) [W, H, L] + (3,3) R -> (8, 3) camera-frame corners.

    Axis convention (Omni3D / cubercnn):
      L -> X-extent, H -> Y-extent, W -> Z-extent
      see cubercnn/util/math_util.py:172-181
    """
    W, H, L = (float(dims_whl[0]), float(dims_whl[1]), float(dims_whl[2]))
    local = np.array([
        [-L/2, -H/2, -W/2], [+L/2, -H/2, -W/2],
        [+L/2, +H/2, -W/2], [-L/2, +H/2, -W/2],
        [-L/2, -H/2, +W/2], [+L/2, -H/2, +W/2],
        [+L/2, +H/2, +W/2], [-L/2, +H/2, +W/2],
    ], dtype=np.float64)
    return (R @ local.T).T + np.asarray(center, dtype=np.float64)


def bev_footprint(center: np.ndarray, dims_whl: np.ndarray,
                  R: np.ndarray) -> Optional[np.ndarray]:
    """Compute the (rotated) bird's-eye-view rectangle of an oriented cuboid.

    Drops camera-Y axis, returns the 4 corner points of the footprint in
    (x, z) in the order suitable for shapely.Polygon. Returns None if the
    cuboid is degenerate (zero volume in the BEV plane).
    """
    corners_3d = cuboid_corners(center, dims_whl, R)        # (8, 3)
    xz = corners_3d[:, [0, 2]]                               # (8, 2)
    # Convex hull. For a cuboid the BEV is a rotated rectangle -> hull
    # has 4 points, but degenerate cases may have fewer.
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(xz)
        pts = xz[hull.vertices]
    except Exception:
        pts = xz  # fallback: let shapely handle non-convex
    if len(pts) < 3:
        return None
    return pts


def rotated_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """IoU between two rotated BEV rectangles given as (N_i, 2) vertex arrays."""
    try:
        pa = Polygon(poly_a)
        pb = Polygon(poly_b)
        if not pa.is_valid:
            pa = pa.buffer(0)
        if not pb.is_valid:
            pb = pb.buffer(0)
        if pa.area <= 0 or pb.area <= 0:
            return 0.0
        inter = pa.intersection(pb).area
        union = pa.area + pb.area - inter
        return float(inter / union) if union > 0 else 0.0
    except (GEOSException, Exception):
        return 0.0


# -----------------------------------------------------------------------
# Extract BEV footprints from GT and predictions
# -----------------------------------------------------------------------

def gt_bev_by_img(gt: Dict) -> Dict[int, List[Dict]]:
    """For each image_id, list of dicts with BEV polygon, category_id, and
    2D bbox (used for display / debug only)."""
    out: Dict[int, List[Dict]] = {}
    for ann in gt["annotations"]:
        if not ann.get("valid3D", True) or ann.get("behind_camera", False):
            continue
        try:
            center = np.array(ann["center_cam"], dtype=np.float64)
            dims = np.array(ann["dimensions"], dtype=np.float64)
            R = np.array(ann["R_cam"], dtype=np.float64)
        except KeyError:
            continue
        poly = bev_footprint(center, dims, R)
        if poly is None:
            continue
        out.setdefault(ann["image_id"], []).append({
            "poly": poly,
            "category_id": int(ann["category_id"]),
        })
    return out


def pred_bev_by_img(preds: List[Dict],
                    score_min: float = 0.0) -> Dict[int, List[Dict]]:
    """Same as GT but for predictions: reads OVMono3D's center_cam + pose
    + dimensions and builds the BEV polygon."""
    out: Dict[int, List[Dict]] = {}
    for im in preds:
        img_id = im["image_id"]
        for inst in im.get("instances", []):
            score = float(inst.get("score", 0))
            if score < score_min:
                continue
            try:
                center = np.array(inst["center_cam"], dtype=np.float64).reshape(-1)
                dims = np.array(inst["dimensions"], dtype=np.float64).reshape(-1)
                pose = np.array(inst["pose"], dtype=np.float64)
                if pose.size == 9:
                    pose = pose.reshape(3, 3)
                if center.size != 3 or dims.size != 3 or pose.shape != (3, 3):
                    continue
            except Exception:
                continue
            poly = bev_footprint(center, dims, pose)
            if poly is None:
                continue
            out.setdefault(img_id, []).append({
                "poly": poly,
                "score": score,
                "category_id": int(inst["category_id"]),
            })
    return out


# -----------------------------------------------------------------------
# AP computation (standard 11-point interpolation for stability)
# -----------------------------------------------------------------------

def voc_11point_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recall >= t
        ap += float(precision[mask].max()) / 11 if mask.any() else 0.0
    return ap


def compute_bev_ap_at_iou(
    preds_by_img: Dict[int, List[Dict]],
    gts_by_img: Dict[int, List[Dict]],
    iou_thresh: float,
    cat_filter: Optional[int] = None,
) -> Tuple[float, int, int]:
    """Compute BEV AP at a single IoU threshold, optionally filtered to one category.

    Returns (ap, n_preds, n_gt).
    """
    # Flatten predictions, filter by category if requested, sort by score desc
    flat = []
    for img_id, insts in preds_by_img.items():
        for d in insts:
            if cat_filter is not None and d["category_id"] != cat_filter:
                continue
            flat.append((d["score"], img_id, d["poly"], d["category_id"]))
    if not flat:
        return 0.0, 0, 0
    flat.sort(key=lambda x: -x[0])

    # Per-image GT, with filtering
    img_to_gt: Dict[int, List[Dict]] = {}
    n_gt_total = 0
    for img_id, gts in gts_by_img.items():
        kept = [g for g in gts if cat_filter is None or g["category_id"] == cat_filter]
        if kept:
            img_to_gt[img_id] = [{"poly": g["poly"], "matched": False} for g in kept]
            n_gt_total += len(kept)
    if n_gt_total == 0:
        return 0.0, len(flat), 0

    tp = np.zeros(len(flat), dtype=np.float32)
    fp = np.zeros(len(flat), dtype=np.float32)

    for i, (score, img_id, p_poly, cat_id) in enumerate(flat):
        gts = img_to_gt.get(img_id, [])
        if not gts:
            fp[i] = 1
            continue
        best_j, best_iou = -1, -1.0
        for j, g in enumerate(gts):
            if g["matched"]:
                continue
            iou = rotated_iou(p_poly, g["poly"])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thresh:
            tp[i] = 1
            gts[best_j]["matched"] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt_total
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-10)
    ap = voc_11point_ap(precision, recall)
    return float(ap), len(flat), n_gt_total


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None,
                   help="Save metrics to JSON at this path")
    p.add_argument("--iou-thresholds", type=float, nargs="+",
                   default=[0.25, 0.5])
    p.add_argument("--score-min", type=float, default=0.0)
    args = p.parse_args()

    print(f"Loading predictions {args.preds} ...", flush=True)
    preds = torch.load(args.preds, weights_only=False)
    print(f"Loading GT {args.gt} ...", flush=True)
    gt = json.load(open(args.gt))

    # Build per-image BEV polygons for GT + preds
    print("Building BEV footprints ...", flush=True)
    gts_by_img = gt_bev_by_img(gt)
    preds_by_img = pred_bev_by_img(preds, score_min=args.score_min)

    n_preds_total = sum(len(v) for v in preds_by_img.values())
    n_gt_total = sum(len(v) for v in gts_by_img.values())
    print(f"  GT BEV footprints:   {n_gt_total} across {len(gts_by_img)} images")
    print(f"  Pred BEV footprints: {n_preds_total} across {len(preds_by_img)} images")

    # Build contiguous-id -> name mapping from category_meta (training-time
    # mapping) so per-class rows are correctly labeled. Falls back to GT
    # category ordering if category_meta isn't available.
    meta_path = Path("configs/category_meta.json")
    contiguous_to_name: Dict[int, str] = {}
    if meta_path.exists():
        try:
            meta = json.load(open(meta_path))
            thing_classes = meta.get("thing_classes", [])
            cat_id_map = {int(k): int(v)
                          for k, v in meta.get("thing_dataset_id_to_contiguous_id", {}).items()}
            cat_id_to_name = {c["id"]: c["name"] for c in gt["categories"]}
            for dataset_id, cid in cat_id_map.items():
                if cid < len(thing_classes):
                    contiguous_to_name[cid] = thing_classes[cid]
                elif dataset_id in cat_id_to_name:
                    contiguous_to_name[cid] = cat_id_to_name[dataset_id]
        except Exception as e:
            print(f"  warn: couldn't read category_meta.json ({e})")
    if not contiguous_to_name:
        cat_id_to_name = {c["id"]: c["name"] for c in gt["categories"]}
        for i, cid in enumerate(sorted(cat_id_to_name.keys())):
            contiguous_to_name[i] = cat_id_to_name[cid]

    print(f"  Per-class mapping: {sorted(contiguous_to_name.items())}")

    results: Dict[str, Dict] = {
        "iou_thresholds": list(args.iou_thresholds),
        "mapping_contiguous_to_name": contiguous_to_name,
        "score_min": args.score_min,
        "n_preds": n_preds_total,
        "n_gt": n_gt_total,
        "per_class": {},
        "micro": {},
        "macro": {},
    }

    for iou_t in args.iou_thresholds:
        print(f"\n=== AP_BEV @ IoU {iou_t:.2f} ===")
        # Micro: pool all categories
        ap_micro, npr, ngt = compute_bev_ap_at_iou(
            preds_by_img, gts_by_img, iou_t, cat_filter=None)
        print(f"  micro: {100*ap_micro:6.2f} ({npr} preds, {ngt} GT)")
        results["micro"][f"IoU={iou_t:.2f}"] = 100 * ap_micro

        # Per-class + macro
        per_class_aps: List[float] = []
        for cid, name in sorted(contiguous_to_name.items()):
            ap, npr, ngt = compute_bev_ap_at_iou(
                preds_by_img, gts_by_img, iou_t, cat_filter=cid)
            per_class_aps.append(ap)
            print(f"  {name:12s}: {100*ap:6.2f} ({npr} preds, {ngt} GT)")
            results["per_class"].setdefault(name, {})[f"IoU={iou_t:.2f}"] = 100 * ap
        macro = float(np.mean(per_class_aps)) if per_class_aps else 0.0
        print(f"  macro (mean per-class): {100*macro:6.2f}")
        results["macro"][f"IoU={iou_t:.2f}"] = 100 * macro

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote metrics -> {args.out}")


if __name__ == "__main__":
    main()
