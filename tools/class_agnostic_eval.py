#!/usr/bin/env python
"""Class-agnostic 2D/3D AP on WildBox_val.

For zero-shot evaluation of a closed-vocab pretrained model on open-vocab
categories (rhino/elephant not in Omni3D). The model predicts Omni3D class
labels like 'chair' on wildlife images — the standard evaluator filters those
out because the class names don't match the dataset. This script instead asks:
"did the model LOCATE the animal at all, regardless of what it called it?"

Usage:
    python tools/class_agnostic_eval.py \\
        --preds output/wildbox_re_zeroshot/inference/iter_final/WildBox_val/instances_predictions.pth \\
        --gt    datasets/Omni3D/WildBox_val.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def cuboid_corners_from_center_dims_R(
    center: np.ndarray, dims_whl: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """(3,) center + (3,) [W, H, L] + (3,3) R -> (8, 3) corner points.

    Uses Omni3D's axis convention: X-extent=L, Y-extent=H, Z-extent=W
    (see cubercnn/util/math_util.py:172-181).
    """
    W, H, L = float(dims_whl[0]), float(dims_whl[1]), float(dims_whl[2])
    corners_local = np.array([
        [-L/2, -H/2, -W/2], [+L/2, -H/2, -W/2],
        [+L/2, +H/2, -W/2], [-L/2, +H/2, -W/2],
        [-L/2, -H/2, +W/2], [+L/2, -H/2, +W/2],
        [+L/2, +H/2, +W/2], [-L/2, +H/2, +W/2],
    ], dtype=np.float64)
    return (R @ corners_local.T).T + center


def normalized_hausdorff(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two (8, 3) corner sets,
    normalized by the mean diagonal of the two cuboids.

    This is a pytorch3d-free surrogate for 3D IoU that captures all three
    sources of error (center, dimensions, pose) in one number. Lower is
    better. 0.0 = perfect overlap; 1.0 means corners are typically one box
    diagonal apart.
    """
    # pairwise distances
    d = np.linalg.norm(corners_a[:, None, :] - corners_b[None, :, :], axis=2)
    h = max(d.min(axis=1).max(), d.min(axis=0).max())
    # normalize by mean box diagonal
    diag_a = np.linalg.norm(corners_a.max(axis=0) - corners_a.min(axis=0))
    diag_b = np.linalg.norm(corners_b.max(axis=0) - corners_b.min(axis=0))
    return float(h / (0.5 * (diag_a + diag_b) + 1e-8))


def iou_2d(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized IoU for boxes in XYXY. (N, 4) x (M, 4) -> (N, M)."""
    ax1, ay1, ax2, ay2 = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3], boxes_a[:, 3:4]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def compute_ap_at_iou(preds_per_img, gts_per_img, iou_thresh: float) -> float:
    """Class-agnostic AP at a fixed IoU threshold, standard greedy matching.

    preds_per_img: list of (scores[N], boxes[N,4]) per image.
    gts_per_img:   list of boxes[M,4] per image.
    """
    # Flatten predictions with image index and sort by descending score.
    all_scores = []
    all_boxes = []
    all_imgs = []
    for i, (s, b) in enumerate(preds_per_img):
        if len(s) == 0:
            continue
        all_scores.extend(s.tolist() if hasattr(s, "tolist") else list(s))
        all_boxes.append(b)
        all_imgs.extend([i] * len(s))
    if not all_scores:
        return 0.0
    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.array(all_scores)
    all_imgs = np.array(all_imgs)

    order = np.argsort(-all_scores)
    all_boxes = all_boxes[order]
    all_imgs = all_imgs[order]

    gt_matched = [np.zeros(len(g), dtype=bool) for g in gts_per_img]
    n_gt = sum(len(g) for g in gts_per_img)
    if n_gt == 0:
        return 0.0

    tp = np.zeros(len(all_scores))
    fp = np.zeros(len(all_scores))
    for i, (box, img_idx) in enumerate(zip(all_boxes, all_imgs)):
        gts = gts_per_img[img_idx]
        if len(gts) == 0:
            fp[i] = 1
            continue
        ious = iou_2d(box[None, :], gts)[0]
        j = int(np.argmax(ious))
        if ious[j] >= iou_thresh and not gt_matched[img_idx][j]:
            tp[i] = 1
            gt_matched[img_idx][j] = True
        else:
            fp[i] = 1

    # PR curve
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-10)
    recall = tp_cum / n_gt
    # 11-point VOC interpolation (simple, adequate for a diagnostic)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t].max() if np.any(recall >= t) else 0.0
        ap += p / 11
    return float(ap)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds", type=Path, required=True,
                   help="instances_predictions.pth from an eval run")
    p.add_argument("--gt", type=Path, required=True,
                   help="Omni3D-format val JSON (e.g. WildBox_val.json)")
    p.add_argument("--iou-thresholds", type=float, nargs="+",
                   default=[0.25, 0.5, 0.75])
    p.add_argument("--score-min", type=float, default=0.0,
                   help="Drop predictions below this score before evaluating")
    p.add_argument("--nhd", action="store_true",
                   help="Also compute class-agnostic 3D NHD (pytorch3d-free "
                        "surrogate for 3D IoU — lower is better; scale-invariant "
                        "via automatic global-scale search). Useful for synthetic-"
                        "scale VGGT data.")
    p.add_argument("--nhd-scales", type=str, default="0.1,5.0,50",
                   help="Scale search for NHD: 'min,max,n_log_steps'")
    args = p.parse_args()

    print(f"Loading predictions from {args.preds}...")
    preds = torch.load(args.preds, weights_only=False)
    print(f"Loading GT from {args.gt}...")
    gt = json.load(open(args.gt))

    # Build per-image GT boxes (XYXY). Omni3D bbox is XYWH.
    gt_by_img: dict[int, list[list[float]]] = {}
    for ann in gt["annotations"]:
        x, y, w, h = ann["bbox"]
        gt_by_img.setdefault(ann["image_id"], []).append([x, y, x + w, y + h])

    # Total raw prediction stats
    total = sum(len(x.get("instances", [])) for x in preds)
    print(f"\n=== raw predictions ===")
    print(f"  images: {len(preds)}")
    print(f"  total boxes: {total}")
    if total == 0:
        print("  no predictions — model's RPN fired on nothing. Class-agnostic AP = 0.")
        return

    # Score distribution + top classes (diagnostic only)
    from collections import Counter
    score_samples = []
    class_counts: Counter = Counter()
    for im in preds:
        for inst in im.get("instances", []):
            score_samples.append(float(inst["score"]))
            class_counts[int(inst["category_id"])] += 1
    score_samples = np.array(score_samples)
    print(f"  score: min={score_samples.min():.3f} "
          f"median={np.median(score_samples):.3f} max={score_samples.max():.3f}")
    print(f"  top predicted class-ids (closed-vocab): {class_counts.most_common(10)}")

    # Build aligned preds/gts lists.
    preds_per_img = []
    gts_per_img = []
    matched_imgs = 0
    for im_obj in preds:
        img_id = im_obj["image_id"]
        if img_id not in gt_by_img:
            continue
        matched_imgs += 1
        insts = [i for i in im_obj.get("instances", [])
                 if float(i["score"]) >= args.score_min]
        if insts:
            scores = np.array([float(i["score"]) for i in insts])
            # Omni3D saves bbox as XYWH.
            boxes = np.array([i["bbox"] for i in insts])
            if boxes.shape[1] == 4 and (boxes[:, 2:] >= boxes[:, :2]).all():
                # already XYXY
                pass
            else:
                boxes[:, 2] += boxes[:, 0]
                boxes[:, 3] += boxes[:, 1]
        else:
            scores = np.array([])
            boxes = np.zeros((0, 4))
        preds_per_img.append((scores, boxes))
        gts_per_img.append(np.array(gt_by_img[img_id]))

    print(f"\n=== matched images (have both GT and predictions): {matched_imgs} ===")

    # Per-threshold AP
    print(f"\n=== class-agnostic 2D AP ===")
    print(f"  (ignores class labels; 'did any prediction overlap any GT box?')")
    for t in args.iou_thresholds:
        ap = compute_ap_at_iou(preds_per_img, gts_per_img, t)
        print(f"  AP@{t:.2f} = {100*ap:6.2f}")

    # Per-class class-agnostic: split preds+gts by class index using the
    # model's OWN contiguous ids. Useful after fine-tuning, where the model
    # learned which slot = which species -- tells you which class the
    # localizer struggles with.
    classes_in_gt = {ann["category_id"] for ann in gt["annotations"]}
    cat_id_to_name = {c["id"]: c["name"] for c in gt["categories"]}
    # Build dataset-id -> contiguous-id mapping from GT categories (sorted by id).
    sorted_ids = sorted(c["id"] for c in gt["categories"])
    dataset_to_contiguous = {cid: i for i, cid in enumerate(sorted_ids)}
    contiguous_to_name = {i: cat_id_to_name[cid] for cid, i in dataset_to_contiguous.items()}

    # Group GT by (image, contiguous_class)
    per_class_gt = {c: {} for c in contiguous_to_name}
    for ann in gt["annotations"]:
        ctg = dataset_to_contiguous[ann["category_id"]]
        x, y, w, h = ann["bbox"]
        per_class_gt[ctg].setdefault(ann["image_id"], []).append([x, y, x + w, y + h])

    # Group predictions by contiguous_class
    per_class_preds: dict[int, dict] = {c: {} for c in contiguous_to_name}
    for im in preds:
        img_id = im["image_id"]
        for inst in im.get("instances", []):
            c = int(inst["category_id"])
            if c not in per_class_preds:
                continue  # OOV contiguous class from 50-head
            box = list(inst["bbox"])
            box[2] += box[0]
            box[3] += box[1]
            per_class_preds[c].setdefault(img_id, []).append((float(inst["score"]), box))

    if args.nhd:
        print(f"\n=== class-agnostic 3D NHD (pytorch3d-free Rel-AP3D surrogate) ===")
        # Collect (center_cam, dims, R) for every GT ann and every prediction.
        # Match by 2D IoU (>= 0.5) within each image, then compute NHD for
        # matched pairs. Finally, search a single global scale factor that
        # minimizes mean NHD across all matched pairs -- that's the
        # synthetic-scale correction, analogous to Rel-AP3D.
        gt_3d_by_img: dict[int, list] = {}
        for ann in gt["annotations"]:
            if not ann.get("valid3D", True) or ann.get("behind_camera", False):
                continue
            c = np.array(ann["center_cam"], dtype=np.float64)
            # Omni3D stores dimensions as [W, H, L]
            d = np.array(ann["dimensions"], dtype=np.float64)
            R = np.array(ann["R_cam"], dtype=np.float64)
            x, y, w, h = ann["bbox"]
            gt_3d_by_img.setdefault(ann["image_id"], []).append({
                "corners": cuboid_corners_from_center_dims_R(c, d, R),
                "box2d": np.array([x, y, x+w, y+h]),
            })

        pred_3d_by_img: dict[int, list] = {}
        # Diagnostics: what keys are actually present on predictions?
        sample_keys = None
        skipped_reasons: dict[str, int] = {}
        kept = 0
        for im in preds:
            img_id = im["image_id"]
            for inst in im.get("instances", []):
                if sample_keys is None:
                    sample_keys = sorted(inst.keys())
                score = float(inst["score"])
                if score < args.score_min:
                    continue
                # OVMono3D saves predictions as center_cam + dimensions + pose
                # (not as pre-built 8-corner array). Build corners from those.
                corners = None
                if "bbox3D_cam" in inst and inst["bbox3D_cam"] is not None:
                    try:
                        c = np.array(inst["bbox3D_cam"], dtype=np.float64)
                        if c.shape == (8, 3):
                            corners = c
                    except Exception:
                        pass
                if corners is None and all(k in inst for k in ("center_cam", "dimensions", "pose")):
                    try:
                        center = np.array(inst["center_cam"], dtype=np.float64).reshape(-1)
                        dims = np.array(inst["dimensions"], dtype=np.float64).reshape(-1)
                        pose = np.array(inst["pose"], dtype=np.float64)
                        # pose may be (3,3) or flattened (9,)
                        if pose.size == 9:
                            pose = pose.reshape(3, 3)
                        if center.size == 3 and dims.size == 3 and pose.shape == (3, 3):
                            corners = cuboid_corners_from_center_dims_R(
                                center, dims, pose)
                    except Exception as e:
                        skipped_reasons[f"build_err:{type(e).__name__}"] = \
                            skipped_reasons.get(f"build_err:{type(e).__name__}", 0) + 1
                if corners is None:
                    missing = [k for k in ("bbox3D_cam", "center_cam",
                                           "dimensions", "pose") if k not in inst]
                    key = "missing:" + ",".join(missing) if missing else "bad_shape"
                    skipped_reasons[key] = skipped_reasons.get(key, 0) + 1
                    continue
                x, y, w, h = inst["bbox"]
                pred_3d_by_img.setdefault(img_id, []).append({
                    "corners": corners,
                    "box2d": np.array([x, y, x+w, y+h]),
                    "score": score,
                })
                kept += 1
        if sample_keys is not None:
            print(f"  prediction keys seen: {sample_keys}")
        print(f"  3D-valid preds kept:  {kept}")
        if skipped_reasons:
            print(f"  skipped (first 5 reasons): "
                  f"{sorted(skipped_reasons.items(), key=lambda x: -x[1])[:5]}")

        # Match predictions to GT boxes via 2D IoU and collect pairs.
        pairs = []  # (pred_corners, gt_corners)
        for img_id, gts in gt_3d_by_img.items():
            preds_here = pred_3d_by_img.get(img_id, [])
            if not preds_here:
                continue
            gt_boxes = np.array([g["box2d"] for g in gts])
            pd_boxes = np.array([p["box2d"] for p in preds_here])
            iou = iou_2d(pd_boxes, gt_boxes)  # (P, G)
            # greedy match by highest score * best IoU
            order = np.argsort(-np.array([p["score"] for p in preds_here]))
            gt_used = np.zeros(len(gts), dtype=bool)
            for pi in order:
                if not len(gts):
                    break
                j = int(np.argmax(iou[pi]))
                if iou[pi, j] >= 0.5 and not gt_used[j]:
                    pairs.append((preds_here[pi]["corners"], gts[j]["corners"]))
                    gt_used[j] = True
        if not pairs:
            print("  no 2D-matched 3D pairs -- cannot compute NHD")
        else:
            # At scale s, scale pred corners uniformly about origin.
            lo, hi, n = args.nhd_scales.split(",")
            scales = np.logspace(np.log10(float(lo)), np.log10(float(hi)), int(n))
            best_s, best_mean = 1.0, float("inf")
            for s in scales:
                nhds = [normalized_hausdorff(s * p, g) for p, g in pairs]
                m = float(np.mean(nhds))
                if m < best_mean:
                    best_mean, best_s = m, s
            print(f"  matched pairs:      {len(pairs)}")
            print(f"  mean NHD @ s=1:     {np.mean([normalized_hausdorff(p, g) for p, g in pairs]):6.3f}")
            print(f"  best global scale:  {best_s:6.3f}")
            print(f"  mean NHD @ best s:  {best_mean:6.3f}   (lower = better; ~0.5 is decent, <0.3 is good)")
            # Fraction of pairs below NHD thresholds (like AP@IoU but for NHD).
            nhds = np.array([normalized_hausdorff(best_s * p, g) for p, g in pairs])
            for thr in (0.3, 0.5, 1.0):
                frac = (nhds < thr).mean() * 100
                print(f"  frac pairs NHD<{thr}: {frac:6.2f}%")

    print(f"\n=== per-class AP (model's own class assignments) ===")
    for c, name in contiguous_to_name.items():
        # Build aligned lists using GT images for this class
        ids = sorted(per_class_gt[c].keys())
        if not ids:
            continue
        ppi = []
        gpi = []
        for iid in ids:
            entries = per_class_preds[c].get(iid, [])
            if entries:
                s = np.array([e[0] for e in entries])
                b = np.array([e[1] for e in entries])
            else:
                s = np.array([])
                b = np.zeros((0, 4))
            ppi.append((s, b))
            gpi.append(np.array(per_class_gt[c][iid]))
        for t in args.iou_thresholds:
            ap = compute_ap_at_iou(ppi, gpi, t)
            print(f"  {name:12s} AP@{t:.2f} = {100*ap:6.2f}  "
                  f"(gt_images={len(ids)}, preds_seen={sum(len(v) for v in per_class_preds[c].values())})")


if __name__ == "__main__":
    main()
