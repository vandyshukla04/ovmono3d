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


if __name__ == "__main__":
    main()
