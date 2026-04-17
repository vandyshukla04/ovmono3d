#!/usr/bin/env python
"""Class-agnostic visualization of predictions on top of ground truth.

Works for both zero-shot (predictions have out-of-vocab class_ids that the
standard OVMono3D visualizer filters out) and fine-tuned runs. Draws:
  - GT 2D boxes in green
  - Top-K predictions per image in red (score shown), class label ignored
  - Optional: 3D cuboid wireframes for predictions with valid pose

Usage:
    python tools/visualize_class_agnostic.py \\
        --preds output/wildbox_wl_zeroshot/inference/iter_final/WildBox_val/instances_predictions.pth \\
        --gt    datasets/Omni3D/WildBox_val.json \\
        --out   output/wildbox_wl_zeroshot/vis_agnostic \\
        --top-k 5 \\
        --every 100
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def project(points3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(N,3) cam points -> (N,2) pixels via K."""
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


# edges of the cuboid (pairs of corner indices in our ordering)
CUBOID_EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7)]


def draw_cuboid_wireframe(img, corners2d, color, thickness=2):
    for a, b in CUBOID_EDGES:
        pa = tuple(int(v) for v in corners2d[a])
        pb = tuple(int(v) for v in corners2d[b])
        cv2.line(img, pa, pb, color, thickness)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=5,
                   help="Draw the top-K highest-scoring predictions per image")
    p.add_argument("--every", type=int, default=50,
                   help="Save every Nth image (default 50 -> ~46 images per 2300-img val)")
    p.add_argument("--score-min", type=float, default=0.0,
                   help="Skip predictions below this score")
    p.add_argument("--no-3d", action="store_true",
                   help="Draw only 2D boxes, skip 3D cuboid projection")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after saving this many images (0 = no limit)")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from {args.preds}...")
    preds = torch.load(args.preds, weights_only=False)
    print(f"Loading GT from {args.gt}...")
    gt = json.load(open(args.gt))

    # Build per-image lookups
    ann_by_img: dict[int, list] = {}
    for ann in gt["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)
    img_by_id = {im["id"]: im for im in gt["images"]}

    pred_by_img = {im["image_id"]: im for im in preds}

    saved = 0
    for i, img_id in enumerate(sorted(img_by_id.keys())):
        if i % args.every != 0:
            continue
        if args.limit and saved >= args.limit:
            break
        img_info = img_by_id[img_id]
        img_path = Path(img_info["file_path"])
        if not img_path.is_absolute():
            # Omni3D loader prepends "datasets/"
            img_path = Path("datasets") / img_path
        if not img_path.exists():
            continue

        im = cv2.imread(str(img_path))
        if im is None:
            continue

        K = np.array(img_info["K"], dtype=np.float64)

        # --- GT in green (2D) + dark green (3D) ---
        for ann in ann_by_img.get(img_id, []):
            x, y, w, h = ann["bbox"]
            cv2.rectangle(im, (int(x), int(y)), (int(x+w), int(y+h)),
                          (0, 220, 0), 2)
            cv2.putText(im, ann.get("category_name", ""),
                        (int(x), max(15, int(y) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)
            if not args.no_3d and ann.get("valid3D", True):
                try:
                    c = np.array(ann["center_cam"], dtype=np.float64)
                    d = np.array(ann["dimensions"], dtype=np.float64)
                    R = np.array(ann["R_cam"], dtype=np.float64)
                    corners3d = cuboid_corners(c, d, R)
                    if (corners3d[:, 2] > 0).all():
                        corners2d = project(corners3d, K)
                        draw_cuboid_wireframe(im, corners2d, (0, 140, 0), 1)
                except Exception:
                    pass

        # --- Top-K predictions in red (2D) + dark red (3D) ---
        im_obj = pred_by_img.get(img_id)
        if im_obj:
            insts = [inst for inst in im_obj.get("instances", [])
                     if float(inst.get("score", 0)) >= args.score_min]
            insts.sort(key=lambda x: -float(x["score"]))
            for inst in insts[:args.top_k]:
                score = float(inst["score"])
                x, y, w, h = inst["bbox"]
                cv2.rectangle(im, (int(x), int(y)), (int(x+w), int(y+h)),
                              (0, 0, 220), 2)
                cv2.putText(im, f"{score:.2f}",
                            (int(x), max(15, int(y) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 2)
                if args.no_3d:
                    continue
                # 3D wireframe from center_cam + dimensions + pose
                try:
                    c = np.array(inst["center_cam"], dtype=np.float64).reshape(-1)
                    d = np.array(inst["dimensions"], dtype=np.float64).reshape(-1)
                    pose = np.array(inst["pose"], dtype=np.float64)
                    if pose.size == 9:
                        pose = pose.reshape(3, 3)
                    if c.size == 3 and d.size == 3 and pose.shape == (3, 3):
                        corners3d = cuboid_corners(c, d, pose)
                        if (corners3d[:, 2] > 0).all():
                            corners2d = project(corners3d, K)
                            draw_cuboid_wireframe(im, corners2d, (0, 0, 140), 1)
                except Exception:
                    pass

        # Legend
        cv2.putText(im, "GT (green) | pred (red)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_path = args.out / f"combined_{img_id:06d}.jpg"
        cv2.imwrite(str(out_path), im)
        saved += 1

    print(f"wrote {saved} images to {args.out}")


if __name__ == "__main__":
    main()
