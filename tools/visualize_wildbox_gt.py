#!/usr/bin/env python
"""
Gate-2 sanity check for WildBox fine-tuning: overlay GT cuboids from a
prepared Omni3D JSON onto their source frames and save them to disk.

This is the manual check that catches coordinate-conversion bugs in the
KITTI -> Omni3D conversion (center shift, dim ordering, rotation sign)
before wasting training time on garbage labels. Run this on the output
of tools/prepare_wildbox_dataset.py and eyeball the results.

Usage:
  python tools/visualize_wildbox_gt.py \
      --json datasets/Omni3D/WildBox_train.json \
      --out  output/wildbox_gt_vis \
      --num  10
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


# Edge pairs for the 8-corner cuboid layout used by
# cubercnn.util.math_util.get_cuboid_verts_faces:
#   0-1-2-3 is the front face, 4-5-6-7 is the back face.
CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # front
    (4, 5), (5, 6), (6, 7), (7, 4),  # back
    (0, 4), (1, 5), (2, 6), (3, 7),  # connectors
]


def project_corners(corners_cam, K):
    """Project (8, 3) camera-space corners to (8, 2) pixels."""
    corners = np.array(corners_cam, dtype=np.float64)
    K = np.array(K, dtype=np.float64)
    xs = corners[:, 0] / corners[:, 2]
    ys = corners[:, 1] / corners[:, 2]
    u = K[0, 0] * xs + K[0, 2]
    v = K[1, 1] * ys + K[1, 2]
    return np.stack([u, v], axis=1)


def draw_cuboid(img, corners_2d, color=(0, 255, 0), thickness=2):
    for i, j in CUBOID_EDGES:
        p0 = tuple(np.round(corners_2d[i]).astype(int))
        p1 = tuple(np.round(corners_2d[j]).astype(int))
        cv2.line(img, p0, p1, color, thickness, lineType=cv2.LINE_AA)


def draw_bbox(img, xywh, color=(0, 0, 255), thickness=2):
    x, y, w, h = xywh
    p0 = (int(round(x)), int(round(y)))
    p1 = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(img, p0, p1, color, thickness)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num", type=int, default=10,
                    help="Number of frames to visualize")
    ap.add_argument("--image-root", default="datasets",
                    help="Root prepended to image file_path (matches "
                         "simple_register image_root at datasets.py:131)")
    ap.add_argument("--step", type=int, default=1,
                    help="Take every Nth frame (avoid near-duplicates)")
    args = ap.parse_args()

    with open(args.json, "r") as f:
        data = json.load(f)

    args.out.mkdir(parents=True, exist_ok=True)

    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    picked = 0
    for idx, img_info in enumerate(data["images"]):
        if idx % args.step != 0:
            continue
        img_id = img_info["id"]
        if img_id not in anns_by_image:
            continue

        img_path = Path(args.image_root) / img_info["file_path"]
        if not img_path.exists():
            # fall back to the original absolute path hint
            img_path = Path(img_info["file_path"])
        if not img_path.exists():
            print(f"MISSING: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"UNREADABLE: {img_path}")
            continue

        K = np.array(img_info["K"], dtype=np.float64)

        for ann in anns_by_image[img_id]:
            corners_cam = np.array(ann["bbox3D_cam"], dtype=np.float64)
            corners_2d = project_corners(corners_cam, K)
            draw_cuboid(img, corners_2d, color=(0, 255, 0), thickness=2)
            draw_bbox(img, ann["bbox"], color=(0, 0, 255), thickness=2)

        out_path = args.out / f"gtvis_{picked:03d}_{Path(img_info['file_path']).name}"
        cv2.imwrite(str(out_path), img)
        picked += 1
        if picked >= args.num:
            break

    print(f"Wrote {picked} visualizations to {args.out}")
    print("Green = projected 3D cuboid, Red = 2D bbox.")
    print("PASS if green cuboid visually wraps the animal and aligns with the red box.")


if __name__ == "__main__":
    main()
