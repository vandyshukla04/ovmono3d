#!/usr/bin/env python
"""D0 (the gate): how well does a TRIVIAL predictor score on WildBox?

Per-detection it emits: the GROUND-TRUTH 2D box (so 2D is perfect and the 3D head is isolated),
depth z = 1.0 (the per-segment normalisation makes median depth exactly 1.0 by construction),
per-class MEDIAN dimensions and per-class MEAN rotation -- all estimated on TRAIN, applied to VAL.
It contains no learning and no image content whatsoever.

If this scores near the fine-tuned model (13.17 macro 3D AP / 8.68 BEV@0.50), then the trained 3D
head is contributing almost nothing on this data, and that is the headline finding.

    python tools/aeroview/d0_trivial_predictor.py --out /mnt/d/aeroview/d0_trivial_preds.pth
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def mean_rotation(mats):
    """Proper average of rotation matrices: project the arithmetic mean back onto SO(3) via SVD."""
    M = np.mean(np.stack(mats), axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:                      # reflection -> flip the least-significant axis
        U[:, -1] *= -1
        R = U @ Vt
    return R


def class_priors(train_json):
    d = json.load(open(train_json))
    dims, poses = defaultdict(list), defaultdict(list)
    for a in d["annotations"]:
        c = int(a["category_id"])
        dims[c].append([float(x) for x in a["dimensions"]])
        poses[c].append(np.asarray(a["R_cam"], float))
    return ({c: np.median(np.array(v), axis=0) for c, v in dims.items()},
            {c: mean_rotation(v) for c, v in poses.items()})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="/mnt/d/3DBOX/papersubdata/WildBox_train_paper.json")
    ap.add_argument("--val", default="/mnt/d/3DBOX/papersubdata/WildBox_val_paper.json")
    ap.add_argument("--out", default="/mnt/d/aeroview/d0_trivial_preds.pth")
    ap.add_argument("--z", type=float, default=1.0, help="constant depth (GT median is exactly 1.0)")
    args = ap.parse_args()

    import torch
    dims_p, pose_p = class_priors(args.train)
    print(f"class priors from TRAIN ({len(dims_p)} classes):")
    for c in sorted(dims_p):
        print(f"  cat {c}: dims median {np.round(dims_p[c], 4).tolist()}")

    d = json.load(open(args.val))
    cats = sorted({int(c["id"]) for c in d["categories"]})
    ds2cont = {ds: i for i, ds in enumerate(cats)}          # dataset id -> contiguous, as the eval expects
    imgs = {i["id"]: i for i in d["images"]}
    by_img = defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)

    out, n_inst = [], 0
    for iid, anns in by_img.items():
        im = imgs[iid]
        K = np.asarray(im["K"], float)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        inst = []
        for a in anns:
            c = int(a["category_id"])
            x, y, w, h = [float(v) for v in a["bbox"]]      # GT 2D box: 2D is made perfect on purpose
            u, v = x + w / 2.0, y + h / 2.0
            z = float(args.z)
            centre = [z * (u - cx) / fx, z * (v - cy) / fy, z]
            inst.append({
                "image_id": str(iid),
                "category_id": ds2cont[c],
                "bbox": [x, y, w, h],
                "score": 1.0,
                "depth": z,
                "center_cam": centre,
                "center_2D": [u, v],
                "dimensions": dims_p[c].tolist(),
                "pose": pose_p[c].tolist(),
            })
            n_inst += 1
        out.append({"image_id": iid, "K": im["K"], "width": im["width"],
                    "height": im["height"], "instances": inst})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"\nwrote {args.out}: {len(out)} images, {n_inst} instances "
          f"(z={args.z}, GT 2D boxes, per-class median dims + mean pose)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
