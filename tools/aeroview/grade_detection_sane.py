"""Grade detection on the LABEL-SANE val annotations only.  [CPU, ~2 min/run]

    python tools/aeroview/grade_detection_sane.py \
        --val datasets/Omni3D/WildBox_val.json \
        --preds run1=/path/alpha_s0/.../instances_predictions.pth \
        --preds ctrl=/path/run2_control_s0/.../instances_predictions.pth \
        --preds token=/path/run2_token_s0/.../instances_predictions.pth

WHY: run 2 masks junk labels from TRAIN, but val still contains the same junk (human-audited: 95% of the
off-plane population are fragments). A model trained NOT to reproduce those errors is then punished by a
val set full of them, so standard val numbers cannot say whether the cleaning helped. This grader drops the
same junk population (|off-plane| > 1.0x own height, per segment) from GRADING and reports, per run:

    2D recall on sane GT | z rel err (raw and per-frame-anchor-free) | dims log err | xy err

It is a diagnostic companion to the official numbers, not a replacement -- the official val json is never
modified (the benchmark stays comparable).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

JUNK_OFF_H = 1.0


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by); x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    i = (x2-x1)*(y2-y1)
    return i / (aw*ah + bw*bh - i)


def main() -> int:
    import torch
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--preds", action="append", required=True, help="name=path, repeatable")
    ap.add_argument("--score-thresh", type=float, default=0.25)
    args = ap.parse_args()

    d = json.loads(args.val.read_text())
    ims = {i["id"]: i for i in d["images"]}
    by_seg = defaultdict(list)
    for a in d["annotations"]:
        if a.get("behind_camera"):
            continue
        p = ims[a["image_id"]]["file_path"].replace("\\", "/").split("/")
        by_seg[(p[-3], p[-2])].append(a)

    sane = set()
    for seg, lst in by_seg.items():
        if len(lst) < 30:
            for a in lst:
                sane.add(id(a))
            continue
        up = -np.array(lst[0]["R_cam"], float)[:, 1]
        C = np.array([a["center_cam"] for a in lst]); H = np.array([a["dimensions"][1] for a in lst])
        t = (C - (H[:, None]/2)*up) @ up
        off = (t - np.median(t)) / np.maximum(H, 1e-6)
        for a, o in zip(lst, off):
            if abs(float(o)) <= JUNK_OFF_H:
                sane.add(id(a))

    gt_by_img = defaultdict(list)
    n_all = n_sane = 0
    for lst in by_seg.values():
        for a in lst:
            n_all += 1
            if id(a) in sane:
                gt_by_img[a["image_id"]].append(a)
                n_sane += 1
    print(f"val GT: {n_all:,} total -> {n_sane:,} label-sane graded ({100*n_sane/n_all:.1f}%)\n")

    hdr = (f"{'run':8s} {'recall@.5':>9s} | {'z relerr':>9s} {'z(anchor-free)':>14s} | "
           f"{'dims log-err':>12s} {'xy px-err':>10s}")
    print(hdr); print("-" * len(hdr))
    for spec in args.preds:
        name, _, path = spec.partition("=")
        preds = defaultdict(list)
        for r in torch.load(path, map_location="cpu"):
            for inst in r["instances"]:
                if inst.get("score", 0) >= args.score_thresh:
                    preds[r["image_id"]].append(inst)
        ze, zea, de, xe, nm, ng = [], [], [], [], 0, 0
        for iid, gts in gt_by_img.items():
            K = np.array(ims[iid]["K"], float).reshape(3, 3)
            rows = []
            for g in gts:
                ng += 1
                best, bi = None, 0.5
                for p in preds.get(iid, []):
                    v = iou(g["bbox"], p["bbox"])
                    if v > bi:
                        best, bi = p, v
                if best is None:
                    continue
                nm += 1
                rows.append((g, best))
            if not rows:
                continue
            zg = np.array([r[0]["center_cam"][2] for r in rows])
            zp = np.array([r[1]["center_cam"][2] for r in rows])
            ze += (np.abs(zp - zg) / zg).tolist()
            s = np.median(zg / zp)                      # one free scalar per frame
            zea += (np.abs(zp*s - zg) / zg).tolist()
            for g, p in rows:
                dg = np.array(g["dimensions"]); dp = np.array(p["dimensions"])
                de.append(float(np.mean(np.abs(np.log(np.maximum(dp, 1e-6) / np.maximum(dg, 1e-6))))))
                cg = K @ np.array(g["center_cam"]); cg = cg[:2]/cg[2]
                cp = K @ np.array(p["center_cam"]); cp = cp[:2]/cp[2]
                xe.append(float(np.hypot(*(cp - cg))))
        print(f"{name:8s} {100*nm/max(ng,1):8.1f}% | {100*np.median(ze):8.2f}% {100*np.median(zea):13.2f}% | "
              f"{np.median(de):12.3f} {np.median(xe):9.1f}")
    print("\nread: z(anchor-free) is the within-frame depth quality; raw z includes the per-frame anchor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
