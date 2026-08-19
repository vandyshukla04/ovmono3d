#!/usr/bin/env python
"""Score every prediction by its TRUE BEV IoU -> the ceiling any confidence function could reach.

Correctness note: this IMPORTS `bev_footprint` / `rotated_iou` from tools/bev_ap_eval.py rather than
reimplementing them. An earlier hand-rolled version got two things wrong -- it used a 4-corner
mid-height slice instead of the convex hull of all 8 corners, and it put dims[1] on the Z-extent when
the Omni3D convention is dims=[W,H,L] with W->Z, H->Y, L->X. Importing removes the whole class of bug.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, "/home/shuklva/ovmono3d")
from tools.bev_ap_eval import bev_footprint, rotated_iou     # the evaluator's own geometry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", default="/mnt/d/3DBOX/papersubdata/WildBox_val_paper.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores-npy", default=None, help="also save just the scores (small)")
    args = ap.parse_args()

    import torch
    gtj = json.load(open(args.gt))
    cats = sorted({int(c["id"]) for c in gtj["categories"]})
    ds2c = {ds: i for i, ds in enumerate(cats)}
    gt = defaultdict(list)
    for a in gtj["annotations"]:
        if not a.get("valid3D", True) or a.get("behind_camera", False):
            continue
        poly = bev_footprint(np.asarray(a["center_cam"], float),
                             np.asarray(a["dimensions"], float),
                             np.asarray(a["R_cam"], float))
        if poly is not None:
            gt[a["image_id"]].append((ds2c[int(a["category_id"])], poly))

    preds = torch.load(args.preds, map_location="cpu", weights_only=False)
    ids = {int(x["category_id"]) for e in preds for x in e.get("instances", [])}
    contig = not (ids & set(cats))
    scores = []
    for e in preds:
        g = gt.get(int(e["image_id"]), [])
        for inst in e.get("instances", []):
            c = int(inst["category_id"]); c = c if contig else ds2c.get(c, -1)
            best = 0.0
            try:
                pp = bev_footprint(np.asarray(inst["center_cam"], float).reshape(-1),
                                   np.asarray(inst["dimensions"], float).reshape(-1),
                                   np.asarray(inst["pose"], float))
                if pp is not None:
                    for gc, gp in g:
                        if gc == c:
                            best = max(best, rotated_iou(pp, gp))
            except Exception:
                best = 0.0
            inst["score"] = float(best)
            scores.append(best)
    torch.save(preds, args.out)
    s = np.array(scores)
    if args.scores_npy:
        np.save(args.scores_npy, s.astype(np.float32))
    print(f"{len(s)} preds | true BEV IoU: med {np.median(s):.3f} "
          f"frac>0.25 {100*(s>0.25).mean():.1f}%  frac>0.5 {100*(s>0.5).mean():.1f}%")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
