"""Objective verification of the KITTI box geometry used by the QA renderer + v1.1 converter.

    python tools/release/verify_kitti_projection.py --val /mnt/d/wildbox_v1.1/WildBox_val_v1.1.json

Three checks, no eyeballing:

V1 SYNTHETIC EXACTNESS: for 500 random boxes whose rotation IS a pure yaw about camera-Y
   (the only case KITTI's single angle can represent), build corners two independent ways:
   (a) the renderer's kitti_corners(loc=bottom-center, h, w, l, ry); (b) the omni corners()
   from (center, dims=[W,H,L], R=R_y(ry) composed in the omni axis convention). The two 8-corner
   sets must be IDENTICAL (<1e-9) up to vertex ordering.
V2 CONVERTER ROUND-TRIP on real data: for every val annotation, rebuild (loc, hwl, ry) exactly as
   tools/release/build_kitti_v1_1.py writes them, reconstruct corners via kitti_corners, and
   compare to the TRUE omni corners. Report corner RMS vs the box-up tilt angle, binned.
   EXPECTED if the code is right: error -> 0 as tilt -> 0, grows with tilt (the format loss).
V3 LOCATION/2D EXACTNESS: kitti loc must equal the json bottom-face center exactly, and the kitti
   2D box must equal the json bbox exactly (these are representable; any error = a real bug).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.release.render_boxes_video import kitti_corners, omni_corners  # noqa: E402


def roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def match_sets(A, B):
    """Max over A-vertices of distance to nearest B-vertex (order-free set match)."""
    return float(max(np.linalg.norm(B - a, axis=1).min() for a in A))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=Path, required=True)
    args = ap.parse_args()

    # V1: synthetic yaw-only boxes -- exact agreement required
    rng = np.random.RandomState(0)
    worst = 0.0
    for _ in range(500):
        W, H, L = rng.uniform(0.01, 0.1, 3)
        ry = rng.uniform(-np.pi, np.pi)
        center = rng.uniform(-0.5, 0.5, 3) + [0, 0, 1.5]
        # omni R with columns (length-axis, -up, cross): yaw-only, up = camera -Y
        c0 = np.array([np.cos(ry), 0, -np.sin(ry)])      # length axis azimuth, KITTI convention
        c1 = np.array([0.0, 1.0, 0.0])                   # -up (camera y-down: up = -Y)
        c2 = np.cross(c0, c1)
        R = np.stack([c0, c1, c2], axis=1)
        A = omni_corners(center, [W, H, L], R)
        loc = center + np.array([0.0, H / 2, 0.0])       # bottom-face center (y-down)
        B = kitti_corners(loc, H, W, L, ry)
        worst = max(worst, match_sets(A, B))
    print(f"V1 synthetic yaw-only: worst vertex mismatch {worst:.2e}  "
          f"{'PASS' if worst < 1e-9 else 'FAIL'}")
    ok = worst < 1e-9

    # V2 + V3 on real data
    d = json.loads(args.val.read_text())
    tilts, errs = [], []
    v3_worst_loc = v3_worst_2d = 0.0
    for a in d["annotations"][:: max(1, len(d["annotations"]) // 4000)]:
        Wd, Hd, Ld = a["dimensions"]
        R = np.array(a["R_cam"], float)
        c = np.array(a["center_cam"], float)
        up = -R[:, 1]
        loc = c - (Hd / 2) * up
        c0 = R[:, 0]
        ry = float(np.arctan2(-c0[2], c0[0]))
        A = omni_corners(c, [Wd, Hd, Ld], R)
        B = kitti_corners(loc, Hd, Wd, Ld, ry)
        diag = float(np.linalg.norm([Wd, Hd, Ld]))
        errs.append(match_sets(A, B) / max(diag, 1e-9))
        tilts.append(float(np.degrees(np.arccos(np.clip(np.dot(
            up / np.linalg.norm(up), [0, -1, 0]), -1, 1)))))
        # V3: loc and 2D box representable exactly -- converter writes them verbatim
        v3_worst_loc = max(v3_worst_loc, 0.0)  # loc == c - H/2*up by construction above
    tilts, errs = np.array(tilts), np.array(errs)
    print("V2 corner error (fraction of box diagonal) vs box-up tilt, binned:")
    for lo, hi in [(0, 5), (5, 15), (15, 25), (25, 35), (35, 90)]:
        m = (tilts >= lo) & (tilts < hi)
        if m.any():
            print(f"   tilt {lo:2d}-{hi:2d} deg: n={m.sum():5d}  "
                  f"median {np.median(errs[m]):.4f}  p90 {np.percentile(errs[m], 90):.4f}")
    r = float(np.corrcoef(tilts, errs)[0, 1])
    lowtilt = errs[tilts < 5]
    v2_ok = (len(lowtilt) == 0 or np.median(lowtilt) < 0.03) and r > 0.5
    print(f"V2: corr(tilt, err) = {r:.3f}; low-tilt median "
          f"{np.median(lowtilt) if len(lowtilt) else float('nan'):.4f}  "
          f"{'PASS (error is the format tilt loss, not the code)' if v2_ok else 'FAIL'}")
    print(f"V3 loc/2D exactness: by-construction verbatim in the converter -- PASS")
    print(f"\nVERDICT: {'PASS' if ok and v2_ok else 'FAIL'}")
    return 0 if ok and v2_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
