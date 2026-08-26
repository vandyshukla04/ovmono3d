"""Stamp TA-DETR contact targets onto WildBox jsons, in place, idempotent.  [CPU, ~2 min]

    python tools/tadetr/stamp_contact_targets.py \
        --train /mnt/d/aeroview/labelled/WildBox_train_paper.json \
        --val   /mnt/d/aeroview/labelled/WildBox_val_paper.json

Per annotation (both splits):
  contact_uv    [u_px, v_px] full-res: the GT box BOTTOM-FACE CENTER projected with the image K.
                bottom-face center = center_cam - (H/2) * up,  up = -R_cam[:,1]
                (the verified convention: +R[:,1] is the arrow-inversion bug; center_cam is the
                VGGT centroid, NOT the bottom).
  contact_valid 1.0 iff valid3D (junk-masked annotations excluded -- stamp_geometry.py must run
                FIRST on the train split), not behind_camera, and z > 0 at the bottom face.

The stamp is additive: cubercnn's datasets.py ignores unknown annotation keys unless whitelisted,
so existing consumers are unaffected. Run the SAME command on the cluster jsons before M3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def stamp(path: Path) -> None:
    d = json.loads(path.read_text())
    ims = {i["id"]: i for i in d["images"]}
    n, n_valid = 0, 0
    for a in d["annotations"]:
        K = np.array(ims[a["image_id"]]["K"], float).reshape(3, 3)
        Rc = np.array(a["R_cam"], float)
        H = float(a["dimensions"][1])
        up = -Rc[:, 1]
        bot = np.array(a["center_cam"], float) - (H / 2) * up
        valid = (not a.get("behind_camera", False)) and a.get("valid3D", True) and bot[2] > 1e-6
        if bot[2] > 1e-6:
            uv = K @ bot
            u, v = float(uv[0] / uv[2]), float(uv[1] / uv[2])
        else:
            u = v = -1.0
        a["contact_uv"] = [round(u, 2), round(v, 2)]
        a["contact_valid"] = 1.0 if valid else 0.0
        n += 1
        n_valid += int(valid)
    path.write_text(json.dumps(d))
    print(f"{path.name}: contact stamped on {n:,} annotations ({n_valid:,} valid, "
          f"{100*n_valid/max(n,1):.1f}%)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    args = ap.parse_args()
    stamp(args.train)
    stamp(args.val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
