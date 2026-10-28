"""Regenerate the KITTI label channel for v1.1 — FROM the v1.1 jsons, full precision.  [CPU, ~10 min]

    python tools/release/build_kitti_v1_1.py \
        --train /mnt/d/wildbox_v1.1/WildBox_train_v1.1.json \
        --val   /mnt/d/wildbox_v1.1/WildBox_val_v1.1.json \
        --out-root /mnt/d/wildbox_v1.1/kitti_labels

v1.1 channel-consistency rule: KITTI labels are DERIVED from the omni3d jsons (the canonical
channel), so the two can never disagree again (v1.0's did: json carried raw dims, kitti carried
corrected-but-quantized dims). Per annotation:

  type       = category_name
  truncated, occluded = 0.00 0
  alpha      = allocentric observation angle of the KITTI yaw (ry - atan2(x, z))
  bbox       = json 'bbox' (full-res XYWH -> x1 y1 x2 y2), full precision
  h w l      = dimensions[1], dimensions[0], dimensions[2]   (json dims are [W, H, L])
  x y z      = BOTTOM-face center = center_cam - (H/2) * up, with up = -R_cam[:, 1]
               (KITTI's location convention; the verified project up-convention)
  ry         = yaw of the body length axis (R_cam[:,0]) about camera-Y:
               ry = atan2(-c0_z, c0_x)  so that R_y(ry) @ [1,0,0] matches the axis azimuth
  score      = 1.00

All values at 6 decimals (v1.0's 2-decimal quantization is gone). Gauge: SAME per-segment
normalized gauge as the jsons (median z = 1) -- one gauge across both channels, stated in
VERSIONS.md. SCOPE (also for VERSIONS.md): single-yaw KITTI geometry cannot represent the
camera-frame tilt of oblique-aerial boxes (~15-60 deg); the KITTI channel serves species +
2D boxes + coarse 3D for pipeline compatibility; all faithful 3D use must read the jsons.
Per-frame empty files are written for annotated-empty frames so loaders see a complete channel.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tadetr.data.wildbox_paths import parse_label_path  # noqa: E402


def rect_rotation(ups_cam: np.ndarray) -> np.ndarray:
    """cam->rect rotation R (x_rect = R @ x_cam) for an UPRIGHT frame: rect -Y = the frame's
    consensus animal-up (median of per-annotation up vectors), rect Z = camera forward
    re-orthogonalized. In this frame standing animals are yaw-only, so KITTI's single
    rotation_y represents them ~exactly (residual = individual-vs-consensus tilt)."""
    u = np.median(ups_cam, axis=0)
    u = u / max(np.linalg.norm(u), 1e-9)
    y_r = -u
    z_r = np.array([0.0, 0.0, 1.0])
    z_r = z_r - np.dot(z_r, y_r) * y_r
    z_r = z_r / max(np.linalg.norm(z_r), 1e-9)
    x_r = np.cross(y_r, z_r)
    return np.stack([x_r, y_r, z_r])        # rows


def convert_split(json_path: Path, out_root: Path, frame_mode: str = "camera",
                  only: str = "") -> tuple[int, int]:
    d = json.loads(json_path.read_text())
    ims = {i["id"]: i for i in d["images"]}
    anns_by_img = defaultdict(list)
    for a in d["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    n_files = n_objs = 0
    for im in d["images"]:
        group, video, seg, fname = parse_label_path(im["file_path"])
        if only and only not in f"{group}/{video}/{seg}":
            continue
        out_dir = out_root / group / video / seg / "kitti_labels"
        out_dir.mkdir(parents=True, exist_ok=True)
        anns = anns_by_img.get(im["id"], [])
        K = np.array(im["K"], float).reshape(3, 3)
        Rr = np.eye(3)
        if frame_mode == "rectified" and anns:
            ups = np.array([-np.array(a["R_cam"], float)[:, 1] for a in anns])
            Rr = rect_rotation(ups)
            # per-frame KITTI calib: P2 projects RECT coords to pixels (x_img ~ P2 @ [X_rect;1])
            cdir = out_root / group / video / seg / "calib"
            cdir.mkdir(parents=True, exist_ok=True)
            P2 = np.hstack([K @ Rr.T, np.zeros((3, 1))])
            with open(cdir / fname.replace(".jpg", ".txt"), "w") as cf:
                cf.write("P2: " + " ".join(f"{v:.9f}" for v in P2.flatten()) + "\n")
                cf.write("R0_rect: " + " ".join(f"{v:.9f}" for v in np.eye(3).flatten()) + "\n")
        lines = []
        for a in anns:
            W, H, L = a["dimensions"]
            R = Rr @ np.array(a["R_cam"], float)
            c = Rr @ np.array(a["center_cam"], float)
            up = -R[:, 1]
            loc = c - (H / 2) * up
            c0 = R[:, 0]
            ry = float(np.arctan2(-c0[2], c0[0]))
            alpha = float(np.arctan2(np.sin(ry - np.arctan2(loc[0], loc[2])),
                                     np.cos(ry - np.arctan2(loc[0], loc[2]))))
            x, y, w, h2 = a["bbox"]
            lines.append(
                f"{a['category_name']} 0.00 0 {alpha:.6f} "
                f"{x:.4f} {y:.4f} {x + w:.4f} {y + h2:.4f} "
                f"{H:.6f} {W:.6f} {L:.6f} "
                f"{loc[0]:.6f} {loc[1]:.6f} {loc[2]:.6f} {ry:.6f} 1.00")
            n_objs += 1
        (out_dir / fname.replace(".jpg", ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        n_files += 1
    return n_files, n_objs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--frame", choices=["camera", "rectified"], default="camera",
                    help="rectified: upright per-frame reference (yaw-only boxes, EXACT in "
                         "single-ry) + per-frame KITTI calib with P2; standard consumers "
                         "projecting with P2 render correct boxes")
    ap.add_argument("--only", type=str, default="",
                    help="substring filter on group/video/seg (trial runs)")
    args = ap.parse_args()
    for split, p in (("train", args.train), ("val", args.val)):
        nf, no = convert_split(p, args.out_root, frame_mode=args.frame, only=args.only)
        print(f"[{split}] {nf:,} label files, {no:,} objects -> {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
