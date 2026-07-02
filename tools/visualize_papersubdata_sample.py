#!/usr/bin/env python
"""Sanity-check visualizer for the rescaled papersubdata.

Loads ONE frame, the corresponding rescaled `cameras.json` extrinsic+K, and
the rescaled `kitti_labels/frame_*.txt` 3D + 2D bboxes; draws:
  - the per-track 3D wireframe (12 edges with halo, matching the paper-viz
    style in vggt/annotator_paper_viz.py),
  - the 2D KITTI bbox in dashed green (so we can verify projected-cuboid
    coverage matches the 2D label).

If the rescale is correct, both should land on the same animal at the same
pixel coordinates.

Usage:
    python tools/visualize_papersubdata_sample.py \\
        --segment /mnt/d/3DBOX/papersubdata/zebr1/DJI_20230607092100_0001_V/seg1 \\
        --frame   1 \\
        --out     /tmp/sample_overlay.jpg
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np


# ---------- Cuboid-corner / projection helpers (mirror annotator_paper_viz) --

# KITTI dimension order: h w l — Y-up, then Z-extent, then X-extent.
# Local-frame corner layout matches BBox3D.get_corners in annotator_tool.py.
EDGES_BOTTOM = ((0, 1), (1, 2), (2, 3), (3, 0))
EDGES_TOP    = ((4, 5), (5, 6), (6, 7), (7, 4))
EDGES_VERT   = ((0, 4), (1, 5), (2, 6), (3, 7))
ALL_EDGES = EDGES_BOTTOM + EDGES_TOP + EDGES_VERT


def kitti_corners_world(h, w, l, x, y, z, ry):
    """KITTI label → 8 corners in world (== camera) coords.

    Convention: camera-frame Y is down, so the "bottom" of the box is at
    y = y_kitti and the "top" is at y_kitti - h. ry rotates around the
    Y axis. dimensions order is (h, w, l) per the KITTI spec.
    """
    R = np.array([
        [ np.cos(ry), 0.0, np.sin(ry)],
        [ 0.0,        1.0, 0.0       ],
        [-np.sin(ry), 0.0, np.cos(ry)],
    ])
    corners_local = np.array([
        # bottom face
        [-l/2, 0,    -w/2],
        [ l/2, 0,    -w/2],
        [ l/2, 0,     w/2],
        [-l/2, 0,     w/2],
        # top face (shifted up by -h in Y-down camera frame)
        [-l/2, -h,   -w/2],
        [ l/2, -h,   -w/2],
        [ l/2, -h,    w/2],
        [-l/2, -h,    w/2],
    ])
    return (R @ corners_local.T).T + np.array([x, y, z])


def project_to_pixels(corners_3d_world, K, extrinsic):
    """corners_3d_world: (8,3) in world coords. Return (8,2) pixel coords
    + a (8,) bool mask of whether each corner is in front of the camera."""
    corners_h = np.concatenate([corners_3d_world, np.ones((8, 1))], axis=1)
    corners_cam = (extrinsic @ corners_h.T).T  # (8, 3)
    z_pos = corners_cam[:, 2] > 0.01
    proj = (K @ corners_cam.T).T
    pix = proj[:, :2] / np.where(proj[:, 2:3] != 0, proj[:, 2:3], 1)
    return pix, z_pos


# ---------- Drawing helpers (matching annotator_paper_viz colors/thickness) --

PALETTE = [
    (231,  76,  60), ( 46, 204, 113), ( 52, 152, 219),
    (241, 196,  15), (155,  89, 182), ( 26, 188, 156),
]
def color_for(idx):
    r, g, b = PALETTE[idx % len(PALETTE)]
    return (b, g, r)  # OpenCV BGR


def draw_wireframe(img, corners_2d, color_bgr, thickness=3):
    halo_t = thickness + 2
    pts = corners_2d.astype(np.int32)
    for a, b in ALL_EDGES:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (0, 0, 0), halo_t, cv2.LINE_AA)
    for a, b in ALL_EDGES:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color_bgr, thickness, cv2.LINE_AA)


def draw_dashed_rect(img, x1, y1, x2, y2, color_bgr, thickness=2, dash=12):
    """Dashed XYXY rectangle for the 2D KITTI label."""
    def line(a, b):
        v = np.array(b) - np.array(a)
        L = float(np.linalg.norm(v))
        if L < 1: return
        n = max(1, int(L // dash))
        u = v / n
        for i in range(0, n, 2):
            p1 = (int(a[0] + u[0] * i),     int(a[1] + u[1] * i))
            p2 = (int(a[0] + u[0] * (i+1)), int(a[1] + u[1] * (i+1)))
            cv2.line(img, p1, p2, color_bgr, thickness, cv2.LINE_AA)
    line((x1, y1), (x2, y1))
    line((x2, y1), (x2, y2))
    line((x2, y2), (x1, y2))
    line((x1, y2), (x1, y1))


# ---------- Main -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segment", type=Path, required=True,
                    help="Per-segment dir (contains cameras.json, kitti_labels/, frame_*.jpg)")
    ap.add_argument("--frame", type=int, default=1,
                    help="Original frame number to visualize (e.g. 1 → frame_000001.jpg)")
    ap.add_argument("--cameras-name", default="cameras.json",
                    help="cameras.json filename inside segment dir (default: cameras.json). "
                         "Pass cameras.rescaled.json to use the dry-run output of fix_papersubdata.")
    ap.add_argument("--kitti-dir", default="kitti_labels",
                    help="kitti_labels dir name (default: kitti_labels). "
                         "Pass kitti_labels.rescaled to use the dry-run output.")
    ap.add_argument("--out", type=Path, default=Path("/tmp/sample_overlay.jpg"))
    args = ap.parse_args()

    seg = args.segment
    cam_path = seg / args.cameras_name
    kit_path = seg / args.kitti_dir / f"frame_{args.frame:06d}.txt"
    img_path = seg / f"frame_{args.frame:06d}.jpg"

    for p in [cam_path, kit_path, img_path]:
        if not p.exists():
            sys.exit(f"missing: {p}")

    img = cv2.imread(str(img_path))
    if img is None:
        sys.exit(f"failed to load {img_path}")
    img_h, img_w = img.shape[:2]
    print(f"Loaded frame {img_path.name}: {img_w}×{img_h}")

    cameras = json.loads(cam_path.read_text())
    # match by image_name (preferred) then frame_index fallback
    cam_entry = next((c for c in cameras["cameras"]
                      if c.get("image_name") == img_path.name), None)
    if cam_entry is None:
        cam_entry = cameras["cameras"][args.frame - 1]
    K = np.array(cam_entry["intrinsic"], dtype=np.float64)
    ext = np.array(cam_entry["extrinsic"], dtype=np.float64)
    # Keep extrinsic at (3,4) so that ext @ [x,y,z,1]^T → (3,) camera coords;
    # padding to (4,4) caused the projection to fold an extra dim back in.
    if ext.shape == (4, 4):
        ext = ext[:3, :]
    cam_w, cam_h = cam_entry.get("image_width", "?"), cam_entry.get("image_height", "?")
    print(f"K = {K[0,0]:.1f}, {K[1,1]:.1f}, {K[0,2]:.1f}, {K[1,2]:.1f}  "
          f"(intrinsics calibrated for {cam_w}×{cam_h})")
    if cam_w != img_w or cam_h != img_h:
        print(f"  ⚠ K image-size {cam_w}×{cam_h} ≠ frame size {img_w}×{img_h} — "
              "projection will be off; run tools/fix_papersubdata.py first.")

    # Load KITTI labels
    objects = []
    for line in kit_path.read_text().splitlines():
        cols = line.split()
        if len(cols) < 15: continue
        cls = cols[0]
        bbox2d = tuple(float(c) for c in cols[4:8])      # x1, y1, x2, y2
        h, w, l = float(cols[8]), float(cols[9]), float(cols[10])
        x, y, z = float(cols[11]), float(cols[12]), float(cols[13])
        ry = float(cols[14])
        objects.append((cls, bbox2d, (h, w, l), (x, y, z), ry))
    print(f"Loaded {len(objects)} KITTI objects from {kit_path.name}")

    # Draw each object
    for i, (cls, b2d, dims, ctr, ry) in enumerate(objects):
        # cuboid (project)
        corners_3d = kitti_corners_world(*dims, *ctr, ry)
        pix, z_ok = project_to_pixels(corners_3d, K, ext)
        col = color_for(i)
        if z_ok.all():
            draw_wireframe(img, pix, col, thickness=3)
        else:
            print(f"  obj {i} ({cls}): some corners behind camera ({(~z_ok).sum()}/8) — skipped wireframe")
        # 2D KITTI bbox
        x1, y1, x2, y2 = b2d
        draw_dashed_rect(img, x1, y1, x2, y2, col, thickness=2)
        # Label
        text = f"{cls} #{i}"
        cv2.putText(img, text, (int(x1), max(15, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    # Add a debug strip
    debug = (f"frame {img_path.name}  {img_w}×{img_h}  "
             f"K={K[0,0]:.0f}/{K[1,1]:.0f}/{K[0,2]:.0f}/{K[1,2]:.0f}  "
             f"objs={len(objects)}")
    cv2.putText(img, debug, (10, img_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, debug, (10, img_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), img)
    print(f"\nWrote {args.out}  (open this to verify alignment)")
    print("If the cuboid wireframe and the dashed 2D box land on the same animal,")
    print("the rescale is correct. If they're shifted, K and KITTI are still in")
    print("different coordinate spaces — re-check fix_papersubdata.py output.")


if __name__ == "__main__":
    main()
