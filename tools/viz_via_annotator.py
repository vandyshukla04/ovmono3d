"""Visualize a segment frame using the canonical vggt-annotator pipeline.

Imports `compute_bbox_corners`, `render_paper_frame`, and
`_scale_intrinsics_to_image` from /home/shuklva/vggt/annotator_paper_viz.py
to render exactly what the vggt annotator UI produces. Reads from
tracking_summary.json (full-precision annotator format), NOT kitti_labels.

Usage:
    python tools/viz_via_annotator.py <segment_dir> <frame_no> <out_path>
"""
import sys, json
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, "/home/shuklva/vggt")
from annotator_paper_viz import (
    render_paper_frame,
    _scale_intrinsics_to_image,
)

if len(sys.argv) < 4:
    print("usage: viz_via_annotator.py <segment_dir> <frame_no> <out_path>")
    sys.exit(1)
SEG = Path(sys.argv[1])
FRAME_NO = int(sys.argv[2])
OUT = Path(sys.argv[3])

img_path = SEG / f"frame_{FRAME_NO:06d}.jpg"
img = cv2.imread(str(img_path))
img_h, img_w = img.shape[:2]
print(f"Loaded {img_path.name}: {img_w}x{img_h}")

cameras = json.load(open(SEG / "cameras.json"))
cam = next(c for c in cameras["cameras"] if c["image_name"] == img_path.name)
K_model = np.array(cam["intrinsic"], dtype=np.float64)
ext = np.array(cam["extrinsic"], dtype=np.float64)
if ext.shape == (4, 4):
    ext = ext[:3, :]
model_h = cam.get("image_height", 294)
model_w = cam.get("image_width", 518)
print(f"Original K (calibrated for {model_w}x{model_h}): "
      f"fx={K_model[0,0]:.1f} fy={K_model[1,1]:.1f} cx={K_model[0,2]:.1f} cy={K_model[1,2]:.1f}")

K = _scale_intrinsics_to_image(K_model, (model_h, model_w), (img_h, img_w))
print(f"Annotator-rescaled K (for {img_w}x{img_h}): "
      f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

ts = json.load(open(SEG / "tracking_summary.json"))
tracks = ts.get("tracks", {})
print(f"tracks: {len(tracks)}")

meta = json.load(open(SEG / "vggt_metadata.json"))
frame_numbers = meta["frame_numbers"]
try:
    frame_idx = frame_numbers.index(FRAME_NO)
except ValueError:
    print(f"frame {FRAME_NO} not in vggt_metadata frame_numbers list; aborting")
    sys.exit(1)
print(f"frame_number={FRAME_NO} -> sequential frame_idx={frame_idx}")

all_track_ids = sorted(int(t) for t in tracks.keys())
bboxes = []
for tid_str, track in tracks.items():
    tid = int(tid_str)
    cls = track.get("class_name", "object")
    frames = track.get("frames", [])
    if frame_idx not in frames: continue
    i = frames.index(frame_idx)
    centers = track.get("centers", [])
    dims    = track.get("dimensions", [])
    rots    = track.get("rotation_matrices", [])
    if i >= len(centers) or i >= len(dims) or i >= len(rots): continue
    bboxes.append({
        "track_id": tid,
        "center": np.asarray(centers[i], dtype=np.float64),
        "dimensions": np.asarray(dims[i], dtype=np.float64),
        "rotation": np.asarray(rots[i], dtype=np.float64),
        "class_name": cls,
    })

if not bboxes:
    print(f"no bboxes for frame_idx={frame_idx} — abort")
    sys.exit(1)

n = render_paper_frame(img, bboxes, ext, K, all_track_ids)
print(f"render_paper_frame drew {n} of {len(bboxes)} bboxes (rest behind cam / out of bounds)")

cv2.putText(img, f"K: fx={K[0,0]:.0f} fy={K[1,1]:.0f} cx={K[0,2]:.0f} cy={K[1,2]:.0f}  "
                 f"frame {img_w}x{img_h}",
            (10, img_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
cv2.putText(img, f"K: fx={K[0,0]:.0f} fy={K[1,1]:.0f} cx={K[0,2]:.0f} cy={K[1,2]:.0f}  "
                 f"frame {img_w}x{img_h}",
            (10, img_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

OUT.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT), img)
print(f"\nWrote {OUT}")
