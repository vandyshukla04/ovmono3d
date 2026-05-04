#!/usr/bin/env python
"""Rescale per-segment papersubdata to match the native 1920×1080 frames
AND convert the per-frame kitti_labels from a hybrid annotator/KITTI format
into proper standard KITTI.

Two corrections per segment:

(A) Coordinate-space rescale (intrinsics + 2D bboxes)
    The papersubdata's `cameras.json` intrinsics and `kitti_labels` 2D bboxes
    are at the VGGT internal resolution (518×294, the model-input size); the
    released frames are 1920×1080. Without this, any user who loads a frame
    and applies the released K gets projections off by ~3.7×.

(B) KITTI 3D location convention
    Standard KITTI stores cols 12 (y) as the BOTTOM-CENTER of the cuboid.
    The papersubdata version stores it as the CENTROID (annotator's
    convention). Conversion: y_bottom_center = y_centroid + h/2 (Y points
    down in camera frame). This makes the per-frame labels usable with
    standard KITTI tooling. The full-rotation, centroid-form labels are
    independently available in WildBox_train.json (Omni3D format) and
    tracking_summary.json (annotator format) — both unaffected.

Operations per segment:
  1. Walks every segment under --root.
  2. cameras.json: for each per-frame entry, scales K from
     (image_width, image_height) → (target_w, target_h):
        K[0,:] *= target_w / image_width
        K[1,:] *= target_h / image_height
     Updates image_width/image_height to the target.
  3. kitti_labels/frame_*.txt: for each line,
     - scales 2D bbox columns 4..7 (XYXY) by the same per-frame factors
     - shifts column 12 (y) by +h/2 to convert centroid → bottom-center
     - cols 8..10 (h, w, l), col 14 (ry), col 15 (score) untouched
  4. Backs up the originals to `cameras.json.orig518` and
     `kitti_labels.orig518/` (only if --in-place; otherwise writes to
     --out-suffix sibling files).

Usage:
    python tools/fix_papersubdata.py --root /mnt/d/3DBOX/papersubdata --in-place
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
from typing import Tuple


def rescale_one_segment(seg_dir: Path, in_place: bool, target_wh: Tuple[int, int],
                        out_suffix: str = ".rescaled") -> dict:
    """Returns a stats dict: {ok|skipped, n_cameras, n_kitti_files}."""
    cameras_path = seg_dir / "cameras.json"
    kitti_dir    = seg_dir / "kitti_labels"
    if not cameras_path.exists() or not kitti_dir.is_dir():
        return {"status": "skipped", "reason": "missing cameras.json or kitti_labels/"}

    target_w, target_h = target_wh
    cameras = json.loads(cameras_path.read_text())
    if "cameras" not in cameras:
        return {"status": "skipped", "reason": "no 'cameras' key"}

    # Build per-frame scale factors from the existing cameras.json's
    # image_width/image_height. We allow these to vary per frame in case
    # VGGT recorded different resolutions per frame (it usually doesn't).
    scales = {}  # frame_index → (sx, sy)
    for cam in cameras["cameras"]:
        fi = int(cam["frame_index"])
        cur_w = int(cam.get("image_width",  518))
        cur_h = int(cam.get("image_height", 294))
        scales[fi] = (target_w / cur_w, target_h / cur_h)

    if any(abs(sx - sy) > 0.05 for sx, sy in scales.values()):
        # warn but don't abort — non-uniform scale is legitimate for crop+pad
        # resizes, but rare.
        print(f"  [warn] {seg_dir.name}: non-uniform scale across frames "
              f"(max sx-sy = {max(abs(sx-sy) for sx,sy in scales.values()):.3f})",
              file=sys.stderr)

    # 1. Rescale cameras.json
    new_cameras = json.loads(cameras_path.read_text())  # fresh copy
    for cam in new_cameras["cameras"]:
        fi = int(cam["frame_index"])
        sx, sy = scales[fi]
        K = cam["intrinsic"]
        cam["intrinsic"] = [
            [K[0][0] * sx, 0.0,           K[0][2] * sx],
            [0.0,           K[1][1] * sy, K[1][2] * sy],
            [0.0,           0.0,           1.0],
        ]
        cam["image_width"]  = target_w
        cam["image_height"] = target_h

    # 2. Rescale every kitti_labels/frame_*.txt
    new_kitti_lines = {}  # path → list[str]
    for txt_path in sorted(kitti_dir.glob("frame_*.txt")):
        # Map kitti filename → frame_index by matching the cameras.json
        # image_name field. This is robust whether the segment uses 0-based
        # or 1-based frame numbering.
        target_fi = None
        for cam in cameras["cameras"]:
            if Path(cam.get("image_name", "")).stem == txt_path.stem:
                target_fi = int(cam["frame_index"])
                break
        if target_fi is None:
            # fallback: parse the integer from the filename
            try:
                target_fi = int(txt_path.stem.split("_")[-1]) - 1
            except Exception:
                continue
        sx, sy = scales.get(target_fi, (target_w / 518, target_h / 294))

        rescaled = []
        for line in txt_path.read_text().splitlines():
            cols = line.split()
            if len(cols) < 15:
                rescaled.append(line); continue
            # Columns (0-indexed):
            # 0:class 1:trunc 2:occ 3:alpha 4:x1 5:y1 6:x2 7:y2
            # 8:h 9:w 10:l 11:X 12:Y 13:Z 14:ry [15:score]
            try:
                cols[4] = f"{float(cols[4]) * sx:.4f}"   # x1
                cols[5] = f"{float(cols[5]) * sy:.4f}"   # y1
                cols[6] = f"{float(cols[6]) * sx:.4f}"   # x2
                cols[7] = f"{float(cols[7]) * sy:.4f}"   # y2
                # 3D location: convert centroid → bottom-center for standard
                # KITTI. Y points DOWN in the camera frame, so the bottom of
                # the cuboid (animal's feet) is at centroid_y + h/2.
                h = float(cols[8])
                cols[12] = f"{float(cols[12]) + h / 2:.4f}"   # y centroid → bottom
            except ValueError:
                rescaled.append(line); continue
            rescaled.append(" ".join(cols))
        new_kitti_lines[txt_path] = rescaled

    # 3. Write — either in-place (with backup) or to a new sibling.
    if in_place:
        # backup originals — use shutil.copyfile (content only, no utime
        # copy) since /mnt/d (NTFS via WSL) rejects the chown/utime that
        # shutil.copy2 attempts on POSIX filesystems.
        bak = cameras_path.with_suffix(".json.orig518")
        if not bak.exists():
            shutil.copyfile(cameras_path, bak)
        kitti_bak = kitti_dir.parent / "kitti_labels.orig518"
        if not kitti_bak.exists():
            kitti_bak.mkdir()
            for src in kitti_dir.iterdir():
                if src.is_file():
                    shutil.copyfile(src, kitti_bak / src.name)
        cameras_path.write_text(json.dumps(new_cameras, indent=2))
        for p, lines in new_kitti_lines.items():
            p.write_text("\n".join(lines) + "\n")
    else:
        out_cam = seg_dir / f"cameras{out_suffix}.json"
        out_kit = seg_dir / f"kitti_labels{out_suffix}"
        out_kit.mkdir(exist_ok=True)
        out_cam.write_text(json.dumps(new_cameras, indent=2))
        for p, lines in new_kitti_lines.items():
            (out_kit / p.name).write_text("\n".join(lines) + "\n")

    return {"status": "ok", "n_cameras": len(new_cameras["cameras"]),
            "n_kitti_files": len(new_kitti_lines),
            "first_scale": scales[next(iter(scales))]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="papersubdata root (contains <video>/<seg>/ subdirs)")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite files (backups go to .orig518)")
    ap.add_argument("--out-suffix", default=".rescaled",
                    help="when not --in-place, sibling-file suffix")
    ap.add_argument("--target-w", type=int, default=1920)
    ap.add_argument("--target-h", type=int, default=1080)
    args = ap.parse_args()

    if not args.root.is_dir():
        sys.exit(f"missing root dir: {args.root}")

    # Write the format spec at the root so reviewers see it once on opening
    # the release. Idempotent — overwritten on every run, always reflects
    # the latest convention this script enforces.
    spec = f"""# WildBox per-segment data — format specification

Generated by tools/fix_papersubdata.py on {args.root}.
Last applied: target resolution {args.target_w}×{args.target_h}.

## Files in each segment directory

| File                              | Contents |
|-----------------------------------|----------|
| frame_NNNNNN.jpg                  | RGB frames at 1920×1080 native resolution |
| cameras.json                      | Per-frame intrinsics K + extrinsics [R|t] in 1920×1080 coords |
| kitti_labels/frame_NNNNNN.txt     | STRICT KITTI 3D-detection format (one line per object) |
| tracking_summary.json             | Annotator-native format with FULL 3×3 rotation matrices |
| vggt_metadata.json                | Segment-level VGGT reconstruction metadata |
| cameras.json.orig518              | (backup) original cameras.json calibrated for 518×294 |
| kitti_labels.orig518/             | (backup) original kitti_labels in 518×294 + centroid form |

## kitti_labels/*.txt — strict KITTI 3D detection format

One line per object, space-separated, 16 fields:

    class trunc occ alpha x1 y1 x2 y2 h w l X Y_bot Z ry score

| Field         | Meaning |
|---------------|---------|
| class         | species name (e.g. giraffe, plains_zebra) |
| trunc         | truncation [0,1] |
| occ           | occlusion {{0,1,2,3}} |
| alpha         | observation angle, radians |
| x1 y1 x2 y2   | 2D bbox in image pixels at 1920×1080 |
| h w l         | cuboid extents along camera Y / Z / X axes |
| X Y_bot Z     | BOTTOM-CENTER of cuboid in camera coords (Y axis points down, so bottom = max Y) |
| ry            | yaw rotation around camera Y axis (radians) |
| score         | 1.0 for GT |

## Coordinate conventions

* Image coordinates: 1920 × 1080, origin top-left, X right, Y down.
* Camera frame: X right, Y down, Z forward (OpenCV / Omni3D / KITTI).
* Per-segment scale normalisation: every 3D coordinate is divided by the
  median |z_cam| of valid GT cuboids in that segment, so median |z_cam| ≈ 1
  per segment. Per-segment scale factors are listed in
  WildBox_{{train,val}}.json `info.scene_scales`.

## kitti_labels vs. tracking_summary.json (full data)

The strict-KITTI export is a LOSSY conversion of the annotator's native
data. The annotator format is preserved alongside in tracking_summary.json
and (consolidated) WildBox_train.json / WildBox_val.json:

| Property         | kitti_labels                                | tracking_summary.json / Omni3D JSON       |
|------------------|---------------------------------------------|-------------------------------------------|
| Location         | BOTTOM-center of cuboid                     | CENTROID of cuboid                        |
| Dimensions order | (h, w, l)  — KITTI standard                 | (l, w, h) — annotator native              |
| Rotation         | yaw `ry` only (lossy)                       | full 3×3 matrix (lossless)                |

To convert annotator centroid → KITTI bottom-center:

    y_bot = y_centroid + h / 2     (Y axis points down)

To convert annotator full R → KITTI yaw (lossy):

    ry = arctan2(R[0,2], R[0,0])

For drone-oblique data the dropped roll and pitch from R are non-trivial
(roll std ≈ 56°, pitch std ≈ 41° across all annotations). The strict-KITTI
labels are useful for KITTI-tool compatibility, but for full-precision 3D
inference use the JSON / tracking_summary form.

## Data history

This release was rescaled from the original VGGT internal resolution
(518×294) to native frame resolution ({args.target_w}×{args.target_h}) by
tools/fix_papersubdata.py. Per-frame scale factors (~3.71×, ~3.67×) varied
slightly per segment and were applied uniformly across cameras.json
intrinsics and kitti_labels 2D bbox columns.

The centroid → bottom-center conversion of kitti_labels Y was applied at
the same step. Full-precision (centroid + 3×3 R) annotations remain
unchanged in tracking_summary.json and the consolidated Omni3D JSONs.

## Verification

To verify the data renders correctly:

    python tools/visualize_segment.py <segment_dir> <frame_no> <out.jpg>

The 3D wireframes (from KITTI bottom-center labels OR from
tracking_summary centroid labels) should both project onto the same
animals at the same image positions if the rescale + bottom-shift is
correct.
"""
    spec_path = args.root / "FORMAT_SPEC.md"
    spec_path.write_text(spec)
    print(f"Wrote format spec → {spec_path}\n")

    # find every segment dir (looks like .../seg<N>)
    seg_dirs = sorted(p for p in args.root.rglob("seg*")
                      if p.is_dir() and (p / "cameras.json").exists())
    print(f"Found {len(seg_dirs)} segment dirs under {args.root}")

    n_ok, n_skip = 0, 0
    for sd in seg_dirs:
        stats = rescale_one_segment(
            sd, in_place=args.in_place,
            target_wh=(args.target_w, args.target_h),
            out_suffix=args.out_suffix,
        )
        if stats["status"] == "ok":
            print(f"  ✓ {sd.relative_to(args.root)}  "
                  f"({stats['n_cameras']} cameras, "
                  f"{stats['n_kitti_files']} kitti files, "
                  f"scale x={stats['first_scale'][0]:.3f}, y={stats['first_scale'][1]:.3f})")
            n_ok += 1
        else:
            print(f"  - {sd.relative_to(args.root)}  SKIP ({stats['reason']})")
            n_skip += 1

    print(f"\nDone: {n_ok} segments rescaled, {n_skip} skipped.")
    if not args.in_place:
        print(f"Look for cameras{args.out_suffix}.json and "
              f"kitti_labels{args.out_suffix}/ inside each segment dir.")


if __name__ == "__main__":
    main()
