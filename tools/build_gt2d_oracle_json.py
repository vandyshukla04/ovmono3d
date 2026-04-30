#!/usr/bin/env python
"""Convert an Omni3D-format WildBox GT JSON into an oracle-2D-shaped JSON
that the OVMono3D zero-shot inference path consumes via TEST.ORACLE2D=True.

This produces the GT-2D ceiling condition: 2D boxes + species labels are
provided perfectly from ground-truth annotations, so any remaining error is
attributable purely to the 3D head (cube regression: depth, dimensions, pose).

The output schema mirrors the existing gdino_WildBox_val_oracle_2d.json
format expected by cubercnn:
    [
      { "image_id": int,
        "instances": [
          { "bbox": [x, y, w, h],
            "category_id": int,    # WildBox dataset_id (1000-1005)
            "score": 1.0 }, ...
        ] }, ...
    ]

Usage:
    python tools/build_gt2d_oracle_json.py \\
        --gt  datasets/Omni3D/WildBox_val.json \\
        --out datasets/Omni3D/gt2d_WildBox_val_oracle_2d.json
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True,
                    help="Omni3D-format GT JSON (e.g. WildBox_val.json).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output oracle-2D-shaped JSON path.")
    ap.add_argument("--score", type=float, default=1.0,
                    help="Score to assign every box (default 1.0 — the box "
                         "came from GT, so confidence is by definition 1).")
    args = ap.parse_args()

    g = json.loads(args.gt.read_text())
    by_img: dict[int, list] = defaultdict(list)
    n_skipped = 0
    for ann in g["annotations"]:
        # Drop annotations the standard evaluator would reject — keeps the
        # oracle JSON consistent with what the model would have seen if it
        # had perfect detection (no behind-camera, no invalid 3D).
        if not ann.get("valid3D", True) or ann.get("behind_camera", False):
            n_skipped += 1
            continue
        by_img[int(ann["image_id"])].append({
            "bbox": list(ann["bbox"]),  # XYWH (Omni3D convention)
            "category_id": int(ann["category_id"]),
            "score": float(args.score),
        })

    out = []
    for im in g["images"]:
        out.append({
            "image_id": int(im["id"]),
            "instances": by_img.get(int(im["id"]), []),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))

    n_imgs_with_boxes = sum(1 for entry in out if entry["instances"])
    n_total_boxes = sum(len(entry["instances"]) for entry in out)
    print(f"wrote {args.out}")
    print(f"  total images:           {len(out)}")
    print(f"  images with ≥1 box:     {n_imgs_with_boxes}")
    print(f"  total boxes:            {n_total_boxes}")
    print(f"  skipped (invalid3D / behind-camera): {n_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
