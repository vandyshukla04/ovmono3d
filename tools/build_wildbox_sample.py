#!/usr/bin/env python
"""Stage a ~1.7 GB reviewer data-only sample of the WildBox release.

Picks 5 species/videos hand-chosen for breadth + compactness and copies
the corresponding zips into ``sample/<group>/<video>.zip``. No JSONs and
no checkpoints are included — the sample is purely for letting reviewers
inspect the data format without downloading the full ~50 GB release.

Run from anywhere (writes to ``--out`` under the source tree).

Usage:
    python tools/build_wildbox_sample.py \\
        --root /mnt/d/3DBOX/papersubdata \\
        --out  /mnt/d/3DBOX/papersubdata/sample
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# (group, video) tuples — one zip per species, ~1.7 GB total.
SAMPLE_VIDEOS = [
    ("gira1", "DJI_20240118140338_0007_V"),       # giraffe — first vid (richer scenes)
    ("elep3", "DJI_20260227084724_0003_V"),       # elephant — specific pick
    ("zebr3", "DJI_20250802084740_0006_V"),       # plains_zebra — smallest in zebr3
    ("rhin1", "DJI_20250304162356_0004_D"),       # rhino — smallest in rhin1
    ("gaze1", "DJI_20240624152216_0004_V"),       # gazelle — smallest in gaze1
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="Root of the full release (contains the 11 group folders).")
    ap.add_argument("--out",  type=Path, required=True,
                    help="Destination sample/ folder (created if missing).")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("=== copying sample zips ===")
    total_bytes = 0
    for group, video in SAMPLE_VIDEOS:
        src = args.root / group / f"{video}.zip"
        dst = args.out / group / f"{video}.zip"
        if not src.exists():
            raise SystemExit(f"missing source zip: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            print(f"  ↻ already staged: {dst.relative_to(args.out)} ({src.stat().st_size/1e6:.1f} MB)")
        else:
            shutil.copyfile(src, dst)   # NTFS-safe: no metadata copy
            print(f"  ✓ staged: {dst.relative_to(args.out)} ({src.stat().st_size/1e6:.1f} MB)")
        total_bytes += src.stat().st_size

    print(f"  total: {total_bytes/1e9:.2f} GB")
    print(f"\noutput: {args.out}")


if __name__ == "__main__":
    main()
