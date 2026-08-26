"""Extract rel_alt + roll from the full local telemetry cache into a small committable bundle.
[CPU, seconds; run LOCALLY -- /mnt/d/aeroview/telemetry.npz does not exist on the cluster]

    python tools/tadetr/build_telemetry_alt_bundle.py \
        --telemetry /mnt/d/aeroview/telemetry.npz \
        --out tools/tadetr/data/telemetry_alt_min.npz

Bundle layout mirrors tools/aeroview/data/telemetry_min.npz: `_videos` + per-video
`<video>::rel_alt` (metres, per video frame number) and `<video>::roll` (degrees).
The TA-DETR telemetry token wants [sin pitch, cos pitch, roll, log altitude]; pitch/focal come from
the existing telemetry_min.npz; this bundle adds the two missing fields.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--telemetry", type=Path, default=Path("/mnt/d/aeroview/telemetry.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "data" / "telemetry_alt_min.npz")
    args = ap.parse_args()

    z = np.load(args.telemetry, allow_pickle=True)
    vids = z["_videos"].tolist()
    out = {"_videos": np.array(vids, dtype=object)}
    n_alt = n_roll = 0
    for v in vids:
        for field, cnt in (("rel_alt", "alt"), ("roll", "roll")):
            key = f"{v}::{field}"
            if key in z.files:
                out[key] = z[key].astype(np.float32)
                if np.isfinite(z[key]).any():
                    if field == "rel_alt":
                        n_alt += 1
                    else:
                        n_roll += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"{args.out}: {len(vids)} videos (rel_alt usable {n_alt}, roll usable {n_roll}), "
          f"{args.out.stat().st_size/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
