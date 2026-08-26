"""Stamp geo_alt = [rel_alt_m, roll_deg, valid] per image, in place, idempotent.  [CPU, ~1 min]

    python tools/tadetr/stamp_telemetry_alt.py \
        --train /mnt/d/aeroview/labelled/WildBox_train_paper.json \
        --val   /mnt/d/aeroview/labelled/WildBox_val_paper.json

Join: video frame number (frame_XXXXXX.jpg <-> SRT FrameCnt), the stamp_geometry.py convention.
Complements the existing per-image `geo` = [fx_tel, pitch_rad, pitch_valid]; together they feed the
TA-DETR telemetry token [sin pitch, cos pitch, roll, log alt]. Missing telemetry -> valid = 0
(the model substitutes its learned no_telemetry embedding). Runs anywhere once
tools/tadetr/data/telemetry_alt_min.npz is committed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_bundle(bundle: Path) -> dict:
    z = np.load(bundle, allow_pickle=True)
    vids = z["_videos"].tolist()
    return {v: {"rel_alt": z[f"{v}::rel_alt"] if f"{v}::rel_alt" in z.files else None,
                "roll": z[f"{v}::roll"] if f"{v}::roll" in z.files else None} for v in vids}


def stamp(path: Path, tel: dict) -> None:
    d = json.loads(path.read_text())
    n_ok = 0
    for im in d["images"]:
        parts = im["file_path"].replace("\\", "/").split("/")
        v, f = parts[-3], int(parts[-1].split("_")[-1].split(".")[0])
        alt, roll, valid = 0.0, 0.0, 0.0
        t = tel.get(v)
        if t is not None and t["rel_alt"] is not None:
            arr = t["rel_alt"]
            a = arr[f] if f < len(arr) else np.nan
            if not np.isfinite(a):
                a = float(np.nanmedian(arr)) if np.isfinite(arr).any() else np.nan
            if np.isfinite(a) and a > 0:
                alt, valid = float(a), 1.0
                n_ok += 1
            r = t["roll"]
            if r is not None and f < len(r) and np.isfinite(r[f]):
                roll = float(r[f])
        im["geo_alt"] = [round(alt, 2), round(roll, 3), valid]
    path.write_text(json.dumps(d))
    print(f"{path.name}: geo_alt stamped on {len(d['images']):,} images ({n_ok:,} with altitude)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--bundle", type=Path,
                    default=Path(__file__).resolve().parent / "data" / "telemetry_alt_min.npz")
    args = ap.parse_args()
    tel = load_bundle(args.bundle)
    print(f"bundle: {len(tel)} videos")
    stamp(args.train, tel)
    stamp(args.val, tel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
