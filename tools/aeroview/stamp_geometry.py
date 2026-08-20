"""Run-2 stamp: per-image telemetry geometry + human-confirmed junk-label masking.  [CPU, ~2 min]

    python tools/aeroview/stamp_geometry.py \
        --train datasets/Omni3D/WildBox_train.json \
        --val   datasets/Omni3D/WildBox_val.json

Writes IN PLACE (like build_heading_labels.py). Idempotent. Two things:

1. **geo per image**: `[fx_tel_px, pitch_rad, pitch_valid]` from the drone's own SRT telemetry, joined on
   the video frame number (frame_XXXXXX.jpg <-> FrameCnt). fx_tel = focal_len(35mm-equiv)/36mm x width_px
   (focal_len already contains the digital zoom). Phase-A measured facts this rests on: the gimbal normal
   matches the GT normal to p50 0.73 deg; the json K is per-segment fiction (ratio to fx_tel scatters
   0.10-0.71). The model reads `geo` through datasets.py (img_keys_optional) -> rcnn3d -> roi_heads, where
   it becomes the 10-dim per-RoI token. Missing focal falls back to the video median, then to json fx
   (fx_tel=0 => roi_heads falls back itself); missing pitch => pitch_valid=0 and the token's pitch/plane
   entries are zeroed.

2. **junk masking (TRAIN ONLY)**: annotations sitting >1.0x their own height off their segment's ground
   plane get `valid3D=false`, which datasets.is_ignore() already turns into an ignore region -- no code
   path change. The threshold is the HUMAN-AUDITED population: 60 flagged boxes were reviewed and 95% are
   fragments/wrong, 0% lying animals (audit_results.json, 2026-08-20). Val is NEVER touched: the benchmark
   stays comparable; label-sane filtering on val is a reporting choice, not a data edit.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MM_FULL_FRAME_WIDTH = 36.0
JUNK_OFF_H = 1.0          # the audited threshold


def load_telemetry(bundle: Path) -> dict:
    z = np.load(bundle, allow_pickle=True)
    vids = z["_videos"].tolist()
    return {v: {"focal": z[f"{v}::focal"], "pitch": z[f"{v}::pitch"]} for v in vids
            if f"{v}::focal" in z.files}


def stamp(path: Path, tel: dict, *, mask_junk: bool) -> None:
    d = json.loads(path.read_text())
    ims = {i["id"]: i for i in d["images"]}

    # ---- 1. geo per image -----------------------------------------------------------------------
    n_fx = n_pitch = 0
    med_focal = {v: float(np.nanmedian(t["focal"])) for v, t in tel.items()}
    for im in d["images"]:
        parts = im["file_path"].replace("\\", "/").split("/")
        v, f = parts[-3], int(parts[-1].split("_")[-1].split(".")[0])
        fx_tel, pitch, valid = 0.0, 0.0, 0.0
        t = tel.get(v)
        if t is not None:
            focal = t["focal"][f] if f < len(t["focal"]) else np.nan
            if np.isnan(focal):
                focal = med_focal.get(v, np.nan)
            if np.isfinite(focal) and focal > 0:
                fx_tel = float(focal) / MM_FULL_FRAME_WIDTH * im["width"]
                n_fx += 1
            p = t["pitch"][f] if f < len(t["pitch"]) else np.nan
            if np.isfinite(p):
                pitch, valid = float(np.radians(p)), 1.0
                n_pitch += 1
        im["geo"] = [round(fx_tel, 2), round(pitch, 5), valid]

    # ---- 2. junk masking (train only) -----------------------------------------------------------
    n_masked = 0
    per_class = defaultdict(int)
    if mask_junk:
        by_seg = defaultdict(list)
        for a in d["annotations"]:
            p = ims[a["image_id"]]["file_path"].replace("\\", "/").split("/")
            by_seg[(p[-3], p[-2])].append(a)
        for seg, lst in by_seg.items():
            if len(lst) < 30:
                continue
            up = -np.array(lst[0]["R_cam"], float)[:, 1]
            C = np.array([a["center_cam"] for a in lst])
            H = np.array([a["dimensions"][1] for a in lst])
            t = (C - (H[:, None] / 2) * up) @ up
            off = (t - np.median(t)) / np.maximum(H, 1e-6)
            for a, o in zip(lst, off):
                if abs(float(o)) > JUNK_OFF_H and a.get("valid3D", True):
                    a["valid3D"] = False
                    n_masked += 1
                    per_class[a["category_name"]] += 1

    path.write_text(json.dumps(d))
    tag = "train" if mask_junk else "val"
    print(f"[{tag}] {path.name}: geo stamped on {len(d['images']):,} images "
          f"(fx {n_fx:,}, pitch {n_pitch:,})", end="")
    if mask_junk:
        print(f"  | junk masked (valid3D=false): {n_masked:,} "
              f"({', '.join(f'{k} {v:,}' for k, v in sorted(per_class.items()))})")
    else:
        print("  | annotations untouched (benchmark stays comparable)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--telemetry", type=Path,
                    default=Path(__file__).resolve().parent / "data" / "telemetry_min.npz")
    ap.add_argument("--no-mask", action="store_true", help="stamp geo only; skip junk masking")
    args = ap.parse_args()

    tel = load_telemetry(args.telemetry)
    print(f"telemetry: {len(tel)} videos from {args.telemetry.name}")
    stamp(args.train, tel, mask_junk=not args.no_mask)
    stamp(args.val, tel, mask_junk=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
