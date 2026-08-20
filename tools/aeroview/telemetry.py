"""Parse DJI SRT telemetry for WildBox videos and cache it as one npz.  [CPU]

    python tools/aeroview/telemetry.py \
        --gt /mnt/d/aeroview/labelled/WildBox_val_paper.json /mnt/d/aeroview/labelled/WildBox_train_paper.json \
        --srt-root /mnt/d/3DBOX --out /mnt/d/aeroview/telemetry.npz

WHY THIS EXISTS (Phase A, gate S0)
----------------------------------
Every GroundCast geometric quantity is derived from three telemetry fields:
    focal_len   -> the true fx (two optical lenses x a known digital-zoom factor)
    gimbal_pitch-> gravity in camera coordinates (the ground normal)
    rel_alt     -> only ever used for diagnostics; h cancels in the model itself
The join key is the video frame number: WildBox file_path carries `frame_XXXXXX.jpg` where XXXXXX is the
VIDEO frame number, and SRT blocks carry `FrameCnt: N` (1-based). So SRT block N describes video frame N,
and `frame_000123.jpg` joins to FrameCnt 123.  (This is the same video-frame-number fact that once broke
the label join -- crops.npz 'frame' is segment-local, the filename is not.)

DJI SRT layouts vary across firmware. This parser is defensive: it reads whichever of the known fields are
present per block and records per-video field coverage, so a video with a sparse layout is visible rather
than silently zero-filled.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

# Two DJI SRT dialects: modern `FrameCnt: N` + `[field: v]`, and the 2023-era (KABR) `SrtCnt : N` +
# `[field : v]` with `altitude:` in place of `rel_alt:`. One pattern set speaks both.
FIELDS = {
    "focal":   re.compile(r"\[focal_len\s*:\s*([\d.]+)\]"),
    "dzoom":   re.compile(r"dzoom_ratio\s*:\s*([\d.]+)", re.I),
    "rel_alt": re.compile(r"(?:rel_alt|\[altitude)\s*:\s*([-\d.]+)"),
    "pitch":   re.compile(r"gimbal_pitch\(degrees\)\s*:\s*([-\d.]+)"),
    "yaw":     re.compile(r"gimbal_heading\(degrees\)\s*:\s*([-\d.]+)"),
    "roll":    re.compile(r"gimbal_roll\(degrees\)\s*:\s*([-\d.]+)"),
    "lat":     re.compile(r"\[latitude\s*:\s*([-\d.]+)\]"),
    "lon":     re.compile(r"\[longitude\s*:\s*([-\d.]+)\]"),
}
FRAME = re.compile(r"(?:FrameCnt|SrtCnt)\s*:\s*(\d+)")


def wildbox_videos(gt_paths) -> set:
    vids = set()
    for p in gt_paths:
        d = json.loads(Path(p).read_text())
        for im in d["images"]:
            vids.add(im["file_path"].replace("\\", "/").split("/")[-3])
    return vids


def parse_srt(path: Path) -> dict:
    """One array per field, indexed by video frame number (FrameCnt)."""
    n_max = 0
    rows = {}
    block_frame = None
    buf = []

    def flush():
        nonlocal block_frame, buf
        if block_frame is None:
            return
        text = "\n".join(buf)
        rec = {}
        for k, pat in FIELDS.items():
            m = pat.search(text)
            if m:
                rec[k] = float(m.group(1))
        rows[block_frame] = rec
        block_frame, buf = None, []

    with open(path, errors="ignore") as fh:
        for line in fh:
            m = FRAME.search(line)
            if m:
                flush()
                block_frame = int(m.group(1))
                n_max = max(n_max, block_frame)
                buf = [line]
            elif block_frame is not None:
                buf.append(line)
    flush()
    if not rows:
        return {}

    out = {k: np.full(n_max + 1, np.nan) for k in FIELDS}
    for f, rec in rows.items():
        for k, v in rec.items():
            out[k][f] = v
    out["_n"] = n_max
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", nargs="+", required=True)
    ap.add_argument("--srt-root", type=Path, default=Path("/mnt/d/3DBOX"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    vids = wildbox_videos(args.gt)
    print(f"WildBox videos: {len(vids)}")
    srts = {}
    for f in glob.glob(str(args.srt_root / "**" / "*.SRT"), recursive=True):
        name = Path(f).stem
        if name in vids and name not in srts:
            srts[name] = f
    print(f"SRT found for {len(srts)}/{len(vids)} videos")
    missing = sorted(vids - set(srts))
    if missing:
        print("  missing:", ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))

    store = {}
    cov = []
    for v, f in sorted(srts.items()):
        t = parse_srt(Path(f))
        if not t:
            print(f"  {v}: PARSE FAILED")
            continue
        for k in FIELDS:
            store[f"{v}::{k}"] = t[k].astype(np.float32)
        have = {k: float(np.mean(~np.isnan(t[k][1:]))) for k in FIELDS}
        cov.append((v, t["_n"], have))

    print(f"\nper-video field coverage (fraction of frames carrying the field):")
    print(f"{'video':28s} {'frames':>7s}  " + "  ".join(f"{k:>7s}" for k in FIELDS))
    for v, n, have in cov:
        print(f"{v:28s} {n:7d}  " + "  ".join(f"{have[k]:7.2f}" for k in FIELDS))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **store,
                        _videos=np.array(sorted(srts), dtype=object))
    print(f"\ncached -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
