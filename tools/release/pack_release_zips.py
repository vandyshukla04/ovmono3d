"""Pack the WildBox v1.1 release zips.  [CPU+disk, ~1-2h for ~36 GB]

    python tools/release/pack_release_zips.py \
        --train /mnt/d/wildbox_v1.1/WildBox_train_v1.1.json \
        --val   /mnt/d/wildbox_v1.1/WildBox_val_v1.1.json \
        --kitti-root /mnt/d/wildbox_v1.1/kitti_labels \
        --out-root /mnt/d/wildbox_v1.1/release [--workers 4] [--only <substr>]

One zip per video at <out-root>/<alias>/<video>.zip, inner layout identical to v1.0
(`<video>/<seg>/...`, no group prefix). Per segment (only segments present in the v1.1 jsons —
this INCLUDES rhin1 seg4, closing the v1.0 packaging gap):
  frame_*.jpg          verbatim from the dense source tree (only frames referenced by the jsons)
  kitti_labels/*.txt   the NEW v1.1 channel (built by build_kitti_v1_1.py)
  cameras.json         full-resolution K conversion of the dense-tree cameras (per-axis sx, sy —
                       the v1.0 release convention), extrinsics verbatim
  tracking_summary.json  the CORRECTED generation, verbatim (v1.0 shipped raw)
Writes <out-root>/zips_MANIFEST.jsonl (per zip: file count, bytes, sha256).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tadetr.data.wildbox_paths import index_by_video_seg  # noqa: E402
from tadetr.data.wildbox_paths import parse_label_path  # noqa: E402

G = {}


def fullres_cameras(sp, sample_frame: Path) -> dict:
    import struct

    def jpeg_dims(path):
        with open(path, "rb") as f:
            assert f.read(2) == b"\xff\xd8"
            while True:
                marker = f.read(2)
                m = marker[1]
                if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                         0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    struct.unpack(">H", f.read(2))
                    f.read(1)
                    h, w = struct.unpack(">HH", f.read(4))
                    return h, w
                length = struct.unpack(">H", f.read(2))[0]
                f.seek(length - 2, 1)

    cams = json.loads(sp.cameras_json.read_text())
    real_h, real_w = jpeg_dims(sample_frame)
    vw = cams["cameras"][0]["image_width"]
    vh = cams["cameras"][0]["image_height"]
    sx, sy = real_w / vw, real_h / vh
    for c in cams["cameras"]:
        K = c["intrinsic"]
        c["intrinsic"] = [[K[0][0] * sx, 0.0, K[0][2] * sx],
                          [0.0, K[1][1] * sy, K[1][2] * sy],
                          [0.0, 0.0, 1.0]]
        c["image_width"], c["image_height"] = real_w, real_h
    return cams


def pack_video(job):
    (alias, video), segs = job
    args = G["args"]
    index = G["index"]
    out = Path(args.out_root) / alias / f"{video}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    n_files = 0
    try:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for seg, frames in sorted(segs.items()):
                sp = index.get((video, seg))
                if sp is None:
                    return {"zip": str(out), "status": f"FAILED: no dense source {video}/{seg}"}
                frame_paths = sorted(sp.seg_dir / f for f in frames)
                # frames (stored, jpgs don't compress)
                for fp in frame_paths:
                    z.write(fp, f"{video}/{seg}/{fp.name}",
                            compress_type=zipfile.ZIP_STORED)
                    n_files += 1
                # new kitti channel (rectified labels + per-frame calib)
                for sub in ("kitti_labels", "calib"):
                    kdir = Path(args.kitti_root) / alias / video / seg / sub
                    for kt in sorted(kdir.glob("*.txt")):
                        z.write(kt, f"{video}/{seg}/{sub}/{kt.name}")
                        n_files += 1
                # full-res cameras + corrected tracking summary
                cams = fullres_cameras(sp, frame_paths[0])
                z.writestr(f"{video}/{seg}/cameras.json", json.dumps(cams, indent=1))
                z.write(sp.tracking_summary, f"{video}/{seg}/tracking_summary.json")
                n_files += 2
        sha = hashlib.sha256()
        with open(out, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                sha.update(chunk)
        return {"zip": f"{alias}/{video}.zip", "files": n_files,
                "bytes": out.stat().st_size, "sha256": sha.hexdigest(), "status": "ok"}
    except Exception as e:
        return {"zip": str(out), "status": f"FAILED: {e!r}"}


def init_worker(args, index):
    G["args"] = args
    G["index"] = index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--kitti-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()

    by_video = defaultdict(lambda: defaultdict(set))
    for p in (args.train, args.val):
        d = json.loads(p.read_text())
        for im in d["images"]:
            alias, video, seg, fname = parse_label_path(im["file_path"])
            by_video[(alias, video)][seg].add(fname)
    jobs = sorted(by_video.items())
    if args.only:
        jobs = [j for j in jobs if args.only in f"{j[0][0]}/{j[0][1]}"]
    print(f"packing {len(jobs)} video zips -> {args.out_root}")
    index = index_by_video_seg()

    manifest = Path(args.out_root) / "zips_MANIFEST.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with Pool(args.workers, initializer=init_worker, initargs=(args, index)) as pool:
        with open(manifest, "w") as mf:
            for i, r in enumerate(pool.imap_unordered(pack_video, jobs)):
                if r["status"] != "ok":
                    failed += 1
                    print(f"[{i+1}/{len(jobs)}] {r['zip']}: {r['status']}", flush=True)
                else:
                    mf.write(json.dumps(r) + "\n")
                    print(f"[{i+1}/{len(jobs)}] {r['zip']}: {r['files']} files "
                          f"{r['bytes']/1e6:.0f} MB", flush=True)
    print(f"done; {failed} failed; manifest -> {manifest}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
