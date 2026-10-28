"""Release QA: render EVERY 3D box from BOTH annotation channels onto frames, per-segment videos.

    python tools/release/render_boxes_video.py \
        --train /mnt/d/3DBOX/papersubdata/WildBox_train_paper.json \
        --val   /mnt/d/3DBOX/papersubdata/WildBox_val_paper.json \
        --image-root /mnt/d/3DBOX/papersubdata \
        --out /mnt/d/wildbox_qa [--workers 6] [--only <substr>] [--limit N] [--fps 10]

Each frame is read ONCE and drawn into two writers:
  <out>/omni/<split>__<group>__<video>__<seg>.mp4   GREEN  = Omni3D json cuboids
                                                     (center_cam/dimensions[W,H,L]/R_cam, json K)
  <out>/kitti/<split>__<group>__<video>__<seg>.mp4  CYAN   = KITTI-label cuboids
                                                     (release kitti_labels/: location = BOTTOM-face
                                                     center, dims h w l, rotation_y; thin ORANGE =
                                                     the kitti 2D box)
Projection uses the per-image json K for both channels: the KITTI 3D content lives in the
unnormalized gauge, and perspective projection is uniform-scale-invariant, so the overlay is exact.
A frame counter + box count is stamped top-left. No intermediate stills are written.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tadetr.data.wildbox_paths import parse_label_path  # noqa: E402

EDGES_OMNI = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]
EDGES_KITTI = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
               (0, 4), (1, 5), (2, 6), (3, 7)]

G = {}   # per-worker globals


def omni_corners(center, dims, R):
    w, h, l = dims
    lx, ly, lz = l / 2, h / 2, w / 2
    local = np.array([[sx, sy, sz] for sx in (-lx, lx) for sy in (-ly, ly)
                      for sz in (-lz, lz)])
    return np.asarray(center) + local @ np.asarray(R).T


def kitti_corners(loc, h, w, l, ry):
    x = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    c, s = np.cos(ry), np.sin(ry)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return (R @ np.stack([x, y, z])).T + np.asarray(loc)


def draw_cuboid(img, c8, K, color, edges, thick=2):
    import cv2
    P = np.asarray(K, float)
    if P.shape == (3, 4):                       # KITTI P2: project homogeneous rect coords
        uv = np.hstack([c8, np.ones((len(c8), 1))]) @ P.T
    else:
        uv = c8 @ P.T
    z = uv[:, 2]
    if (z <= 1e-6).any():
        return 0
    uv = (uv[:, :2] / z[:, None]).astype(int)
    for a, b in edges:
        cv2.line(img, tuple(uv[a]), tuple(uv[b]), color, thick, cv2.LINE_AA)
    return 1


def parse_kitti(path: Path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text().strip().splitlines():
        f = line.split()
        if len(f) < 15:
            continue
        out.append({"cls": f[0], "bbox": [float(x) for x in f[4:8]],
                    "hwl": [float(x) for x in f[8:11]],
                    "loc": [float(x) for x in f[11:14]], "ry": float(f[14])})
    return out


def render_segment(job):
    import cv2
    (split, group, video, seg), frames = job
    args = G["args"]
    name = f"{split}__{group}__{video}__{seg}.mp4"
    outs = {}
    n_drawn = {"omni": 0, "kitti": 0}
    try:
        for fname, im, anns in frames:
            img_path = Path(args.image_root) / im["file_path"].replace("\\", "/")
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            K = np.array(im["K"], float).reshape(3, 3)
            if not outs:
                H, W = frame.shape[:2]
                for ch in ("omni", "kitti"):
                    p = Path(args.out) / ch / name
                    p.parent.mkdir(parents=True, exist_ok=True)
                    outs[ch] = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"),
                                               args.fps, (W, H))
            f_omni = frame.copy()
            for a in anns:
                n_drawn["omni"] += draw_cuboid(
                    f_omni, omni_corners(a["center_cam"], a["dimensions"], a["R_cam"]),
                    K, (40, 220, 40), EDGES_OMNI)
            kitti_root = Path(args.kitti_root) if args.kitti_root else Path(args.image_root)
            kitti_path = (kitti_root / group / video / seg / "kitti_labels"
                          / fname.replace(".jpg", ".txt"))
            K_kitti = K
            if args.kitti_calib:
                cpath = kitti_root / group / video / seg / "calib" / fname.replace(".jpg", ".txt")
                if cpath.exists():
                    for line in cpath.read_text().splitlines():
                        if line.startswith("P2:"):
                            K_kitti = np.array([float(v) for v in line.split()[1:]]).reshape(3, 4)
            f_kitti = frame
            for k in parse_kitti(kitti_path):
                x1, y1, x2, y2 = [int(v) for v in k["bbox"]]
                cv2.rectangle(f_kitti, (x1, y1), (x2, y2), (0, 165, 255), 1)
                h, w, l = k["hwl"]
                n_drawn["kitti"] += draw_cuboid(
                    f_kitti, kitti_corners(k["loc"], h, w, l, k["ry"]),
                    K_kitti, (255, 220, 40), EDGES_KITTI)
            for ch, f in (("omni", f_omni), ("kitti", f_kitti)):
                cv2.putText(f, f"{ch.upper()} {group}/{video}/{seg} {fname}",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                outs[ch].write(f)
        for o in outs.values():
            o.release()
        return {"name": name, "frames": len(frames), **n_drawn, "status": "ok"}
    except Exception as e:
        for o in outs.values():
            o.release()
        return {"name": name, "status": f"FAILED: {e!r}"}


def init_worker(args):
    G["args"] = args


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--kitti-calib", action="store_true",
                    help="project kitti boxes with the per-frame calib P2 (rectified-frame labels)")
    ap.add_argument("--kitti-root", type=str, default="",
                    help="read kitti_labels from <root>/<group>/<video>/<seg>/kitti_labels "
                         "instead of the image root (v1.1 QA: the regenerated channel)")
    args = ap.parse_args()

    segs = defaultdict(list)
    for split, path in (("train", args.train), ("val", args.val)):
        d = json.loads(path.read_text())
        anns_by_img = defaultdict(list)
        for a in d["annotations"]:
            if not a.get("behind_camera"):
                anns_by_img[a["image_id"]].append(a)
        for im in d["images"]:
            group, video, seg, fname = parse_label_path(im["file_path"])
            segs[(split, group, video, seg)].append((fname, im, anns_by_img.get(im["id"], [])))
        print(f"{split}: loaded")
    jobs = []
    for key in sorted(segs):
        if args.only and args.only not in "__".join(key):
            continue
        jobs.append((key, sorted(segs[key], key=lambda t: t[0])))
    if args.limit:
        jobs = jobs[:args.limit]
    total_frames = sum(len(f) for _, f in jobs)
    print(f"{len(jobs)} segment videos x2 channels, {total_frames:,} frames -> {args.out}")

    done = failed = 0
    with Pool(args.workers, initializer=init_worker, initargs=(args,)) as pool:
        for r in pool.imap_unordered(render_segment, jobs):
            done += 1
            if "FAILED" in str(r["status"]):
                failed += 1
                print(f"[{done}/{len(jobs)}] {r['name']}: {r['status']}", flush=True)
            elif done % 10 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] latest {r['name']}: {r['frames']} frames, "
                      f"omni {r['omni']} kitti {r['kitti']} boxes", flush=True)
    print(f"done: {done - failed} ok, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
