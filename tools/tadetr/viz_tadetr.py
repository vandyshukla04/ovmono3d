"""Render TA-DETR predictions on real frames — the look-at-it check.  [CPU]

    python tools/tadetr/viz_tadetr.py \
        --val datasets/Omni3D/WildBox_val.json --image-root datasets/wildbox_hf \
        --preds /storage2/3DOM/vshukla/runs/tadetr_a1/preds/own/WildBox_val.pth \
        --out /storage2/3DOM/vshukla/runs/tadetr_a1/viz [--video V] [--n 12] [--crops] \
        [--score-thresh 0.4]

Full frames: GT cuboids GREEN, predicted cuboids MAGENTA, predicted ground-contact (bottom-face
center) as a YELLOW dot, body-axis segment with an arrowhead at the predicted-sign end.
--crops adds a per-animal contact sheet (GT-matched tiles) titled with z_gt / z_pred / error.

Conventions carried from the project's hard lessons (do not "fix"):
  up = -R[:, 1] (the +R[:,1] arrow-inversion bug); dims = [W, H, L] with L on local x (column 0).
⚠ A1 CAVEAT drawn into every sheet: the A1 checkpoint's SIGN head trained on inverted labels
(fixed for A2) -- trust the AXIS of the arrow, not which end has the head.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tadetr.data.wildbox_paths import parse_label_path  # noqa: E402

EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def corners(center, dims, R):
    w, h, l = dims
    lx, ly, lz = l / 2, h / 2, w / 2
    local = np.array([[sx, sy, sz] for sx in (-lx, lx) for sy in (-ly, ly)
                      for sz in (-lz, lz)])
    return center + local @ np.asarray(R).T


def project(pts, K):
    uv = pts @ np.asarray(K).T
    z = np.clip(uv[:, 2], 1e-6, None)
    return np.stack([uv[:, 0] / z, uv[:, 1] / z], 1), pts[:, 2]


def draw_cuboid(d: ImageDraw.ImageDraw, center, dims, R, K, color, width=2):
    c8 = corners(np.asarray(center, float), dims, R)
    uv, z = project(c8, K)
    if (z <= 0).any():
        return
    for a, b in EDGES:
        d.line([tuple(uv[a]), tuple(uv[b])], fill=color, width=width)


def draw_axis(d: ImageDraw.ImageDraw, center, R, K, length, color):
    """Body axis (mod-pi trusted); arrowhead at the +col0 end (sign — untrusted in A1)."""
    c = np.asarray(center, float)
    ax = np.asarray(R)[:, 0]
    p = np.stack([c - ax * length / 2, c + ax * length / 2, c])
    uv, z = project(p, K)
    if (z <= 0).any():
        return
    d.line([tuple(uv[0]), tuple(uv[1])], fill=color, width=3)
    tip, base = uv[1], uv[0]
    v = tip - base
    n = np.linalg.norm(v)
    if n > 1e-3:
        v = v / n
        perp = np.array([-v[1], v[0]])
        for s in (+1, -1):
            q = tip - v * 12 + perp * 6 * s
            d.line([tuple(tip), tuple(q)], fill=color, width=3)


def main() -> int:
    import torch
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--video", type=str, default="")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--score-thresh", type=float, default=0.4)
    ap.add_argument("--crops", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = json.loads(args.val.read_text())
    ims = {i["id"]: i for i in d["images"]}
    anns_by_img = defaultdict(list)
    for a in d["annotations"]:
        if not a.get("behind_camera"):
            anns_by_img[a["image_id"]].append(a)

    preds_by_img = defaultdict(list)
    for r in torch.load(args.preds, map_location="cpu"):
        for inst in r.get("instances", []):
            if inst.get("score", 0.0) >= args.score_thresh:
                preds_by_img[r["image_id"]].append(inst)

    pool = [iid for iid in preds_by_img
            if anns_by_img.get(iid) and (not args.video
                                         or parse_label_path(ims[iid]["file_path"])[1] == args.video)]
    random.Random(args.seed).shuffle(pool)
    pool = pool[:args.n]
    if not pool:
        print("no images matched (check --video / --score-thresh)")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    tiles = []
    for iid in pool:
        im = ims[iid]
        K = np.array(im["K"], float).reshape(3, 3)
        path = args.image_root / im["file_path"].replace("\\", "/")
        img = Image.open(path).convert("RGB")
        dr = ImageDraw.Draw(img)
        for a in anns_by_img[iid]:
            draw_cuboid(dr, a["center_cam"], a["dimensions"], a["R_cam"], K,
                        (40, 220, 40), width=2)
        for p in preds_by_img[iid]:
            R = np.array(p["pose"], float)
            c = np.array(p["center_cam"], float)
            dims = p["dimensions"]
            draw_cuboid(dr, c, dims, R, K, (255, 60, 255), width=2)
            up = -R[:, 1]
            foot = c - (dims[1] / 2) * up
            uv, z = project(foot[None], K)
            if z[0] > 0:
                u, v = uv[0]
                dr.ellipse([u - 4, v - 4, u + 4, v + 4], fill=(255, 220, 0))
            draw_axis(dr, c, R, K, length=dims[2] * 1.3, color=(80, 160, 255))
        _, video, seg, name = parse_label_path(im["file_path"])

        if args.crops:
            for a in anns_by_img[iid]:
                best, bi = None, 0.5
                ax1, ay1, aw, ah = a["bbox"]
                for p in preds_by_img[iid]:
                    px, py, pw, ph = p["bbox"]
                    x1, y1 = max(ax1, px), max(ay1, py)
                    x2, y2 = min(ax1 + aw, px + pw), min(ay1 + ah, py + ph)
                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                    v = inter / (aw * ah + pw * ph - inter + 1e-9)
                    if v > bi:
                        best, bi = p, v
                if best is None:
                    continue
                cx, cy = ax1 + aw / 2, ay1 + ah / 2
                side = max(aw, ah) * 2.2
                box = (int(cx - side / 2), int(cy - side / 2),
                       int(cx + side / 2), int(cy + side / 2))
                tile = img.crop(box).resize((360, 360))
                zg, zp = a["center_cam"][2], best["center_cam"][2]
                td = ImageDraw.Draw(tile)
                td.rectangle([0, 0, 359, 14], fill=(0, 0, 0))
                td.text((4, 2), f"{a['category_name'][:12]} z {zg:.3f}->{zp:.3f} "
                                f"({100 * (zp - zg) / zg:+.1f}%)  s={best['score']:.2f}",
                        fill=(255, 255, 255))
                tiles.append(tile)

        # caption AFTER tiles are cut, so it never bleeds into a crop
        dr.text((8, 6), f"{video}/{seg}/{name}  GT=green pred=magenta contact=yellow "
                        f"axis=blue (A1: arrowhead sign NOT trusted)", fill=(255, 255, 255))
        img.save(args.out / f"{video}__{seg}__{name}".replace(".jpg", ".png"))

    if args.crops and tiles:
        cols = 6
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 360, rows * 360), (20, 20, 20))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * 360, (i // cols) * 360))
        sheet.save(args.out / "contact_sheet.png")
        print(f"contact sheet: {len(tiles)} tiles -> {args.out / 'contact_sheet.png'}")
    print(f"{len(pool)} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
