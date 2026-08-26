"""M1: build the per-segment terrain height-field caches from the dense VGGT tree.  [LOCAL CPU]

    python tools/tadetr/build_terrain_cache.py \
        --train /mnt/d/3DBOX/papersubdata/WildBox_train_paper.json \
        --val   /mnt/d/3DBOX/papersubdata/WildBox_val_paper.json \
        --out-dir datasets/tadetr/terrain --workers 4

Per segment (contract: tadetr/data/terrain_cache.py):
  depth_maps.npz + cameras.json -> conf-filtered unprojection -> background points
  (animal pixels removed via DILATED bbox_2d rectangles in 518-space, dilation 8 px ~ 30 full-res px:
   a strict SUPERSET of the spec's dilated masks -- more conservative against contamination, and it
   avoids ~2,400 PNG reads per segment over drvfs; recorded in meta)
  -> RANSAC ground plane (PCA init; inlier threshold 2% of median camera height; sign toward cameras;
     cross-checked against the box-up consensus from tracking rotations, warn > 10 deg)
  -> plane-tangent frame R_grid -> 256^2 confidence-weighted median height grid + H_var + n_points
  -> NN-fill + Laplacian smoothing (lambda 0.5, 20 iters)   [fill is LOAD-BEARING: cells under
     animals are empty by construction -- measured]
  -> s_seg (gauge bridge) from track_id/image-matched detector-json annotations
  -> per-frame extrinsics, K_518, cam_height, frame_names
Writes <out-dir>/<group>__<video>__<seg>.npz + terrain_MANIFEST.json (sha256 + stats; committed).

Idempotent: existing npz are skipped unless --force. Failures are recorded in the manifest, never
silently dropped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tadetr.data.terrain_cache import GRID_SIZE, write_terrain_cache  # noqa: E402
from tadetr.data.wildbox_paths import index_by_video_seg, parse_label_path  # noqa: E402
from tadetr.geometry.unproject import camera_center, unproject_depth  # noqa: E402

DILATE_PX = 8              # 518-space; ~30 full-res px, superset of spec's 15 px mask dilation
CONF_MIN = 1.5             # floor (the project-wide reader threshold); raised to p20 if higher
MAX_PTS_PER_FRAME = 25000
RANSAC_ITERS = 300
RANSAC_SUBSAMPLE = 200000

LABELS: dict = {}          # (video, seg) -> list[(tid, image, center_cam(3))]; set in main, fork-inherited


def _weighted_median_grid(ia, ib, h, w, G):
    """Per-cell weighted median / variance / count over a G x G grid. Vectorized sort + bounded loop."""
    cell = ia.astype(np.int64) * G + ib
    order = np.argsort(cell, kind="stable")
    cell, h, w = cell[order], h[order], w[order]
    starts = np.searchsorted(cell, np.arange(G * G))
    ends = np.searchsorted(cell, np.arange(G * G), side="right")
    H = np.full((G, G), np.nan, np.float32)
    V = np.full((G, G), np.nan, np.float32)
    N = np.zeros((G, G), np.int32)
    occupied = np.nonzero(ends > starts)[0]
    for c in occupied:
        s, e = starts[c], ends[c]
        hs, ws = h[s:e], w[s:e]
        o = np.argsort(hs)
        hs, ws = hs[o], ws[o]
        cw = np.cumsum(ws)
        med = hs[np.searchsorted(cw, 0.5 * cw[-1])]
        i, j = divmod(c, G)
        H[i, j] = med
        mean_conf = float(ws.mean())
        V[i, j] = (float(np.var(hs)) if len(hs) > 1 else 0.0) / max(mean_conf, 1e-6)
        N[i, j] = e - s
    return H, V, N


def _ransac_plane(P, thresh, iters=RANSAC_ITERS, rng=None):
    rng = rng or np.random.RandomState(0)
    best_n, best_c, best_in = None, None, -1
    for _ in range(iters):
        idx = rng.choice(len(P), 3, replace=False)
        a, b, c = P[idx]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        d = np.abs((P - a) @ n)
        n_in = int((d < thresh).sum())
        if n_in > best_in:
            best_in, best_n, best_c = n_in, n, a
    inl = np.abs((P - best_c) @ best_n) < thresh
    Pi = P[inl]
    ctr = Pi.mean(0)
    _, _, Vt = np.linalg.svd(Pi - ctr, full_matrices=False)
    return Vt[2], ctr, float(inl.mean())


def build_one(args_tuple):
    sp, out_dir, force = args_tuple
    out_path = out_dir / f"{sp.name}.npz"
    if out_path.exists() and not force:
        return {"name": sp.name, "status": "exists"}
    warnings = []
    try:
        cams = json.loads(sp.cameras_json.read_text())["cameras"]
        ts = json.loads(sp.tracking_summary.read_text())
        Nf = len(cams)
        Ks = np.array([c["intrinsic"] for c in cams], np.float64)
        Es = np.array([c["extrinsic"] for c in cams], np.float64)
        names = [c["image_name"] for c in cams]
        idx_of_frame = {c["frame_index"]: i for i, c in enumerate(cams)}

        boxes_by_pos = defaultdict(list)
        rotations = []
        for tr in ts["tracks"].values():
            rotations.extend(tr.get("rotation_matrices", []))
            for f, b in zip(tr["frames"], tr["bbox_2d"]):
                pos = idx_of_frame.get(int(f))
                if pos is not None:
                    boxes_by_pos[pos].append(b)

        z = np.load(sp.depth_npz)
        depth, conf = z["depth"], z["depth_conf"]
        conf_thresh = max(CONF_MIN, float(np.percentile(conf, 20)))

        rng = np.random.RandomState(0)
        pts_list, w_list = [], []
        for pos in range(Nf):
            D, C = depth[pos], conf[pos]
            m = (C >= conf_thresh) & (D > 1e-6)
            for b in boxes_by_pos.get(pos, []):
                x1, y1 = int(b[0]) - DILATE_PX, int(b[1]) - DILATE_PX
                x2, y2 = int(b[2]) + DILATE_PX, int(b[3]) + DILATE_PX
                m[max(y1, 0):y2, max(x1, 0):x2] = False
            if not m.any():
                continue
            pw = unproject_depth(D, Ks[pos], Es[pos], valid=m)
            ww = C[m]
            if len(pw) > MAX_PTS_PER_FRAME:
                sel = rng.choice(len(pw), MAX_PTS_PER_FRAME, replace=False)
                pw, ww = pw[sel], ww[sel]
            pts_list.append(pw.astype(np.float32))
            w_list.append(ww.astype(np.float32))
        if not pts_list:
            return {"name": sp.name, "status": "failed", "error": "no background points"}
        P = np.concatenate(pts_list)
        W = np.concatenate(w_list)

        # ground plane: PCA init -> camera-height scale -> RANSAC -> sign toward cameras
        sub = P[rng.choice(len(P), min(RANSAC_SUBSAMPLE, len(P)), replace=False)]
        ctr0 = sub.mean(0)
        _, _, Vt = np.linalg.svd(sub - ctr0, full_matrices=False)
        n0 = Vt[2]
        cam_centers = np.array([camera_center(E) for E in Es])
        if np.median((cam_centers - ctr0) @ n0) < 0:
            n0 = -n0
        h_cam0 = float(np.median((cam_centers - ctr0) @ n0))
        n, ctr, inlier_frac = _ransac_plane(sub, thresh=0.02 * max(h_cam0, 1e-6), rng=rng)
        if np.median((cam_centers - ctr) @ n) < 0:
            n = -n
        pca_angle = float(np.degrees(np.arccos(np.clip(abs(n @ n0), 0, 1))))

        # box-up consensus cross-check: per stored rotation, the axis most aligned with n
        if rotations:
            R = np.array(rotations, np.float64)          # (M,3,3), columns = box axes
            dots = np.abs(np.einsum("mij,i->mj", R, n))  # |axis_j . n|
            best = dots.max(axis=1)
            box_up_angle = float(np.degrees(np.arccos(np.clip(np.median(best), 0, 1))))
        else:
            box_up_angle = float("nan")
        if not (box_up_angle < 10.0):
            warnings.append(f"box-up consensus angle {box_up_angle:.1f} deg (>10)")

        # tangent frame
        e1 = np.array([1.0, 0.0, 0.0]) - n[0] * n
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.array([0.0, 0.0, 1.0]) - n[2] * n
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        R_grid = np.stack([e1, e2, n])                   # world -> tangent: x_t = R_grid @ (x - ctr)

        Pl = (P - ctr) @ R_grid.T
        a, b, h = Pl[:, 0], Pl[:, 1], Pl[:, 2]
        a_lo, a_hi = np.percentile(a, [0.5, 99.5])
        b_lo, b_hi = np.percentile(b, [0.5, 99.5])
        side = max(a_hi - a_lo, b_hi - b_lo)
        origin = np.array([(a_lo + a_hi) / 2 - side / 2, (b_lo + b_hi) / 2 - side / 2])
        scale = side / GRID_SIZE
        inside = (a >= origin[0]) & (a < origin[0] + side) & (b >= origin[1]) & (b < origin[1] + side)
        ia = np.clip(((a[inside] - origin[0]) / scale).astype(int), 0, GRID_SIZE - 1)
        ib = np.clip(((b[inside] - origin[1]) / scale).astype(int), 0, GRID_SIZE - 1)
        H, V, Ncnt = _weighted_median_grid(ia, ib, h[inside], W[inside], GRID_SIZE)
        fill_frac = float(np.isfinite(H).mean())

        # NN-fill + Laplacian smoothing (spec: lambda 0.5, 20 iters)
        missing = ~np.isfinite(H)
        if missing.any():
            _, (ii, jj) = ndimage.distance_transform_edt(missing, return_indices=True)
            H = H[ii, jj]
            V_filled = V.copy()
            V_filled[missing] = np.nanmax(V) if np.isfinite(V).any() else 1.0
            V = V_filled
        for _ in range(20):
            neigh = (np.roll(H, 1, 0) + np.roll(H, -1, 0) + np.roll(H, 1, 1) + np.roll(H, -1, 1)) / 4
            neigh[0, :], neigh[-1, :] = H[1, :], H[-2, :]
            neigh[:, 0], neigh[:, -1] = H[:, 1], H[:, -2]
            H = H + 0.5 * (neigh - H)

        # s_seg: gauge bridge from matched detector-json annotations
        obs = {}
        for tid, tr in ts["tracks"].items():
            try:
                tnum = int(tid.split("::")[-1]) if "::" in str(tid) else int(tid)
            except ValueError:
                tnum = tid
            for f, c in zip(tr["frames"], tr["centers"]):
                pos = idx_of_frame.get(int(f))
                if pos is not None:
                    obs[(tnum, names[pos])] = (pos, np.array(c, float))
        ratios = []
        for tid, image, center_cam in LABELS.get(sp.key, []):
            hit = obs.get((tid, image))
            if hit is None:
                continue
            pos, cw = hit
            zc = (Es[pos][:, :3] @ cw + Es[pos][:, 3])[2]
            if zc > 1e-9 and center_cam[2] > 0:
                ratios.append(center_cam[2] / zc)
        if len(ratios) >= 5:
            s_seg = float(np.median(ratios))
            s_sd = float(np.std(np.log(np.array(ratios))))
            if s_sd > 0.05:
                warnings.append(f"s_seg sd(log)={s_sd:.3f} (>0.05)")
        else:
            s_seg, s_sd = float("nan"), float("nan")
            warnings.append(f"s_seg unrecovered (only {len(ratios)} matches)")

        # per-frame camera height above local terrain
        cam_l = (cam_centers - ctr) @ R_grid.T
        ga = np.clip(((cam_l[:, 0] - origin[0]) / scale).astype(int), 0, GRID_SIZE - 1)
        gb = np.clip(((cam_l[:, 1] - origin[1]) / scale).astype(int), 0, GRID_SIZE - 1)
        cam_height = cam_l[:, 2] - H[ga, gb]

        plane = np.concatenate([n, [-float(n @ ctr)]])
        write_terrain_cache(
            out_path, H_grid=H, H_var=V, n_points=Ncnt, grid_origin=origin, grid_scale=scale,
            R_grid=R_grid, ctr=ctr, plane=plane, s_seg=s_seg, extrinsics=Es, K_518=Ks,
            cam_height=cam_height, frame_names=np.array(names), meta={
                "shoot": sp.shoot, "group": sp.group, "video": sp.video, "seg": sp.seg,
                "conf_thresh": conf_thresh, "n_bg_points": int(len(P)),
                "animal_removal": f"bbox_2d dilated {DILATE_PX}px (518-space; superset of masks+15px)",
                "tracking_corrected": sp.tracking_is_corrected,
                "ransac_inlier_frac": round(inlier_frac, 3),
                "pca_vs_ransac_deg": round(pca_angle, 2),
                "box_up_angle_deg": round(box_up_angle, 2),
                "cell_fill_frac_prefill": round(fill_frac, 3),
                "s_seg_sd_log": None if np.isnan(s_sd) else round(s_sd, 4),
                "s_seg_n_matches": len(ratios),
                "cam_height_med": round(float(np.median(cam_height)), 4),
                "warnings": warnings,
            })
        sha = hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]
        return {"name": sp.name, "status": "ok", "sha256_16": sha,
                "s_seg": None if np.isnan(s_seg) else round(s_seg, 5),
                "fill_frac": round(fill_frac, 3), "box_up_deg": round(box_up_angle, 2),
                "warnings": warnings}
    except Exception:
        return {"name": sp.name, "status": "failed", "error": traceback.format_exc(limit=3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("datasets/tadetr/terrain"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", type=str, default="", help="substring filter on segment name")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    global LABELS
    lab = defaultdict(list)
    for path in (args.train, args.val):
        d = json.loads(path.read_text())
        ims = {i["id"]: i for i in d["images"]}
        for a in d["annotations"]:
            _, video, seg, image = parse_label_path(ims[a["image_id"]]["file_path"])
            lab[(video, seg)].append((a.get("track_id"), image,
                                      np.array(a["center_cam"], float)))
        print(f"labels: {path.name} loaded")
    LABELS = dict(lab)

    segs = sorted(index_by_video_seg().values(), key=lambda s: s.name)
    if args.only:
        segs = [s for s in segs if args.only in s.name]
    if args.limit:
        segs = segs[:args.limit]
    print(f"building {len(segs)} segments -> {args.out_dir} (workers={args.workers})")

    jobs = [(sp, args.out_dir, args.force) for sp in segs]
    results = []
    if args.workers <= 1:
        for i, j in enumerate(jobs):
            r = build_one(j)
            results.append(r)
            print(f"[{i+1}/{len(jobs)}] {r['name']}: {r['status']}"
                  + (f" (s_seg {r.get('s_seg')}, fill {r.get('fill_frac')})" if r["status"] == "ok" else ""),
                  flush=True)
    else:
        with Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(build_one, jobs)):
                results.append(r)
                print(f"[{i+1}/{len(jobs)}] {r['name']}: {r['status']}"
                      + (f" (s_seg {r.get('s_seg')}, fill {r.get('fill_frac')})" if r["status"] == "ok" else ""),
                      flush=True)

    ok = [r for r in results if r["status"] == "ok"]
    exists = [r for r in results if r["status"] == "exists"]
    failed = [r for r in results if r["status"] == "failed"]
    warned = [r for r in ok if r.get("warnings")]
    manifest_path = args.out_dir / "terrain_MANIFEST.json"
    prior = json.loads(manifest_path.read_text())["segments"] if manifest_path.exists() else {}
    for r in ok:
        prior[r["name"]] = {k: v for k, v in r.items() if k != "name"}
    for r in failed:
        prior[r["name"]] = {"status": "failed", "error": r["error"].splitlines()[-1] if r.get("error") else "?"}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(
        {"schema": 1, "n": len(prior), "segments": dict(sorted(prior.items()))}, indent=1))
    print(f"\nbuilt {len(ok)}, skipped-existing {len(exists)}, FAILED {len(failed)}, warned {len(warned)}")
    for r in failed:
        print(f"  FAILED {r['name']}: {r['error'].splitlines()[-1] if r.get('error') else '?'}")
    for r in warned[:20]:
        print(f"  warn {r['name']}: {r['warnings']}")
    print(f"manifest -> {manifest_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
