"""TA-DETR preflight: CPU checks that must PASS before believing any lift/eval number.  [seconds-minutes]

    python tools/tadetr/preflight_tadetr.py            # M2 subset (P6, P13, P14, P15)

Numbering follows the plan's P1-P13 list; training-time checks (P1-P4 flips, P7 O(1), P8
init-equivalence, P9-P12) join at M3 when the transforms/model exist. Implemented here:

  P6   intersection gradient: autograd d(z)/d(u,v) through the unrolled secant vs central finite
       difference on a REAL terrain cache (rel err < 1e-3, fp64).
  P13  vendored unprojection == vggt.utils.geometry.unproject_depth_map_to_point_map on a real
       segment (max abs diff < 1e-6).
  P14  torch bilinear height (TerrainField.height) == numpy sample_height on random points (< 1e-6).
  P15  one-segment lift smoke: GT bbox bottom-centers -> geometric_lift -> z_label vs GT z on a real
       val segment; median relative error must be < 15% (catches frame/gauge/ray sign errors long
       before the full A0 run; NOT a Gate-1 substitute).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tadetr.data.terrain_cache import load_terrain_cache, sample_height  # noqa: E402
from tadetr.data.wildbox_paths import index_by_video_seg, parse_label_path  # noqa: E402
from tadetr.geometry.heightfield import TerrainField  # noqa: E402
from tadetr.geometry.lift import geometric_lift  # noqa: E402
from tadetr.geometry.unproject import unproject_depth  # noqa: E402

TERRAIN = Path("datasets/tadetr/terrain")
VAL_JSON = Path("/mnt/d/aeroview/labelled/WildBox_val_paper.json")
RESULTS = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def find_val_cache():
    """A val-split terrain cache with plenty of annotations (elep1 0001_V is val -- measured)."""
    d = json.loads(VAL_JSON.read_text())
    ims = {i["id"]: i for i in d["images"]}
    per_seg = defaultdict(list)
    for a in d["annotations"]:
        if a.get("behind_camera"):
            continue
        _, video, seg, image = parse_label_path(ims[a["image_id"]]["file_path"])
        per_seg[(video, seg)].append((image, a))
    for p in sorted(TERRAIN.glob("*.npz")):
        c = load_terrain_cache(p)
        key = (c["meta"]["video"], c["meta"]["seg"])
        if len(per_seg.get(key, [])) > 300 and np.isfinite(float(c["s_seg"])):
            return p, c, per_seg[key]
    raise RuntimeError("no val cache with >300 annotations found")


def p13_unprojection():
    """vggt needs py>=3.9 (builtin-generic annotations); run the comparison under the vggt env."""
    import subprocess
    sp = next(iter(index_by_video_seg().values()))
    script = f"""
import sys, json
import numpy as np
sys.path.insert(0, "/home/shuklva/vggt")
sys.path.insert(0, "{Path(__file__).resolve().parents[2]}")
from vggt.utils.geometry import unproject_depth_map_to_point_map
from tadetr.geometry.unproject import unproject_depth
cams = json.load(open("{sp.cameras_json}"))["cameras"][:2]
z = np.load("{sp.depth_npz}")
worst = 0.0
for i, cam in enumerate(cams):
    D = z["depth"][i]; K = np.array(cam["intrinsic"], float); E = np.array(cam["extrinsic"], float)
    ref = unproject_depth_map_to_point_map(D[None, ..., None], E[None], K[None])[0]
    ours = unproject_depth(D, K, E, valid=np.ones_like(D, bool))
    worst = max(worst, float(np.abs(ref.reshape(-1, 3) - ours).max()))
print(worst)
"""
    r = subprocess.run(["/home/shuklva/miniconda3/envs/vggt/bin/python", "-c", script],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        check("P13 unprojection == vggt", False, f"subprocess failed: {r.stderr.strip()[-200:]}")
        return
    worst = float(r.stdout.strip().splitlines()[-1])
    check("P13 unprojection == vggt", worst < 1e-6, f"max abs diff {worst:.2e} ({sp.name})")


def p14_bilinear(cache):
    f = TerrainField(cache, dtype=torch.float64)
    rng = np.random.RandomState(0)
    G = cache["H_grid"].shape[0]
    a = cache["grid_origin"][0] + rng.uniform(0, G * float(cache["grid_scale"]), 2000)
    b = cache["grid_origin"][1] + rng.uniform(0, G * float(cache["grid_scale"]), 2000)
    ref = sample_height(cache, a, b)
    ours = f.height(torch.tensor(a), torch.tensor(b)).numpy()
    d = float(np.abs(ref - ours).max())
    check("P14 torch bilinear == numpy", d < 1e-6, f"max abs diff {d:.2e}")


def lift_batch(cache, anns, dtype=torch.float64, du=0.0):
    """Run geometric_lift on GT bbox bottom-centers of (image, ann) pairs."""
    f = TerrainField(cache, dtype=dtype)
    E_by = {str(n): cache["extrinsics"][i] for i, n in enumerate(cache["frame_names"])}
    ch_by = {str(n): cache["cam_height"][i] for i, n in enumerate(cache["frame_names"])}
    uv, Ks, Es, chs, zg = [], [], [], [], []
    for image, a in anns:
        E = E_by.get(image)
        if E is None:
            continue
        bx = a["bbox"]
        uv.append([bx[0] + bx[2] / 2 + du, bx[1] + bx[3]])
        Ks.append(np.array(a_K(a), float))
        Es.append(E)
        chs.append(ch_by[image])
        zg.append(a["center_cam"][2])
    T = lambda x: torch.tensor(np.array(x), dtype=dtype)
    out = geometric_lift(T(uv), T(Ks), T(Es), f, T(chs))
    return out, np.array(zg)


_IMS_K = {}


def a_K(a):
    return _IMS_K[a["image_id"]]


def p15_p6(cache_path, cache, anns):
    d = json.loads(VAL_JSON.read_text())
    for i in d["images"]:
        _IMS_K[i["id"]] = np.array(i["K"], float).reshape(3, 3)
    sub = anns[:400]
    out, zg = lift_batch(cache, sub)
    z = out["z_label"].numpy()
    rel = np.abs(z - zg) / zg
    med = float(np.median(rel))
    fb = float(out["fallback"].float().mean())
    check("P15 one-segment lift smoke", med < 0.15,
          f"median |z-z_gt|/z_gt {100*med:.2f}% (fallback {100*fb:.1f}%) on {cache_path.stem}")

    # P6: gradient of z w.r.t. u through the unrolled secant vs finite difference
    f64 = TerrainField(cache, dtype=torch.float64)
    E_by = {str(n): cache["extrinsics"][i] for i, n in enumerate(cache["frame_names"])}
    ch_by = {str(n): cache["cam_height"][i] for i, n in enumerate(cache["frame_names"])}
    image, a = sub[0]
    bx = a["bbox"]
    uv = torch.tensor([[bx[0] + bx[2] / 2, bx[1] + bx[3]]], dtype=torch.float64,
                      requires_grad=True)
    K = torch.tensor(np.array(a_K(a), float)[None], dtype=torch.float64)
    E = torch.tensor(np.array(E_by[image])[None], dtype=torch.float64)
    ch = torch.tensor(np.array([ch_by[image]]), dtype=torch.float64)
    out1 = geometric_lift(uv, K, E, f64, ch)
    out1["z_label"].sum().backward()
    g_auto = uv.grad[0].numpy()
    eps = 0.5
    g_fd = []
    for k, dvec in enumerate(([eps, 0.0], [0.0, eps])):
        zp, _ = lift_one(cache, image, a, dvec)
        zm, _ = lift_one(cache, image, a, [-dvec[0], -dvec[1]])
        g_fd.append((zp - zm) / (2 * eps))
    g_fd = np.array(g_fd)
    denom = max(float(np.abs(g_fd).max()), 1e-12)
    rel = float(np.abs(g_auto - g_fd).max()) / denom
    check("P6 secant gradient vs finite diff", rel < 1e-3,
          f"rel err {rel:.2e} (auto {g_auto}, fd {g_fd})")


def lift_one(cache, image, a, duv):
    f = TerrainField(cache, dtype=torch.float64)
    E_by = {str(n): cache["extrinsics"][i] for i, n in enumerate(cache["frame_names"])}
    ch_by = {str(n): cache["cam_height"][i] for i, n in enumerate(cache["frame_names"])}
    bx = a["bbox"]
    uv = torch.tensor([[bx[0] + bx[2] / 2 + duv[0], bx[1] + bx[3] + duv[1]]], dtype=torch.float64)
    K = torch.tensor(np.array(a_K(a), float)[None], dtype=torch.float64)
    E = torch.tensor(np.array(E_by[image])[None], dtype=torch.float64)
    ch = torch.tensor(np.array([ch_by[image]]), dtype=torch.float64)
    out = geometric_lift(uv, K, E, f, ch)
    return float(out["z_label"][0]), out


def training_checks():
    """P1-P4, P7, P8, P10, P12 -- the M3 additions (flips, O(1) inputs, init-equivalence,
    matcher determinism, join audit). Uses the LOCAL site config."""
    from tadetr.config import TADETRConfig
    from tadetr.data.dataset import WildBoxTADETR, collate
    from tadetr.data.transforms import ViewBridge
    from tadetr.geometry.heightfield import TerrainField as TF32
    from tadetr.modeling.matcher import hungarian_match

    cfg = TADETRConfig.load("configs/tadetr/a1.yaml,configs/tadetr/local_paths.yaml",
                            ["data.num_workers=0"])
    ds = WildBoxTADETR(cfg.data, "val", training=False)
    # P12: join audit -- every frame of both splits resolves to a terrain cache
    ds_tr = WildBoxTADETR(cfg.data, "train", training=False)
    ok12 = (len(ds.samples) == len(ds.images)) and (len(ds_tr.samples) == len(ds_tr.images))
    check("P12 join audit (all frames have terrain)", ok12,
          f"val {len(ds.samples)}/{len(ds.images)}, train {len(ds_tr.samples)}/{len(ds_tr.images)}")

    # P1-P4: flip consistency straight through the REAL target builder
    key = next(iter(ds.by_segment))
    im, seg_key, image_name = ds.samples[ds.by_segment[key][0]]
    anns = [a for a in ds.anns_by_img[im["id"]] if a.get("valid3D", True)][:8]
    W, H = cfg.data.input_w, cfg.data.input_h
    br_n = ViewBridge(sx=1920 / W, sy=1080 / H, ox=0, oy=0, flip=False, w_in=W, h_in=H)
    br_f = ViewBridge(sx=1920 / W, sy=1080 / H, ox=0, oy=0, flip=True, w_in=W, h_in=H)
    tn = ds._targets(list(anns), br_n, seg_key, ds.cache(seg_key))
    tf = ds._targets(list(anns), br_f, seg_key, ds.cache(seg_key))
    du = (tf["contact"][:, 0] - (1 - (W - 1) / W - tn["contact"][:, 0] * -1)).abs()  # u' ~ 1-u
    err_u = (tf["contact"][:, 0] + tn["contact"][:, 0] - (W - 1) / W).abs().max()
    check("P1 hflip contact u' = 1-u", float(err_u) < 2e-3, f"max dev {float(err_u):.2e}")
    e_axis = ((tf["axis_embed"] - torch.stack([-tn["axis_embed"][:, 0],
                                               tn["axis_embed"][:, 1]], 1)).abs().max())
    check("P3 hflip axis (sin2psi flips)", float(e_axis) < 1e-6, f"max dev {float(e_axis):.2e}")
    e_sign = (tf["sign_target"] - tn["sign_target"]).abs().max()
    check("P2 sign bit flip-invariant", float(e_sign) < 1e-6, f"max dev {float(e_sign):.2e}")
    uf, vf = br_f.view_to_full(tf["contact"][:, 0].numpy() * W, tf["contact"][:, 1].numpy() * H)
    bx = np.array([a["bbox"] for a in anns], float)
    err4 = np.abs(uf - (bx[:, 0] + bx[:, 2] / 2)).max()
    check("P4 bridge round-trip to full-res", err4 < 4.0,
          f"max |u_full - bbox bottom-center| {err4:.2f} px (view px = {1920/W:.2f} full px)")

    # P7: O(1) audit on real inputs
    batch = collate([ds[i] for i in ds.by_segment[key][:2]])
    mags = {"telemetry": float(batch["telemetry"].abs().max()),
            "cam_feats": float(batch["cam_feats"].abs().max())}
    check("P7 O(1) model inputs", all(v < 3.5 for v in mags.values()), str(mags))

    # P16: target/export round-trip for orientation -- THE CHECK THAT WOULD HAVE CAUGHT THE A1
    # SIGN BUG. Decode psi from the axis embed exactly as eval_tadetr does (atan2(s2,c2)/2, the
    # canonical branch), apply the sign target, and demand the reconstructed alpha lands within
    # 90 deg of the GT heading on EVERY motion-labelled instance (axis is within 45 deg of true
    # heading by construction, so the correct branch is always within 90 deg).
    n_checked, worst = 0, 0.0
    for k2 in list(ds_tr.by_segment):
        for idx in ds_tr.by_segment[k2][:20]:
            im2, sk2, _ = ds_tr.samples[idx]
            anns2 = [a for a in ds_tr.anns_by_img[im2["id"]]
                     if a.get("heading_valid", 0) and a.get("valid3D", True)]
            if not anns2:
                continue
            t2 = ds_tr._targets(list(anns2), br_n, sk2, ds_tr.cache(sk2))
            m2 = t2["sign_valid"] > 0
            if not m2.any():
                continue
            psi_dec = 0.5 * torch.atan2(t2["axis_embed"][m2, 0], t2["axis_embed"][m2, 1])
            a_rec = psi_dec + torch.pi * t2["sign_target"][m2]
            a_gt = torch.tensor([float(a["heading_alpha"]) for a, mm in
                                 zip(anns2, m2.tolist()) if mm])
            dev = torch.atan2(torch.sin(a_rec - a_gt), torch.cos(a_rec - a_gt)).abs()
            worst = max(worst, float(dev.max()))
            n_checked += int(m2.sum())
        if n_checked >= 300:
            break
    check("P16 orientation target/export round-trip", n_checked > 50 and worst < torch.pi / 2 + 1e-4,
          f"{n_checked} labelled instances, worst |alpha_rec - alpha_gt| = "
          f"{torch.rad2deg(torch.tensor(worst)):.1f} deg (must be < 90)")

    # P10: matcher determinism
    Q, C = 50, 6
    g = torch.Generator().manual_seed(0)
    lg = torch.randn(2, Q, C + 1, generator=g)
    bx2 = torch.rand(2, Q, 4, generator=g)
    ct = torch.rand(2, Q, 2, generator=g)
    tgts = [{"cls": torch.tensor([1, 2]), "boxes": torch.rand(2, 4, generator=g),
             "contact": torch.rand(2, 2, generator=g)} for _ in range(2)]
    m1 = hungarian_match(lg, bx2, ct, tgts, 2, 5, 2)
    m2 = hungarian_match(lg, bx2, ct, tgts, 2, 5, 2)
    ok10 = all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(m1, m2))
    check("P10 matcher determinism", ok10, "two identical calls agree")

    # P8: init-equivalence -- composed z with GT contact == geometric_lift z (A1 flags, zero-init)
    c = ds.cache(seg_key)
    f32 = TF32(c, dtype=torch.float32)
    E_by = {str(n): i for i, n in enumerate(c["frame_names"])}
    fi = E_by[image_name]
    n_t = min(len(anns), 6)
    uv_full = torch.tensor([[a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3]]
                            for a in anns[:n_t]], dtype=torch.float32)
    K = torch.tensor(np.array(im["K"], np.float32).reshape(3, 3))
    E = torch.tensor(np.array(c["extrinsics"][fi], np.float32))
    ch = torch.tensor(float(c["cam_height"][fi])).expand(n_t)
    lift = geometric_lift(uv_full, K.expand(n_t, 3, 3), E.expand(n_t, 3, 4), f32, ch)
    # composed (A1): z of foot + H/2 lift -- compare center z vs lift z + analytic H/2 term
    s = f32.s_seg
    p_t = f32.world_to_tangent(lift["p_world"])
    n_w = f32.normal_world(p_t[:, 0], p_t[:, 1])
    Hmed = torch.tensor([a["dimensions"][1] for a in anns[:n_t]], dtype=torch.float32)
    cw = lift["p_world"] + n_w * (Hmed / (2 * s))[:, None]
    z_comp = s * ((E[:, :3] @ cw.T).T + E[:, 3])[:, 2]
    z_gt = torch.tensor([a["center_cam"][2] for a in anns[:n_t]])
    med = float((z_comp - z_gt).abs().median() / z_gt.median())
    check("P8 composition == lift+H/2 (A1 path, GT contact)", med < 0.05,
          f"median |z_comp - z_gt|/z {100*med:.2f}% on {n_t} anns")


def main() -> int:
    print("TA-DETR preflight")
    cache_path, cache, anns = find_val_cache()
    p13_unprojection()
    p14_bilinear(cache)
    p15_p6(cache_path, cache, anns)
    training_checks()
    n_ok = sum(RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
