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


def main() -> int:
    print("TA-DETR preflight (M2 subset)")
    cache_path, cache, anns = find_val_cache()
    p13_unprojection()
    p14_bilinear(cache)
    p15_p6(cache_path, cache, anns)
    n_ok = sum(RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
