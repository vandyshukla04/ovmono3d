"""M2 / ablation A0: training-free geometric_lift over WildBox val -> instances_predictions.pth.

    python tools/tadetr/run_geometric_lift.py \
        --val   /mnt/d/aeroview/labelled/WildBox_val_paper.json \
        --train /mnt/d/aeroview/labelled/WildBox_train_paper.json \
        --terrain datasets/tadetr/terrain --mode gt2d \
        --out datasets/tadetr/runs/a0_gt2d/WildBox_val.pth

Per annotation (gt2d mode = GT 2D boxes, the Gate-1 protocol):
  contact pixel  = GT bbox bottom-center (u = cx, v = y2)
  dh             = species offset (world units) = offset_frac[species] * H_med[class] / s_seg
                   (offset_frac from the Gate-B report: animals sit above the VGGT terrain by a
                   species-dependent constant -- legs under-reconstructed at 518 res)
  depth          = geometric_lift ray-terrain intersection  ->  z in the LABEL gauge via s_seg
  center         = foot3d + n_surface * (H_med[class] / (2 s_seg))  ->  label cam via s_seg * E
  dims           = per-class LABEL-SANE train medians (valid3D only; medians not means -- robust)
  pose           = terrain-normal frame, yaw 0: columns [c0, -up_cam, c0 x -up_cam]; alpha 0.0
  score          = 1.0 (oracle-2D protocol), bbox = GT XYWH, category_id = GT (dataset ids)

Class stats + species offsets are derived once and cached to tools/tadetr/data/a0_class_stats.json.
Output format matches tools/ovmono3d_geo.py records exactly (bbox3D corners via
cubercnn.util.get_cuboid_verts_faces -- never hand-rolled).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cubercnn.util import get_cuboid_verts_faces  # noqa: E402
from tadetr.data.terrain_cache import load_terrain_cache  # noqa: E402
from tadetr.data.wildbox_paths import parse_label_path  # noqa: E402
from tadetr.geometry.heightfield import TerrainField  # noqa: E402
from tadetr.geometry.lift import geometric_lift  # noqa: E402

STATS_PATH = Path(__file__).resolve().parent / "data" / "a0_class_stats.json"


def build_stats(train_json: Path, unit_report: Path) -> dict:
    d = json.loads(train_json.read_text())
    ims = {i["id"]: i for i in d["images"]}
    dims = defaultdict(list)
    for a in d["annotations"]:
        if a.get("behind_camera") or not a.get("valid3D", True):
            continue
        dims[a["category_name"]].append(a["dimensions"])
    cls = {k: {"dims_median": np.median(np.array(v), axis=0).tolist(), "n": len(v)}
           for k, v in dims.items()}

    r = json.loads(unit_report.read_text())
    per_sp = defaultdict(list)
    for name, splits in r["per_segment"].items():
        v = splits.get("train")
        if v and v.get("terr_med_signed") is not None:
            per_sp[v["species_mode"]].append(v["terr_med_signed"])
    offsets = {sp: float(np.median(vals)) for sp, vals in per_sp.items()}
    return {"classes": cls, "species_offset_frac": offsets,
            "source": {"train": str(train_json), "unit_report": str(unit_report)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--terrain", type=Path, default=Path("datasets/tadetr/terrain"))
    ap.add_argument("--mode", choices=["gt2d", "gdino"], default="gt2d")
    ap.add_argument("--gdino-json", type=Path,
                    default=Path("datasets/Omni3D/gdino_WildBox_val_oracle_2d.json"),
                    help="gdino mode: per-image detections (image_id join; bbox XYWH, score, "
                         "category_name) -- the frozen GroundingDINO oracle file, never regenerated")
    ap.add_argument("--out", type=Path, default=Path("datasets/tadetr/runs/a0_gt2d/WildBox_val.pth"))
    ap.add_argument("--unit-report", type=Path,
                    default=Path("datasets/tadetr/reports/terrain_unit_test.json"))
    ap.add_argument("--no-species-offset", action="store_true")
    args = ap.parse_args()

    if STATS_PATH.exists():
        stats = json.loads(STATS_PATH.read_text())
    else:
        stats = build_stats(args.train, args.unit_report)
        STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATS_PATH.write_text(json.dumps(stats, indent=1))
    print(f"class stats: { {k: [round(x,4) for x in v['dims_median']] for k, v in stats['classes'].items()} }")
    print(f"species offsets: { {k: round(v,3) for k, v in stats['species_offset_frac'].items()} }")

    caches = {}
    for p in sorted(args.terrain.glob("*.npz")):
        c = load_terrain_cache(p)
        caches[(c["meta"]["video"], c["meta"]["seg"])] = c

    d = json.loads(args.val.read_text())
    ims = {i["id"]: i for i in d["images"]}
    per_seg = defaultdict(list)
    if args.mode == "gt2d":
        for a in d["annotations"]:
            if a.get("behind_camera"):
                continue
            _, video, seg, image = parse_label_path(ims[a["image_id"]]["file_path"])
            per_seg[(video, seg)].append((image, a))
    else:  # gdino: real detections; synthesize gt2d-shaped anns (bbox/score/category only)
        for rec in json.loads(args.gdino_json.read_text()):
            im = ims.get(rec["image_id"])
            if im is None:
                continue
            _, video, seg, image = parse_label_path(im["file_path"])
            for det in rec["instances"]:
                per_seg[(video, seg)].append((image, {
                    "image_id": rec["image_id"], "bbox": det["bbox"],
                    "score": float(det.get("score", 1.0)),
                    "category_id": det["category_id"], "category_name": det["category_name"],
                }))

    inst_by_img: dict[int, list] = defaultdict(list)
    n_done = n_skip = n_fb = 0
    for key, lst in sorted(per_seg.items()):
        c = caches.get(key)
        if c is None or not np.isfinite(float(c["s_seg"])):
            n_skip += len(lst)
            continue
        field = TerrainField(c, dtype=torch.float64)
        s_seg = float(c["s_seg"])
        E_by = {str(n): c["extrinsics"][i] for i, n in enumerate(c["frame_names"])}
        ch_by = {str(n): float(c["cam_height"][i]) for i, n in enumerate(c["frame_names"])}

        uv, Ks, Es, chs, dhs, meta = [], [], [], [], [], []
        for image, a in lst:
            E = E_by.get(image)
            if E is None:
                n_skip += 1
                continue
            cls = stats["classes"].get(a["category_name"])
            if cls is None:
                n_skip += 1
                continue
            H_med = cls["dims_median"][1]
            off = 0.0 if args.no_species_offset else \
                stats["species_offset_frac"].get(a["category_name"], 0.0)
            b = a["bbox"]
            uv.append([b[0] + b[2] / 2, b[1] + b[3]])
            Ks.append(np.array(ims[a["image_id"]]["K"], float).reshape(3, 3))
            Es.append(np.array(E, float))
            chs.append(ch_by[image])
            dhs.append(off * H_med / s_seg)
            meta.append((image, a, H_med, cls["dims_median"]))
        if not uv:
            continue
        T = lambda x: torch.tensor(np.array(x), dtype=torch.float64)
        out = geometric_lift(T(uv), T(Ks), T(Es), field, T(chs), dh=T(dhs))
        p_world = out["p_world"].numpy()
        n_fb += int(out["fallback"].sum())

        # per-instance surface normal in world, then assemble the record
        p_t = field.world_to_tangent(out["p_world"])
        n_world = field.normal_world(p_t[:, 0], p_t[:, 1]).numpy()
        for i, (image, a, H_med, dims_med) in enumerate(meta):
            E = Es[i]
            center_world = p_world[i] + n_world[i] * (H_med / (2 * s_seg))
            center_cam = s_seg * (E[:, :3] @ center_world + E[:, 3])
            up_cam = E[:, :3] @ n_world[i]
            up_cam = up_cam / max(np.linalg.norm(up_cam), 1e-12)
            c0 = np.cross(up_cam, np.array([0.0, 0.0, 1.0]))
            c0 = c0 / max(np.linalg.norm(c0), 1e-12)
            c1 = -up_cam
            c2 = np.cross(c0, c1)
            pose = np.stack([c0, c1, c2], axis=1)
            w, h, l = dims_med
            verts, _ = get_cuboid_verts_faces(
                box3d=[float(center_cam[0]), float(center_cam[1]), float(center_cam[2]),
                       float(w), float(h), float(l)], R=torch.tensor(pose, dtype=torch.float32))
            K = Ks[i]
            uvz = K @ center_cam
            inst_by_img[a["image_id"]].append({
                "image_id": a["image_id"],
                "category_id": a["category_id"],
                "category_name": a["category_name"],
                "bbox": [float(x) for x in a["bbox"]],
                "score": float(a.get("score", 1.0)),
                "depth": float(center_cam[2]),
                "bbox3D": verts.numpy().tolist() if hasattr(verts, "numpy") else np.asarray(verts).tolist(),
                "center_cam": center_cam.tolist(),
                "center_2D": [float(uvz[0] / uvz[2]), float(uvz[1] / uvz[2])],
                "dimensions": [float(w), float(h), float(l)],
                "pose": pose.tolist(),
                "alpha": 0.0,
                "lift_fallback": bool(out["fallback"][i]),
                "lift_sin_theta": float(out["sin_theta_g"][i]),
            })
            n_done += 1
        print(f"  {key[0]}/{key[1]}: {len(meta)} lifted", flush=True)

    dataset = []
    for img in d["images"]:
        rec = dict(img)
        rec["image_id"] = img["id"]              # instances_predictions.pth top-level key
        rec["instances"] = inst_by_img.get(img["id"], [])
        dataset.append(rec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, args.out)
    print(f"\nlifted {n_done:,} annotations ({n_skip:,} skipped, {n_fb:,} plane-fallback) "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
