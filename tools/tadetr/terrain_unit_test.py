"""M1 GATE B: do GT boxes stand on the terrain caches?  [CPU, minutes]

    python tools/tadetr/terrain_unit_test.py \
        --train /mnt/d/3DBOX/papersubdata/WildBox_train_paper.json \
        --val   /mnt/d/3DBOX/papersubdata/WildBox_val_paper.json \
        --terrain datasets/tadetr/terrain \
        --out datasets/tadetr/reports/terrain_unit_test.json

Per annotation: bottom-face center = center_cam - (H/2)*up with up = -R_cam[:,1] (the verified
convention; +R[:,1] is the arrow-inversion bug) -> label gauge -> /s_seg -> world via extrinsic^-1
-> tangent frame -> signed distance to the bilinear-sampled H_grid, normalized by the box height.

Populations, kept separate (the human-audited junk MUST be excluded from the gate):
  sane  = |off-plane vs the segment's own per-split plane| <= 1.0*H   (stamp_geometry convention)
  junk  = the rest (95% human-confirmed fragments -- reported for information, never gated)

GATE B (spec 0.2 amended): median |signed dist|/H < 0.3 over SANE TRAIN annotations.
Also reported: per-segment medians + histogram tails, val numbers, the consistency-flagged segments
from Gate A individually, and terrain-vs-plane comparison (the terrain should win in the tail).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tadetr.data.terrain_cache import load_terrain_cache, sample_height  # noqa: E402
from tadetr.data.wildbox_paths import parse_label_path  # noqa: E402

JUNK_OFF_H = 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--terrain", type=Path, default=Path("datasets/tadetr/terrain"))
    ap.add_argument("--out", type=Path, default=Path("datasets/tadetr/reports/terrain_unit_test.json"))
    ap.add_argument("--flagged", type=Path,
                    default=Path("datasets/tadetr/reports/resolution_mapping.json"),
                    help="Gate-A report; its consistency_flagged list is reported individually")
    args = ap.parse_args()

    caches: dict[tuple[str, str], dict] = {}
    name_of: dict[tuple[str, str], str] = {}
    for p in sorted(args.terrain.glob("*.npz")):
        c = load_terrain_cache(p)
        key = (c["meta"]["video"], c["meta"]["seg"])
        caches[key] = c
        name_of[key] = p.stem
    print(f"terrain caches loaded: {len(caches)}")

    flagged = set()
    if args.flagged.exists():
        flagged = set(json.loads(args.flagged.read_text())["summary"].get("consistency_flagged", []))

    report = {"splits": {}, "per_segment": {}, "flagged_segments": {}}
    gate_val = None
    for split, path in (("train", args.train), ("val", args.val)):
        d = json.loads(path.read_text())
        ims = {i["id"]: i for i in d["images"]}
        by_seg = defaultdict(list)
        for a in d["annotations"]:
            if a.get("behind_camera"):
                continue
            _, video, seg, image = parse_label_path(ims[a["image_id"]]["file_path"])
            by_seg[(video, seg)].append((image, a))

        sane_terr, sane_plane, junk_terr, missing = [], [], [], 0
        n_lookup_nan = 0
        for key, lst in sorted(by_seg.items()):
            c = caches.get(key)
            if c is None:
                missing += len(lst)
                continue
            s_seg = float(c["s_seg"])
            if not np.isfinite(s_seg):
                missing += len(lst)
                continue
            E_by_name = {str(n): c["extrinsics"][i] for i, n in enumerate(c["frame_names"])}
            R_grid, ctr = c["R_grid"].astype(float), c["ctr"].astype(float)

            offs_t, offs_p, Hs, images_ok = [], [], [], []
            for image, a in lst:
                E = E_by_name.get(image)
                if E is None:
                    missing += 1
                    continue
                Rc = np.array(a["R_cam"], float)
                up = -Rc[:, 1]
                Hh = float(a["dimensions"][1])
                bot_cam = np.array(a["center_cam"], float) - (Hh / 2) * up
                bw = (bot_cam / s_seg - E[:, 3]) @ E[:, :3]      # world
                aa, bb, hh = R_grid @ (bw - ctr)                 # tangent frame of the cache
                ht = float(sample_height(c, np.array([aa]), np.array([bb]))[0])
                off_t = (hh - ht) / max(Hh / s_seg, 1e-9)
                off_p = hh / max(Hh / s_seg, 1e-9)
                if not np.isfinite(off_t):
                    n_lookup_nan += 1
                    continue
                offs_t.append(off_t)
                offs_p.append(off_p)
                Hs.append(Hh)
                images_ok.append(image)
            if not offs_t:
                continue
            offs_t = np.array(offs_t)
            offs_p = np.array(offs_p)
            med_p = np.median(offs_p)
            sane = np.abs(offs_p - med_p) <= JUNK_OFF_H
            sane_terr += offs_t[sane].tolist()
            sane_plane += (offs_p[sane] - med_p).tolist()
            junk_terr += offs_t[~sane].tolist()
            name = name_of[key]
            seg_med = float(np.median(np.abs(offs_t[sane]))) if sane.any() else float("nan")
            report["per_segment"].setdefault(name, {})[split] = {
                "n_sane": int(sane.sum()), "n_junk": int((~sane).sum()),
                "terr_med_abs": round(seg_med, 4),
                # signed median feeds the M2 species-offset table: animals sit ABOVE the VGGT
                # background terrain by a species-dependent constant (legs under-reconstructed
                # at 518-res; +0.14H measured on elephants, ~0 on zebras)
                "terr_med_signed": round(float(np.median(offs_t[sane])), 4) if sane.any() else None,
                "species_mode": max(set(cat := [x[1].get("category_name", "?") for x in lst]),
                                    key=cat.count),
            }
            if name in flagged:
                report["flagged_segments"].setdefault(name, {})[split] = round(seg_med, 4)

        st, sp_, jt = map(np.array, (sane_terr, sane_plane, junk_terr))
        if len(st) == 0:
            report["splits"][split] = {"n_sane": 0, "n_missing_join": missing}
            print(f"[{split}] no joined annotations (missing-join {missing:,})")
            continue
        res = {
            "n_sane": len(st), "n_junk": len(jt), "n_missing_join": missing,
            "n_lookup_nan": n_lookup_nan,
            "terrain_med_abs": round(float(np.median(np.abs(st))), 4),
            "terrain_p90_abs": round(float(np.percentile(np.abs(st), 90)), 4),
            "terrain_frac_lt_0.3": round(float((np.abs(st) < 0.3).mean()), 4),
            "plane_med_abs": round(float(np.median(np.abs(sp_))), 4),
            "plane_p90_abs": round(float(np.percentile(np.abs(sp_), 90)), 4),
            "plane_frac_lt_0.3": round(float((np.abs(sp_) < 0.3).mean()), 4),
            "junk_terr_med_abs": round(float(np.median(np.abs(jt))), 4) if len(jt) else None,
            "signed_median": round(float(np.median(st)), 4),
        }
        report["splits"][split] = res
        if split == "train":
            gate_val = res["terrain_med_abs"]
        print(f"[{split}] sane n={res['n_sane']:,}: terrain med|d|/H {res['terrain_med_abs']} "
              f"(p90 {res['terrain_p90_abs']}, <0.3: {100*res['terrain_frac_lt_0.3']:.1f}%) | "
              f"plane med {res['plane_med_abs']} (p90 {res['plane_p90_abs']}) | "
              f"junk n={res['n_junk']:,} med {res['junk_terr_med_abs']} | "
              f"missing-join {res['n_missing_join']:,}")

    worst = sorted(((v.get("train", v.get("val", {})).get("terr_med_abs", 0), k)
                    for k, v in report["per_segment"].items()), reverse=True)[:10]
    report["worst_segments"] = [{"name": k, "med": m} for m, k in worst]
    report["gate_B"] = {"train_terrain_med_abs": gate_val, "pass": bool(gate_val is not None and gate_val < 0.3)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print("\nworst segments (sane med |d|/H):")
    for m, k in worst:
        print(f"  {m:8.3f}  {k}" + ("   [GATE-A FLAGGED]" if k in flagged else ""))
    if report["flagged_segments"]:
        print("Gate-A consistency-flagged segments:", json.dumps(report["flagged_segments"]))
    print(f"\nGATE B: {'PASS' if report['gate_B']['pass'] else 'FAIL'} "
          f"(train sane median |d|/H = {gate_val}, bar 0.3)  report -> {args.out}")
    return 0 if report["gate_B"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
