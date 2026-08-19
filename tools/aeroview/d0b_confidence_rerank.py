#!/usr/bin/env python
"""D0b: how much of the model's geometry can a BETTER RANKING recover, using only
information available at inference time?

D0 showed the model's geometry supports BEV@0.25 macro 42.41 under oracle ranking, but its own
confidence realises only 24.31. This script re-ranks the SAME predictions (geometry untouched) with
several inference-available scores and writes one predictions file per ranker for evaluation.

Rankers (none uses ground truth):
  current   the model's own score (baseline, = 24.31)
  depth     plausibility of the predicted depth: WildBox depth is scene-normalised so the prior is
            z ~ 1; rank by -|log z|
  dims      plausibility of predicted dimensions vs the per-class TRAIN median (log-ratio distance)
  size2d    larger 2D boxes are better resolved -> rank by box area
  combo     current x depth-plausibility x dims-plausibility (product of normalised terms)
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def train_dim_medians(train_json, cats):
    d = json.load(open(train_json))
    ds2c = {ds: i for i, ds in enumerate(cats)}
    acc = defaultdict(list)
    for a in d["annotations"]:
        acc[ds2c[int(a["category_id"])]].append([float(x) for x in a["dimensions"]])
    return {c: np.median(np.array(v), axis=0) for c, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", default="/mnt/d/ovmono3d-lift/wl6_init5sp_multiseed/seed0/eval/inference/iter_final/WildBox_val/instances_predictions.pth")
    ap.add_argument("--train", default="/mnt/d/3DBOX/papersubdata/WildBox_train_paper.json")
    ap.add_argument("--val", default="/mnt/d/3DBOX/papersubdata/WildBox_val_paper.json")
    ap.add_argument("--outdir", default="/mnt/d/aeroview")
    args = ap.parse_args()

    import torch, copy
    cats = sorted({int(c["id"]) for c in json.load(open(args.val))["categories"]})
    dmed = train_dim_medians(args.train, cats)
    raw = torch.load(args.preds, map_location="cpu", weights_only=False)

    feats = []
    for e in raw:
        for inst in e.get("instances", []):
            z = float(inst.get("depth", np.nan))
            dims = np.asarray(inst["dimensions"], float).reshape(-1)
            c = int(inst["category_id"])
            m = dmed.get(c)
            dd = float(np.abs(np.log(np.clip(dims, 1e-6, None) / np.clip(m, 1e-6, None))).mean()) if m is not None else np.nan
            b = inst["bbox"]
            feats.append((float(inst.get("score", 0.0)),
                          abs(np.log(max(z, 1e-6))) if np.isfinite(z) else 9.9,
                          dd, float(b[2]) * float(b[3])))
    F = np.array(feats, float)
    s0, dz, dd, area = F[:, 0], F[:, 1], F[:, 2], F[:, 3]
    print(f"{len(F)} predictions")
    print(f"  |log z|      med {np.nanmedian(dz):.3f}  p90 {np.nanpercentile(dz,90):.3f}")
    print(f"  dims log-dev med {np.nanmedian(dd):.3f}  p90 {np.nanpercentile(dd,90):.3f}")

    nrm = lambda v: (v - np.nanmin(v)) / max(np.nanmax(v) - np.nanmin(v), 1e-9)
    rankers = {
        "depth":  -dz,
        "dims":   -np.nan_to_num(dd, nan=9.9),
        "size2d": area,
        "combo":  nrm(s0) * nrm(-dz) * nrm(-np.nan_to_num(dd, nan=9.9)),
    }
    outs = {}
    for name, sc in rankers.items():
        sc = np.asarray(sc, float)
        sc = (sc - np.nanmin(sc)) / max(np.nanmax(sc) - np.nanmin(sc), 1e-9)   # -> [0,1], monotone only
        p = copy.deepcopy(raw); k = 0
        for e in p:
            for inst in e.get("instances", []):
                inst["score"] = float(sc[k]); k += 1
        o = str(Path(args.outdir) / f"rerank_{name}.pth")
        torch.save(p, o); outs[name] = o
        print(f"  wrote {name:<7} -> {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
