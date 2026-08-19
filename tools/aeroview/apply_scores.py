#!/usr/bin/env python
"""Rebuild a predictions .pth from the base predictions + a saved score vector.

Re-ranking experiments change ONLY the `score` field, so storing a full 74 MB copy per variant is
pure waste. We keep the scores as a ~475 KB .npy and materialise the .pth on demand for evaluation.

    python tools/aeroview/apply_scores.py --scores /mnt/d/aeroview/rerank_combo_scores.npy --out /tmp/x.pth
"""
import argparse, numpy as np, torch
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--base", default="/mnt/d/ovmono3d-lift/wl6_init5sp_multiseed/seed0/eval/inference/iter_final/WildBox_val/instances_predictions.pth")
ap.add_argument("--scores", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
d = torch.load(a.base, map_location="cpu", weights_only=False); s = np.load(a.scores); k = 0
for e in d:
    for i in e.get("instances", []):
        i["score"] = float(s[k]); k += 1
assert k == len(s), f"score count {len(s)} != predictions {k}"
torch.save(d, a.out); print(f"wrote {a.out} ({k} preds)")
