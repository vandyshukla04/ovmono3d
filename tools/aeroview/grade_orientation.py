"""Grade the predicted heading against held-out labels.  [CPU, ~1 min]

    python tools/aeroview/grade_orientation.py \
        --preds  <run>/eval/inference/iter_final/WildBox_val/instances_predictions.pth \
        --val    datasets/Omni3D/WildBox_val.json \
        --train  datasets/Omni3D/WildBox_train.json

WHY THIS SCRIPT AND NOT THE USUAL METRICS
-----------------------------------------
NHD, 3D IoU and BEV AP are EXACTLY invariant to a 180 degree flip about the box's vertical axis (proven:
min BEV IoU 1.000 over 3000 random boxes; flipping all 118,809 seed0 predictions gives a byte-identical
evaluator report). So none of them can measure heading AT ALL. This is the only grader that can.

THE FOUR RULES, each earned by a way of getting a believable wrong number
------------------------------------------------------------------------
1. **The floor is the TRAIN-TRANSFERRED best constant**, never 50%, never 45.7%, never a constant fitted on
   the evaluation set. Fitting the constant on the eval set is an ORACLE: on the gold locks it scores 75.1%
   where the honest transferred floor is 35.2%. This script fits the constant on TRAIN and applies it here.
2. **Never pool across species.** Rhino is 4,500 of the 7,460 val labels with circular concentration
   R=0.847, so a CONSTANT scores 98.3% on it and 87.9% on the pooled set. A pooled number is meaningless.
   Elephant and zebra carry the result; rhino cannot.
3. **Cluster by track, not by instance.** Heading is near-constant within a track, so 7,460 instances are
   ~208 independent observations. CIs are bootstrapped over TRACKS.
4. **Band-resolve by |sin alpha|.** Near end-on (|sin a| < 0.35) the flank bit is genuinely degenerate and
   nobody -- model or human -- should be trusted there. Report it separately, never averaged in silently.

MATCHING: predictions carry no track_id, so each labelled GT box is matched to its best-IoU prediction
above --score-thresh. Unmatched GT is reported as coverage, never silently dropped.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

USABLE_SIN = 0.35


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def sign_ok(pred, tgt):
    """The head/tail bit: is the predicted heading within 90 deg of the truth?"""
    return np.abs(wrap(np.asarray(pred) - np.asarray(tgt))) < np.pi / 2


def flank_ok(pred, tgt):
    """The left/right bit."""
    return np.sign(np.sin(np.asarray(pred))) == np.sign(np.sin(np.asarray(tgt)))


def best_constant(train_alpha):
    """The single alpha that maximises SIGN accuracy on the training labels. Transferred, never re-fitted."""
    grid = np.linspace(-np.pi, np.pi, 1441)
    return float(grid[int(np.argmax([sign_ok(np.full_like(train_alpha, c), train_alpha).mean()
                                     for c in grid]))])


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    i = (x2 - x1) * (y2 - y1)
    return i / (aw * ah + bw * bh - i)


def boot_ci(values, groups, n_boot=2000, seed=0):
    """95% CI of a mean, resampling GROUPS (tracks) not instances -- see rule 3."""
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return (float("nan"), float("nan"))
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        means.append(values[sel].mean())
    return tuple(np.percentile(means, [2.5, 97.5]) * 100)


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True, help="labelled val json")
    ap.add_argument("--train", type=Path, required=True, help="labelled train json (for the constant floor)")
    ap.add_argument("--score-thresh", type=float, default=0.25)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    args = ap.parse_args()

    # ---- the floor, fitted on TRAIN and transferred (rule 1) --------------------------------------
    tr = json.loads(args.train.read_text())
    tr_alpha = np.array([a["heading_alpha"] for a in tr["annotations"] if a.get("heading_valid", 0)])
    const = best_constant(tr_alpha)
    print(f"train labels {len(tr_alpha):,}  ->  best-constant alpha = {math.degrees(const):+.1f} deg "
          f"(fitted on TRAIN, transferred; never re-fitted on val)")

    # ---- val GT ----------------------------------------------------------------------------------
    va = json.loads(args.val.read_text())
    images = {im["id"]: im for im in va["images"]}
    gt_by_img = defaultdict(list)
    for a in va["annotations"]:
        if a.get("heading_valid", 0):
            gt_by_img[a["image_id"]].append(a)
    n_gt = sum(len(v) for v in gt_by_img.values())
    print(f"val labelled GT: {n_gt:,} across {len(gt_by_img):,} images")

    # ---- predictions -----------------------------------------------------------------------------
    raw = torch.load(args.preds, map_location="cpu")
    pred_by_img = defaultdict(list)
    n_alpha = 0
    for r in raw:
        for inst in r["instances"]:
            if inst.get("score", 0) < args.score_thresh:
                continue
            if "alpha" in inst:
                n_alpha += 1
            pred_by_img[r["image_id"]].append(inst)
    if n_alpha == 0:
        raise SystemExit("No prediction carries an 'alpha' field -- the model predates the orientation head, "
                         "or omni3d_evaluation.instances_to_coco_json dropped it.")
    print(f"predictions above score {args.score_thresh}: "
          f"{sum(len(v) for v in pred_by_img.values()):,} ({n_alpha:,} with alpha)")

    # ---- match, then collect ---------------------------------------------------------------------
    rows = []
    for iid, gts in gt_by_img.items():
        preds = pred_by_img.get(iid, [])
        for g in gts:
            best, best_iou = None, 0.0
            for p in preds:
                if "alpha" not in p:
                    continue
                v = iou_xywh(g["bbox"], p["bbox"])
                if v > best_iou:
                    best, best_iou = p, v
            if best is None or best_iou < args.iou_thresh:
                continue
            vid = images[iid]["file_path"].replace("\\", "/").split("/")[-3]
            seg = images[iid]["file_path"].replace("\\", "/").split("/")[-2]
            rows.append(dict(species=g["category_name"], track=(vid, seg, g["track_id"]),
                             tgt=float(g["heading_alpha"]), pred=float(best["alpha"])))
    if not rows:
        raise SystemExit("nothing matched -- check --iou-thresh / --score-thresh")
    print(f"matched {len(rows):,}/{n_gt:,} labelled GT ({100*len(rows)/n_gt:.1f}% coverage) "
          f"at IoU>={args.iou_thresh}\n")

    tgt = np.array([r["tgt"] for r in rows])
    pred = np.array([r["pred"] for r in rows])
    sp = np.array([r["species"] for r in rows])
    trk = np.array([f"{t[0]}|{t[1]}|{t[2]}" for t in (r["track"] for r in rows)])
    usable = np.abs(np.sin(tgt)) >= USABLE_SIN

    # ---- report, per species, never pooled (rule 2) ----------------------------------------------
    hdr = (f"{'species':14s} {'n':>6s} {'trk':>5s} | {'sign%':>6s} {'floor':>6s} {'delta':>7s} "
           f"{'95% CI':>16s} | {'flank%':>6s} {'floor':>6s} {'delta':>7s}")
    print(hdr); print("-" * len(hdr))

    def line(name, m):
        if m.sum() < 5:
            return
        s = sign_ok(pred[m], tgt[m]); f = flank_ok(pred[m], tgt[m])
        s0 = sign_ok(np.full(m.sum(), const), tgt[m]).mean() * 100
        f0 = max(np.mean(np.sin(tgt[m]) > 0), np.mean(np.sin(tgt[m]) < 0)) * 100
        lo, hi = boot_ci(s, trk[m])
        beats = "" if np.isnan(lo) else ("  <-- BEATS FLOOR" if lo > s0 else "")
        print(f"{name:14s} {m.sum():6d} {len(np.unique(trk[m])):5d} | "
              f"{s.mean()*100:6.1f} {s0:6.1f} {s.mean()*100-s0:+7.1f} "
              f"[{lo:5.1f},{hi:5.1f}] | {f.mean()*100:6.1f} {f0:6.1f} {f.mean()*100-f0:+7.1f}{beats}")

    for s in sorted(set(sp)):
        line(s, sp == s)
    print("-" * len(hdr))
    line("ALL (do not", np.ones(len(tgt), bool))
    print("   ^ pooled row is diagnostic ONLY -- rhino dominates and saturates it. Do not quote it.\n")

    # ---- band-resolved (rule 4) ------------------------------------------------------------------
    print("band-resolved by |sin alpha| (the flank bit is degenerate near end-on):")
    for name, m in (("usable  |sin|>=0.35", usable), ("end-on  |sin| <0.35", ~usable)):
        if m.sum() < 5:
            continue
        s = sign_ok(pred[m], tgt[m]); f = flank_ok(pred[m], tgt[m])
        s0 = sign_ok(np.full(m.sum(), const), tgt[m]).mean() * 100
        print(f"  {name:22s} n={m.sum():5d}  sign {s.mean()*100:5.1f}% (floor {s0:5.1f})  "
              f"flank {f.mean()*100:5.1f}%")

    # ---- degeneracy guards -----------------------------------------------------------------------
    R = float(np.abs(np.mean(np.exp(1j * pred))))
    print(f"\nconcentration R of the PREDICTED alpha = {R:.3f}")
    if R > 0.9:
        print("  ** WARNING: the head has collapsed to a near-constant output. Any accuracy it shows is the")
        print("     constant floor, not learning. Compare the delta column, never the raw accuracy. **")
    print(f"predicted-alpha circular std = {math.degrees(math.sqrt(max(-2*math.log(max(R,1e-9)),0))):.1f} deg "
          f"(GT: {math.degrees(math.sqrt(max(-2*math.log(max(float(np.abs(np.mean(np.exp(1j*tgt)))),1e-9)),0))):.1f} deg)")
    print("\nREMINDER: a gap under ~5 points is not resolvable on this data (design effect ~7x).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
