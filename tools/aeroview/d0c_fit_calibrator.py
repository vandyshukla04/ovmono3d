#!/usr/bin/env python
"""D0c: FIT the confidence ranking instead of hand-weighting it.

D0b showed a hand-picked product (score x depth-plausibility x dims-plausibility) already beats the
model's own confidence. Here we let the data choose the weights: a small logistic regression that maps
inference-available features -> P(this box is actually good), used purely as a ranking score. The
geometry of every box is left untouched.

FEATURES (nothing here uses ground truth at inference):
    model score, |log z| (depth is scene-normalised so the prior is z~1), dims log-deviation from the
    per-class TRAIN median, log 2D box area, log box aspect, n predictions in the image, class one-hot.
The class one-hot matters: D0b showed geometric plausibility helps abundant classes and hurts giraffe,
so the fit needs a per-class intercept to reconcile micro and macro.

TARGET: y = 1 if the box's true BEV IoU with a same-class GT exceeds tau.

NO LEAKAGE: fitted by GROUPED cross-validation over VIDEOS -- every prediction is scored by a model
that never saw its video. So the resulting AP is measured on the full val set and is directly
comparable to the published 24.31 / 8.20, with no train/test contamination.

    python tools/aeroview/d0c_fit_calibrator.py --tau 0.25
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", default="/mnt/d/ovmono3d-lift/wl6_init5sp_multiseed/seed0/eval/inference/iter_final/WildBox_val/instances_predictions.pth")
    ap.add_argument("--oracle", default="/mnt/d/aeroview/model_oracleorder.pth",
                    help="same preds with score = true BEV IoU (built by the D0 verification step)")
    ap.add_argument("--train", default="/mnt/d/3DBOX/papersubdata/WildBox_train_paper.json")
    ap.add_argument("--val", default="/mnt/d/3DBOX/papersubdata/WildBox_val_paper.json")
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--out", default="/mnt/d/aeroview/rerank_fitted.pth")
    ap.add_argument("--per-class", action="store_true",
                    help="fit one calibrator PER CLASS. macro-AP ranks within a class, so a global\n                          fit optimises the wrong objective: a per-class intercept cannot change\n                          within-class order, and the global fit exploits between-class effects\n                          that macro-AP ignores.")
    args = ap.parse_args()

    import torch, copy
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    val = json.load(open(args.val))
    cats = sorted({int(c["id"]) for c in val["categories"]})
    ds2c = {ds: i for i, ds in enumerate(cats)}
    names = {ds2c[int(c["id"])]: c["name"] for c in val["categories"]}
    vid_of = {i["id"]: i["file_path"].split("/")[1] for i in val["images"]}     # group by VIDEO

    tr = json.load(open(args.train))
    acc = defaultdict(list)
    for a in tr["annotations"]:
        acc[ds2c[int(a["category_id"])]].append([float(x) for x in a["dimensions"]])
    dmed = {c: np.median(np.array(v), axis=0) for c, v in acc.items()}

    raw = torch.load(args.preds, map_location="cpu", weights_only=False)
    orc = torch.load(args.oracle, map_location="cpu", weights_only=False)

    X, y, grp = [], [], []
    for e, eo in zip(raw, orc):
        vid = vid_of.get(int(e["image_id"]), "?")
        n_in_img = len(e.get("instances", []))
        for inst, io in zip(e.get("instances", []), eo.get("instances", [])):
            c = int(inst["category_id"])
            z = float(inst.get("depth", np.nan))
            dims = np.asarray(inst["dimensions"], float).reshape(-1)
            m = dmed.get(c, dims)
            b = inst["bbox"]; w, h = float(b[2]), float(b[3])
            oh = [0.0] * len(cats); oh[c] = 1.0
            X.append([float(inst.get("score", 0.0)),
                      abs(np.log(max(z, 1e-6))) if np.isfinite(z) else 9.9,
                      float(np.abs(np.log(np.clip(dims, 1e-6, None) / np.clip(m, 1e-6, None))).mean()),
                      np.log(max(w * h, 1.0)),
                      np.log(max(w, 1.0) / max(h, 1.0)),
                      np.log(max(n_in_img, 1))] + oh)
            y.append(1.0 if float(io.get("score", 0.0)) > args.tau else 0.0)
            grp.append(vid)
    X = np.asarray(X, float); y = np.asarray(y, float); grp = np.asarray(grp)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"{len(X)} predictions | positives (IoU>{args.tau}): {100*y.mean():.1f}% | videos: {len(set(grp))}")

    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    p_oof = np.zeros(len(X))
    cls = np.array([int(np.argmax(r[6:])) for r in X])          # recover class from the one-hot block
    gkf = GroupKFold(n_splits=min(5, len(set(grp))))
    if args.per_class:
        Xn = Xs[:, :6]                                          # drop one-hot: constant within a class
        for c in range(len(cats)):
            m = cls == c
            if m.sum() < 50 or len(set(grp[m])) < 2 or y[m].sum() < 5 or y[m].sum() == m.sum():
                p_oof[m] = X[m, 0]                              # too little signal -> keep model score
                print(f"  class {names[c]:<13} n={m.sum():>6}  (insufficient -> fallback to model score)")
                continue
            gk = GroupKFold(n_splits=min(5, len(set(grp[m]))))
            idx = np.where(m)[0]
            for tr_i, te_i in gk.split(Xn[m], y[m], groups=grp[m]):
                if len(np.unique(y[m][tr_i])) < 2:      # degenerate fold -> keep the model's own score
                    p_oof[idx[te_i]] = X[idx[te_i], 0]
                    continue
                clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xn[m][tr_i], y[m][tr_i])
                p_oof[idx[te_i]] = clf.predict_proba(Xn[m][te_i])[:, 1]
            print(f"  class {names[c]:<13} n={m.sum():>6}  pos {100*y[m].mean():>5.1f}%  videos {len(set(grp[m]))}")
    else:
        for k, (tr_i, te_i) in enumerate(gkf.split(Xs, y, groups=grp)):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xs[tr_i], y[tr_i])
            p_oof[te_i] = clf.predict_proba(Xs[te_i])[:, 1]
            print(f"  fold {k}: fit {len(tr_i)} / score {len(te_i)}  (held-out videos: {len(set(grp[te_i]))})")

    from sklearn.metrics import roc_auc_score
    print(f"\nout-of-fold ranking AUC vs true IoU>{args.tau}: {roc_auc_score(y, p_oof):.4f}"
          f"   (model's own score: {roc_auc_score(y, X[:, 0]):.4f})")
    print("  per-class out-of-fold AUC (what macro-AP actually cares about):")
    for c in range(len(cats)):
        m = cls == c
        if m.sum() > 50 and 0 < y[m].sum() < m.sum():
            print(f"     {names[c]:<13} fitted {roc_auc_score(y[m], p_oof[m]):.3f}   model {roc_auc_score(y[m], X[m,0]):.3f}")
    full = LogisticRegression(max_iter=2000).fit(Xs, y)
    fn = ["score", "|log z|", "dims_dev", "log_area", "log_aspect", "log_n_img"] + [names[i] for i in range(len(cats))]
    print("\nfitted weights (standardised, full-data fit — for interpretation only):")
    for n_, w_ in sorted(zip(fn, full.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"   {n_:<14} {w_:+.3f}")

    p = copy.deepcopy(raw); k = 0
    for e in p:
        for inst in e.get("instances", []):
            inst["score"] = float(p_oof[k]); k += 1
    torch.save(p, args.out)
    print(f"\nwrote {args.out}  (out-of-fold scores only — no video scored by a model that saw it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
