#!/usr/bin/env python
"""Plot training curves from detectron2's metrics.json.

Usage:
    python tools/plot_training.py output/wildbox_wl_finetune/metrics.json
    # -> writes <dir>/training_curves.png next to the input file.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def series(rows, key):
    xs, ys = [], []
    for r in rows:
        if key in r and "iteration" in r:
            try:
                ys.append(float(r[key]))
                xs.append(int(r["iteration"]))
            except (TypeError, ValueError):
                pass
    return xs, ys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("metrics", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    rows = load_metrics(args.metrics)
    print(f"loaded {len(rows)} log rows")
    if not rows:
        return

    out_path = args.out or args.metrics.with_name("training_curves.png")

    # Discover which eval keys are present for any dataset name.
    eval_keys = {
        "AP2D": [k for k in rows[-1] if k.endswith("mode=2D/AP") or "/AP" == k[-3:]],
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # 1. Total loss
    ax = axes[0, 0]
    for key, label in [("total_loss", "total"),
                       ("loss_cls", "cls"),
                       ("loss_box_reg", "box2d"),
                       ("loss_rpn_cls", "rpn_cls"),
                       ("loss_rpn_loc", "rpn_loc")]:
        xs, ys = series(rows, key)
        if ys:
            ax.plot(xs, ys, label=label, alpha=0.7, linewidth=0.8)
    ax.set_title("2D losses")
    ax.set_xlabel("iter")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2. 3D cube head losses
    ax = axes[0, 1]
    for key in ("loss_xy", "loss_z", "loss_dims", "loss_pose", "loss_joint"):
        xs, ys = series(rows, key)
        if ys:
            ax.plot(xs, ys, label=key.replace("loss_", ""), alpha=0.7, linewidth=0.8)
    ax.set_title("3D cube-head losses")
    ax.set_xlabel("iter")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. LR
    ax = axes[0, 2]
    xs, ys = series(rows, "lr")
    if ys:
        ax.plot(xs, ys, color="black")
    ax.set_title("learning rate")
    ax.set_xlabel("iter")
    ax.grid(alpha=0.3)

    # 4. 2D AP @ eval points
    ax = axes[1, 0]
    for k in rows[0].keys() if rows else []:
        pass
    plotted = 0
    for r in rows:
        for k in r:
            if "mode=2D" in k and k.endswith(("AP", "AP50", "AP75")):
                pass
    # Collect all eval keys across rows
    ap2d_keys = sorted({k for r in rows for k in r
                        if "mode=2D" in k and k.split("/")[-1] in ("AP", "AP50", "AP75")})
    for key in ap2d_keys:
        xs, ys = series(rows, key)
        if ys:
            label = key.split("/")[-1]
            ax.plot(xs, ys, marker="o", label=label)
            plotted += 1
    ax.set_title("2D AP @ eval")
    ax.set_xlabel("iter")
    ax.set_ylabel("AP")
    if plotted:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 5. 3D AP @ eval points
    ax = axes[1, 1]
    ap3d_keys = sorted({k for r in rows for k in r
                        if "mode=3D" in k and k.split("/")[-1] in ("AP", "AP15", "AP25", "AP50")})
    plotted = 0
    for key in ap3d_keys:
        xs, ys = series(rows, key)
        if ys:
            label = key.split("/")[-1]
            ax.plot(xs, ys, marker="o", label=label)
            plotted += 1
    ax.set_title("3D AP @ eval")
    ax.set_xlabel("iter")
    ax.set_ylabel("AP")
    if plotted:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 6. NHD decomposition @ eval
    ax = axes[1, 2]
    nhd_keys = sorted({k for r in rows for k in r
                       if "nhd" in k.lower() or "NHD" in k})
    plotted = 0
    for key in nhd_keys[:6]:
        xs, ys = series(rows, key)
        if ys:
            label = key.split("/")[-1]
            ax.plot(xs, ys, marker="o", label=label, alpha=0.8)
            plotted += 1
    ax.set_title("NHD (lower = better)")
    ax.set_xlabel("iter")
    if plotted:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
