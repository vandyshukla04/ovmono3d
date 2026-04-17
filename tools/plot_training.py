#!/usr/bin/env python
"""Plot training curves from detectron2's metrics.json.

Usage:
    python tools/plot_training.py output/wildbox_wl_finetune/metrics.json
    python tools/plot_training.py ... --list-keys    # just print all keys
"""
import argparse
import json
import re
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


def all_keys(rows):
    s = set()
    for r in rows:
        s.update(r.keys())
    return sorted(s)


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


def keys_matching(rows, regex):
    rx = re.compile(regex)
    keys = all_keys(rows)
    return [k for k in keys if rx.search(k)]


def plot_group(ax, rows, keys, title, y_label="", marker=None):
    plotted = 0
    for key in keys:
        xs, ys = series(rows, key)
        if ys:
            label = key.split("/")[-1] if "/" in key else key
            ax.plot(xs, ys, label=label, alpha=0.8, linewidth=1,
                    marker=marker, markersize=4 if marker else 0)
            plotted += 1
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("iter")
    if y_label:
        ax.set_ylabel(y_label)
    if plotted:
        ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("metrics", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--list-keys", action="store_true",
                   help="Just print all keys present in metrics.json and exit.")
    args = p.parse_args()
    rows = load_metrics(args.metrics)
    print(f"loaded {len(rows)} log rows")
    if not rows:
        return

    if args.list_keys:
        print("\nAll keys observed:")
        for k in all_keys(rows):
            print(f"  {k}")
        return

    out_path = args.out or args.metrics.with_name("training_curves.png")

    # Auto-discover keys by regex on actual names observed in the file.
    loss_2d = keys_matching(rows, r"^loss_(?!.*3d)(?!.*box3d)|total_loss")
    loss_3d = keys_matching(rows, r"loss.*(3d|cube|xy|_z$|dims|pose|joint)")
    ap2d_final = keys_matching(rows, r"(mode=2D|2D)/?(AP|AP50|AP75)$")
    ap3d_final = keys_matching(rows, r"(mode=3D|3D)/?(AP|AP15|AP25|AP50)$")
    if not ap2d_final:
        ap2d_final = keys_matching(rows, r"^bbox_2D_?AP")
    if not ap3d_final:
        ap3d_final = keys_matching(rows, r"^bbox_3D_?AP")
    nhd_keys = keys_matching(rows, r"(nhd|NHD)")
    lr_keys = keys_matching(rows, r"^lr$")

    print("Auto-discovered keys:")
    print(f"  2D losses ({len(loss_2d)}):", loss_2d[:10])
    print(f"  3D losses ({len(loss_3d)}):", loss_3d[:10])
    print(f"  2D AP    ({len(ap2d_final)}):", ap2d_final[:10])
    print(f"  3D AP    ({len(ap3d_final)}):", ap3d_final[:10])
    print(f"  NHD      ({len(nhd_keys)}):", nhd_keys[:10])

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    plot_group(axes[0, 0], rows, loss_2d, "2D losses")
    plot_group(axes[0, 1], rows, loss_3d, "3D losses")
    plot_group(axes[0, 2], rows, lr_keys, "learning rate")
    plot_group(axes[1, 0], rows, ap2d_final, "2D AP (eval)", y_label="AP", marker="o")
    plot_group(axes[1, 1], rows, ap3d_final, "3D AP (eval)", y_label="AP", marker="o")
    plot_group(axes[1, 2], rows, nhd_keys, "NHD (lower=better, eval)", marker="o")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
