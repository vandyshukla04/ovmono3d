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
    # Detectron2 uses a variety of naming conventions; match them all.
    all_k = all_keys(rows)

    # 2D losses: anything that looks like a 2D-detection loss.
    #   total_loss, loss_cls, loss_box_reg, loss_rpn_cls, loss_rpn_loc,
    #   BoxHead/loss_cls, BoxHead/loss_box_reg, ...
    loss_2d = [k for k in all_k if (
        k == "total_loss"
        or re.search(r"^loss_(cls|box_reg|rpn)", k)
        or re.search(r"BoxHead/loss_", k)
        or re.search(r"rpn/loss_", k, re.I)
    )]

    # 3D losses: cube head components.
    loss_3d = [k for k in all_k if re.search(
        r"(^|/)loss_(xy|z|dims|pose|joint|3d)\b|Cube/|total_3D_loss", k, re.I
    )]

    # LR
    lr_keys = [k for k in all_k if k == "lr"]

    # Eval AP. detectron2 emits one per metric, typically under bbox/AP,
    # bbox/AP50, bbox/AP75 (2D), and custom keys for 3D / per-class.
    def _ap_keys(mode_hint):
        hits = []
        for k in all_k:
            kl = k.lower()
            if mode_hint == "2d":
                if re.search(r"(^|[/_ ])(bbox[_-]?2?d?|2d)[/_ ]?(ap|ap50|ap75|ap95)\b", kl):
                    hits.append(k)
                elif re.match(r"^(ap|ap50|ap75)$", kl):
                    hits.append(k)
            else:  # 3d
                if re.search(r"(3d)[/_ ]?(ap|ap15|ap25|ap50)\b", kl):
                    hits.append(k)
                elif re.search(r"bbox_3d", kl):
                    hits.append(k)
        return sorted(set(hits))

    ap2d_final = _ap_keys("2d")
    ap3d_final = _ap_keys("3d")
    nhd_keys = [k for k in all_k if re.search(r"nhd", k, re.I)]

    print(f"\nAll metric keys in file ({len(all_k)} total):")
    for k in all_k:
        print(f"  {k}")

    print("\nAuto-discovered keys:")
    print(f"  2D losses ({len(loss_2d)}):", loss_2d[:15])
    print(f"  3D losses ({len(loss_3d)}):", loss_3d[:15])
    print(f"  2D AP    ({len(ap2d_final)}):", ap2d_final[:15])
    print(f"  3D AP    ({len(ap3d_final)}):", ap3d_final[:15])
    print(f"  NHD      ({len(nhd_keys)}):", nhd_keys[:15])

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
