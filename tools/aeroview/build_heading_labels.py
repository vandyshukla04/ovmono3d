"""Stamp `heading_alpha` / `heading_valid` onto every WildBox annotation.  [CPU, minutes]

    # normal use -- the 0.15 MB label bundle ships with the repo, so no data transfer is needed
    python tools/aeroview/build_heading_labels.py \
        --train datasets/Omni3D/WildBox_train.json \
        --val   datasets/Omni3D/WildBox_val.json \
        --out-dir datasets/Omni3D

    # only to regenerate the bundle from the full 508 MB crop files
    python tools/aeroview/build_heading_labels.py --heading-dir /mnt/d/detany3d/heading ...

WHAT THIS IS FOR
----------------
The detector has no orientation supervision at all: cubercnn's `loss_pose` is a chamfer over the cuboid's
8-corner set (`roi_heads.py:624`), and that set is EXACTLY invariant under a 180 deg flip about the box's own
vertical axis -- so the existing loss cannot express a head/tail error even in principle. This script produces
the target for a separate, MASKED orientation head.

THE THREE TRAPS THIS SCRIPT EXISTS TO AVOID (each one silently produces a believable wrong number)
--------------------------------------------------------------------------------------------------
1. **The join key.** `crops.npz['frame']` is the SEGMENT-LOCAL index (0..199) while the released `file_path`
   carries the VIDEO frame number, so joining on `frame` matches ~0.1% and looks like "the labels don't join".
   The unique key is `(video, seg_name, track_id, image_name)` -- verified collision-free across all 237,505
   annotations, matching 100.00% of all three crop files.

2. **The target array.** Use `Y` = [cos alpha, sin alpha], which is the MOTION-derived allocentric heading and
   is already on disk. Do NOT use `face_alpha[i, y_face]`: that is the heading SNAPPED to the nearest of the
   4 PCA box faces, and it flips the FLANK bit on 9.11% of `crops` / 11.19% of `crops_stand`. It flips 0.00%
   on `crops_human`, which is exactly why the error is invisible if you only check the gold set.
   (The SIGN bit is unaffected either way -- faces are 90 deg apart, so the nearest is always within 45 deg.)

3. **The gold overlap.** Both human-lock videos are 100% inside WildBox_train, and 142 exact keys on 12 of the
   66 gold tracks (1,105 gold instances = 19.9%) also appear in the motion/standing label pool. Training on
   them makes the headline number uninterpretable, so they are subtracted here, at the source.

WHAT IT EMITS, per annotation (every annotation is stamped, so `datasets.py:404`'s `if key in anno` can never
silently drop the field and the mapper can never KeyError):
    heading_alpha        float   allocentric angle in radians; 0.0 where invalid
    heading_valid        int     1 = carries supervision, 0 = masked out of the loss
    heading_long_axis    int     1 = the labelled head sits on the box's LONG axis (see --require-long-axis)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

# The two videos carrying the 5,542 human face locks. They are the (secondary) grading set, so no label
# from them may ever carry a gradient -- see trap 3.
LOCK_VIDEOS = {"DJI_20250802085130_0007_V", "DJI_20250802085520_0008_V"}


def _load_bundle(path: Path):
    """The stripped label bundle shipped with the repo: 0.15 MB instead of 508 MB of crop JPEGs.

    The three crops_*.npz files are ~508 MB, of which the fields this join actually needs are 8 MB --
    the rest is encoded crop imagery used only by the DINOv3 template. So the labels travel with the
    code and the cluster needs no separate data transfer.
    """
    with np.load(path, allow_pickle=True) as z:
        out = {}
        for tag in ("motion", "standing", "gold"):
            keys = list(zip(z[f"{tag}_video"].astype(str), z[f"{tag}_seg"].astype(str),
                            z[f"{tag}_track"].astype(int).tolist(), z[f"{tag}_image"].astype(str)))
            out[tag] = dict(zip(keys, z[f"{tag}_alpha"].astype(float).tolist()))
    return out["motion"], out["standing"], out["gold"]


def _load_crops(path: Path):
    """(key -> alpha) for one crop file. Key = (video, seg_name, track_id, image_name)."""
    with np.load(path, allow_pickle=True) as z:
        alpha = np.arctan2(z["Y"][:, 1], z["Y"][:, 0]).astype(np.float64)  # trap 2
        tid = np.array([int(t.split("::")[-1]) for t in z["track"]])
        keys = list(zip(z["video"].astype(str), z["seg_name"].astype(str),
                        tid.tolist(), z["image_name"].astype(str)))
    return dict(zip(keys, alpha.tolist()))


def _long_axis_flags(heading_dir: Path) -> dict:
    """labels.npz carries `on_long_axis`; a head on a SHORT box face is anatomically impossible for a
    quadruped and flags a bad PCA box. Keyed the same way so it can be joined or split at report time."""
    p = heading_dir / "labels.npz"
    if not p.is_file():
        return {}
    with np.load(p, allow_pickle=True) as z:
        if "on_long_axis" not in z.files:
            return {}
        # labels.npz has no image_name; key on (video, seg, track, frame-index) is not joinable to the
        # released file_path, so this is returned per (video, seg, track) as a track-level flag.
        tid = np.array([int(t) for t in z["track"]])
        keys = list(zip(z["video"].astype(str), z["seg"].astype(str), tid.tolist()))
        return dict(zip(keys, z["on_long_axis"].tolist()))


def ann_key(image: dict, anno: dict):
    """The unique join key, built from the LAST THREE path components -- never the group prefix, which
    differs between the relative-path paper jsons and the absolute-path cluster jsons."""
    parts = image["file_path"].replace("\\", "/").split("/")
    video, seg, image_name = parts[-3], parts[-2], parts[-1]
    return (video, seg, int(anno["track_id"]), image_name)


def stamp(json_path: Path, labels: dict, out_path: Path, *, split: str, verbose=True):
    d = json.loads(json_path.read_text())
    by_id = {im["id"]: im for im in d["images"]}

    n_valid = 0
    per_class = Counter()
    per_video = Counter()
    for anno in d["annotations"]:
        im = by_id[anno["image_id"]]
        a = labels.get(ann_key(im, anno))
        anno["heading_alpha"] = float(a) if a is not None else 0.0
        anno["heading_valid"] = int(a is not None)
        anno["heading_long_axis"] = 1  # populated below only where a track-level flag exists
        if a is not None:
            n_valid += 1
            per_class[anno["category_name"]] += 1
            per_video[im["file_path"].replace("\\", "/").split("/")[-3]] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(d))
    if verbose:
        print(f"\n[{split}] {json_path.name} -> {out_path}")
        print(f"  annotations {len(d['annotations']):,}   labelled {n_valid:,} "
              f"({100*n_valid/len(d['annotations']):.2f}%)")
        for c, n in per_class.most_common():
            print(f"    {c:16s} {n:6,}")
        leaked = sorted(set(per_video) & LOCK_VIDEOS)
        print(f"  lock videos carrying labels: {leaked if leaked else 'NONE'}")
    return n_valid, per_class, set(per_video)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path,
                    default=Path(__file__).resolve().parent / "data" / "heading_labels_min.npz",
                    help="stripped label bundle shipped with the repo (default). Use --heading-dir to "
                         "rebuild from the full crops_*.npz instead.")
    ap.add_argument("--heading-dir", type=Path, default=None,
                    help="directory holding crops.npz / crops_stand.npz / crops_human.npz. Only needed "
                         "to regenerate labels from source; the bundle is normally enough.")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--expect-train", type=int, default=9887)
    ap.add_argument("--expect-val", type=int, default=7460)
    ap.add_argument("--no-assert", action="store_true", help="report counts but do not enforce them")
    args = ap.parse_args()

    if args.heading_dir is not None:
        motion = _load_crops(args.heading_dir / "crops.npz")
        stand = _load_crops(args.heading_dir / "crops_stand.npz")
        gold = _load_crops(args.heading_dir / "crops_human.npz")
        print(f"source: {args.heading_dir}")
    else:
        if not args.bundle.is_file():
            raise SystemExit(f"bundle not found: {args.bundle}\n"
                             f"pass --heading-dir <dir with crops*.npz> to rebuild from source.")
        motion, stand, gold = _load_bundle(args.bundle)
        print(f"source: {args.bundle} (stripped bundle)")
    print(f"loaded  motion {len(motion):,}  standing {len(stand):,}  gold {len(gold):,}")

    # union, motion preferred over standing where a key appears in both
    pool = dict(stand)
    pool.update(motion)
    print(f"union (motion preferred)          {len(pool):,}")

    # ---- trap 3: subtract the gold keys AND everything from the two lock videos -------------------
    n_gold_overlap = sum(k in gold for k in pool)
    train_labels = {k: v for k, v in pool.items() if k not in gold and k[0] not in LOCK_VIDEOS}
    print(f"minus {n_gold_overlap} gold-overlapping keys and all lock-video keys -> {len(train_labels):,}")

    # val json gets the FULL pool: it never carries a gradient (DATASETS.TRAIN=('WildBox_train',)),
    # so these become the held-out multi-species orientation grader.
    n_tr, cls_tr, vids_tr = stamp(args.train, train_labels, args.out_dir / args.train.name, split="train")
    n_va, cls_va, vids_va = stamp(args.val, pool, args.out_dir / args.val.name, split="val")

    print("\n=== ASSERTIONS ===")
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: {got:,} (expected {want:,})")

    check("train labelled", n_tr, args.expect_train)
    check("val labelled", n_va, args.expect_val)
    check("gazelle labels (must be 0)", cls_tr.get("gazelle", 0) + cls_va.get("gazelle", 0), 0)

    leaked = sorted(vids_tr & LOCK_VIDEOS)
    good = not leaked
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] no lock video carries a TRAIN label: {leaked or 'NONE'}")

    if not ok and not args.no_assert:
        raise SystemExit("\nASSERTIONS FAILED -- do not train on this output.")
    print("\nall assertions passed." if ok else "\n(assertions failed; --no-assert set)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
