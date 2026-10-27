"""Build WildBox v1.1 — the corrected-source re-export.  [LOCAL CPU; hours (SAM3 mask reads)]

    python tools/release/build_v1_1.py --out-dir /mnt/d/wildbox_v1.1 [--smoke N] [--skip-build]

What it does, auditable step by step:
 1. Derives the EXACT v1.0 split (segment membership per split) from the frozen v1.0 paper jsons,
    so train/val membership is preserved bit-for-bit. seg4 (the v1.0 packaging gap) is in train.
 2. Assembles --source entries from wildbox_alias_map.json (shoot -> category) with the v1.0
    category-id mapping read from the v1.0 json itself.
 3. Runs tools/prepare_wildbox_dataset.py with --prefer-corrected --dataset-version 1.1
    (the ONLY semantic change vs the v1.0 build is the annotation source).
 4. Rewrites absolute file_paths to the release-relative '<alias>/<video>/<seg>/<frame>' form and
    re-dumps compact.
 5. Gates: (a) DIMS-CONSTANCY GATE -- every corrected-provenance track must have within-track dims
    cv == 0 (the v1.0 bug's signature, now an automated release gate); (b) counts + per-class
    deltas vs v1.0 (reported, not asserted -- corrected tracks legitimately change membership);
    (c) provenance summary (corrected vs raw segments).
Then: fingerprint with tools/release/fingerprint_wildbox.py, render QA videos, regen kitti, pack.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tadetr.data.wildbox_paths import parse_label_path  # noqa: E402

V10_TRAIN = Path("/mnt/d/3DBOX/papersubdata/WildBox_train_paper.json")
V10_VAL = Path("/mnt/d/3DBOX/papersubdata/WildBox_val_paper.json")
ALIAS_MAP = Path("/mnt/d/3DBOX/wildbox_alias_map.json")
DATA_ROOT = Path("/mnt/d/3DBOX/Data/WildBox/data")
PROCESSED = "WildBox_sam3-vggtv1_processed"


def derive_v10(out_dir: Path):
    """Split membership + category-id map + per-split counts from the frozen v1.0 jsons."""
    amap = json.loads(ALIAS_MAP.read_text())["datasets"]
    alias_to = {rec["alias"]: (shoot, rec["category"]) for shoot, rec in amap.items()}
    split_ids = {}
    cat_ids = None
    v10_counts = {}
    for split, path in (("train", V10_TRAIN), ("val", V10_VAL)):
        d = json.loads(path.read_text())
        if cat_ids is None:
            cat_ids = {c["name"]: c["id"] for c in d["categories"]}
        segs = set()
        for im in d["images"]:
            group, video, seg, _ = parse_label_path(im["file_path"])
            segs.add((group, video, seg))
        ids = []
        for group, video, seg in sorted(segs):
            shoot, category = alias_to[group]
            ids.append(f"{category}/{video}/{seg}")
        split_ids[split] = ids
        v10_counts[split] = {
            "images": len(d["images"]), "annotations": len(d["annotations"]),
            "per_class": dict(Counter(a["category_name"] for a in d["annotations"])),
            "segments": len(segs),
        }
        print(f"v1.0 {split}: {len(segs)} segments, {len(d['images'])} images, "
              f"{len(d['annotations'])} anns")
    sources = []
    for shoot, rec in amap.items():
        p = DATA_ROOT / shoot / PROCESSED / "WildBox"
        if p.is_dir():
            sources.append(f"{p}={rec['category']}:{cat_ids[rec['category']]}")
        else:
            print(f"!! source missing on disk: {p}")
    (out_dir / "v10_reference.json").write_text(json.dumps(
        {"counts": v10_counts, "cat_ids": cat_ids}, indent=1))
    return split_ids, sources, v10_counts


def rewrite_paths(path: Path, out_path: Path):
    amap = json.loads(ALIAS_MAP.read_text())["datasets"]
    prefix_to_alias = {}
    for shoot, rec in amap.items():
        pref = str((DATA_ROOT / shoot / PROCESSED / "WildBox").resolve())
        prefix_to_alias[pref] = rec["alias"]
    d = json.loads(path.read_text())
    n_fixed = 0
    for im in d["images"]:
        fp = im["file_path"]
        for pref, alias in prefix_to_alias.items():
            if fp.startswith(pref):
                im["file_path"] = alias + fp[len(pref):]
                n_fixed += 1
                break
    out_path.write_text(json.dumps(d, separators=(",", ":")))
    print(f"  {path.name}: {n_fixed}/{len(d['images'])} paths rewritten -> {out_path.name} "
          f"({out_path.stat().st_size/1e6:.0f} MB)")
    return d


def gates(d: dict, split: str, v10: dict):
    prov = d["info"].get("tracking_provenance", {})
    n_corr = sum(1 for v in prov.values() if v == "corrected")
    print(f"\n[{split}] provenance: {n_corr} corrected / {len(prov) - n_corr} raw segments")

    ims = {i["id"]: i for i in d["images"]}
    dims_by_track = defaultdict(list)
    seg_of_track = {}
    for a in d["annotations"]:
        g, v, s, _ = parse_label_path(ims[a["image_id"]]["file_path"])
        key = (v, s, a.get("track_id"))
        dims_by_track[key].append(a["dimensions"])
        seg_of_track[key] = (v, s)
    seg_prov = {}
    for sid, p in prov.items():
        parts = sid.split("/")
        seg_prov[(parts[1], parts[2])] = p
    bad = []
    cvs_corr = []
    for key, dims in dims_by_track.items():
        if len(dims) < 4:
            continue
        arr = np.array(dims, float)
        cv = float((arr.std(0) / np.maximum(arr.mean(0), 1e-9)).mean())
        if seg_prov.get(seg_of_track[key]) == "corrected":
            cvs_corr.append(cv)
            if cv > 1e-6:
                bad.append((key, cv))
    ok = len(bad) == 0
    print(f"[{split}] DIMS-CONSTANCY GATE on corrected segments: "
          f"{len(cvs_corr)} tracks, max cv {max(cvs_corr) if cvs_corr else 0:.2e} "
          f"-> {'PASS' if ok else f'FAIL ({len(bad)} tracks fluctuate)'}")
    for key, cv in bad[:5]:
        print(f"    FLUCTUATING {key}: cv={cv:.4f}")

    new_counts = {"images": len(d["images"]), "annotations": len(d["annotations"]),
                  "per_class": dict(Counter(a["category_name"] for a in d["annotations"]))}
    print(f"[{split}] v1.0 -> v1.1: images {v10['images']} -> {new_counts['images']}  "
          f"anns {v10['annotations']} -> {new_counts['annotations']}")
    for cls in sorted(set(v10["per_class"]) | set(new_counts["per_class"])):
        a, b = v10["per_class"].get(cls, 0), new_counts["per_class"].get(cls, 0)
        print(f"    {cls:14s} {a:7,} -> {b:7,}  ({b-a:+,})")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/d/wildbox_v1.1"))
    ap.add_argument("--smoke", type=int, default=0,
                    help="build only the first N train + N val segments (pipeline test)")
    ap.add_argument("--skip-build", action="store_true",
                    help="only re-run path rewrite + gates on existing raw outputs")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    split_ids, sources, v10_counts = derive_v10(args.out_dir)
    train_ids, val_ids = split_ids["train"], split_ids["val"]
    if args.smoke:
        train_ids, val_ids = train_ids[:args.smoke], val_ids[:args.smoke]

    raw_train = args.out_dir / "_raw_WildBox_train.json"
    raw_val = args.out_dir / "_raw_WildBox_val.json"
    if not args.skip_build:
        cmd = [sys.executable, "tools/prepare_wildbox_dataset.py",
               "--split-mode", "manual",
               "--prefer-corrected", "--dataset-version", "1.1",
               "--output-train", str(raw_train), "--output-val", str(raw_val),
               "--train-segments", *train_ids, "--val-segments", *val_ids]
        for s in sources:
            cmd += ["--source", s]
        print(f"\nlaunching prepare_wildbox_dataset.py "
              f"({len(train_ids)} train + {len(val_ids)} val segments)...")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print("BUILD FAILED")
            return 1

    all_ok = True
    for split, raw, v10 in (("train", raw_train, v10_counts["train"]),
                            ("val", raw_val, v10_counts["val"])):
        if not raw.exists():
            print(f"missing {raw}")
            return 1
        out = args.out_dir / f"WildBox_{split}_v1.1.json"
        d = rewrite_paths(raw, out)
        all_ok &= gates(d, split, v10)
    print(f"\n{'ALL GATES PASS' if all_ok else 'GATE FAILURES -- do not release'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
