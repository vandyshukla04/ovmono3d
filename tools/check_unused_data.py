#!/usr/bin/env python
"""Identify processed segments on cluster that were NOT used in training.

Walks the cluster's source data tree (auto-detected from the train JSON's
absolute file_paths), enumerates every (video, segment) tuple, then diffs
against what `WildBox_train.json` ∪ `WildBox_val.json` actually reference.

Run from anywhere on cluster:
    python tools/check_unused_data.py \\
        --train datasets/Omni3D/WildBox_train.json \\
        --val   datasets/Omni3D/WildBox_val.json
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path


def vidseg_in_json(p: Path):
    """Return: set of (video, seg), set of source-root paths inferred from
    the file_paths in the JSON."""
    g = json.loads(p.read_text())
    vidseg = set()
    roots = set()
    for im in g["images"]:
        fp = im["file_path"]
        # Infer the source root: everything before /WildBox_sam3-...
        m = re.search(r"^(.*?)/WildBox_sam3-vggtv\d+_processed[_a-z]*/WildBox(?:_v\d+)?/([^/]+)/(seg\d+)/", fp)
        if m:
            roots.add(m.group(1))
            vidseg.add((m.group(2), m.group(3)))
    return vidseg, roots


def walk_source_root(roots):
    """For each root path, walk it and return {group_name: set((video, seg))}."""
    out = defaultdict(set)
    for root in roots:
        rp = Path(root)
        if not rp.is_dir():
            continue
        # The structure is: <root> = path-up-to-but-not-including-the-group-dir.
        # The next-up parent is the group dir (e.g. data202501KGiraffes).
        # Then inside the group: WildBox_sam3-vggt*_processed*/WildBox*/<video>/<seg>
        parent = rp.parent  # the dir that contains all groups
        # rp.name is the group name (e.g. "data202501KGiraffes")
        group_name = rp.name
        for proc in rp.glob("WildBox_sam3-vggtv*_processed*"):
            if not proc.is_dir(): continue
            for wb in proc.glob("WildBox*"):
                if not wb.is_dir(): continue
                for vid in wb.iterdir():
                    if not vid.is_dir(): continue
                    for seg in vid.iterdir():
                        if seg.is_dir() and seg.name.startswith("seg"):
                            out[group_name].add((vid.name, seg.name))

    # If single root inferred (more common): walk one level higher to find
    # OTHER group dirs that *could* have been used but weren't even referenced
    # by the JSON (e.g. processed but never registered).
    if len(roots) == 1:
        parent = Path(next(iter(roots))).parent
        for sibling_group in parent.iterdir():
            if not sibling_group.is_dir(): continue
            if sibling_group.name in out: continue   # already walked
            for proc in sibling_group.glob("WildBox_sam3-vggtv*_processed*"):
                if not proc.is_dir(): continue
                for wb in proc.glob("WildBox*"):
                    if not wb.is_dir(): continue
                    for vid in wb.iterdir():
                        if not vid.is_dir(): continue
                        for seg in vid.iterdir():
                            if seg.is_dir() and seg.name.startswith("seg"):
                                out[sibling_group.name].add((vid.name, seg.name))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, default=Path("datasets/Omni3D/WildBox_train.json"))
    ap.add_argument("--val",   type=Path, default=Path("datasets/Omni3D/WildBox_val.json"))
    args = ap.parse_args()

    if not args.train.exists() or not args.val.exists():
        sys.exit(f"Cannot find train/val JSONs.\n"
                 f"  train: {args.train} {'EXISTS' if args.train.exists() else 'MISSING'}\n"
                 f"  val:   {args.val} {'EXISTS' if args.val.exists() else 'MISSING'}")

    print(f"Reading train: {args.train}")
    train_used, train_roots = vidseg_in_json(args.train)
    print(f"Reading val:   {args.val}")
    val_used,   val_roots   = vidseg_in_json(args.val)

    used = train_used | val_used
    roots = train_roots | val_roots
    print(f"\n=== USED in cluster training+val ===")
    print(f"  train (video, seg) tuples: {len(train_used)}")
    print(f"  val   (video, seg) tuples: {len(val_used)}")
    print(f"  union (unique):            {len(used)}")
    print(f"\n=== Source roots inferred from JSON file_paths ===")
    for r in sorted(roots):
        print(f"  {r}")

    print(f"\n=== Walking source roots to enumerate ALL processed segments ===")
    on_disk = walk_source_root(roots)
    all_disk = set()
    for grp, segs in on_disk.items():
        all_disk |= segs

    print(f"\n=== Per-group summary (disk vs used) ===")
    print(f"{'Group':40s} {'on_disk':>8s} {'used':>6s} {'unused':>7s}")
    for grp in sorted(on_disk):
        d = on_disk[grp]
        u = d & used
        unused = d - used
        print(f"  {grp:40s} {len(d):>8} {len(u):>6} {len(unused):>7}")
    print(f"  {'TOTAL':40s} {len(all_disk):>8} {len(all_disk & used):>6} {len(all_disk - used):>7}")

    print(f"\n=== Local-only (on disk but NOT used in train/val) ===")
    unused = all_disk - used
    print(f"  count: {len(unused)}")
    if unused:
        # Tag each tuple with its group for human readability
        gv = {(v, s): g for g, segs in on_disk.items() for v, s in segs}
        by_group_video = defaultdict(lambda: defaultdict(list))
        for v, s in unused:
            by_group_video[gv.get((v, s), "?")][v].append(s)
        for grp in sorted(by_group_video):
            print(f"  [{grp}]")
            for v in sorted(by_group_video[grp]):
                print(f"    {v}: {sorted(by_group_video[grp][v])}")

    cluster_only = used - all_disk
    print(f"\n=== Cluster-only (in train/val JSON but NOT on disk under enumerated roots) ===")
    print(f"  count: {len(cluster_only)}")
    if cluster_only:
        print("  (these would indicate JSON references to data that has been deleted/moved)")
        for v, s in sorted(cluster_only):
            print(f"    {v}/{s}")


if __name__ == "__main__":
    main()
