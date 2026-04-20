#!/usr/bin/env python
"""Rewrite absolute file_path entries in a generated WildBox_{train,val}.json
when the underlying data has been moved on disk.

Three modes (pick one):

1. `--map OLD=NEW` (repeatable): substring substitution. Any file_path
   containing OLD has that substring replaced with NEW. Simple, safe.

2. `--old-prefix X --new-prefix Y`: same thing, single pair, prefix-only match.

3. `--search-root DIR`: fallback auto-repair. For each file_path that
   doesn't exist, search DIR recursively for a file with the same
   basename + last 2 path components (video/seg/frame) and rewrite.
   Slow on huge trees but robust when you don't know exact prefixes.

By default we write to <input>.remapped.json. Pass --in-place to overwrite
after backing up the original to <input>.bak.

Usage:
    # Explicit swap:
    python tools/remap_wildbox_paths.py datasets/Omni3D/WildBox_val.json \
        --map /storage3/.../dataRhinoCami2025/WildBoxVGGT-v1/WildBox=/storage3/.../archive/data202502KRhinoCamiV2/WildBox_sam3-vggtv1_processed_unzipped/WildBox \
        --map /storage3/.../data202406KElephants/WildBox_vggtv1/WildBox=/storage3/.../archive/data202406KElephants/WildBox_sam3-vggtv1_processed_unzipped/WildBox \
        --in-place

    # Or auto-search under a single root:
    python tools/remap_wildbox_paths.py datasets/Omni3D/WildBox_val.json \
        --search-root /storage3/3DOM/vshukla/sam3/wd_data/wildbox/archive \
        --in-place
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def apply_substring_maps(p: str, maps: List[Tuple[str, str]]) -> str:
    for old, new in maps:
        if old in p:
            p = p.replace(old, new, 1)
    return p


def build_basename_index(search_root: Path) -> Dict[str, List[Path]]:
    """Build {"frame_000001.jpg": [/abs/path, /abs/other], ...}. Only walks
    .jpg files to keep the index small."""
    idx: Dict[str, List[Path]] = defaultdict(list)
    for p in search_root.rglob("*.jpg"):
        idx[p.name].append(p)
    return idx


def auto_repair_with_index(orig: str,
                           idx: Dict[str, List[Path]]) -> Optional[str]:
    """Try to match by last 3 path components: <video>/<seg>/frame.jpg.
    More specific than just basename, to disambiguate when the same
    frame number appears in many segments."""
    orig_p = Path(orig)
    basename = orig_p.name
    candidates = idx.get(basename, [])
    if not candidates:
        return None
    # Filter by matching parent (seg) name
    target_seg = orig_p.parent.name
    target_video = orig_p.parent.parent.name
    best: Optional[Path] = None
    for c in candidates:
        if c.parent.name != target_seg:
            continue
        if c.parent.parent.name != target_video:
            continue
        best = c
        break
    return str(best.resolve()) if best else None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json_path", type=Path,
                   help="Path to WildBox_train.json / WildBox_val.json")
    p.add_argument("--map", action="append", default=[],
                   help="Repeatable OLD=NEW substring substitution.")
    p.add_argument("--old-prefix", type=str, default=None)
    p.add_argument("--new-prefix", type=str, default=None)
    p.add_argument("--search-root", type=Path, default=None,
                   help="Fallback: search here for missing files by basename.")
    p.add_argument("--in-place", action="store_true",
                   help="Overwrite input (after writing a .bak copy).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be changed, don't write.")
    return p.parse_args()


def main():
    args = parse_args()

    maps: List[Tuple[str, str]] = []
    for entry in args.map:
        if "=" not in entry:
            sys.exit(f"--map needs OLD=NEW, got {entry!r}")
        lhs, _, rhs = entry.partition("=")
        maps.append((lhs, rhs))
    if args.old_prefix and args.new_prefix:
        maps.append((args.old_prefix, args.new_prefix))

    if not maps and not args.search_root:
        sys.exit("Provide at least one of: --map, --old-prefix/--new-prefix, or --search-root")

    print(f"Loading {args.json_path} ...", flush=True)
    d = json.load(open(args.json_path))
    total = len(d["images"])
    print(f"  {total} images")

    index: Dict[str, List[Path]] = {}
    if args.search_root:
        print(f"Building basename index under {args.search_root} ...", flush=True)
        index = build_basename_index(args.search_root)
        print(f"  indexed {sum(len(v) for v in index.values())} .jpg files "
              f"across {len(index)} unique basenames")

    changed = 0
    still_missing = 0
    verified = 0
    sample_changes: List[Tuple[str, str]] = []

    for img in d["images"]:
        orig = img["file_path"]
        new = apply_substring_maps(orig, maps)

        # Prefer substring map outcome if it exists on disk
        if new != orig and Path(new).exists():
            img["file_path"] = new
            changed += 1
            if len(sample_changes) < 5:
                sample_changes.append((orig, new))
            continue

        # Check: does original path exist as-is?
        if Path(orig).exists():
            verified += 1
            continue

        # Try auto-search
        if index:
            found = auto_repair_with_index(orig, index)
            if found:
                img["file_path"] = found
                changed += 1
                if len(sample_changes) < 5:
                    sample_changes.append((orig, found))
                continue

        still_missing += 1

    print(f"\nSummary:")
    print(f"  already OK (unchanged):   {verified}")
    print(f"  rewritten:                {changed}")
    print(f"  still missing after scan: {still_missing}")
    if sample_changes:
        print(f"\nFirst few rewrites:")
        for a, b in sample_changes:
            print(f"  - {a}\n    -> {b}")

    if still_missing > 0:
        print(f"\nWARNING: {still_missing} file_paths still don't resolve. "
              f"Add more --map entries or extend --search-root.")

    if args.dry_run:
        print("\n(--dry-run: not writing)")
        return

    out_path: Path
    if args.in_place:
        bak = args.json_path.with_suffix(args.json_path.suffix + ".bak")
        shutil.copy2(args.json_path, bak)
        print(f"\nBackup: {bak}")
        out_path = args.json_path
    else:
        out_path = args.json_path.with_suffix(".remapped" + args.json_path.suffix)

    with open(out_path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
