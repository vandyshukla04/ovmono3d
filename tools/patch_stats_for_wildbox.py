#!/usr/bin/env python
"""
Append WildBox wildlife categories (or any custom categories) to the
Omni3D stats.json so register_and_store_model_metadata can find them.

stats.json ships with Omni3D and lives at datasets/Omni3D/stats.json. It
is NOT checked into this repo — run sh datasets/Omni3D/download_omni3d_json.sh
first. Then run this patcher on the cluster before fine-tuning.

Idempotent: re-running with the same categories is a no-op.

Usage (Phase 1, just giraffe):
  python tools/patch_stats_for_wildbox.py \
      --stats datasets/Omni3D/stats.json \
      --add giraffe:1000

Phase 2 (all six savanna species):
  python tools/patch_stats_for_wildbox.py \
      --stats datasets/Omni3D/stats.json \
      --add giraffe:1000 zebra:1001 elephant:1002 \
            lion:1003 rhino:1004 gazelle:1005
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, default=Path("datasets/Omni3D/stats.json"))
    ap.add_argument("--add", nargs="+", required=True,
                    help="Pairs like giraffe:1000 zebra:1001 ...")
    args = ap.parse_args()

    if not args.stats.exists():
        print(f"ERROR: {args.stats} not found. Run download_omni3d_json.sh first.",
              file=sys.stderr)
        sys.exit(2)

    with open(args.stats, "r") as f:
        stats = json.load(f)

    existing_names = set(stats.get("category_names", []))
    existing_ids = {c["id"] for c in stats.get("categories", [])}

    added = []
    for entry in args.add:
        name, _, cat_id_str = entry.partition(":")
        if not name or not cat_id_str:
            print(f"SKIP malformed entry: {entry}", file=sys.stderr)
            continue
        cat_id = int(cat_id_str)
        if name in existing_names:
            print(f"skip (exists): {name}")
            continue
        if cat_id in existing_ids:
            print(f"ERROR: id {cat_id} already in use for another category",
                  file=sys.stderr)
            sys.exit(3)

        stats.setdefault("category_names", []).append(name)
        stats.setdefault("categories", []).append({
            "id": cat_id,
            "name": name,
            "supercategory": "animal",
        })
        existing_names.add(name)
        existing_ids.add(cat_id)
        added.append((name, cat_id))

    if not added:
        print("Nothing to add; stats.json unchanged.")
        return

    backup = args.stats.with_suffix(".json.bak")
    if not backup.exists():
        backup.write_text(Path(args.stats).read_text())
        print(f"Backup written: {backup}")

    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Added {len(added)} categories to {args.stats}:")
    for name, cat_id in added:
        print(f"  {name} -> id {cat_id}")


if __name__ == "__main__":
    main()
