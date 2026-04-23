#!/usr/bin/env python
"""Rename the `zebra` class to `grevys_zebra` throughout the WildBox
corpus for paper accuracy: all current zebra training data is Grévy's
zebra (Equus grevyi) sourced from the KABR reserve. No plains zebra
present.

This is a pure string rename — the dataset_id (1001) stays the same,
the contiguous mapping stays the same, no data re-prep needed.
Predictions and checkpoints saved before this rename remain fully
usable (they reference ids, not names).

Idempotent — safe to run twice. Run from repo root:

    python tools/rename_zebra_to_grevys.py
    # add --dry-run to see what would change without writing
"""
import argparse
import json
from pathlib import Path

OLD = "zebra"
NEW = "grevys_zebra"

# Files we will touch. Missing files are skipped with a note.
FILES = [
    Path("datasets/Omni3D/stats.json"),
    Path("datasets/Omni3D/WildBox_train.json"),
    Path("datasets/Omni3D/WildBox_val.json"),
    Path("configs/wildbox/category_meta_wildlife5.json"),
    Path("configs/wildbox/category_meta_wildlife.json"),
]


def _rename_in_stats(d: dict) -> bool:
    """stats.json has a top-level `category_names` list and a nested
    per-category dict keyed by name. Rewrite both."""
    changed = False
    if isinstance(d.get("category_names"), list) and OLD in d["category_names"]:
        d["category_names"] = [NEW if x == OLD else x for x in d["category_names"]]
        changed = True
    if isinstance(d.get("category_frequency"), dict) and OLD in d["category_frequency"]:
        d["category_frequency"][NEW] = d["category_frequency"].pop(OLD)
        changed = True
    if isinstance(d.get("category_stats"), dict) and OLD in d["category_stats"]:
        d["category_stats"][NEW] = d["category_stats"].pop(OLD)
        changed = True
    # Generic pass — rename any top-level key that is exactly OLD
    if OLD in d:
        d[NEW] = d.pop(OLD)
        changed = True
    return changed


def _rename_in_omni3d_json(d: dict) -> bool:
    """Train/val GT JSONs have `categories: [{id, name}, ...]` where
    name is the human-readable label."""
    changed = False
    cats = d.get("categories", [])
    for c in cats:
        if c.get("name") == OLD:
            c["name"] = NEW
            changed = True
    return changed


def _rename_in_category_meta(d: dict) -> bool:
    """thing_classes is a list; dataset_id->contiguous_id mapping uses
    numeric ids (no rename needed there)."""
    changed = False
    tc = d.get("thing_classes", [])
    if OLD in tc:
        d["thing_classes"] = [NEW if x == OLD else x for x in tc]
        changed = True
    return changed


def _detect_and_rename(path: Path, dry_run: bool) -> str:
    """Returns a human-readable status string."""
    if not path.exists():
        return f"  skip (missing): {path}"
    try:
        d = json.load(open(path))
    except Exception as e:
        return f"  error reading {path}: {e}"

    # Idempotency check — if NEW is already present and OLD isn't, we're done
    serialized = json.dumps(d)
    if OLD not in serialized and NEW in serialized:
        return f"  already renamed: {path}"

    changed = False
    # stats.json has its own shape
    if path.name == "stats.json":
        changed = _rename_in_stats(d)
    elif path.name.startswith("category_meta"):
        changed = _rename_in_category_meta(d)
    else:
        changed = _rename_in_omni3d_json(d)

    if not changed:
        return f"  no-op (no zebra references): {path}"
    if dry_run:
        return f"  WOULD RENAME: {path}"
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    return f"  renamed: {path}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would change without writing anything.")
    args = p.parse_args()

    print(f"Renaming '{OLD}' -> '{NEW}' across WildBox artifacts"
          f"{' (DRY RUN)' if args.dry_run else ''}")
    for f in FILES:
        print(_detect_and_rename(f, args.dry_run))

    # Verify symlinks still resolve
    top_symlink = Path("configs/category_meta.json")
    cfg_symlink = Path("configs/wildbox/category_meta.json")
    for s in [top_symlink, cfg_symlink]:
        if s.is_symlink():
            tgt = s.resolve()
            tc = []
            try:
                tc = json.load(open(s)).get("thing_classes", [])
            except Exception:
                pass
            status = "OK" if NEW in tc else f"WARN (thing_classes={tc})"
            print(f"  symlink: {s} -> {tgt.name}  [{status}]")

    print("\nNext step: re-register datasets via tools/patch_stats_for_wildbox.py "
          "if it reads stats.json by name, OR just verify the run:")
    print(f"  python -c \"import json; "
          f"print(json.load(open('configs/wildbox/category_meta_wildlife5.json'))"
          f"['thing_classes'])\"")


if __name__ == "__main__":
    main()
