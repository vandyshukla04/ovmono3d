"""Fingerprint WildBox annotation jsons so dataset versions can be compared across machines.
[CPU; ~1 min per json]

    python tools/release/fingerprint_wildbox.py <json> [<json> ...] [--out report.json]

Prints, per file, three levels of identity:
  raw_sha256    -- byte identity (matches HuggingFace LFS sha256; detects ANY difference)
  core_sha256   -- semantic identity of the RELEASE content: images (id, file_path, width,
                   height, K) + annotations (id, image_id, category_id, bbox, center_cam,
                   dimensions, R_cam, behind_camera), all rounded to 6 decimals and sorted by id.
                   IGNORES the in-place stamp fields this project added for training
                   (geo, geo_alt, heading_alpha/valid, contact_uv/valid) AND valid3D (the
                   train-side junk mask edits it). Two files with equal core_sha256 carry the
                   same dataset even if one was stamped.
  summary       -- human-readable: counts, per-class counts, id ranges, which stamp fields are
                   present, valid3D=false count (0 = unmasked/pristine, ~38.5k = masked train).

Run the SAME script on every machine (cluster: git pull first) and diff the printed lines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

STAMP_KEYS_ANN = ["heading_alpha", "heading_valid", "contact_uv", "contact_valid"]
STAMP_KEYS_IMG = ["geo", "geo_alt", "p2"]


def r6(x):
    if isinstance(x, float):
        return round(x, 6)
    if isinstance(x, list):
        return [r6(v) for v in x]
    return x


def fingerprint(path: Path) -> dict:
    raw = path.read_bytes()
    raw_sha = hashlib.sha256(raw).hexdigest()
    d = json.loads(raw)

    core_imgs = sorted(
        ([im["id"], im.get("file_path", ""), im.get("width"), im.get("height"),
          r6(im.get("K"))] for im in d["images"]), key=lambda x: x[0])
    core_anns = sorted(
        ([a["id"], a["image_id"], a.get("category_id"),
          r6(a.get("bbox")), r6(a.get("center_cam")), r6(a.get("dimensions")),
          r6(a.get("R_cam")), bool(a.get("behind_camera", False))]
         for a in d["annotations"]), key=lambda x: x[0])
    core_sha = hashlib.sha256(
        json.dumps([core_imgs, core_anns], separators=(",", ":")).encode()).hexdigest()

    cls = Counter(a.get("category_name", str(a.get("category_id"))) for a in d["annotations"])
    ann0 = d["annotations"][0] if d["annotations"] else {}
    img0 = d["images"][0] if d["images"] else {}
    stamps = ([k for k in STAMP_KEYS_ANN if k in ann0]
              + [k for k in STAMP_KEYS_IMG if k in img0])
    n_masked = sum(1 for a in d["annotations"] if a.get("valid3D", True) is False)
    return {
        "path": str(path),
        "bytes": len(raw),
        "raw_sha256": raw_sha,
        "core_sha256": core_sha,
        "n_images": len(d["images"]),
        "n_annotations": len(d["annotations"]),
        "per_class": dict(sorted(cls.items())),
        "image_id_range": [min(i["id"] for i in d["images"]),
                           max(i["id"] for i in d["images"])] if d["images"] else None,
        "stamp_fields_present": stamps,
        "valid3D_false_count": n_masked,
        "categories_block": [c.get("name") for c in d.get("categories", [])],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsons", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    reports = []
    for p in args.jsons:
        if not p.exists():
            print(f"!! missing: {p}")
            continue
        r = fingerprint(p)
        reports.append(r)
        print(f"\n== {r['path']}")
        print(f"   bytes {r['bytes']:,}  images {r['n_images']:,}  anns {r['n_annotations']:,}")
        print(f"   raw  {r['raw_sha256'][:16]}…   CORE {r['core_sha256'][:16]}…")
        print(f"   per-class {r['per_class']}")
        print(f"   stamps {r['stamp_fields_present'] or 'none'}   "
              f"valid3D=false {r['valid3D_false_count']:,}")
    if args.out:
        args.out.write_text(json.dumps(reports, indent=1))
        print(f"\nreport -> {args.out}")
    if len(reports) > 1:
        print("\nCORE identity groups:")
        groups = {}
        for r in reports:
            groups.setdefault(r["core_sha256"], []).append(r["path"])
        for i, (sha, paths) in enumerate(groups.items()):
            print(f"  group {i} ({sha[:16]}…):")
            for p in paths:
                print(f"    {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
