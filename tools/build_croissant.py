#!/usr/bin/env python
"""Build a Croissant 1.1 JSON-LD descriptor for the WildBox HF dataset.

Pulls the live file inventory from Hugging Face's tree API (so SHA-256s
come from LFS oids, byte-preserved by definition), assembles all required
Core Croissant fields per the NeurIPS spec, and writes ``croissant.json``.

Run after every meaningful change to the HF repo (file additions / edits)
to refresh hashes and sizes.

Usage:
    python tools/build_croissant.py --out /mnt/d/3DBOX/papersubdata/croissant.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REPO = "wildbox-anon-2026/wildbox-review"
DATASET_URL = f"https://huggingface.co/datasets/{REPO}"
RESOLVE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"

ENCODING = {
    ".zip":  "application/zip",
    ".json": "application/json",
    ".md":   "text/markdown",
    ".pth":  "application/octet-stream",
}


def list_hf_tree(path: str = "") -> list[dict]:
    """Recursively list every file in the HF dataset (excluding directories)."""
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{path}".rstrip("/")
    out: list[dict] = []
    with urllib.request.urlopen(url, timeout=30) as r:
        entries = json.load(r)
    for e in entries:
        if e["type"] == "directory":
            out.extend(list_hf_tree(e["path"]))
        elif e["type"] == "file":
            out.append(e)
    return out


def file_id(path: str) -> str:
    """Stable @id from a file path: replace path separators."""
    return path.replace("/", "__")


def fetch_sha256_via_url(url: str) -> str:
    """Stream a file from its HF resolve URL and compute SHA-256.

    Used for non-LFS files (<10 MB) which lack an lfs.oid in the tree API
    response. Identity-encoding to avoid gzip-on-the-wire variance.
    """
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=120) as r:
        while True:
            chunk = r.read(1 << 20)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()


def make_file_object(f: dict) -> dict:
    """Map an HF tree entry into a Croissant FileObject.

    SHA-256 source preference:
      1. ``lfs.oid`` — for files >10 MB; HF stores them in Git LFS, oid is sha256.
      2. Streamed fetch from contentUrl — for small files without LFS storage.
    """
    p = f["path"]
    ext = "." + p.rsplit(".", 1)[-1].lower() if "." in p else ""
    enc = ENCODING.get(ext, "application/octet-stream")
    url = f"{RESOLVE_URL}/{p}"
    obj = {
        "@type": "cr:FileObject",
        "@id": file_id(p),
        "name": p.rsplit("/", 1)[-1],
        "contentUrl": url,
        "encodingFormat": enc,
        "contentSize": str(f.get("size", 0)),
    }
    lfs = f.get("lfs") or {}
    if lfs.get("oid"):
        obj["sha256"] = lfs["oid"]
    else:
        # small file — fetch + hash; sizes are <10 MB so this is fast
        print(f"  hashing non-LFS file: {p}")
        obj["sha256"] = fetch_sha256_via_url(url)
    return obj


def build_record_sets() -> list[dict]:
    """One ``images`` table sourced from WildBox_train_paper.json — minimal but
    valid (validator only requires ≥1 RecordSet with ≥1 Field)."""
    json_obj_id = file_id("WildBox_train_paper.json")
    return [
        {
            "@type": "cr:RecordSet",
            "@id": "images",
            "name": "images",
            "description": "Per-frame image records sourced from the train Omni3D JSON.",
            "field": [
                {
                    "@type": "cr:Field",
                    "@id": "images/id",
                    "name": "id",
                    "description": "Unique image identifier.",
                    "dataType": "sc:Integer",
                    "source": {
                        "fileObject": {"@id": json_obj_id},
                        "extract": {"jsonPath": "$.images[*].id"},
                    },
                },
                {
                    "@type": "cr:Field",
                    "@id": "images/file_path",
                    "name": "file_path",
                    "description": "Relative path to the JPEG inside the per-video zip.",
                    "dataType": "sc:Text",
                    "source": {
                        "fileObject": {"@id": json_obj_id},
                        "extract": {"jsonPath": "$.images[*].file_path"},
                    },
                },
                {
                    "@type": "cr:Field",
                    "@id": "images/height",
                    "name": "height",
                    "description": "Image height in pixels (1080 for all WildBox frames).",
                    "dataType": "sc:Integer",
                    "source": {
                        "fileObject": {"@id": json_obj_id},
                        "extract": {"jsonPath": "$.images[*].height"},
                    },
                },
                {
                    "@type": "cr:Field",
                    "@id": "images/width",
                    "name": "width",
                    "description": "Image width in pixels (1920 for all WildBox frames).",
                    "dataType": "sc:Integer",
                    "source": {
                        "fileObject": {"@id": json_obj_id},
                        "extract": {"jsonPath": "$.images[*].width"},
                    },
                },
            ],
        }
    ]


def build_croissant(files: list[dict]) -> dict:
    return {
        "@context": {
            "@vocab": "https://schema.org/",
            "@language": "en",
            "sc": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "dct": "http://purl.org/dc/terms/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "conformsTo": "dct:conformsTo",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "extract": "cr:extract",
            "field": "cr:field",
            "fileObject": "cr:fileObject",
            "fileSet": "cr:fileSet",
            "format": "cr:format",
            "includes": "cr:includes",
            "jsonPath": "cr:jsonPath",
            "key": "cr:key",
            "parentField": "cr:parentField",
            "path": "cr:path",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "regex": "cr:regex",
            "repeated": "cr:repeated",
            "replace": "cr:replace",
            "separator": "cr:separator",
            "source": "cr:source",
            "subField": "cr:subField",
            "transform": "cr:transform",
        },
        "@type": "sc:Dataset",
        "name": "WildBox",
        "alternateName": ["WildBox", "wildbox-review"],
        "description": (
            "WildBox: a 6-class monocular 3D wildlife detection benchmark. "
            "Drone footage of African megafauna (giraffe, plains zebra, Grevy's "
            "zebra, elephant, rhino, gazelle) with KITTI / Omni3D-compatible 3D "
            "bounding-box annotations. 45,979 train + 13,779 val frames at "
            "1920x1080, across 64 per-video segments organised into 11 "
            "species-site groups. Annotations are released in two redundant "
            "forms: strict-KITTI per-frame txt files (lossy yaw-only rotation) "
            "and full-precision Omni3D / tracking-summary JSON (3x3 rotation "
            "matrix, centroid location)."
        ),
        "url": DATASET_URL,
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "version": "1.0.0",
        "datePublished": "2026-05-06",
        "citeAs": (
            "@inproceedings{wildbox_anonymous_2026, "
            "title={WildBox: a monocular 3D wildlife detection benchmark}, "
            "author={{Anonymous Authors}}, "
            "booktitle={Submitted to NeurIPS 2026 Datasets and Benchmarks Track}, "
            "year={2026}, note={Under double-blind review}}"
        ),
        "creator": {
            "@type": "sc:Organization",
            "name": "Anonymous (under review)",
        },
        "keywords": [
            "monocular 3D detection",
            "wildlife",
            "open-vocabulary detection",
            "KITTI format",
            "Omni3D",
            "drone imagery",
        ],
        "distribution": [make_file_object(f) for f in files],
        "recordSet": build_record_sets(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True,
                    help="Path to write the croissant JSON-LD.")
    args = ap.parse_args()

    print(f"Fetching live file inventory from {REPO} ...")
    files = list_hf_tree()
    print(f"  {len(files)} files found")

    descriptor = build_croissant(files)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(descriptor, f, indent=2)
    sz = args.out.stat().st_size
    print(f"\nWrote {args.out} ({sz/1024:.1f} KB)")
    print(f"  - {len(descriptor['distribution'])} FileObject entries")
    print(f"  - {len(descriptor['recordSet'])} RecordSet entries")


if __name__ == "__main__":
    main()
