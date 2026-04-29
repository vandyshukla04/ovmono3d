#!/usr/bin/env python
"""Build a Croissant 1.0 JSON-LD descriptor from WildBox_{train,val}.json.

Croissant (https://github.com/mlcommons/croissant) is the MLCommons spec for
machine-readable ML-dataset metadata. Hubs like HuggingFace, Kaggle, and
OpenML auto-consume it. This script writes a single `croissant.json` next to
the Omni3D JSONs that:

  - declares only the zips actually referenced by train ∪ val (so the
    distribution doesn't mention zips you never used),
  - exposes the per-frame-instance record schema used in the paper,
  - encodes the train/val split as a Croissant Split,
  - propagates per-segment scale factors from each split's `info.scene_scales`,
  - validates with `mlcroissant validate --jsonld croissant.json`.

Usage:
    python tools/build_croissant.py \\
        --train datasets/Omni3D/WildBox_train.json \\
        --val   datasets/Omni3D/WildBox_val.json \\
        --release-base-url https://example.org/wildbox/v1.0 \\
        --license  https://creativecommons.org/licenses/by/4.0/ \\
        --citation "@inproceedings{...}" \\
        --version  1.0 \\
        --out      croissant.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional


ZIP_RE = re.compile(r"/(data[^/]+)/WildBox_sam3-vggtv1_processed")


def _zip_slugs_in_split(gt_path: Path) -> tuple[set[str], dict]:
    """Return (set of zip slug names, parsed json)."""
    g = json.loads(gt_path.read_text())
    slugs: set[str] = set()
    for im in g["images"]:
        m = ZIP_RE.search(im["file_path"])
        if m:
            slugs.add(m.group(1))
    return slugs, g


def _sha256(p: Path) -> Optional[str]:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_object(*, fid: str, name: str, content_url: str,
                 encoding_format: str, sha256: Optional[str] = None,
                 contained_in: Optional[str] = None,
                 description: str = "") -> dict:
    obj = {
        "@type": "cr:FileObject",
        "@id": fid,
        "name": name,
        "contentUrl": content_url,
        "encodingFormat": encoding_format,
    }
    if description:
        obj["description"] = description
    if sha256:
        obj["sha256"] = sha256
    if contained_in:
        obj["containedIn"] = {"@id": contained_in}
    return obj


def _field(fid: str, dtype: str, *, repeated: bool = False,
           description: str = "", source_field: Optional[str] = None,
           extract_from: Optional[str] = None) -> dict:
    f: dict = {"@type": "cr:Field", "@id": fid, "dataType": dtype}
    if description:
        f["description"] = description
    if repeated:
        f["repeated"] = True
    if source_field is not None:
        # tells loaders which JSON key to read; conservative — Croissant 1.0
        # supports cr:source for FileObject-backed field extraction.
        f["source"] = {"extract": {"jsonPath": extract_from or f"$.{source_field}"}}
    return f


def build(*, train: Path, val: Path,
          release_base_url: str,
          license_url: str,
          citation: str,
          version: str,
          dataset_name: str,
          dataset_description: str,
          dataset_url: str,
          add_sha256: bool,
          zip_dir: Optional[Path]) -> dict:
    train_slugs, train_g = _zip_slugs_in_split(train)
    val_slugs, val_g = _zip_slugs_in_split(val)
    used_slugs = sorted(train_slugs | val_slugs)
    train_only = sorted(train_slugs - val_slugs)
    val_only = sorted(val_slugs - train_slugs)
    both = sorted(train_slugs & val_slugs)
    print(f"  train slugs: {len(train_slugs)}  val slugs: {len(val_slugs)}  "
          f"union: {len(used_slugs)}", file=sys.stderr)
    print(f"  train-only: {train_only}", file=sys.stderr)
    print(f"  val-only:   {val_only}", file=sys.stderr)
    print(f"  both:       {both}", file=sys.stderr)

    # ----- distribution: zips + the two JSONs ------------------------------
    distribution: list[dict] = []
    for slug in used_slugs:
        url = f"{release_base_url.rstrip('/')}/{slug}.zip"
        sha = (_sha256(zip_dir / f"{slug}.zip")
               if (add_sha256 and zip_dir is not None) else None)
        distribution.append(_file_object(
            fid=f"{slug}.zip",
            name=f"{slug}.zip",
            content_url=url,
            encoding_format="application/zip",
            sha256=sha,
            description=(f"Frame JPEGs, SAM3 masks, VGGT depth maps + point "
                         f"clouds, KITTI-style cuboid labels, and per-frame "
                         f"camera matrices for source group `{slug}`."),
        ))

    distribution.append(_file_object(
        fid="wildbox-train-json",
        name="WildBox_train.json",
        content_url=f"{release_base_url.rstrip('/')}/WildBox_train.json",
        encoding_format="application/json",
        sha256=(_sha256(train) if add_sha256 else None),
        description="Omni3D-format training annotations (frame-instance records).",
    ))
    distribution.append(_file_object(
        fid="wildbox-val-json",
        name="WildBox_val.json",
        content_url=f"{release_base_url.rstrip('/')}/WildBox_val.json",
        encoding_format="application/json",
        sha256=(_sha256(val) if add_sha256 else None),
        description="Omni3D-format validation annotations (frame-instance records).",
    ))

    # ----- recordSet: categories ------------------------------------------
    cat_records = []
    for c in val_g["categories"]:
        rec = {
            "@type": "cr:Record",
            "@id": f"category/{c['name']}",
            "name": c["name"],
            "id": c["id"],
            "supercategory": c.get("supercategory", ""),
        }
        cat_records.append(rec)

    # ----- recordSet: scene_scales (per-segment) --------------------------
    # Each split carries its own scene_scales mapping in info.scene_scales.
    # We materialise both as ascendant Records under one RecordSet so loaders
    # can join on `segment_path` to denormalise the scale factor.
    scale_records: list[dict] = []
    for split_name, g in (("train", train_g), ("val", val_g)):
        for seg_path, s in g.get("info", {}).get("scene_scales", {}).items():
            scale_records.append({
                "@type": "cr:Record",
                "@id": f"scale/{split_name}/{seg_path}",
                "split": split_name,
                "segment_path": seg_path,
                "scale_factor": float(s),
            })

    # ----- recordSet: frame_annotations schema ----------------------------
    ann_fields = [
        _field("video_id", "sc:Text",
               description="Source-video identifier (e.g. DJI_20230607092100_0001_V)."),
        _field("segment_id", "sc:Text",
               description="Segment within the source video (segN)."),
        _field("frame_index", "sc:Integer",
               description="Per-segment frame index. Maps to extracted frame "
                           "number via the segment metadata's frame_step."),
        _field("image_id", "sc:Integer",
               source_field="image_id",
               description="Stable per-record image identifier matching "
                           "Omni3D's images[*].id."),
        _field("track_id", "sc:Integer",
               source_field="track_id",
               description="Per-segment track ID (links the same instance "
                           "across frames within a segment)."),
        _field("species", "sc:Text",
               source_field="category_name",
               description="Species label: one of giraffe, grevys_zebra, "
                           "elephant, plains_zebra, rhino, gazelle."),
        _field("subspecies", "sc:Text",
               description="Optional within-species attribute. Currently used "
                           "only for `rhino` to mark Diceros bicornis vs "
                           "Ceratotherium simum (v1.0 trains/evaluates rhino "
                           "as a single class — attribute is for stratified "
                           "analysis only)."),
        _field("image", "sc:ImageObject",
               description="Path to the frame JPEG inside the corresponding "
                           "data{group}.zip."),
        _field("mask", "sc:ImageObject",
               description="Path to the SAM3 binary mask PNG for this "
                           "instance: sam3_masks/masks/obj_<i>/frame_<N>.png."),
        _field("bbox_2d", "cr:BoundingBox",
               source_field="bbox",
               description="Tight 2D bounding box derived from the SAM3 mask, "
                           "stored as [x, y, w, h] in pixels."),
        _field("center_cam", "cr:Float32", repeated=True,
               source_field="center_cam",
               description="3D cuboid centre (X, Y, Z) in the per-frame "
                           "camera frame, scene units rescaled by the "
                           "segment's scale_factor."),
        _field("dimensions", "cr:Float32", repeated=True,
               source_field="dimensions",
               description="Cuboid extents [W, H, L] in the same scaled "
                           "scene units as center_cam."),
        _field("R_cam", "cr:Float32", repeated=True,
               source_field="R_cam",
               description="3x3 row-major rotation matrix bringing the "
                           "object frame into the camera frame."),
        _field("K", "cr:Float32", repeated=True,
               description="Per-frame 3x3 intrinsics (Omni3D images[*].K)."),
        _field("scale_factor", "sc:Float",
               description="Segment-level scale factor applied to all 3D "
                           "coordinates. Joinable with the `scene_scales` "
                           "record set on (split, segment_path)."),
    ]

    # ----- recordSet: splits ----------------------------------------------
    split_records = [
        {"@type": "cr:Record", "@id": "split/train", "name": "train"},
        {"@type": "cr:Record", "@id": "split/val", "name": "val"},
    ]

    record_sets = [
        {
            "@type": "cr:RecordSet",
            "@id": "categories",
            "name": "categories",
            "description": "Six species labels used for training and evaluation.",
            "field": [
                _field("name", "sc:Text"),
                _field("id", "sc:Integer"),
                _field("supercategory", "sc:Text"),
            ],
            "data": cat_records,
        },
        {
            "@type": "cr:RecordSet",
            "@id": "splits",
            "name": "splits",
            "description": "Split labels (video-level 80/20).",
            "field": [_field("name", "sc:Text")],
            "data": split_records,
        },
        {
            "@type": "cr:RecordSet",
            "@id": "scene_scales",
            "name": "scene_scales",
            "description": ("Per-segment scale factor used to normalise the "
                            "VGGT reconstruction so that median |z| of GT "
                            "cuboids is 1 within each segment. Join with "
                            "frame_annotations on (split, segment_path)."),
            "field": [
                _field("split", "sc:Text"),
                _field("segment_path", "sc:Text"),
                _field("scale_factor", "sc:Float"),
            ],
            "data": scale_records,
        },
        {
            "@type": "cr:RecordSet",
            "@id": "frame_annotations",
            "name": "frame_annotations",
            "description": ("One record per (frame, instance). Train and val "
                            "are sourced from WildBox_train.json and "
                            "WildBox_val.json respectively; both carry the "
                            "schema below."),
            "field": ann_fields,
            "split": [{"@id": "splits/train"}, {"@id": "splits/val"}],
        },
    ]

    # ----- top-level dataset block ----------------------------------------
    croissant: dict = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "sc": "https://schema.org/",
            "data": {"@id": "cr:data", "@type": "@json"},
        },
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": dataset_name,
        "version": version,
        "description": dataset_description,
        "license": license_url,
        "url": dataset_url,
        "citeAs": citation,
        "keywords": ["3D object detection", "wildlife", "monocular 3D",
                     "drone footage", "open-vocabulary", "WildBox", "Omni3D"],
        "creator": {"@type": "sc:Organization",
                    "name": "WildBox dataset authors"},
        "distribution": distribution,
        "recordSet": record_sets,
    }
    return croissant


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True,
                    help="Path to WildBox_train.json (Omni3D format).")
    ap.add_argument("--val", type=Path, required=True,
                    help="Path to WildBox_val.json (Omni3D format).")
    ap.add_argument("--release-base-url", required=True,
                    help="Public URL prefix where the zips and JSONs will be "
                         "hosted (e.g. Zenodo record URL).")
    ap.add_argument("--license",
                    default="https://creativecommons.org/licenses/by/4.0/",
                    help="Dataset license URL (default: CC-BY-4.0).")
    ap.add_argument("--citation", default="",
                    help="BibTeX entry for the dataset paper.")
    ap.add_argument("--version", default="1.0", help="Dataset version.")
    ap.add_argument("--name", default="WildBox",
                    help="Dataset name in the Croissant `name` field.")
    ap.add_argument("--description",
                    default=("WildBox: a 6-species wildlife 3D object "
                             "detection benchmark built on drone footage "
                             "with VGGT-derived 3D cuboids and SAM3 masks. "
                             "Per-segment 3D coordinates are normalised so "
                             "that the median |Z| of GT cuboids equals 1."),
                    help="Top-level dataset description.")
    ap.add_argument("--dataset-url", default="",
                    help="Canonical URL for the dataset homepage.")
    ap.add_argument("--add-sha256", action="store_true",
                    help="Include sha256 for each zip + JSON. Requires --zip-dir "
                         "for the zips and reads the JSONs from --train/--val.")
    ap.add_argument("--zip-dir", type=Path, default=None,
                    help="Directory containing data{slug}.zip files (used only "
                         "with --add-sha256).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSON-LD path (e.g. croissant.json).")
    args = ap.parse_args(argv)

    print(f"Building Croissant from:\n  train={args.train}\n  val={args.val}",
          file=sys.stderr)
    doc = build(
        train=args.train, val=args.val,
        release_base_url=args.release_base_url,
        license_url=args.license,
        citation=args.citation,
        version=args.version,
        dataset_name=args.name,
        dataset_description=args.description,
        dataset_url=args.dataset_url,
        add_sha256=args.add_sha256,
        zip_dir=args.zip_dir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}", file=sys.stderr)
    print(f"  -> {len(doc['distribution'])} FileObjects "
          f"({len(doc['distribution']) - 2} zips + 2 JSONs)", file=sys.stderr)
    print(f"  -> {len(doc['recordSet'])} RecordSets",
          file=sys.stderr)
    print(f"\nValidate with:\n"
          f"  pip install mlcroissant\n"
          f"  mlcroissant validate --jsonld {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
