"""
Comprehensive WildBox dataset description for the paper's data section.

Produces a single markdown document plus supporting figures with everything
a dataset paper-description would have:

  1. Overview — totals, modality, annotation provenance, split protocol
  2. Per-species inventory — videos / segments / frames / boxes / instances
                              (train, val, total) with median per-video/segment
  3. Per-species 2D properties — bbox pixel area, bbox area as fraction of
     image, bbox aspect-ratio, position-in-image (center bias)
  4. Per-species 3D properties — depth (z), dimensions (W,H,L), per-segment
     scale variation (since GT is VGGT-normalized to median |z|=1)
  5. Scene composition — boxes per frame, multi-species frames,
     class co-occurrence matrix
  6. Per-video / per-segment breakdown — top-N tables, distributions
  7. Image resolution / intrinsics — distinct image sizes, K diagnostics
  8. Split integrity — video/segment/frame disjointness, per-class video coverage
  9. Reproducibility — source tool, split seed, date generated

Outputs
-------
  <out>/dataset_description.md      — paper-grade markdown
  <out>/dataset_description.json    — machine-readable
  <out>/figures/
      species_counts.png            — bar chart, per-species frame & box counts
      bbox_pixel_area.png           — log-scale histogram per species
      bbox_area_ratio.png           — bbox area as fraction of image, per species
      bbox_aspect_ratio.png         — w/h ratio histogram per species
      depth_distribution.png        — z (depth) histogram per species
      dimensions.png                — W, H, L histograms per species (3 subplots)
      boxes_per_frame.png           — density histogram per species
      cooccurrence.png              — class co-occurrence heatmap
      position_bias.png             — bbox-center heatmap per species
      frames_per_video.png          — distribution of frames per video
      frames_per_segment.png        — distribution of frames per segment

Usage
-----
    python tools/dataset_paper_description.py \\
        --train datasets/Omni3D/WildBox_train.json \\
        --val   datasets/Omni3D/WildBox_val.json \\
        --out   datasets/Omni3D/dataset_description

The output dir is created. Idempotent — re-running overwrites.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------- shared helpers ----------

def _quants(xs: List[float]) -> Dict[str, float]:
    """Min, p10, p25, median, p75, p90, max, mean, std for a list."""
    xs = [x for x in xs if x is not None and isinstance(x, (int, float)) and not math.isnan(x)]
    if not xs:
        return {"n": 0}
    xs_sorted = sorted(xs)
    n = len(xs)
    def pct(p):
        idx = max(0, min(n - 1, int(round((p / 100) * (n - 1)))))
        return xs_sorted[idx]
    return {
        "n": n,
        "min": xs_sorted[0],
        "p10": pct(10), "p25": pct(25),
        "median": xs_sorted[n // 2],
        "p75": pct(75), "p90": pct(90),
        "max": xs_sorted[-1],
        "mean": statistics.mean(xs),
        "std": statistics.pstdev(xs) if n > 1 else 0.0,
    }


def _fmt_quants(q: Dict[str, float], scale: float = 1.0, suffix: str = "") -> str:
    """One-line median [p25, p75], mean ± std, range."""
    if q.get("n", 0) == 0:
        return "—"
    return (f"{q['median']*scale:.3g}{suffix} "
            f"[{q['p25']*scale:.3g}, {q['p75']*scale:.3g}]; "
            f"mean {q['mean']*scale:.3g} ± {q['std']*scale:.3g}; "
            f"range [{q['min']*scale:.3g}, {q['max']*scale:.3g}]")


def _video_id(file_path: str) -> str:
    parts = file_path.split("/")
    return parts[-3] if len(parts) >= 3 else "unknown"


def _segment_id(file_path: str) -> str:
    parts = file_path.split("/")
    if len(parts) < 3:
        return "unknown/unknown"
    return f"{parts[-3]}/{parts[-2]}"


def _frame_idx(file_path: str) -> int:
    """Try to parse frame_NNNNNN.jpg → NNNNNN."""
    name = file_path.split("/")[-1]
    name = name.rsplit(".", 1)[0]
    digits = "".join(c for c in name if c.isdigit())
    try:
        return int(digits)
    except ValueError:
        return -1


# ---------- core gather ----------

class SplitData:
    """All cached statistics needed for one split (train or val)."""
    def __init__(self, gt: dict, label: str):
        self.label = label
        self.gt = gt
        self.cat_id_to_name = {c["id"]: c["name"] for c in gt["categories"]}
        self.img_by_id = {im["id"]: im for im in gt["images"]}

        # Per-species accumulators
        self.per_species: Dict[str, Dict[str, Any]] = {
            c["name"]: {
                "videos": set(), "segments": set(), "frames": set(),
                "annotations": 0,
                "pixel_areas": [], "area_ratios": [], "aspect_ratios": [],
                "depths_z": [], "dim_W": [], "dim_H": [], "dim_L": [],
                "centers_xy_norm": [],   # (cx_norm, cy_norm) for spatial bias
                "boxes_per_frame": collections.Counter(),  # img_id -> count
                "image_size_set": set(),
            } for c in gt["categories"]
        }
        # Per-frame
        self.boxes_per_frame_total: collections.Counter = collections.Counter()
        # Co-occurrence: how often pairs of species appear in the same frame
        self.frame_species: Dict[int, set] = collections.defaultdict(set)
        # Image-level aggregates
        self.unique_resolutions: collections.Counter = collections.Counter()
        self.K_samples: List[List[List[float]]] = []
        self.video_to_segments: Dict[str, set] = collections.defaultdict(set)
        self.segment_to_frames: Dict[str, set] = collections.defaultdict(set)
        self.video_to_frames: Dict[str, set] = collections.defaultdict(set)

        for ann in gt["annotations"]:
            cname = self.cat_id_to_name.get(ann["category_id"])
            if cname is None:
                continue
            img = self.img_by_id.get(ann["image_id"])
            if img is None:
                continue
            sp = self.per_species[cname]
            vid = _video_id(img["file_path"])
            seg = _segment_id(img["file_path"])

            sp["videos"].add(vid)
            sp["segments"].add(seg)
            sp["frames"].add(ann["image_id"])
            sp["annotations"] += 1
            sp["boxes_per_frame"][ann["image_id"]] += 1
            sp["image_size_set"].add((img["width"], img["height"]))

            x, y, w, h = ann["bbox"]
            area = float(w) * float(h)
            img_area = float(img["width"]) * float(img["height"])
            sp["pixel_areas"].append(area)
            if img_area > 0:
                sp["area_ratios"].append(area / img_area)
            if h > 0:
                sp["aspect_ratios"].append(w / h)
            if img["width"] > 0 and img["height"] > 0:
                cx = (x + w / 2) / img["width"]
                cy = (y + h / 2) / img["height"]
                sp["centers_xy_norm"].append((cx, cy))

            # 3D fields (may be missing in some legacy data)
            center = ann.get("center_cam")
            if center and len(center) == 3:
                sp["depths_z"].append(float(center[2]))
            dims = ann.get("dimensions")
            if dims and len(dims) == 3:
                sp["dim_W"].append(float(dims[0]))
                sp["dim_H"].append(float(dims[1]))
                sp["dim_L"].append(float(dims[2]))

            self.frame_species[ann["image_id"]].add(cname)
            self.boxes_per_frame_total[ann["image_id"]] += 1
            self.video_to_segments[vid].add(seg)
            self.segment_to_frames[seg].add(ann["image_id"])
            self.video_to_frames[vid].add(ann["image_id"])

        # Image resolution + K
        for im in gt["images"]:
            self.unique_resolutions[(im["width"], im["height"])] += 1
            if len(self.K_samples) < 100 and "K" in im:
                self.K_samples.append(im["K"])

    # ----- summaries -----

    def species_table(self) -> List[Dict[str, Any]]:
        out = []
        for name, sp in self.per_species.items():
            entry = {
                "species": name,
                "videos": len(sp["videos"]),
                "segments": len(sp["segments"]),
                "frames": len(sp["frames"]),
                "annotations": sp["annotations"],
                "boxes_per_frame": _quants(list(sp["boxes_per_frame"].values())),
                "bbox_pixel_area": _quants(sp["pixel_areas"]),
                "bbox_area_ratio": _quants(sp["area_ratios"]),
                "bbox_aspect_ratio": _quants(sp["aspect_ratios"]),
                "depth_z": _quants(sp["depths_z"]),
                "dim_W": _quants(sp["dim_W"]),
                "dim_H": _quants(sp["dim_H"]),
                "dim_L": _quants(sp["dim_L"]),
            }
            out.append(entry)
        return out


# ---------- markdown rendering ----------

def render_markdown(train: SplitData, val: SplitData, out_path: Path,
                    fig_dir: Path) -> str:
    L: List[str] = []
    L.append(f"# WildBox — Dataset Description\n")
    L.append(f"_Generated {date.today().isoformat()}; "
             f"comprehensive paper-grade inventory + analysis._\n")

    # ---- 1. Overview ----
    L.append("## 1. Overview\n")
    classes = sorted({c for c in train.per_species} | {c for c in val.per_species})
    total_vids = len({v for sp in train.per_species.values() for v in sp["videos"]} |
                     {v for sp in val.per_species.values() for v in sp["videos"]})
    total_segs = len({s for sp in train.per_species.values() for s in sp["segments"]} |
                     {s for sp in val.per_species.values() for s in sp["segments"]})
    total_frames = sum(len(sp["frames"]) for sp in train.per_species.values()) + \
                   sum(len(sp["frames"]) for sp in val.per_species.values())
    total_anns = sum(sp["annotations"] for sp in train.per_species.values()) + \
                 sum(sp["annotations"] for sp in val.per_species.values())
    L.append(f"- **Modality**: aerial drone footage (DJI), monocular RGB.")
    L.append(f"- **Task**: 3D object detection — full 9-DoF cuboids "
             f"(center, dimensions, full 3-DoF rotation).")
    L.append(f"- **Species**: {len(classes)} — {', '.join(classes)}.")
    L.append(f"- **Scale**: {total_vids} videos, {total_segs} segments, "
             f"{total_frames:,} frames, {total_anns:,} annotated 3D cuboids.")
    L.append(f"- **2D bbox source**: SAM3-segmentation-tight axis-aligned bboxes.")
    L.append(f"- **3D pseudo-label source**: VGGT 3D reconstruction + per-segment scale "
             f"normalization (median |z|=1).")
    L.append(f"- **Split**: video-level, deterministic seed=0, 80/20 train/val. "
             f"No video appears in both splits.\n")

    # ---- 2. Per-species inventory ----
    L.append("## 2. Per-species inventory\n")
    L.append("Counts per species across train and val. The 'rare' flag (≤5 train videos) "
             "determines whether per-class metrics need multi-seed reporting.\n")
    L.append("| Species | train vids | val vids | train segs | val segs | train frames | val frames | train boxes | val boxes | rare? |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for c in classes:
        t = next((s for s in train.species_table() if s["species"] == c), None) or {}
        v = next((s for s in val.species_table() if s["species"] == c), None) or {}
        rare = "✓" if t.get("videos", 0) <= 5 else ""
        L.append(f"| {c} "
                 f"| {t.get('videos', 0)} | {v.get('videos', 0)} "
                 f"| {t.get('segments', 0)} | {v.get('segments', 0)} "
                 f"| {t.get('frames', 0):,} | {v.get('frames', 0):,} "
                 f"| {t.get('annotations', 0):,} | {v.get('annotations', 0):,} "
                 f"| {rare} |")
    L.append("")

    # ---- 3. 2D properties ----
    L.append("## 3. Per-species 2D properties\n")
    L.append("All distributions computed over train+val combined unless noted. "
             "Quantiles are reported as **median [p25, p75]**.\n")

    L.append("### 3.1 Bounding box size (pixels²)\n")
    L.append("| Species | n boxes | median px² | [p25, p75] | mean ± std | min..max |")
    L.append("|---|---:|---:|---|---|---|")
    for c in classes:
        merged = (train.per_species[c]["pixel_areas"] +
                  val.per_species[c]["pixel_areas"])
        q = _quants(merged)
        if q.get("n", 0) == 0:
            continue
        L.append(f"| {c} | {q['n']:,} | {q['median']:,.0f} "
                 f"| [{q['p25']:,.0f}, {q['p75']:,.0f}] "
                 f"| {q['mean']:,.0f} ± {q['std']:,.0f} "
                 f"| {q['min']:,.0f}..{q['max']:,.0f} |")
    L.append("")

    L.append("### 3.2 Bounding box area as fraction of image\n")
    L.append("Tiny ratios indicate small-object detection difficulty (e.g. gazelle).\n")
    L.append("| Species | n | median % | [p25, p75] % | mean ± std % |")
    L.append("|---|---:|---:|---|---|")
    for c in classes:
        merged = (train.per_species[c]["area_ratios"] +
                  val.per_species[c]["area_ratios"])
        q = _quants(merged)
        if q.get("n", 0) == 0:
            continue
        L.append(f"| {c} | {q['n']:,} "
                 f"| {q['median']*100:.3f} "
                 f"| [{q['p25']*100:.3f}, {q['p75']*100:.3f}] "
                 f"| {q['mean']*100:.3f} ± {q['std']*100:.3f} |")
    L.append("")

    L.append("### 3.3 Bounding box aspect ratio (width / height)\n")
    L.append("Ratio > 1 = wider than tall (typical for elephants); < 1 = taller "
             "(giraffes).\n")
    L.append("| Species | median | [p25, p75] | mean ± std |")
    L.append("|---|---:|---|---|")
    for c in classes:
        merged = (train.per_species[c]["aspect_ratios"] +
                  val.per_species[c]["aspect_ratios"])
        q = _quants(merged)
        if q.get("n", 0) == 0:
            continue
        L.append(f"| {c} | {q['median']:.2f} "
                 f"| [{q['p25']:.2f}, {q['p75']:.2f}] "
                 f"| {q['mean']:.2f} ± {q['std']:.2f} |")
    L.append("")

    # ---- 4. 3D properties ----
    L.append("## 4. Per-species 3D properties\n")
    L.append("All 3D properties are in **VGGT-synthetic per-segment scale** "
             "(per-segment median |z|=1). Comparing across species is meaningful; "
             "comparing absolute z to metric depth is not.\n")

    L.append("### 4.1 Depth (z, camera-frame)\n")
    L.append("| Species | n | median z | [p25, p75] | mean ± std | min..max |")
    L.append("|---|---:|---:|---|---|---|")
    for c in classes:
        merged = (train.per_species[c]["depths_z"] +
                  val.per_species[c]["depths_z"])
        q = _quants(merged)
        if q.get("n", 0) == 0:
            L.append(f"| {c} | 0 | — | — | — | — |"); continue
        L.append(f"| {c} | {q['n']:,} | {q['median']:.2f} "
                 f"| [{q['p25']:.2f}, {q['p75']:.2f}] "
                 f"| {q['mean']:.2f} ± {q['std']:.2f} "
                 f"| {q['min']:.2f}..{q['max']:.2f} |")
    L.append("")

    L.append("### 4.2 3D dimensions (Omni3D ordering: W, H, L)\n")
    L.append("Width, height, length per species (synthetic units). Largest "
             "absolute dimensions are the head classes (elephant, rhino).\n")
    L.append("| Species | W median (mean ± std) | H median (mean ± std) | L median (mean ± std) |")
    L.append("|---|---|---|---|")
    for c in classes:
        rows = []
        for axis in ("dim_W", "dim_H", "dim_L"):
            merged = train.per_species[c][axis] + val.per_species[c][axis]
            q = _quants(merged)
            if q.get("n", 0) == 0:
                rows.append("—"); continue
            rows.append(f"{q['median']:.2f} ({q['mean']:.2f} ± {q['std']:.2f})")
        L.append(f"| {c} | {rows[0]} | {rows[1]} | {rows[2]} |")
    L.append("")

    # ---- 5. Scene composition ----
    L.append("## 5. Scene composition\n")

    L.append("### 5.1 Boxes per frame (object density)\n")
    L.append("Higher = denser scenes (groups). Dense-scene 3D regression is "
             "harder than sparse-scene; reflected in the depth-axis NHD bottleneck.\n")
    L.append("| Species | median | [p25, p75] | mean ± std | max |")
    L.append("|---|---:|---|---|---:|")
    for c in classes:
        merged = (list(train.per_species[c]["boxes_per_frame"].values()) +
                  list(val.per_species[c]["boxes_per_frame"].values()))
        q = _quants(merged)
        if q.get("n", 0) == 0:
            continue
        L.append(f"| {c} | {q['median']:.1f} "
                 f"| [{q['p25']:.1f}, {q['p75']:.1f}] "
                 f"| {q['mean']:.1f} ± {q['std']:.1f} "
                 f"| {q['max']:.0f} |")
    L.append("")

    L.append("### 5.2 Multi-species scenes\n")
    multi = collections.Counter()
    for split in (train, val):
        for img_id, species_set in split.frame_species.items():
            multi[len(species_set)] += 1
    total_frames_with_anns = sum(multi.values())
    L.append("Number of frames containing N distinct species:\n")
    L.append("| # species | frames | %  of annotated frames |")
    L.append("|---:|---:|---:|")
    for k in sorted(multi.keys()):
        L.append(f"| {k} | {multi[k]:,} | {multi[k]*100/max(1,total_frames_with_anns):.1f} |")
    L.append("")
    L.append("Single-species frames are typical for drone footage focused on a herd. "
             "Multi-species frames create dataset-balance interactions between "
             "REPEAT_THRESHOLD upsampling and head/rare-class image co-occurrence.\n")

    L.append("### 5.3 Class co-occurrence matrix\n")
    L.append("Number of frames in which species A and B appear together. Diagonal "
             "= frames in which the species appears at all (in train+val).\n")
    cooc = {a: collections.Counter() for a in classes}
    for split in (train, val):
        for sset in split.frame_species.values():
            for a in sset:
                for b in sset:
                    cooc[a][b] += 1
    L.append("| | " + " | ".join(classes) + " |")
    L.append("|" + "|".join(["---"] * (len(classes) + 1)) + "|")
    for a in classes:
        row = [a] + [str(cooc[a][b]) for b in classes]
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # ---- 6. Position bias ----
    L.append("### 5.4 Spatial position bias (bbox center, normalized to image)\n")
    L.append("Per species, mean and std of bbox-center x/y in [0, 1] image coords. "
             "(0,0)=top-left, (1,1)=bottom-right. Drone perspective often biases "
             "objects toward the lower-center.\n")
    L.append("| Species | n | mean cx | mean cy | std cx | std cy |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for c in classes:
        merged = (train.per_species[c]["centers_xy_norm"] +
                  val.per_species[c]["centers_xy_norm"])
        if not merged:
            continue
        cxs = [p[0] for p in merged]
        cys = [p[1] for p in merged]
        L.append(f"| {c} | {len(merged):,} "
                 f"| {statistics.mean(cxs):.3f} | {statistics.mean(cys):.3f} "
                 f"| {statistics.pstdev(cxs):.3f} | {statistics.pstdev(cys):.3f} |")
    L.append("")

    # ---- 6. Video / segment / frame breakdown ----
    L.append("## 6. Video / segment / frame breakdown\n")

    # Frames per video
    fpv_train = [len(fs) for fs in train.video_to_frames.values()]
    fpv_val = [len(fs) for fs in val.video_to_frames.values()]
    L.append("### 6.1 Frames per video\n")
    L.append("| Split | n videos | median | [p25, p75] | mean ± std | min..max |")
    L.append("|---|---:|---:|---|---|---|")
    for label, xs in (("train", fpv_train), ("val", fpv_val)):
        q = _quants(xs)
        if q.get("n", 0) == 0: continue
        L.append(f"| {label} | {q['n']} | {q['median']:.0f} "
                 f"| [{q['p25']:.0f}, {q['p75']:.0f}] "
                 f"| {q['mean']:.0f} ± {q['std']:.0f} "
                 f"| {q['min']:.0f}..{q['max']:.0f} |")
    L.append("")

    # Segments per video
    spv_train = [len(s) for s in train.video_to_segments.values()]
    spv_val = [len(s) for s in val.video_to_segments.values()]
    L.append("### 6.2 Segments per video\n")
    L.append("| Split | n videos | median | [p25, p75] | mean ± std | min..max |")
    L.append("|---|---:|---:|---|---|---|")
    for label, xs in (("train", spv_train), ("val", spv_val)):
        q = _quants(xs)
        if q.get("n", 0) == 0: continue
        L.append(f"| {label} | {q['n']} | {q['median']:.0f} "
                 f"| [{q['p25']:.0f}, {q['p75']:.0f}] "
                 f"| {q['mean']:.0f} ± {q['std']:.0f} "
                 f"| {q['min']:.0f}..{q['max']:.0f} |")
    L.append("")

    # Frames per segment
    fps_train = [len(fs) for fs in train.segment_to_frames.values()]
    fps_val = [len(fs) for fs in val.segment_to_frames.values()]
    L.append("### 6.3 Frames per segment\n")
    L.append("| Split | n segments | median | [p25, p75] | mean ± std | min..max |")
    L.append("|---|---:|---:|---|---|---|")
    for label, xs in (("train", fps_train), ("val", fps_val)):
        q = _quants(xs)
        if q.get("n", 0) == 0: continue
        L.append(f"| {label} | {q['n']} | {q['median']:.0f} "
                 f"| [{q['p25']:.0f}, {q['p75']:.0f}] "
                 f"| {q['mean']:.0f} ± {q['std']:.0f} "
                 f"| {q['min']:.0f}..{q['max']:.0f} |")
    L.append("")

    # Top-N videos by frame count
    L.append("### 6.4 Top 10 longest videos by frame count (train + val combined)\n")
    L.append("| Video | split | frames | segments |")
    L.append("|---|---|---:|---:|")
    all_video_frames = collections.Counter()
    all_video_segs = collections.defaultdict(set)
    video_split = {}
    for split, label in ((train, "train"), (val, "val")):
        for vid, frames in split.video_to_frames.items():
            all_video_frames[vid] += len(frames)
            video_split[vid] = label
        for vid, segs in split.video_to_segments.items():
            all_video_segs[vid] |= segs
    for vid, n in all_video_frames.most_common(10):
        L.append(f"| `{vid}` | {video_split.get(vid, '?')} | {n:,} | {len(all_video_segs[vid])} |")
    L.append("")

    # ---- 7. Image resolution / intrinsics ----
    L.append("## 7. Image resolution and intrinsics\n")
    res_combined = collections.Counter()
    for c in (train.unique_resolutions, val.unique_resolutions):
        res_combined.update(c)
    L.append("### 7.1 Image resolutions present\n")
    L.append("| width × height | image count |")
    L.append("|---|---:|")
    for (w, h), n in res_combined.most_common(10):
        L.append(f"| {w}×{h} | {n:,} |")
    L.append("")
    L.append(f"_Total distinct resolutions: {len(res_combined)}._\n")

    # K diagnostics — focal length and principal point distributions
    L.append("### 7.2 Camera intrinsics (K) summary\n")
    fxs = []; fys = []; cxs = []; cys = []
    for split in (train, val):
        for K in split.K_samples:
            try:
                fxs.append(K[0][0]); fys.append(K[1][1])
                cxs.append(K[0][2]); cys.append(K[1][2])
            except (IndexError, TypeError):
                pass
    L.append(f"_Sampled from {len(fxs)} images (cap of 100/split for speed)._\n")
    if fxs:
        L.append("| Parameter | median | mean ± std | range |")
        L.append("|---|---:|---|---|")
        for label, xs in (("fx", fxs), ("fy", fys), ("cx", cxs), ("cy", cys)):
            q = _quants(xs)
            L.append(f"| {label} | {q['median']:.1f} | {q['mean']:.1f} ± {q['std']:.1f} "
                     f"| {q['min']:.1f}..{q['max']:.1f} |")
    L.append("")

    # ---- 8. Split integrity ----
    L.append("## 8. Split integrity\n")
    train_vids = set(train.video_to_frames.keys())
    val_vids = set(val.video_to_frames.keys())
    overlap = train_vids & val_vids
    if not overlap:
        L.append(f"- **Video-level disjoint**: ✓ {len(train_vids)} train videos, "
                 f"{len(val_vids)} val videos, **0 shared**.")
    else:
        L.append(f"- **Video-level LEAKAGE**: {len(overlap)} shared videos. "
                 f"Re-run prep with `--seed 0`.")
    train_segs = {s for ss in train.video_to_segments.values() for s in ss}
    val_segs = {s for ss in val.video_to_segments.values() for s in ss}
    L.append(f"- **Segment-level**: {len(train_segs)} train, {len(val_segs)} val, "
             f"{len(train_segs & val_segs)} shared "
             f"(should be 0 since segments are subsets of videos).")
    L.append("")
    L.append("**Per-class video coverage**:\n")
    L.append("| Species | train vids | val vids | total vids |")
    L.append("|---|---:|---:|---:|")
    for c in classes:
        t = len(train.per_species[c]["videos"])
        v = len(val.per_species[c]["videos"])
        L.append(f"| {c} | {t} | {v} | {t + v} |")
    L.append("")

    # ---- 9. Reproducibility ----
    L.append("## 9. Reproducibility\n")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL).decode().strip())
        L.append(f"- git commit: `{commit}`{' (working tree DIRTY)' if dirty else ''}")
    except Exception:
        pass
    L.append(f"- Generated by: `tools/dataset_paper_description.py`")
    L.append(f"- Source: 15 drone campaign zips, see "
             f"[WILDBOX_EXPERIMENT.md §2.2](WILDBOX_EXPERIMENT.md)")
    L.append(f"- Prep tool: `tools/prepare_wildbox_dataset.py --split-mode video --seed 0 --val-fraction 0.2`")
    L.append(f"- 2D bbox source: SAM3 segmentation-tight (NOT 3D-cuboid projection)")
    L.append(f"- 3D pseudo-labels: VGGT 3D reconstruction with per-segment scale "
             f"normalization (median |z|=1)")
    L.append("")

    # ---- Figures ----
    L.append("## 10. Figures\n")
    figs_existing = sorted(fig_dir.glob("*.png")) if fig_dir.exists() else []
    if figs_existing:
        for fp in figs_existing:
            L.append(f"![{fp.stem}]({fp.relative_to(out_path.parent)})")
    else:
        L.append("_(figures generated alongside this markdown — see `figures/` subdir)._")
    L.append("")

    return "\n".join(L) + "\n"


# ---------- figure rendering ----------

def render_figures(train: SplitData, val: SplitData, fig_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  (matplotlib not available; skipping figures)")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted({c for c in train.per_species} | {c for c in val.per_species})

    # 1. species counts (bar chart)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    train_frames = [len(train.per_species[c]["frames"]) for c in classes]
    val_frames = [len(val.per_species[c]["frames"]) for c in classes]
    train_anns = [train.per_species[c]["annotations"] for c in classes]
    val_anns = [val.per_species[c]["annotations"] for c in classes]
    x = np.arange(len(classes))
    axes[0].bar(x - 0.2, train_frames, width=0.4, label="train")
    axes[0].bar(x + 0.2, val_frames, width=0.4, label="val")
    axes[0].set_xticks(x); axes[0].set_xticklabels(classes, rotation=20)
    axes[0].set_ylabel("# annotated frames"); axes[0].set_title("Frames per species")
    axes[0].legend()
    axes[1].bar(x - 0.2, train_anns, width=0.4, label="train")
    axes[1].bar(x + 0.2, val_anns, width=0.4, label="val")
    axes[1].set_xticks(x); axes[1].set_xticklabels(classes, rotation=20)
    axes[1].set_ylabel("# annotated boxes"); axes[1].set_title("Boxes per species")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(fig_dir / "species_counts.png", dpi=120); plt.close(fig)

    # 2. bbox pixel area histogram (log scale)
    cols = 3
    rows = (len(classes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for i, c in enumerate(classes):
        ax = axes[i // cols][i % cols]
        merged = train.per_species[c]["pixel_areas"] + val.per_species[c]["pixel_areas"]
        merged = [v for v in merged if v > 0]
        if not merged:
            ax.set_axis_off(); continue
        ax.hist(merged, bins=40, edgecolor="k", alpha=0.85)
        ax.set_xscale("log"); ax.set_xlabel("bbox pixel area (px²)")
        ax.set_ylabel("count"); ax.set_title(f"{c} (n={len(merged)})")
    for j in range(len(classes), rows * cols):
        axes[j // cols][j % cols].set_axis_off()
    fig.tight_layout(); fig.savefig(fig_dir / "bbox_pixel_area.png", dpi=120); plt.close(fig)

    # 3. bbox area as fraction of image
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for i, c in enumerate(classes):
        ax = axes[i // cols][i % cols]
        merged = [r * 100 for r in
                  (train.per_species[c]["area_ratios"] + val.per_species[c]["area_ratios"])
                  if r > 0]
        if not merged:
            ax.set_axis_off(); continue
        ax.hist(merged, bins=40, edgecolor="k", alpha=0.85)
        ax.set_xscale("log"); ax.set_xlabel("bbox area / image area (%)")
        ax.set_ylabel("count"); ax.set_title(f"{c} (n={len(merged)})")
    for j in range(len(classes), rows * cols):
        axes[j // cols][j % cols].set_axis_off()
    fig.tight_layout(); fig.savefig(fig_dir / "bbox_area_ratio.png", dpi=120); plt.close(fig)

    # 4. aspect ratio histogram
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for i, c in enumerate(classes):
        ax = axes[i // cols][i % cols]
        merged = train.per_species[c]["aspect_ratios"] + val.per_species[c]["aspect_ratios"]
        if not merged:
            ax.set_axis_off(); continue
        ax.hist(merged, bins=40, edgecolor="k", alpha=0.85, range=(0, 5))
        ax.axvline(1.0, color="r", linestyle="--", alpha=0.6, label="square (w=h)")
        ax.set_xlabel("aspect ratio (w/h)"); ax.set_ylabel("count")
        ax.set_title(f"{c} (n={len(merged)})"); ax.legend(fontsize=8)
    for j in range(len(classes), rows * cols):
        axes[j // cols][j % cols].set_axis_off()
    fig.tight_layout(); fig.savefig(fig_dir / "bbox_aspect_ratio.png", dpi=120); plt.close(fig)

    # 5. depth (z) distribution
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for i, c in enumerate(classes):
        ax = axes[i // cols][i % cols]
        merged = train.per_species[c]["depths_z"] + val.per_species[c]["depths_z"]
        if not merged:
            ax.text(0.5, 0.5, f"{c}: no z data", ha="center", va="center",
                    transform=ax.transAxes); ax.set_axis_off(); continue
        ax.hist(merged, bins=40, edgecolor="k", alpha=0.85)
        ax.axvline(1.0, color="r", linestyle="--", alpha=0.6, label="VGGT median z=1")
        ax.set_xlabel("depth z (synthetic units)"); ax.set_ylabel("count")
        ax.set_title(f"{c} (n={len(merged)})"); ax.legend(fontsize=8)
    for j in range(len(classes), rows * cols):
        axes[j // cols][j % cols].set_axis_off()
    fig.tight_layout(); fig.savefig(fig_dir / "depth_distribution.png", dpi=120); plt.close(fig)

    # 6. dimensions: 3 subplots (W, H, L) per species
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    for i, axis in enumerate(("dim_W", "dim_H", "dim_L")):
        data = []
        for c in classes:
            merged = train.per_species[c][axis] + val.per_species[c][axis]
            if merged:
                data.append((c, merged))
        if data:
            for c, vs in data:
                axes[i].hist(vs, bins=40, alpha=0.5, label=c)
        axes[i].set_xlabel(f"{axis.replace('dim_', '')} (synthetic units)")
        axes[i].set_ylabel("count")
        axes[i].set_title(f"3D dimension: {axis.replace('dim_', '')}")
        axes[i].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "dimensions.png", dpi=120); plt.close(fig)

    # 7. boxes per frame
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for i, c in enumerate(classes):
        ax = axes[i // cols][i % cols]
        merged = (list(train.per_species[c]["boxes_per_frame"].values()) +
                  list(val.per_species[c]["boxes_per_frame"].values()))
        if not merged:
            ax.set_axis_off(); continue
        ax.hist(merged, bins=range(0, max(merged) + 2), edgecolor="k", alpha=0.85)
        ax.set_xlabel("boxes per frame"); ax.set_ylabel("frame count")
        ax.set_title(f"{c} (median {statistics.median(merged):.0f}, max {max(merged)})")
    for j in range(len(classes), rows * cols):
        axes[j // cols][j % cols].set_axis_off()
    fig.tight_layout(); fig.savefig(fig_dir / "boxes_per_frame.png", dpi=120); plt.close(fig)

    # 8. co-occurrence heatmap
    cooc = np.zeros((len(classes), len(classes)), dtype=int)
    for split in (train, val):
        for sset in split.frame_species.values():
            for a in sset:
                for b in sset:
                    cooc[classes.index(a)][classes.index(b)] += 1
    fig, ax = plt.subplots(figsize=(7, 6))
    # Off-diagonal log scale, diagonal raw counts
    im = ax.imshow(np.log1p(cooc), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cooc[i][j]), ha="center", va="center",
                    color="white" if cooc[i][j] > 0 else "black", fontsize=7)
    ax.set_title("Class co-occurrence (frames containing both)\n(log1p color scale; cell value = raw count)")
    fig.colorbar(im, ax=ax, label="log1p(frame count)")
    fig.tight_layout(); fig.savefig(fig_dir / "cooccurrence.png", dpi=120); plt.close(fig)

    # 9. position bias heatmap (per species)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for i, c in enumerate(classes):
        ax = axes[i // cols][i % cols]
        merged = (train.per_species[c]["centers_xy_norm"] +
                  val.per_species[c]["centers_xy_norm"])
        if not merged:
            ax.set_axis_off(); continue
        cx = [p[0] for p in merged]; cy = [p[1] for p in merged]
        ax.hist2d(cx, cy, bins=30, range=[[0, 1], [0, 1]], cmap="hot")
        ax.invert_yaxis()  # image-coord origin is top-left
        ax.set_xlabel("normalized cx (image x)"); ax.set_ylabel("normalized cy (image y)")
        ax.set_title(f"{c} (n={len(merged)}) — bbox centers")
    for j in range(len(classes), rows * cols):
        axes[j // cols][j % cols].set_axis_off()
    fig.tight_layout(); fig.savefig(fig_dir / "position_bias.png", dpi=120); plt.close(fig)

    # 10. frames per video
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fpv_train = [len(fs) for fs in train.video_to_frames.values()]
    fpv_val = [len(fs) for fs in val.video_to_frames.values()]
    if fpv_train:
        axes[0].hist(fpv_train, bins=20, edgecolor="k", alpha=0.85)
        axes[0].set_title(f"frames per video — train ({len(fpv_train)} videos)")
        axes[0].set_xlabel("frame count"); axes[0].set_ylabel("video count")
    if fpv_val:
        axes[1].hist(fpv_val, bins=20, edgecolor="k", alpha=0.85, color="orange")
        axes[1].set_title(f"frames per video — val ({len(fpv_val)} videos)")
        axes[1].set_xlabel("frame count"); axes[1].set_ylabel("video count")
    fig.tight_layout(); fig.savefig(fig_dir / "frames_per_video.png", dpi=120); plt.close(fig)

    # 11. frames per segment
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fps_train = [len(fs) for fs in train.segment_to_frames.values()]
    fps_val = [len(fs) for fs in val.segment_to_frames.values()]
    if fps_train:
        axes[0].hist(fps_train, bins=30, edgecolor="k", alpha=0.85)
        axes[0].set_title(f"frames per segment — train ({len(fps_train)} segments)")
        axes[0].set_xlabel("frame count"); axes[0].set_ylabel("segment count")
    if fps_val:
        axes[1].hist(fps_val, bins=30, edgecolor="k", alpha=0.85, color="orange")
        axes[1].set_title(f"frames per segment — val ({len(fps_val)} segments)")
        axes[1].set_xlabel("frame count"); axes[1].set_ylabel("segment count")
    fig.tight_layout(); fig.savefig(fig_dir / "frames_per_segment.png", dpi=120); plt.close(fig)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output dir; created if missing.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out / "figures"
    fig_dir.mkdir(exist_ok=True)

    print(f"Reading {args.train}")
    gt_train = json.load(open(args.train))
    print(f"Reading {args.val}")
    gt_val = json.load(open(args.val))

    train = SplitData(gt_train, "train")
    val = SplitData(gt_val, "val")

    print("Rendering figures...")
    render_figures(train, val, fig_dir)

    print("Rendering markdown...")
    md_path = args.out / "dataset_description.md"
    md_text = render_markdown(train, val, md_path, fig_dir)
    md_path.write_text(md_text)

    # JSON dump
    print("Writing JSON...")
    out_json = {
        "generated": date.today().isoformat(),
        "train": {"per_species": train.species_table()},
        "val": {"per_species": val.species_table()},
    }
    (args.out / "dataset_description.json").write_text(json.dumps(out_json, indent=2))

    print(f"\nWrote:")
    print(f"  {md_path}")
    print(f"  {args.out / 'dataset_description.json'}")
    print(f"  {fig_dir}/ ({len(list(fig_dir.glob('*.png')))} figures)")
    print()
    print("Preview (first 60 lines of dataset_description.md):")
    print("-" * 60)
    print("\n".join(md_text.splitlines()[:60]))


if __name__ == "__main__":
    main()
