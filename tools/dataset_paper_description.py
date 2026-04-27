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

# Canonical species color palette — used consistently across ALL plots so a
# reader can follow each species across panels. Hand-picked to be perceptually
# distinct AND semantically suggestive (giraffe = tan, elephant = gray-blue,
# rhino = brown, gazelle = warm beige, plains_zebra = light slate, grevys_zebra
# = dark slate). Falls back gracefully if a class isn't in the dict.
SPECIES_COLORS = {
    "giraffe":      "#D4A24C",   # giraffe spots tan
    "grevys_zebra": "#2C2C3E",   # dark slate (narrow stripes)
    "elephant":     "#5D7282",   # gray-blue (skin tone)
    "plains_zebra": "#9AA3AE",   # light slate
    "rhino":        "#7B4A2A",   # warm brown
    "gazelle":      "#E8B870",   # warm beige
}
SPECIES_FALLBACK = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                    "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def species_color(name: str, fallback_idx: int = 0) -> str:
    return SPECIES_COLORS.get(name,
                              SPECIES_FALLBACK[fallback_idx % len(SPECIES_FALLBACK)])


def _style_axes(ax, *, grid: bool = True):
    """Common matplotlib axis polish — flat, no top/right spines, light grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)


def _violin_horizontal(ax, data_per_class, classes, palette,
                       *, log_x: bool = False, x_label: str = "",
                       title: str = "", show_means: bool = True,
                       reference_lines: Optional[List[Tuple[float, str, str]]] = None):
    """Horizontal violins, one per class, sorted by median ascending.
    `reference_lines` is a list of (x_value, color, label) for vertical refs."""
    # Sort classes by median for visual storytelling: low → high
    valid = [(c, d) for c, d in zip(classes, data_per_class) if d]
    if not valid:
        ax.set_axis_off(); return
    valid.sort(key=lambda kv: statistics.median(kv[1]))
    sorted_classes = [c for c, _ in valid]
    sorted_data = [d for _, d in valid]

    positions = list(range(1, len(sorted_classes) + 1))
    parts = ax.violinplot(sorted_data, positions=positions, vert=False,
                          showmeans=False, showextrema=False, widths=0.85)
    for body, c in zip(parts["bodies"], sorted_classes):
        body.set_facecolor(palette(c))
        body.set_edgecolor(palette(c))
        body.set_alpha(0.78)

    # Median markers + count badges
    for i, (c, d) in enumerate(valid):
        med = statistics.median(d)
        ax.scatter([med], [positions[i]], color="white", s=30, zorder=5,
                   edgecolor="black", linewidths=1.0)
        if show_means:
            mean = statistics.mean(d)
            ax.scatter([mean], [positions[i]], color="black", s=18, marker="d",
                       zorder=4)
        # n-badge on right edge
        ax.annotate(f"n={len(d):,}", xy=(1.0, positions[i]), xycoords=("axes fraction", "data"),
                    xytext=(-6, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8, alpha=0.65)

    if reference_lines:
        for x, col, lab in reference_lines:
            ax.axvline(x, color=col, linestyle=":", linewidth=1.2, alpha=0.7, label=lab)

    ax.set_yticks(positions)
    ax.set_yticklabels(sorted_classes)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    if title:
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
    if reference_lines:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.7)
    _style_axes(ax, grid=True)
    # Lighter horizontal grid since violins are horizontal
    ax.grid(True, axis="x", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)


def render_figures(train: SplitData, val: SplitData, fig_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Rectangle, Patch
    except ImportError:
        print("  (matplotlib not available; skipping figures)")
        return

    # Apply baseline style for the whole script
    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted({c for c in train.per_species} | {c for c in val.per_species})
    n_cls = len(classes)

    # Color closure that handles "any class" — preserves order-stability
    color_for = lambda c: species_color(
        c, fallback_idx=classes.index(c) if c in classes else 0)

    # =============================================================
    # Fig 1. species_counts.png — overview dashboard (3 panels)
    #        Tells the long-tail story up front.
    # =============================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    train_frames = [len(train.per_species[c]["frames"]) for c in classes]
    val_frames = [len(val.per_species[c]["frames"]) for c in classes]
    train_anns = [train.per_species[c]["annotations"] for c in classes]
    val_anns = [val.per_species[c]["annotations"] for c in classes]
    train_vids = [len(train.per_species[c]["videos"]) for c in classes]
    val_vids = [len(val.per_species[c]["videos"]) for c in classes]

    pos = np.arange(n_cls)
    bar_colors = [color_for(c) for c in classes]

    # (a) Frames stacked
    axes[0].bar(pos, train_frames, color=bar_colors, edgecolor="white", label="train", alpha=0.95)
    axes[0].bar(pos, val_frames, bottom=train_frames, color=bar_colors,
                edgecolor="white", hatch="///", alpha=0.55, label="val")
    axes[0].set_xticks(pos); axes[0].set_xticklabels(classes, rotation=20, ha="right")
    axes[0].set_ylabel("annotated frames")
    axes[0].set_title("(a)  Frames per species (train + val stacked)")
    axes[0].legend(loc="upper left", fontsize=9)
    _style_axes(axes[0])
    for i, (t, v) in enumerate(zip(train_frames, val_frames)):
        axes[0].text(pos[i], t + v + max(train_frames + val_frames) * 0.01,
                     f"{t+v:,}", ha="center", va="bottom", fontsize=8)

    # (b) Boxes log-scale (long-tail dramatic)
    totals_box = [t + v for t, v in zip(train_anns, val_anns)]
    axes[1].bar(pos, totals_box, color=bar_colors, edgecolor="white")
    axes[1].set_xticks(pos); axes[1].set_xticklabels(classes, rotation=20, ha="right")
    axes[1].set_ylabel("annotated boxes (log scale)")
    axes[1].set_yscale("log")
    axes[1].set_title("(b)  Boxes per species — log scale exposes the long tail")
    _style_axes(axes[1])
    for i, t in enumerate(totals_box):
        axes[1].text(pos[i], t * 1.08, f"{t:,}", ha="center", va="bottom", fontsize=8)

    # (c) Videos with rare-class hatching
    axes[2].bar(pos, train_vids, color=bar_colors, edgecolor="white", label="train")
    axes[2].bar(pos, val_vids, bottom=train_vids, color=bar_colors, alpha=0.55,
                edgecolor="white", hatch="///", label="val")
    axes[2].set_xticks(pos); axes[2].set_xticklabels(classes, rotation=20, ha="right")
    axes[2].set_ylabel("source videos")
    axes[2].set_title("(c)  Videos per species — bottleneck on rare-class generalization")
    axes[2].legend(loc="upper left", fontsize=9)
    axes[2].axhline(5, color="red", linestyle=":", linewidth=1.2, alpha=0.6,
                    label="rare-class line (≤5 train vids)")
    _style_axes(axes[2])
    for i, (t, v) in enumerate(zip(train_vids, val_vids)):
        if t <= 5:
            axes[2].annotate("rare", xy=(pos[i], t + v), xytext=(0, 5),
                             textcoords="offset points", ha="center",
                             fontsize=8, color="red", weight="bold")

    fig.suptitle("Dataset composition — long-tail species distribution typical of wildlife domains",
                 fontsize=13, weight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(fig_dir / "species_counts.png", dpi=140,
                                    bbox_inches="tight")
    plt.close(fig)

    # =============================================================
    # Fig 2. bbox_pixel_area.png — sorted horizontal violins
    #        Highlights small-object regime where it lives.
    # =============================================================
    fig, ax = plt.subplots(figsize=(11, 0.9 * n_cls + 1.5))
    data = [(train.per_species[c]["pixel_areas"] + val.per_species[c]["pixel_areas"])
            for c in classes]
    refs = [
        (32 * 32,  "#cc4444", "small-obj limit (32×32 px)"),
        (96 * 96,  "#cc8844", "medium-obj limit (96×96 px)"),
    ]
    _violin_horizontal(
        ax, data, classes, color_for,
        log_x=True, x_label="bounding-box pixel area (log px²)",
        title="Bounding-box size — gazelle and grevys_zebra dominate the small-object regime",
        reference_lines=refs)
    ax.text(0.01, 0.97,
            "● = median   ◆ = mean   |   sorted by median (small → large)",
            transform=ax.transAxes, fontsize=9, va="top", alpha=0.65)
    fig.tight_layout(); fig.savefig(fig_dir / "bbox_pixel_area.png", dpi=140,
                                    bbox_inches="tight")
    plt.close(fig)

    # =============================================================
    # Fig 3. bbox_area_ratio.png — area-as-fraction violins with semantic zones
    # =============================================================
    fig, ax = plt.subplots(figsize=(11, 0.9 * n_cls + 1.5))
    data = [[r * 100 for r in (train.per_species[c]["area_ratios"] +
                               val.per_species[c]["area_ratios"]) if r > 0]
            for c in classes]
    refs = [
        (0.1, "#cc4444", "0.1 % image area"),
        (1.0, "#cc8844", "1.0 % image area"),
        (10.0, "#cccc44", "10 % image area"),
    ]
    _violin_horizontal(
        ax, data, classes, color_for,
        log_x=True, x_label="bbox area as % of image (log scale)",
        title="Object size relative to image — drone altitude shapes the size distribution",
        reference_lines=refs)
    fig.tight_layout(); fig.savefig(fig_dir / "bbox_area_ratio.png", dpi=140,
                                    bbox_inches="tight")
    plt.close(fig)

    # =============================================================
    # Fig 4. bbox_aspect_ratio.png — square reference at w/h=1
    # =============================================================
    fig, ax = plt.subplots(figsize=(11, 0.9 * n_cls + 1.5))
    data = [(train.per_species[c]["aspect_ratios"] + val.per_species[c]["aspect_ratios"])
            for c in classes]
    refs = [(1.0, "#444444", "square (w=h)")]
    _violin_horizontal(
        ax, data, classes, color_for,
        log_x=False, x_label="aspect ratio (width / height)",
        title="Bbox aspect ratios — body shape signal: giraffes are tall (<1), elephants wide (>1)",
        reference_lines=refs)
    ax.set_xlim(0, 4)
    # Annotation zones
    ax.text(0.5, 0.95, "← TALLER", transform=ax.transAxes, fontsize=9,
            color="#666", alpha=0.6, ha="center")
    ax.text(0.95, 0.95, "WIDER →", transform=ax.transAxes, fontsize=9,
            color="#666", alpha=0.6, ha="right")
    fig.tight_layout(); fig.savefig(fig_dir / "bbox_aspect_ratio.png", dpi=140,
                                    bbox_inches="tight")
    plt.close(fig)

    # =============================================================
    # Fig 5. depth_distribution.png — VGGT-scale clustered around 1.0
    # =============================================================
    fig, ax = plt.subplots(figsize=(11, 0.9 * n_cls + 1.5))
    data = [(train.per_species[c]["depths_z"] + val.per_species[c]["depths_z"])
            for c in classes]
    refs = [(1.0, "#cc4444", "VGGT median |z| = 1 (per-segment normalization anchor)")]
    _violin_horizontal(
        ax, data, classes, color_for,
        log_x=False, x_label="camera-frame depth z (synthetic units)",
        title="Per-segment scale-normalized depth — distributions concentrate around z=1 by construction",
        reference_lines=refs)
    fig.tight_layout(); fig.savefig(fig_dir / "depth_distribution.png", dpi=140,
                                    bbox_inches="tight")
    plt.close(fig)

    # =============================================================
    # Fig 6. dimensions.png — 3-panel side-by-side body shape
    # =============================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, 0.7 * n_cls + 1.5)))
    axes_titles = [
        ("dim_W", "(a) Width (W)", "narrow ← W → broad"),
        ("dim_H", "(b) Height (H)", "short ← H → tall"),
        ("dim_L", "(c) Length (L)", "short ← L → long"),
    ]
    for i, (key, title, side) in enumerate(axes_titles):
        ax = axes[i]
        data = [(train.per_species[c][key] + val.per_species[c][key])
                for c in classes]
        _violin_horizontal(ax, data, classes, color_for,
                           log_x=False, x_label=f"{key.replace('dim_', '')} (synth. units)",
                           title=title)
        # Side annotation
        ax.text(0.5, -0.18, side, transform=ax.transAxes, fontsize=9,
                color="#666", ha="center")
    fig.suptitle("3D dimensions per species — body-shape fingerprints (Omni3D ordering W, H, L)",
                 fontsize=13, weight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(fig_dir / "dimensions.png", dpi=140,
                                    bbox_inches="tight")
    plt.close(fig)

    # =============================================================
    # Fig 7. boxes_per_frame.png — scene-density violins
    # =============================================================
    fig, ax = plt.subplots(figsize=(11, 0.9 * n_cls + 1.5))
    data = [(list(train.per_species[c]["boxes_per_frame"].values()) +
             list(val.per_species[c]["boxes_per_frame"].values()))
            for c in classes]
    refs = [
        (1, "#999999", "1 (sparse)"),
        (5, "#cc8844", "5 (group)"),
        (10, "#cc4444", "10 (dense herd)"),
    ]
    _violin_horizontal(
        ax, data, classes, color_for,
        log_x=False, x_label="boxes per annotated frame",
        title="Scene density — group/herd vs solo: dense scenes concentrate the depth-bottleneck error",
        reference_lines=refs)
    fig.tight_layout(); fig.savefig(fig_dir / "boxes_per_frame.png", dpi=140,
                                    bbox_inches="tight")
    plt.close(fig)

    # =============================================================
    # Fig 8. cooccurrence.png — two-panel: raw + conditional probability
    # =============================================================
    cooc = np.zeros((n_cls, n_cls), dtype=int)
    for split in (train, val):
        for sset in split.frame_species.values():
            for a in sset:
                for b in sset:
                    cooc[classes.index(a)][classes.index(b)] += 1
    # Conditional: P(B in frame | A in frame) = cooc[A,B] / cooc[A,A]
    cond = np.where(cooc.diagonal()[:, None] > 0,
                    cooc / np.maximum(cooc.diagonal()[:, None], 1),
                    0.0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # (a) raw counts (log-color)
    im0 = axes[0].imshow(np.log1p(cooc), cmap="viridis", aspect="auto")
    axes[0].set_xticks(range(n_cls)); axes[0].set_xticklabels(classes, rotation=30, ha="right")
    axes[0].set_yticks(range(n_cls)); axes[0].set_yticklabels(classes)
    for i in range(n_cls):
        for j in range(n_cls):
            v = cooc[i][j]
            color = "white" if np.log1p(v) > np.log1p(cooc.max()) * 0.55 else "black"
            axes[0].text(j, i, f"{v:,}", ha="center", va="center", color=color, fontsize=8)
    axes[0].set_title("(a) Co-occurrence — frames containing both species (log color)")
    fig.colorbar(im0, ax=axes[0], label="log1p(frame count)", fraction=0.04)

    # (b) conditional probability — interpretable for ecology audience
    im1 = axes[1].imshow(cond, cmap="rocket_r" if "rocket_r" in plt.colormaps() else "magma",
                         aspect="auto", vmin=0, vmax=1)
    axes[1].set_xticks(range(n_cls)); axes[1].set_xticklabels(classes, rotation=30, ha="right")
    axes[1].set_yticks(range(n_cls)); axes[1].set_yticklabels(classes)
    for i in range(n_cls):
        for j in range(n_cls):
            v = cond[i][j]
            if v < 0.005:
                continue
            color = "white" if v > 0.5 else "black"
            axes[1].text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                         color=color, fontsize=8)
    axes[1].set_title("(b) P(column species in frame | row species in frame)")
    axes[1].set_xlabel("co-occurring species")
    axes[1].set_ylabel("conditional on row species present")
    fig.colorbar(im1, ax=axes[1], label="conditional probability", fraction=0.04)

    fig.suptitle(
        "Class co-occurrence — almost all frames are single-species "
        "(diagonal-dominant); inter-species mixing is rare in drone footage",
        fontsize=13, weight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(fig_dir / "cooccurrence.png", dpi=140,
                                    bbox_inches="tight")
    plt.close(fig)

    # =============================================================
    # Fig 9. position_bias.png — per-species heatmaps with image-frame overlay
    # =============================================================
    cols = 3
    rows = (n_cls + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.5 * rows), squeeze=False)
    for i, c in enumerate(classes):
        ax = axes[i // cols][i % cols]
        merged = (train.per_species[c]["centers_xy_norm"] +
                  val.per_species[c]["centers_xy_norm"])
        if not merged:
            ax.set_axis_off(); continue
        cx = np.array([p[0] for p in merged])
        cy = np.array([p[1] for p in merged])
        h = ax.hist2d(cx, cy, bins=24, range=[[0, 1], [0, 1]],
                       cmap="rocket_r" if "rocket_r" in plt.colormaps() else "magma",
                       cmin=1)
        # image frame outline
        ax.add_patch(Rectangle((0, 0), 1, 1, linewidth=1.2, edgecolor="black",
                               facecolor="none", alpha=0.6))
        # centroid marker
        ax.scatter([cx.mean()], [cy.mean()], s=50, marker="x",
                   color=color_for(c), linewidths=2.5,
                   label=f"{c} centroid", zorder=5)
        ax.invert_yaxis()  # image-coord origin top-left
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(1.05, -0.05)
        ax.set_xlabel("normalized image x"); ax.set_ylabel("normalized image y")
        ax.set_title(f"{c}  (n = {len(merged):,})", fontsize=10,
                     color=color_for(c), weight="bold")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.7)
        ax.set_aspect("equal")
    for j in range(n_cls, rows * cols):
        axes[j // cols][j % cols].set_axis_off()
    fig.suptitle("Where do animals appear in the frame? — drone perspective biases per species",
                 fontsize=13, weight="bold", y=1.005)
    fig.tight_layout(); fig.savefig(fig_dir / "position_bias.png", dpi=140,
                                    bbox_inches="tight")
    plt.close(fig)

    # =============================================================
    # Fig 10. video_segment_distribution.png — combined 4-panel
    # =============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (a) frames per video
    fpv_train = sorted([len(fs) for fs in train.video_to_frames.values()])
    fpv_val = sorted([len(fs) for fs in val.video_to_frames.values()])
    if fpv_train + fpv_val:
        bins = np.logspace(np.log10(max(1, min(fpv_train + fpv_val))),
                           np.log10(max(fpv_train + fpv_val) + 1), 25)
        axes[0, 0].hist([fpv_train, fpv_val], bins=bins, label=["train", "val"],
                        color=["#3a86b8", "#e89742"], edgecolor="white", stacked=False)
        axes[0, 0].set_xscale("log")
        axes[0, 0].set_xlabel("frames per video (log)")
        axes[0, 0].set_ylabel("video count")
        axes[0, 0].set_title("(a) Video duration: train vs val")
        axes[0, 0].legend(fontsize=9, loc="upper left")
        # Median markers
        if fpv_train:
            axes[0, 0].axvline(np.median(fpv_train), color="#3a86b8",
                               linestyle="--", linewidth=1.5, alpha=0.7)
        if fpv_val:
            axes[0, 0].axvline(np.median(fpv_val), color="#e89742",
                               linestyle="--", linewidth=1.5, alpha=0.7)
    _style_axes(axes[0, 0])

    # (b) segments per video
    spv_train = [len(s) for s in train.video_to_segments.values()]
    spv_val = [len(s) for s in val.video_to_segments.values()]
    if spv_train + spv_val:
        all_segs = spv_train + spv_val
        bins = np.arange(0, max(all_segs) + 2)
        axes[0, 1].hist([spv_train, spv_val], bins=bins, label=["train", "val"],
                        color=["#3a86b8", "#e89742"], edgecolor="white")
        axes[0, 1].set_xlabel("segments per video")
        axes[0, 1].set_ylabel("video count")
        axes[0, 1].set_title("(b) Segments per video — multiple sub-clips per recording")
        axes[0, 1].legend(fontsize=9, loc="upper right")
    _style_axes(axes[0, 1])

    # (c) frames per segment
    fps_train = sorted([len(fs) for fs in train.segment_to_frames.values()])
    fps_val = sorted([len(fs) for fs in val.segment_to_frames.values()])
    if fps_train + fps_val:
        bins = np.logspace(np.log10(max(1, min(fps_train + fps_val))),
                           np.log10(max(fps_train + fps_val) + 1), 30)
        axes[1, 0].hist([fps_train, fps_val], bins=bins, label=["train", "val"],
                        color=["#3a86b8", "#e89742"], edgecolor="white")
        axes[1, 0].set_xscale("log")
        axes[1, 0].set_xlabel("frames per segment (log)")
        axes[1, 0].set_ylabel("segment count")
        axes[1, 0].set_title("(c) Segment duration: contiguous-frame chunks for tracking")
        axes[1, 0].legend(fontsize=9, loc="upper left")
    _style_axes(axes[1, 0])

    # (d) Top-10 longest videos
    all_vid_frames = collections.Counter()
    video_split = {}
    for split, label in ((train, "train"), (val, "val")):
        for vid, frames in split.video_to_frames.items():
            all_vid_frames[vid] += len(frames)
            video_split[vid] = label
    top10 = all_vid_frames.most_common(10)
    if top10:
        names = [v[0] for v in top10][::-1]
        counts = [v[1] for v in top10][::-1]
        cols_v = [("#3a86b8" if video_split[v[0]] == "train" else "#e89742")
                  for v in top10][::-1]
        axes[1, 1].barh(range(len(names)), counts, color=cols_v, edgecolor="white")
        axes[1, 1].set_yticks(range(len(names)))
        # Truncate long video names
        labels_short = [(n if len(n) <= 28 else n[:25] + "...") for n in names]
        axes[1, 1].set_yticklabels(labels_short, fontsize=8)
        axes[1, 1].set_xlabel("frame count")
        axes[1, 1].set_title("(d) Top 10 longest videos (color = train / val)")
        for i, c in enumerate(counts):
            axes[1, 1].text(c, i, f"  {c:,}", va="center", fontsize=8)
    _style_axes(axes[1, 1])

    fig.suptitle("Video / segment / frame breakdown — duration is heavy-tailed across the corpus",
                 fontsize=13, weight="bold", y=1.005)
    fig.tight_layout(); fig.savefig(fig_dir / "video_segment_distribution.png", dpi=140,
                                    bbox_inches="tight")
    plt.close(fig)

    # =============================================================
    # Fig 11 (NEW): bbox_size_vs_depth.png — the "small-and-far" story
    #        Log-log scatter of pixel area vs depth, colored per species.
    #        Single most striking figure: shows that small-bbox + large-depth
    #        is the regime, and per-species clusters tell apart the species.
    # =============================================================
    fig, ax = plt.subplots(figsize=(10, 7))
    handles = []
    for c in classes:
        depths = (train.per_species[c]["depths_z"] +
                  val.per_species[c]["depths_z"])
        areas = (train.per_species[c]["pixel_areas"] +
                 val.per_species[c]["pixel_areas"])
        if not depths or not areas:
            continue
        # Subsample if very large for readability
        n = min(len(depths), len(areas))
        depths = depths[:n]; areas = areas[:n]
        if n > 4000:
            stride = n // 4000
            depths = depths[::stride]; areas = areas[::stride]
        col = color_for(c)
        ax.scatter(depths, areas, s=8, alpha=0.35, color=col,
                   edgecolors="none", label=c)
        # Centroid
        med_d = statistics.median(depths)
        med_a = statistics.median(areas)
        ax.scatter([med_d], [med_a], marker="X", s=240, color=col,
                   edgecolors="black", linewidths=1.4, zorder=5)
    ax.set_xlabel("camera-frame depth z (synthetic units, per-segment median = 1)")
    ax.set_ylabel("bbox pixel area  (px²)")
    ax.set_yscale("log")
    ax.set_title("Bbox size vs. depth — closer animals subtend larger pixel area "
                 "(× = per-species median)", loc="left", fontsize=12, weight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.75, markerscale=2.0)
    _style_axes(ax)
    ax.grid(True, which="both", axis="both", alpha=0.2, linestyle="--", linewidth=0.5)
    fig.tight_layout(); fig.savefig(fig_dir / "bbox_size_vs_depth.png", dpi=140,
                                    bbox_inches="tight")
    plt.close(fig)

    # =============================================================
    # Fig 12 (NEW): species_dashboard.png — one-page TL;DR
    #        Single figure for talks, abstract figures, the
    #        "everything at a glance" panel of the dataset paper.
    # =============================================================
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.6, 1.4, 1.4],
                          width_ratios=[1, 1, 1, 1.0])

    # Header strip: species color chips
    ax_hdr = fig.add_subplot(gs[0, :])
    for i, c in enumerate(classes):
        ax_hdr.add_patch(Rectangle((i, 0), 0.85, 1, color=color_for(c)))
        ax_hdr.text(i + 0.425, 0.5, c, ha="center", va="center",
                    fontsize=11, weight="bold",
                    color=("white" if c in ("grevys_zebra", "rhino", "elephant")
                           else "black"))
    ax_hdr.set_xlim(-0.1, n_cls + 0.1); ax_hdr.set_ylim(0, 1)
    ax_hdr.set_axis_off()
    ax_hdr.set_title("WildBox species — color coding consistent across the entire dataset description",
                     fontsize=12, weight="bold")

    # Panel a: count comparison (frames)
    ax_a = fig.add_subplot(gs[1, 0])
    ax_a.bar(range(n_cls), [len(train.per_species[c]["frames"]) +
                            len(val.per_species[c]["frames"]) for c in classes],
             color=[color_for(c) for c in classes], edgecolor="white")
    ax_a.set_xticks(range(n_cls)); ax_a.set_xticklabels(classes, rotation=30, ha="right",
                                                        fontsize=9)
    ax_a.set_ylabel("frames")
    ax_a.set_title("(a) Frame count")
    _style_axes(ax_a)

    # Panel b: median bbox area (px²) per species
    ax_b = fig.add_subplot(gs[1, 1])
    medians = []
    for c in classes:
        merged = train.per_species[c]["pixel_areas"] + val.per_species[c]["pixel_areas"]
        medians.append(statistics.median(merged) if merged else 0)
    ax_b.bar(range(n_cls), medians, color=[color_for(c) for c in classes],
             edgecolor="white")
    ax_b.set_xticks(range(n_cls)); ax_b.set_xticklabels(classes, rotation=30, ha="right",
                                                        fontsize=9)
    ax_b.set_yscale("log")
    ax_b.set_ylabel("median bbox px² (log)")
    ax_b.set_title("(b) Median bbox size")
    _style_axes(ax_b)

    # Panel c: median depth z per species
    ax_c = fig.add_subplot(gs[1, 2])
    medians = []
    for c in classes:
        merged = train.per_species[c]["depths_z"] + val.per_species[c]["depths_z"]
        medians.append(statistics.median(merged) if merged else 0)
    ax_c.bar(range(n_cls), medians, color=[color_for(c) for c in classes],
             edgecolor="white")
    ax_c.axhline(1.0, color="red", linestyle="--", alpha=0.6,
                 label="VGGT median z=1")
    ax_c.set_xticks(range(n_cls)); ax_c.set_xticklabels(classes, rotation=30, ha="right",
                                                        fontsize=9)
    ax_c.set_ylabel("median depth z")
    ax_c.set_title("(c) Median depth (synth.)")
    ax_c.legend(fontsize=8)
    _style_axes(ax_c)

    # Panel d: median boxes per frame
    ax_d = fig.add_subplot(gs[1, 3])
    medians = []
    for c in classes:
        merged = (list(train.per_species[c]["boxes_per_frame"].values()) +
                  list(val.per_species[c]["boxes_per_frame"].values()))
        medians.append(statistics.median(merged) if merged else 0)
    ax_d.bar(range(n_cls), medians, color=[color_for(c) for c in classes],
             edgecolor="white")
    ax_d.set_xticks(range(n_cls)); ax_d.set_xticklabels(classes, rotation=30, ha="right",
                                                        fontsize=9)
    ax_d.set_ylabel("median boxes / frame")
    ax_d.set_title("(d) Scene density")
    _style_axes(ax_d)

    # Bottom row: bbox-size-vs-depth scatter (the "story" panel)
    ax_e = fig.add_subplot(gs[2, :])
    for c in classes:
        depths = (train.per_species[c]["depths_z"] +
                  val.per_species[c]["depths_z"])
        areas = (train.per_species[c]["pixel_areas"] +
                 val.per_species[c]["pixel_areas"])
        if not depths or not areas:
            continue
        n = min(len(depths), len(areas))
        depths = depths[:n]; areas = areas[:n]
        if n > 3000:
            stride = n // 3000
            depths = depths[::stride]; areas = areas[::stride]
        ax_e.scatter(depths, areas, s=10, alpha=0.4,
                     color=color_for(c), edgecolors="none", label=c)
    ax_e.set_yscale("log")
    ax_e.set_xlabel("camera-frame depth z (synthetic units)")
    ax_e.set_ylabel("bbox pixel area  (log px²)")
    ax_e.set_title("(e) Bbox size vs. depth — per-species clusters separate by body size and altitude",
                    loc="left")
    ax_e.legend(loc="upper right", fontsize=9, markerscale=2.0)
    _style_axes(ax_e)
    ax_e.grid(True, which="both", axis="both", alpha=0.2, linestyle="--", linewidth=0.5)

    fig.suptitle("WildBox Dataset Dashboard — TL;DR for ML and ecology audiences",
                 fontsize=14, weight="bold", y=1.005)
    fig.tight_layout(); fig.savefig(fig_dir / "species_dashboard.png", dpi=140,
                                    bbox_inches="tight")
    plt.close(fig)

    # =============================================================
    # Legacy named figures kept as compat aliases (old paths still resolve)
    # =============================================================
    # We unified the two distribution figures into video_segment_distribution.png;
    # save thin alias copies for any code/script still referencing the old names.
    import shutil
    for src, dst in [("video_segment_distribution.png", "frames_per_video.png"),
                     ("video_segment_distribution.png", "frames_per_segment.png")]:
        try:
            shutil.copyfile(fig_dir / src, fig_dir / dst)
        except Exception:
            pass


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
