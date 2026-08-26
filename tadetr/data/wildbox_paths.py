"""Join the three WildBox trees. All of M1 stands on this join, so it fails LOUDLY.

The three trees (verified 2026-08-26 census):

1. DENSE (VGGT outputs; local /mnt/d only, 345 segments, 56 GB of depth):
   /mnt/d/3DBOX/Data/WildBox/data/<shoot>/WildBox_sam3-vggtv1_processed/WildBox/<video>/<seg>/
       vggt_results/{depth_maps.npz, cameras.json, tracking_summary.json, annotations/...}
       sam3_masks/masks/obj_<track_id>/frame_%06d.png     (1920x1080 binary {0,255})

2. RELEASE (images + kitti labels the detector jsons reference):
   /mnt/d/3DBOX/papersubdata/<group>/<video>/<seg>/frame_*.jpg

3. DETECTOR JSONS (Omni3D format): image["file_path"] = <group>/<video>/<seg>/frame_XXXXXX.jpg

<shoot> <-> <group> via /mnt/d/3DBOX/wildbox_alias_map.json (scheme B: DJI video dirnames are
IDENTICAL in all trees, so (video, seg) is the primary join key; the alias only names the shoot).

Known landmines carried from the previous project:
- `_reid_*` experiment dirs sit next to real videos under two zebra shoots -- filtered here.
- `problem segments.txt` in the data root names known-flaky segments -- exposed as KNOWN_FLAKY.
- `vggt_results/annotations/tracking_summary.json` (human-corrected, 287/345) is preferred over the
  raw top-level one when present; which one was used is recorded per segment.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DATA_ROOT = Path("/mnt/d/3DBOX/Data/WildBox/data")
ALIAS_MAP_PATH = Path("/mnt/d/3DBOX/wildbox_alias_map.json")
PROCESSED_DIRNAME = "WildBox_sam3-vggtv1_processed"


@dataclass(frozen=True)
class SegmentPaths:
    shoot: str
    group: str            # paper alias, e.g. "zebr3" ("" if shoot not in the alias map)
    video: str
    seg: str
    seg_dir: Path

    @property
    def key(self) -> tuple[str, str]:
        return (self.video, self.seg)

    @property
    def name(self) -> str:
        return f"{self.group or self.shoot}__{self.video}__{self.seg}"

    @property
    def vggt_dir(self) -> Path:
        return self.seg_dir / "vggt_results"

    @property
    def depth_npz(self) -> Path:
        return self.vggt_dir / "depth_maps.npz"

    @property
    def cameras_json(self) -> Path:
        return self.vggt_dir / "cameras.json"

    @property
    def tracking_summary(self) -> Path:
        """Human-corrected annotations when present (287/345), else raw pipeline output."""
        corrected = self.vggt_dir / "annotations" / "tracking_summary.json"
        return corrected if corrected.exists() else self.vggt_dir / "tracking_summary.json"

    @property
    def tracking_is_corrected(self) -> bool:
        return (self.vggt_dir / "annotations" / "tracking_summary.json").exists()

    @property
    def masks_dir(self) -> Path:
        return self.seg_dir / "sam3_masks" / "masks"


def load_alias_map(path: Path = ALIAS_MAP_PATH) -> dict[str, str]:
    """shoot dirname -> paper group alias (e.g. '2025_07_Zebras_BlTo' -> 'zebr3')."""
    d = json.loads(path.read_text())
    return {shoot: rec["alias"] for shoot, rec in d["datasets"].items()}


def iter_dense_segments(root: Path = DATA_ROOT) -> Iterator[SegmentPaths]:
    """Every segment with dense VGGT outputs. Filters `_reid_*`/underscore experiment dirs."""
    alias = load_alias_map()
    for shoot_dir in sorted(root.iterdir()):
        wb = shoot_dir / PROCESSED_DIRNAME / "WildBox"
        if not wb.is_dir():
            continue
        for video_dir in sorted(wb.iterdir()):
            if not video_dir.is_dir() or video_dir.name.startswith("_"):
                continue
            for seg_dir in sorted(video_dir.iterdir()):
                if not seg_dir.is_dir() or not re.fullmatch(r"seg\d+", seg_dir.name):
                    continue
                if (seg_dir / "vggt_results" / "depth_maps.npz").exists():
                    yield SegmentPaths(
                        shoot=shoot_dir.name,
                        group=alias.get(shoot_dir.name, ""),
                        video=video_dir.name,
                        seg=seg_dir.name,
                        seg_dir=seg_dir,
                    )


def index_by_video_seg(root: Path = DATA_ROOT) -> dict[tuple[str, str], SegmentPaths]:
    """(video, seg) -> SegmentPaths; raises on a duplicate key (would silently corrupt joins)."""
    out: dict[tuple[str, str], SegmentPaths] = {}
    for sp in iter_dense_segments(root):
        if sp.key in out:
            raise ValueError(f"duplicate (video, seg) across shoots: {sp.key} in "
                             f"{out[sp.key].shoot} and {sp.shoot}")
        out[sp.key] = sp
    return out


def parse_label_path(file_path: str) -> tuple[str, str, str, str]:
    """Detector-json image file_path -> (group, video, seg, image_name). Last-3-components
    convention (the build_heading_labels.py join rule -- path prefixes differ across machines)."""
    parts = file_path.replace("\\", "/").split("/")
    return parts[-4] if len(parts) >= 4 else "", parts[-3], parts[-2], parts[-1]


def known_flaky(root: Path = DATA_ROOT) -> list[tuple[str, str, str]]:
    """(shoot, video, seg) triples from `problem segments.txt` (known-flaky, tagged not skipped)."""
    f = root / "problem segments.txt"
    out = []
    if f.exists():
        for line in f.read_text().splitlines():
            m = re.search(r"\]\s*(\S+)\s*/\s*(\S+)\s*/\s*(\S+)", line)
            if m:
                out.append((m.group(1), m.group(2), m.group(3)))
    return out
