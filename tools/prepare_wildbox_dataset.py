#!/usr/bin/env python
"""
Convert WildBox (SAM3 + VGGT) drone wildlife pseudo-labels into an Omni3D-style
JSON so OVMono3D can fine-tune on them.

We read VGGT's full-precision tracking_summary.json + cameras.json directly
rather than the lossy kitti_labels/*.txt export. The KITTI files drop two
things that matter for drone top-down shots:
  - dimensions written with .2f precision (significant for synthetic-scale
    values near 0.1)
  - the full 3x3 rotation matrix collapsed to a scalar rot_y (pitch/roll
    are lost, which is most of the orientation for top-down cameras)

Input: one or more --source entries. Each source points directly to the
directory that contains <video>/<segment>/vggt_results/. For example:

  --source /storage3/.../dataRhinoCami2025/WildBoxVGGT-v1/WildBox=rhino:1004
  --source /storage3/.../data202406KElephants/WildBox_vggtv1/WildBox=elephant:1002

Per segment (produced by the user's VGGT+SAM3 pipeline) we expect:
  <source_path>/<video>/<seg>/
    frame_XXXXXX.jpg
    vggt_results/cameras.json           # per-frame intrinsic + extrinsic
    vggt_results/tracking_summary.json  # per-track 3D state (world coords)

Output:
  datasets/Omni3D/<name>.json           # Omni3D-style dict (absolute file_paths)

file_path is stored as the absolute path to the JPEG. Omni3D's loader does
os.path.join(image_root, file_path); when file_path is absolute, Python's
os.path.join returns it unchanged, so no symlink dance under datasets/ is
required.

Scale: VGGT output is in its own synthetic units. We apply a per-segment
scalar so median |Z_cam| across all boxes in the segment maps to 1.0.
This is uniform — centers, dimensions, and corners all scale together.
Intrinsic K is NOT rescaled: (X/Z, Y/Z) is scale-invariant, so 2D
projections remain valid.

Axis conventions (both VGGT and Omni3D use the SAME local frame layout,
despite different field naming):

  VGGT get_corners() in demo_viser_tracking.py:109-121:
    l, w, h = self.dimensions
    corners_local = [ (+/- l/2,  +/- w/2,  +/- h/2 ) ]
                      X-axis    Y-axis    Z-axis

  Omni3D get_cuboid_verts_faces in cubercnn/util/math_util.py:172-181:
    dimensions = [W, H, L]
    verts[X-axis] = +/- L/2       -> X-ext = L
    verts[Y-axis] = +/- H/2       -> Y-ext = H
    verts[Z-axis] = +/- W/2       -> Z-ext = W

  Matching them axis by axis:
    VGGT.l == Omni3D.L   (X-ext)
    VGGT.w == Omni3D.H   (Y-ext)
    VGGT.h == Omni3D.W   (Z-ext)

  So Omni3D dimensions [W, H, L] = [VGGT.h, VGGT.w, VGGT.l]
                                 = reversed(VGGT.dimensions)

Rotation + center: VGGT stores both in WORLD coordinates. Apply the
per-frame 3x4 extrinsic to transform into camera space:
    R_cam      = R_ext @ R_world
    center_cam = R_ext @ center_world + t_ext
The same R (3x3) then drives Omni3D's get_cuboid_verts_faces because
both frameworks use the same (X, Y, Z) local axis layout.
"""

import argparse
import json
import logging
import os
import random
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("prepare_wildbox_dataset")


def resolve_source_path(path: Path) -> Optional[Path]:
    """Resolve a --source path. If it points at a .zip, auto-extract to a
    sibling directory `<name>_unzipped/` and return that directory. Idempotent:
    if the sibling directory already exists, skip extraction.

    On missing path / corrupt zip / partial transfer, log a warning and
    return None -- the caller should skip that source. This lets the prep
    script proceed when some zips in a batch are still transferring.

    Heuristic for finding <video>/<seg>/vggt_results/ inside the zip:
      1. If the extracted root directly contains <video>/<seg>/vggt_results, use it.
      2. Otherwise, descend into single-child directories until we find one
         whose grandchildren contain vggt_results. That handles zips wrapped
         in a redundant top-level folder (the common case with
         `zip -r archive.zip folder/`).
    """
    if path.is_dir():
        return path
    if path.is_file() and path.suffix.lower() == ".zip":
        extract_dir = path.with_name(path.stem + "_unzipped")
        if not extract_dir.exists():
            logger.info(f"Extracting {path} -> {extract_dir}")
            extract_dir.mkdir(parents=True, exist_ok=False)
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(extract_dir)
            except (zipfile.BadZipFile, OSError, EOFError) as e:
                import shutil
                shutil.rmtree(extract_dir, ignore_errors=True)
                logger.warning(
                    f"SKIP: {path} is not a valid zip ({type(e).__name__}: "
                    f"{e}). Likely a partial/in-progress transfer. Rerun "
                    f"prep later with this source."
                )
                return None
            except Exception:
                import shutil
                shutil.rmtree(extract_dir, ignore_errors=True)
                raise
        else:
            logger.info(f"Reusing existing extraction: {extract_dir}")

        # Try to find the real data root (skip wrapper dirs).
        candidate = extract_dir
        for _ in range(5):  # bound the descent to avoid infinite loops
            # Does candidate already contain <video>/<seg>/vggt_results?
            for video_dir in candidate.iterdir() if candidate.is_dir() else []:
                if not video_dir.is_dir():
                    continue
                for seg_dir in video_dir.iterdir():
                    if seg_dir.is_dir() and (seg_dir / "vggt_results").exists():
                        return candidate
            # Otherwise, if there's exactly one subdirectory, descend.
            subdirs = [p for p in candidate.iterdir() if p.is_dir()] if candidate.is_dir() else []
            if len(subdirs) == 1:
                candidate = subdirs[0]
                continue
            break
        return candidate
    logger.warning(
        f"SKIP: --source path does not exist or is not a .zip: {path}"
    )
    return None


@dataclass
class VggtBox:
    """Full-precision 3D box read from VGGT tracking_summary.json."""
    track_id: int
    frame_seq: int
    class_name: str
    center_world: np.ndarray
    dimensions_lwh: np.ndarray
    R_world: np.ndarray
    bbox_2d_vggt: Tuple[float, float, float, float]
    confidence: float


@dataclass
class Source:
    """One data source on disk. `path` contains <video>/<seg>/ subdirs."""
    path: Path
    category_name: str
    category_id: int


def vggt_corners_world(center: np.ndarray, dims_lwh: np.ndarray,
                       R: np.ndarray) -> np.ndarray:
    l, w, h = float(dims_lwh[0]), float(dims_lwh[1]), float(dims_lwh[2])
    corners_local = np.array([
        [-l/2, -w/2, -h/2], [+l/2, -w/2, -h/2],
        [+l/2, +w/2, -h/2], [-l/2, +w/2, -h/2],
        [-l/2, -w/2, +h/2], [+l/2, -w/2, +h/2],
        [+l/2, +w/2, +h/2], [-l/2, +w/2, +h/2],
    ], dtype=np.float64)
    return (R @ corners_local.T).T + center


def project_points(corners_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    xs = corners_cam[:, 0] / corners_cam[:, 2]
    ys = corners_cam[:, 1] / corners_cam[:, 2]
    u = K[0, 0] * xs + K[0, 2]
    v = K[1, 1] * ys + K[1, 2]
    return np.stack([u, v], axis=1)


def discover_segments(sources: List[Source]) -> List[Dict]:
    """Walk each source path and enumerate its <video>/<seg> subdirs."""
    segments = []
    for src in sources:
        if not src.path.exists():
            logger.warning(f"Source path not found: {src.path}")
            continue
        for video_dir in sorted(src.path.iterdir()):
            if not video_dir.is_dir():
                continue
            for seg_dir in sorted(video_dir.iterdir()):
                if not seg_dir.is_dir():
                    continue
                cameras_json = seg_dir / "vggt_results" / "cameras.json"
                tracking_json = seg_dir / "vggt_results" / "tracking_summary.json"
                if not cameras_json.exists() or not tracking_json.exists():
                    logger.debug(f"Skipping incomplete segment: {seg_dir}")
                    continue
                # Unique id across sources: prefix with category to dodge
                # name clashes (both sources can have the same video/seg
                # name by coincidence).
                seg_id = f"{src.category_name}/{video_dir.name}/{seg_dir.name}"
                segments.append({
                    "seg_id": seg_id,
                    "category_name": src.category_name,
                    "category_id": src.category_id,
                    "video": video_dir.name,
                    "seg": seg_dir.name,
                    "seg_dir": seg_dir,
                    "cameras_json": cameras_json,
                    "tracking_json": tracking_json,
                })
    return segments


def read_jpeg_dimensions(path: Path) -> Tuple[int, int]:
    with open(path, "rb") as f:
        data = f.read(2)
        if data != b"\xff\xd8":
            raise ValueError(f"{path} is not a JPEG")
        while True:
            marker = f.read(2)
            if len(marker) != 2:
                raise ValueError(f"{path}: unexpected EOF")
            if marker[0] != 0xFF:
                raise ValueError(f"{path}: bad marker byte {marker[0]:#x}")
            m = marker[1]
            if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                     0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                _length = struct.unpack(">H", f.read(2))[0]
                _precision = f.read(1)
                h, w = struct.unpack(">HH", f.read(4))
                return h, w
            length = struct.unpack(">H", f.read(2))[0]
            f.seek(length - 2, 1)


def _load_mask_bbox(mask_path: Path) -> Optional[Tuple[float, float, float, float, int]]:
    """Load an 8-bit PNG mask and return (x1, y1, x2, y2, pixel_count).

    Returns None if the mask file is missing, unreadable, or empty.
    Pure-Python (no cv2/PIL dep beyond PIL which is already in the env via
    torchvision). Uses zlib + manual IDAT decode only if PIL isn't
    available, but we assume PIL is present.
    """
    try:
        from PIL import Image
        import numpy as _np
        with Image.open(mask_path) as im:
            a = _np.asarray(im.convert("L"), dtype=_np.uint8)
    except Exception:
        return None
    ys, xs = (a > 0).nonzero()
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()),
            float(xs.max()), float(ys.max()),
            int(xs.size))


def _find_sam3_masks_dir(seg_dir: Path) -> Optional[Path]:
    """Locate the SAM3 per-object mask directory.

    Expected layout produced by the WildBox SAM3 pipeline:
        <seg>/sam3_masks/masks/obj_<id>/frame_NNNNNN.png
    Returns the `sam3_masks/masks/` directory if present, else None.
    """
    candidate = seg_dir / "sam3_masks" / "masks"
    if candidate.is_dir():
        return candidate
    return None


def load_segment(seg: Dict) -> Optional[Dict]:
    with open(seg["cameras_json"], "r") as f:
        cam_data = json.load(f)
    with open(seg["tracking_json"], "r") as f:
        track_data = json.load(f)

    real_h, real_w = None, None
    for cam in cam_data["cameras"]:
        candidate = seg["seg_dir"] / cam["image_name"]
        if candidate.exists():
            try:
                real_h, real_w = read_jpeg_dimensions(candidate)
            except Exception as e:
                logger.warning(f"{candidate}: {e}")
                continue
            break
    if real_h is None:
        logger.warning(f"No readable frame in {seg['seg_id']}; skipping")
        return None

    vggt_w = int(cam_data["cameras"][0]["image_width"])
    vggt_h = int(cam_data["cameras"][0]["image_height"])
    sx = real_w / vggt_w
    sy = real_h / vggt_h

    frames_seq: Dict[int, Dict] = {}
    for cam in cam_data["cameras"]:
        K_vggt = np.array(cam["intrinsic"], dtype=np.float64)
        K_real = K_vggt.copy()
        K_real[0, 0] *= sx
        K_real[1, 1] *= sy
        K_real[0, 2] *= sx
        K_real[1, 2] *= sy

        ext = np.array(cam["extrinsic"], dtype=np.float64)

        frames_seq[int(cam["frame_index"])] = {
            "image_name": cam["image_name"],
            "K_real": K_real,
            "extrinsic": ext,
            "image_height": real_h,
            "image_width": real_w,
        }

    boxes_by_frame: Dict[int, List[VggtBox]] = {}
    for tid_str, track in track_data["tracks"].items():
        tid = int(tid_str)
        centers = track["centers"]
        dims_list = track["dimensions"]
        rots = track["rotation_matrices"]
        bb2d = track["bbox_2d"]
        confs = track.get("confidences", [1.0] * len(centers))
        frames = track["frames"]
        cls_name = track.get("class_name", "object")

        for i, fseq in enumerate(frames):
            box = VggtBox(
                track_id=tid,
                frame_seq=int(fseq),
                class_name=cls_name,
                center_world=np.array(centers[i], dtype=np.float64),
                dimensions_lwh=np.array(dims_list[i], dtype=np.float64),
                R_world=np.array(rots[i], dtype=np.float64),
                bbox_2d_vggt=tuple(bb2d[i]),
                confidence=float(confs[i]),
            )
            boxes_by_frame.setdefault(int(fseq), []).append(box)

    # SAM3 per-object masks give tight 2D boxes; the VGGT-stored bbox_2d
    # is usually the projected 3D cuboid, which is always loose (cuboid
    # extent > animal silhouette). When masks are available, compute tight
    # bboxes once and stash them per (track_id, frame_name) for the
    # annotation builder to prefer.
    sam3_masks_dir = _find_sam3_masks_dir(seg["seg_dir"])
    sam3_boxes: Dict[Tuple[int, str], Tuple[float, float, float, float, int]] = {}
    if sam3_masks_dir is not None:
        # Enumerate obj_<id> dirs; assume VGGT track_id == SAM3 obj_id,
        # which is how the WildBox pipeline emits them.
        for obj_dir in sorted(sam3_masks_dir.iterdir()):
            if not obj_dir.is_dir() or not obj_dir.name.startswith("obj_"):
                continue
            try:
                obj_id = int(obj_dir.name.split("_", 1)[1])
            except ValueError:
                continue
            for png in obj_dir.glob("frame_*.png"):
                bbox = _load_mask_bbox(png)
                if bbox is None:
                    continue
                # Map PNG filename back to the JPEG it corresponds to
                # (cameras.json stores "frame_NNNNNN.jpg").
                jpeg_name = png.stem + ".jpg"
                sam3_boxes[(obj_id, jpeg_name)] = bbox
        logger.debug(f"[{seg['seg_id']}] loaded {len(sam3_boxes)} SAM3 mask "
                     f"bboxes from {sam3_masks_dir}")

    return {
        "frames_seq": frames_seq,
        "boxes_by_frame": boxes_by_frame,
        "real_h": real_h,
        "real_w": real_w,
        "sx": sx,
        "sy": sy,
        "sam3_boxes": sam3_boxes,
    }


def compute_scene_scale_cam(boxes_by_frame: Dict[int, List[VggtBox]],
                            frames_seq: Dict[int, Dict]) -> float:
    zs = []
    for fseq, boxes in boxes_by_frame.items():
        frame = frames_seq.get(fseq)
        if frame is None:
            continue
        ext = frame["extrinsic"]
        R_ext = ext[:3, :3]
        t_ext = ext[:3, 3]
        for b in boxes:
            center_cam = R_ext @ b.center_world + t_ext
            if center_cam[2] > 0:
                zs.append(float(center_cam[2]))
    if not zs:
        return 1.0
    if len(zs) < 3:
        return 1.0 / max(zs)
    return 1.0 / float(np.median(zs))


def build_annotation(
    box: VggtBox,
    frame: Dict,
    sx: float,
    sy: float,
    s_scene: float,
    ann_id: int,
    image_id: int,
    dataset_id: int,
    category_id: int,
    category_name: str,
    sam3_bbox: Optional[Tuple[float, float, float, float, int]] = None,
) -> Optional[Dict]:
    ext = frame["extrinsic"]
    R_ext = ext[:3, :3]
    t_ext = ext[:3, 3]

    center_cam = R_ext @ box.center_world + t_ext
    if center_cam[2] <= 0:
        return None

    R_cam = R_ext @ box.R_world

    dims_omni3d = np.array([
        box.dimensions_lwh[2],
        box.dimensions_lwh[1],
        box.dimensions_lwh[0],
    ], dtype=np.float64)
    if np.any(dims_omni3d <= 0):
        return None

    corners_world = vggt_corners_world(box.center_world, box.dimensions_lwh, box.R_world)
    corners_cam = (R_ext @ corners_world.T).T + t_ext

    center_cam = center_cam * s_scene
    corners_cam = corners_cam * s_scene
    dims_omni3d = dims_omni3d * s_scene

    # Priority for the tight 2D bbox (used for 'bbox' and 'bbox2D_tight'):
    #   1. SAM3 mask -- tightest (silhouette-level). Uses real image coords
    #      directly; no sx/sy scaling needed since masks are at real res.
    #   2. VGGT's stored bbox_2d -- sometimes tight (from SAM), often the
    #      projected 3D cuboid (loose). Scale from VGGT res to real res.
    #   3. Projected 3D cuboid corners -- always loose, last-resort fallback.
    used_sam3 = False
    mask_pixels = 0
    if sam3_bbox is not None:
        bl, bt, br, bb, mask_pixels = sam3_bbox
        used_sam3 = True
    else:
        bl, bt, br, bb = box.bbox_2d_vggt
        if bl < 0 or bt < 0 or br < 0 or bb < 0:
            corners_2d = project_points(corners_cam, frame["K_real"])
            bl = float(corners_2d[:, 0].min())
            bt = float(corners_2d[:, 1].min())
            br = float(corners_2d[:, 0].max())
            bb = float(corners_2d[:, 1].max())
        else:
            bl *= sx
            br *= sx
            bt *= sy
            bb *= sy

    img_h = frame["image_height"]
    img_w = frame["image_width"]
    bl_c = float(np.clip(bl, 0, img_w - 1))
    bt_c = float(np.clip(bt, 0, img_h - 1))
    br_c = float(np.clip(br, 0, img_w - 1))
    bb_c = float(np.clip(bb, 0, img_h - 1))
    if br_c - bl_c <= 1.0 or bb_c - bt_c <= 1.0:
        return None

    bbox2d_xyxy = [bl_c, bt_c, br_c, bb_c]
    bbox_xywh = [bl_c, bt_c, br_c - bl_c, bb_c - bt_c]

    corners_2d = project_points(corners_cam, frame["K_real"])
    u_min = float(np.clip(corners_2d[:, 0].min(), 0, img_w - 1))
    v_min = float(np.clip(corners_2d[:, 1].min(), 0, img_h - 1))
    u_max = float(np.clip(corners_2d[:, 0].max(), 0, img_w - 1))
    v_max = float(np.clip(corners_2d[:, 1].max(), 0, img_h - 1))
    bbox2D_proj = [u_min, v_min, u_max, v_max]

    # When we have a SAM3 mask, also report a realistic segmentation_pts so
    # the annotation filter can treat tiny / occluded animals reasonably.
    # (Cap at 10000 so the default is_ignore() thresholds don't over-trust
    # giant masks.)
    seg_pts = min(mask_pixels, 10000) if used_sam3 and mask_pixels > 0 else 10

    return {
        "id": ann_id,
        "image_id": image_id,
        "dataset_id": dataset_id,
        "category_id": category_id,
        "category_name": category_name,
        "valid3D": True,
        "behind_camera": False,
        "bbox": bbox_xywh,
        "bbox2D_tight": bbox2d_xyxy,
        "bbox2D_trunc": bbox2d_xyxy,
        "bbox2D_proj": bbox2D_proj,
        "bbox3D_cam": corners_cam.tolist(),
        "center_cam": center_cam.tolist(),
        "dimensions": dims_omni3d.tolist(),
        "R_cam": R_cam.tolist(),
        "truncation": 0.0,
        "visibility": 1.0,
        "segmentation_pts": seg_pts,
        "lidar_pts": 10,
        "depth_error": 0.0,
        "area": float(bbox_xywh[2] * bbox_xywh[3]),
        "iscrowd": 0,
        "track_id": box.track_id,
        "bbox_source": "sam3" if used_sam3 else "vggt",
    }


def build_split_json(
    segments: List[Dict],
    dataset_id: int,
    dataset_name: str,
) -> Tuple[Dict, Dict[str, float]]:
    """Build one Omni3D-style dict from a list of segment records."""
    images = []
    annotations = []
    scene_scales = {}
    categories_seen: Dict[int, str] = {}

    next_image_id = 0
    next_ann_id = 0

    for seg in segments:
        category_name = seg["category_name"]
        category_id = seg["category_id"]
        categories_seen[category_id] = category_name

        loaded = load_segment(seg)
        if loaded is None:
            continue

        frames_seq = loaded["frames_seq"]
        boxes_by_frame = loaded["boxes_by_frame"]
        sx, sy = loaded["sx"], loaded["sy"]
        sam3_boxes = loaded.get("sam3_boxes", {})

        s_scene = compute_scene_scale_cam(boxes_by_frame, frames_seq)
        scene_scales[seg["seg_id"]] = s_scene

        for fseq in sorted(frames_seq.keys()):
            frame = frames_seq[fseq]
            boxes = boxes_by_frame.get(fseq, [])
            if not boxes:
                continue

            image_id = next_image_id
            next_image_id += 1

            abs_frame_path = str((seg["seg_dir"] / frame["image_name"]).resolve())

            images.append({
                "id": image_id,
                "dataset_id": dataset_id,
                "file_path": abs_frame_path,
                "height": frame["image_height"],
                "width": frame["image_width"],
                "K": frame["K_real"].tolist(),
                "src_flagged": False,
            })

            for box in boxes:
                # Look up SAM3 tight 2D bbox keyed by (track_id, frame_name).
                # If absent (no SAM3 masks for this segment / this frame
                # missing / this track_id not in SAM3), fall through to the
                # VGGT/projection logic inside build_annotation.
                sam3_key = (box.track_id, frame["image_name"])
                sam3_bbox = sam3_boxes.get(sam3_key)

                ann = build_annotation(
                    box=box,
                    frame=frame,
                    sx=sx,
                    sy=sy,
                    s_scene=s_scene,
                    ann_id=next_ann_id,
                    image_id=image_id,
                    dataset_id=dataset_id,
                    category_id=category_id,
                    category_name=category_name,
                    sam3_bbox=sam3_bbox,
                )
                if ann is None:
                    continue
                annotations.append(ann)
                next_ann_id += 1

    data = {
        "info": {
            "id": dataset_id,
            "source": "wildbox",
            "name": dataset_name,
            "split": dataset_name.split("_")[-1],
            "version": "1.0",
            "url": "",
            "known_category_ids": sorted(categories_seen.keys()),
            "scene_scales": scene_scales,
        },
        "categories": [
            {"id": cid, "name": categories_seen[cid], "supercategory": "animal"}
            for cid in sorted(categories_seen.keys())
        ],
        "images": images,
        "annotations": annotations,
    }
    return data, scene_scales


def auto_split(segments: List[Dict], mode: str, val_fraction: float,
               seed: int) -> Tuple[List[Dict], List[Dict]]:
    """Split segments into train/val deterministically.

    - mode="segment": each segment independently assigned to train or val.
    - mode="video":   all segments of a given (category, video) go the same
                      way. Stricter; prevents frames from adjacent segments
                      of the same scene leaking between splits.

    Grouping is per category so every category appears in both splits even
    at small val fractions.
    """
    rng = random.Random(seed)

    if mode == "segment":
        group_key = lambda s: s["seg_id"]
    elif mode == "video":
        group_key = lambda s: f"{s['category_name']}/{s['video']}"
    else:
        raise ValueError(f"unknown split mode: {mode}")

    # Partition segments by category so we stratify on category.
    by_cat: Dict[str, List[Dict]] = {}
    for seg in segments:
        by_cat.setdefault(seg["category_name"], []).append(seg)

    train: List[Dict] = []
    val: List[Dict] = []
    for cat, segs in by_cat.items():
        # unique groups, sorted for determinism
        groups = sorted({group_key(s) for s in segs})
        rng.shuffle(groups)
        n_val = max(1, int(round(len(groups) * val_fraction))) if groups else 0
        if n_val >= len(groups):
            n_val = max(1, len(groups) - 1)  # always keep at least one train group
        val_groups = set(groups[:n_val])
        for s in segs:
            (val if group_key(s) in val_groups else train).append(s)
        logger.info(f"[split] {cat}: {len(groups)} groups -> "
                    f"{len(groups) - n_val} train / {n_val} val")

    return train, val


def parse_source(entry: str) -> Optional[Source]:
    """Parse 'path=category:category_id' into a Source.

    Returns None if the path can't be resolved (e.g. .zip still
    transferring / corrupt) so the main() loop can skip it with a warning
    rather than aborting the whole run.
    """
    if "=" not in entry or ":" not in entry:
        raise SystemExit(f"--source entry '{entry}' must be "
                         f"'path=category:category_id'")
    path_str, _, spec = entry.partition("=")
    name, _, cid_str = spec.partition(":")
    try:
        cid = int(cid_str)
    except ValueError:
        raise SystemExit(f"--source: '{cid_str}' in '{entry}' is not an integer")
    raw = Path(path_str).expanduser()
    resolved = resolve_source_path(raw)
    if resolved is None:
        return None
    return Source(path=resolved, category_name=name, category_id=cid)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", required=True,
                   dest="sources",
                   help="Data source as 'path=category:category_id'. The path "
                        "is a directory containing <video>/<seg>/vggt_results/. "
                        "Repeatable. Example: "
                        "--source /.../WildBox=rhino:1004 "
                        "--source /.../WildBox=elephant:1002")
    p.add_argument("--split-mode", choices=("segment", "video", "manual"),
                   default="segment",
                   help="segment: random 80/20 split of segments per category "
                        "(default). video: random split by (category, video), "
                        "stricter. manual: use --train-segments/--val-segments.")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="Fraction of groups held out for val (random modes only)")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for reproducible splits")
    p.add_argument("--train-segments", nargs="+", default=[],
                   help="(manual mode) Segment ids 'category/video/seg' for train")
    p.add_argument("--val-segments", nargs="+", default=[],
                   help="(manual mode) Segment ids 'category/video/seg' for val")
    p.add_argument("--output-train", type=Path,
                   default=Path("datasets/Omni3D/WildBox_train.json"))
    p.add_argument("--output-val", type=Path,
                   default=Path("datasets/Omni3D/WildBox_val.json"))
    p.add_argument("--dataset-id", type=int, default=1000,
                   help="Omni3D dataset id for the info section")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Silence PIL's chatty PNG chunk logs -- we read thousands of masks
    # and each one prints IHDR/IDAT at DEBUG level, drowning our own logs.
    logging.getLogger("PIL").setLevel(logging.WARNING)

    raw_sources = [parse_source(e) for e in args.sources]
    sources = [s for s in raw_sources if s is not None]
    n_skipped = len(raw_sources) - len(sources)
    if n_skipped > 0:
        logger.warning(
            f"{n_skipped} source(s) skipped due to missing / corrupt paths. "
            f"Add them to datasets/pending_sources.txt and rerun when ready."
        )
    if not sources:
        sys.exit("No valid sources. Aborting.")
    for s in sources:
        logger.info(f"Source: {s.path} -> {s.category_name} (id={s.category_id})")

    all_segments = discover_segments(sources)
    logger.info(f"Discovered {len(all_segments)} segments total")
    if not all_segments:
        sys.exit(2)

    if args.split_mode == "manual":
        seg_by_id = {s["seg_id"]: s for s in all_segments}
        train_segs = [seg_by_id[sid] for sid in args.train_segments if sid in seg_by_id]
        val_segs = [seg_by_id[sid] for sid in args.val_segments if sid in seg_by_id]
        missing = set(args.train_segments + args.val_segments) - set(seg_by_id.keys())
        if missing:
            logger.warning(f"Segment ids not found on disk: {sorted(missing)}")
    else:
        train_segs, val_segs = auto_split(
            all_segments, mode=args.split_mode,
            val_fraction=args.val_fraction, seed=args.seed,
        )

    logger.info(f"Split: {len(train_segs)} train segments, "
                f"{len(val_segs)} val segments")

    for split_name, segs, out_path in [
        ("train", train_segs, args.output_train),
        ("val", val_segs, args.output_val),
    ]:
        if not segs:
            logger.warning(f"[{split_name}] no segments; skipping output")
            continue
        data, scales = build_split_json(
            segments=segs,
            dataset_id=args.dataset_id,
            dataset_name=f"WildBox_{split_name}",
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        n_sam3 = sum(1 for a in data["annotations"]
                     if a.get("bbox_source") == "sam3")
        n_total = len(data["annotations"])
        pct = (100.0 * n_sam3 / max(n_total, 1))
        logger.info(
            f"[{split_name}] wrote {out_path} — {len(data['images'])} images, "
            f"{n_total} annotations "
            f"({n_sam3} with SAM3 tight 2D bbox = {pct:.1f}%), "
            f"{len(scales)} segments"
        )
        if args.verbose:
            for sid, s in scales.items():
                logger.debug(f"  scene_scale[{sid}] = {s:.4f}")


if __name__ == "__main__":
    main()
