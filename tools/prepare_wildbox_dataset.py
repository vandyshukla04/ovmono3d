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

Input per segment (produced by the user's VGGT+SAM3 pipeline):
  <root>/<species>/WildBox_sam3-vggtv1_processed/WildBox/<video>/<seg>/
    frame_XXXXXX.jpg
    vggt_results/cameras.json           # per-frame intrinsic + extrinsic
    vggt_results/tracking_summary.json  # per-track 3D state (world coords)

Output:
  datasets/Omni3D/<name>.json           # Omni3D-style dict

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
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("prepare_wildbox_dataset")


@dataclass
class VggtBox:
    """Full-precision 3D box read from VGGT tracking_summary.json.

    Fields are in VGGT's WORLD coordinate frame until transformed by the
    per-frame extrinsic. `dimensions_lwh` is (l, w, h) in VGGT's own
    naming, which corresponds to (X-extent, Y-extent, Z-extent) of the
    local object frame (see demo_viser_tracking.py:109-121).
    """
    track_id: int
    frame_seq: int                                 # 0-indexed camera/frame position
    class_name: str
    center_world: np.ndarray                       # (3,)
    dimensions_lwh: np.ndarray                     # (3,) VGGT (l, w, h)
    R_world: np.ndarray                            # (3, 3) object rotation in world
    bbox_2d_vggt: Tuple[float, float, float, float]  # (l,t,r,b) at VGGT resolution
    confidence: float


def vggt_corners_world(center: np.ndarray, dims_lwh: np.ndarray,
                       R: np.ndarray) -> np.ndarray:
    """Compute 8 world-frame corners using VGGT's exact formula
    (demo_viser_tracking.py:109-121). Returns (8, 3)."""
    l, w, h = float(dims_lwh[0]), float(dims_lwh[1]), float(dims_lwh[2])
    corners_local = np.array([
        [-l/2, -w/2, -h/2],
        [+l/2, -w/2, -h/2],
        [+l/2, +w/2, -h/2],
        [-l/2, +w/2, -h/2],
        [-l/2, -w/2, +h/2],
        [+l/2, -w/2, +h/2],
        [+l/2, +w/2, +h/2],
        [-l/2, +w/2, +h/2],
    ], dtype=np.float64)
    return (R @ corners_local.T).T + center


def project_points(corners_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project (N, 3) camera-space points to (N, 2) image pixels."""
    xs = corners_cam[:, 0] / corners_cam[:, 2]
    ys = corners_cam[:, 1] / corners_cam[:, 2]
    u = K[0, 0] * xs + K[0, 2]
    v = K[1, 1] * ys + K[1, 2]
    return np.stack([u, v], axis=1)


def project_points(corners_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project (N, 3) camera-space points to (N, 2) image pixels."""
    xs = corners_cam[:, 0] / corners_cam[:, 2]
    ys = corners_cam[:, 1] / corners_cam[:, 2]
    u = K[0, 0] * xs + K[0, 2]
    v = K[1, 1] * ys + K[1, 2]
    return np.stack([u, v], axis=1)


def discover_segments(root: Path, species: List[str]) -> List[Dict]:
    """Walk the WildBox tree and enumerate segments under each species."""
    segments = []
    for sp in species:
        sp_root = root / sp / "WildBox_sam3-vggtv1_processed" / "WildBox"
        if not sp_root.exists():
            logger.warning(f"Species root not found: {sp_root}")
            continue
        for video_dir in sorted(sp_root.iterdir()):
            if not video_dir.is_dir():
                continue
            for seg_dir in sorted(video_dir.iterdir()):
                if not seg_dir.is_dir() or not seg_dir.name.startswith("seg"):
                    continue
                cameras_json = seg_dir / "vggt_results" / "cameras.json"
                tracking_json = seg_dir / "vggt_results" / "tracking_summary.json"
                if not cameras_json.exists() or not tracking_json.exists():
                    logger.warning(f"Skipping incomplete segment: {seg_dir}")
                    continue
                segments.append({
                    "species": sp,
                    "video": video_dir.name,
                    "seg": seg_dir.name,
                    "seg_id": f"{video_dir.name}/{seg_dir.name}",
                    "seg_dir": seg_dir,
                    "cameras_json": cameras_json,
                    "tracking_json": tracking_json,
                })
    return segments


def read_jpeg_dimensions(path: Path) -> Tuple[int, int]:
    """Parse JPEG SOF marker to read (height, width) without decoding."""
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
            # SOF0..SOF3, SOF5..SOF7, SOF9..SOF11, SOF13..SOF15 are all SOFn
            if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                     0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                length = struct.unpack(">H", f.read(2))[0]
                _precision = f.read(1)
                h, w = struct.unpack(">HH", f.read(4))
                return h, w
            length = struct.unpack(">H", f.read(2))[0]
            f.seek(length - 2, 1)


def load_segment(seg: Dict) -> Optional[Dict]:
    """Read tracking_summary.json and cameras.json for one segment.

    Returns a dict with:
      frames_seq:     list of per-frame dicts (K_real, extrinsic, image name, dims)
      boxes_by_frame: {frame_seq: [VggtBox, ...]}
      real_h/real_w:  actual JPEG dimensions
      sx/sy:          VGGT->real resolution scale factors
    """
    with open(seg["cameras_json"], "r") as f:
        cam_data = json.load(f)
    with open(seg["tracking_json"], "r") as f:
        track_data = json.load(f)

    # Detect real JPEG dimensions (VGGT's cameras.json size is the downsized
    # processing resolution; actual images on disk are usually larger).
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

    # Per-frame camera state, keyed by frame_index (0-based sequence position).
    frames_seq: Dict[int, Dict] = {}
    for cam in cam_data["cameras"]:
        K_vggt = np.array(cam["intrinsic"], dtype=np.float64)
        K_real = K_vggt.copy()
        K_real[0, 0] *= sx
        K_real[1, 1] *= sy
        K_real[0, 2] *= sx
        K_real[1, 2] *= sy

        ext = np.array(cam["extrinsic"], dtype=np.float64)  # (3, 4)

        frames_seq[int(cam["frame_index"])] = {
            "image_name": cam["image_name"],
            "K_real": K_real,
            "extrinsic": ext,
            "image_height": real_h,
            "image_width": real_w,
        }

    # Collect per-frame VggtBox objects from the tracking summary.
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

    return {
        "frames_seq": frames_seq,
        "boxes_by_frame": boxes_by_frame,
        "real_h": real_h,
        "real_w": real_w,
        "sx": sx,
        "sy": sy,
    }


def compute_scene_scale_cam(boxes_by_frame: Dict[int, List[VggtBox]],
                            frames_seq: Dict[int, Dict]) -> float:
    """Map median camera-frame |Z| to 1.0 across the segment."""
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
) -> Optional[Dict]:
    """Convert one VggtBox + per-frame extrinsic into an Omni3D annotation.

    No rotation rebuilding, no axis swapping — we use VGGT's own world-frame
    corner formula and then apply the extrinsic to get camera-frame corners.
    """
    ext = frame["extrinsic"]
    R_ext = ext[:3, :3]
    t_ext = ext[:3, 3]

    center_cam = R_ext @ box.center_world + t_ext
    if center_cam[2] <= 0:
        return None

    R_cam = R_ext @ box.R_world

    # VGGT (l, w, h) maps axis-by-axis to Omni3D (L, H, W). Omni3D stores
    # dimensions in order [W, H, L], which is the reverse of VGGT's tuple.
    dims_omni3d = np.array([
        box.dimensions_lwh[2],  # W = VGGT.h  (Z-extent)
        box.dimensions_lwh[1],  # H = VGGT.w  (Y-extent)
        box.dimensions_lwh[0],  # L = VGGT.l  (X-extent)
    ], dtype=np.float64)
    if np.any(dims_omni3d <= 0):
        return None

    # Full-precision camera-frame corners, via VGGT's own formula.
    corners_world = vggt_corners_world(box.center_world, box.dimensions_lwh, box.R_world)
    corners_cam = (R_ext @ corners_world.T).T + t_ext  # (8, 3)

    # Per-scene uniform scaling: center, corners, dims all scale together.
    center_cam = center_cam * s_scene
    corners_cam = corners_cam * s_scene
    dims_omni3d = dims_omni3d * s_scene

    # 2D bbox: VGGT pipeline stored it at small processing resolution; rescale
    # to the actual image size.
    bl, bt, br, bb = box.bbox_2d_vggt
    if bl < 0 or bt < 0 or br < 0 or bb < 0:
        # sentinel for invalid projection; re-derive from scaled corners
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

    # Also compute the tight projection of the 3D corners (useful when the
    # stored SAM3 2D bbox is larger than the projected cuboid).
    corners_2d = project_points(corners_cam, frame["K_real"])
    u_min = float(np.clip(corners_2d[:, 0].min(), 0, img_w - 1))
    v_min = float(np.clip(corners_2d[:, 1].min(), 0, img_h - 1))
    u_max = float(np.clip(corners_2d[:, 0].max(), 0, img_w - 1))
    v_max = float(np.clip(corners_2d[:, 1].max(), 0, img_h - 1))
    bbox2D_proj = [u_min, v_min, u_max, v_max]

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
        "segmentation_pts": 10,
        "lidar_pts": 10,
        "depth_error": 0.0,
        "area": float(bbox_xywh[2] * bbox_xywh[3]),
        "iscrowd": 0,
        "track_id": box.track_id,
    }


def build_split_json(
    segments: List[Dict],
    species_map: Dict[str, Tuple[str, int]],
    dataset_id: int,
    dataset_name: str,
    image_root_prefix: str,
) -> Tuple[Dict, Dict[str, float]]:
    """Build one Omni3D-style dict from a list of segments.

    species_map maps the folder name under --root (e.g. 'rhinos') to a
    (category_name, category_id) tuple. Segments whose species isn't in
    the map are dropped with a warning.
    """
    images = []
    annotations = []
    scene_scales = {}

    next_image_id = 0
    next_ann_id = 0

    for seg in segments:
        sp_key = seg["species"]
        if sp_key not in species_map:
            logger.warning(f"Skipping segment {seg['seg_id']} — species "
                           f"'{sp_key}' not in --species-map")
            continue
        category_name, category_id = species_map[sp_key]

        loaded = load_segment(seg)
        if loaded is None:
            continue

        frames_seq = loaded["frames_seq"]
        boxes_by_frame = loaded["boxes_by_frame"]
        sx, sy = loaded["sx"], loaded["sy"]

        s_scene = compute_scene_scale_cam(boxes_by_frame, frames_seq)
        scene_scales[seg["seg_id"]] = s_scene

        # Iterate in frame_seq order for reproducibility.
        for fseq in sorted(frames_seq.keys()):
            frame = frames_seq[fseq]
            boxes = boxes_by_frame.get(fseq, [])
            if not boxes:
                continue

            image_id = next_image_id
            next_image_id += 1

            rel_path = os.path.join(
                image_root_prefix,
                seg["species"],
                "WildBox_sam3-vggtv1_processed",
                "WildBox",
                seg["video"],
                seg["seg"],
                frame["image_name"],
            )

            images.append({
                "id": image_id,
                "dataset_id": dataset_id,
                "file_path": rel_path,
                "height": frame["image_height"],
                "width": frame["image_width"],
                "K": frame["K_real"].tolist(),
                "src_flagged": False,
            })

            for box in boxes:
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
                )
                if ann is None:
                    continue
                annotations.append(ann)
                next_ann_id += 1

    category_ids_used = sorted({cid for _, cid in species_map.values()})
    data = {
        "info": {
            "id": dataset_id,
            "source": "wildbox",
            "name": dataset_name,
            "split": dataset_name.split("_")[-1],
            "version": "1.0",
            "url": "",
            "known_category_ids": category_ids_used,
            "scene_scales": scene_scales,
        },
        "categories": [
            {"id": cid, "name": name, "supercategory": "animal"}
            for name, cid in sorted(species_map.values(), key=lambda x: x[1])
        ],
        "images": images,
        "annotations": annotations,
    }
    return data, scene_scales


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True,
                   help="WildBox videos root, e.g. /mnt/d/3DBOX/Data/videos/202401_Kenya")
    p.add_argument("--species-map", nargs="+", required=True,
                   help="Per-species mapping 'folder=category:category_id'. "
                        "Example: 'giraffes=giraffe:1000' 'rhinos=rhino:1004' "
                        "'elephants=elephant:1002'. The folder is the dir under "
                        "--root; category name + id must match stats.json after "
                        "running tools/patch_stats_for_wildbox.py.")
    p.add_argument("--train-segments", nargs="+", default=[],
                   help="Segment ids (video/seg) to place in the train split")
    p.add_argument("--val-segments", nargs="+", default=[],
                   help="Segment ids (video/seg) to place in the val split")
    p.add_argument("--output-train", type=Path,
                   default=Path("datasets/Omni3D/WildBox_train.json"))
    p.add_argument("--output-val", type=Path,
                   default=Path("datasets/Omni3D/WildBox_val.json"))
    p.add_argument("--dataset-id", type=int, default=1000,
                   help="Omni3D dataset id for the info section")
    p.add_argument("--image-root-prefix", default="WildBox",
                   help="Prefix prepended to file_path; must match the "
                        "symlink at datasets/<prefix> -> /mnt/d/.../videos")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def parse_species_map(entries: List[str]) -> Dict[str, Tuple[str, int]]:
    """Parse ['giraffes=giraffe:1000', 'rhinos=rhino:1004'] into a dict."""
    out: Dict[str, Tuple[str, int]] = {}
    for entry in entries:
        if "=" not in entry or ":" not in entry:
            raise SystemExit(f"--species-map entry '{entry}' must be "
                             f"'folder=category:category_id'")
        folder, _, spec = entry.partition("=")
        name, _, cid_str = spec.partition(":")
        try:
            cid = int(cid_str)
        except ValueError:
            raise SystemExit(f"--species-map: '{cid_str}' in '{entry}' "
                             f"is not an integer")
        out[folder] = (name, cid)
    return out


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    species_map = parse_species_map(args.species_map)
    species_folders = list(species_map.keys())

    all_segments = discover_segments(args.root, species_folders)
    logger.info(f"Discovered {len(all_segments)} segments under "
                f"{args.root} species={species_folders}")
    if not all_segments:
        sys.exit(2)

    seg_by_id = {s["seg_id"]: s for s in all_segments}

    train_segs = [seg_by_id[sid] for sid in args.train_segments if sid in seg_by_id]
    val_segs = [seg_by_id[sid] for sid in args.val_segments if sid in seg_by_id]

    missing = set(args.train_segments + args.val_segments) - set(seg_by_id.keys())
    if missing:
        logger.warning(f"Segment ids not found on disk: {sorted(missing)}")

    if not train_segs and not val_segs:
        logger.info("No explicit split given, placing everything into train.")
        train_segs = all_segments

    for split_name, segs, out_path in [
        ("train", train_segs, args.output_train),
        ("val", val_segs, args.output_val),
    ]:
        if not segs:
            continue
        data, scales = build_split_json(
            segments=segs,
            species_map=species_map,
            dataset_id=args.dataset_id,
            dataset_name=f"WildBox_{split_name}",
            image_root_prefix=args.image_root_prefix,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            f"[{split_name}] wrote {out_path} — {len(data['images'])} images, "
            f"{len(data['annotations'])} annotations, {len(scales)} segments"
        )
        if args.verbose:
            for sid, s in scales.items():
                logger.debug(f"  scene_scale[{sid}] = {s:.4f}")


if __name__ == "__main__":
    main()
