"""WildBox dataset for TA-DETR.  [M3]

Emits, per frame (the spec 0.3 sample format, amended by the M1/M2 findings):
  image      (3, H_in, W_in) float32, ImageNet-normalized, VIEW space (bridge-augmented)
  bridge     (5,) [sx, sy, ox, oy, flip] -- view -> full-res mapping (see transforms.py)
  K          (3,3) ORIGINAL full-res label-gauge intrinsics (json K; ray-casting happens here)
  extrinsic  (3,4) world->cam for this frame (from the terrain cache; jsons carry none)
  cam_height ()   camera height above terrain, world units
  seg_key    str  terrain-cache key for this frame's segment
  telemetry  (6,) [sin pitch, cos pitch, roll, std log alt, pitch_valid, alt_valid]
  cam_feats  (4,) O(1)-standardized [log fx_json, log fx_tel, fx_tel_valid, log cam_height]
  targets: cls (contiguous), boxes_view (N,4 cxcywh, normalized), contact (N,2 in [0,1] VIEW),
           contact_valid, center_cam (N,3 REAL coords), dims (N,3), valid3d,
           axis_embed (N,2) VIEW-consistent (sin2psi flipped under mirror), axis_weight,
           sign_target, sign_valid, category_name per target

Junk handling: annotations with valid3D=false (the human-audited fragment population) KEEP their
2D box/class for matching but are masked out of every 3D/contact/axis loss (valid3d=0).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..config import DataCfg
from ..utils.rot import axis_angle_mod_pi
from .terrain_cache import load_terrain_cache
from .transforms import ViewBridge, boxes_full_to_view, photometric, sample_bridge
from .wildbox_paths import parse_label_path

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _std_log(x, center, scale):
    return 0.0 if x <= 0 else (float(np.log(x)) - center) / scale


class WildBoxTADETR(torch.utils.data.Dataset):
    def __init__(self, cfg: DataCfg, split: str, training: bool):
        self.cfg = cfg
        self.training = training
        json_path = cfg.train_json if split == "train" else cfg.val_json
        d = json.loads(Path(json_path).read_text())
        meta = json.loads(Path(cfg.category_meta).read_text())
        self.cat_map = {int(k): v for k, v in meta["thing_dataset_id_to_contiguous_id"].items()}
        self.class_names = meta["thing_classes"]

        self.images = d["images"]
        anns_by_img = defaultdict(list)
        for a in d["annotations"]:
            if a.get("behind_camera"):
                continue
            anns_by_img[a["image_id"]].append(a)
        self.anns_by_img = anns_by_img

        # terrain join + per-segment frame grouping
        self.terrain_dir = Path(cfg.terrain_dir)
        self._cache = {}
        self._cache_paths = {}
        for p in sorted(self.terrain_dir.glob("*.npz")):
            parts = p.stem.split("__")
            self._cache_paths[(parts[1], parts[2])] = p

        self.samples = []
        self.by_segment = defaultdict(list)
        for im in self.images:
            _, video, seg, image_name = parse_label_path(im["file_path"])
            key = (video, seg)
            if key not in self._cache_paths:
                continue
            idx = len(self.samples)
            self.samples.append((im, key, image_name))
            self.by_segment[key].append(idx)
        if cfg.epoch_frame_stride > 1 and training:
            self.by_segment = {k: v[::cfg.epoch_frame_stride]
                               for k, v in self.by_segment.items()}
        n_dropped = len(self.images) - len(self.samples)
        print(f"[dataset {split}] {len(self.samples):,} frames in {len(self.by_segment)} segments "
              f"({n_dropped} frames without terrain cache)")

    def cache(self, key):
        if isinstance(key, str):                      # "video||seg" (samples carry strings:
            key = tuple(key.split("||"))              # pin_memory mangles tuples into lists)
        if key not in self._cache:
            c = load_terrain_cache(self._cache_paths[key])
            c["_frame_index"] = {str(n): i for i, n in enumerate(c["frame_names"])}
            self._cache[key] = c
        return self._cache[key]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        im, key, image_name = self.samples[idx]
        cfg = self.cfg
        rng = random.Random((idx * 1000003 + torch.initial_seed()) % (2 ** 31))
        c = self.cache(key)
        fi = c["_frame_index"][image_name]

        img = Image.open(Path(cfg.image_root) / im["file_path"].replace("\\", "/")).convert("RGB")
        W0, H0 = img.size
        br = sample_bridge(W0, H0, cfg.input_w, cfg.input_h, training=self.training,
                           hflip_p=cfg.hflip_p, rrc_p=cfg.rrc_p,
                           rrc_scale_min=cfg.rrc_scale_min, rng=rng)
        crop = img.crop((br.ox, br.oy, br.ox + br.sx * cfg.input_w, br.oy + br.sy * cfg.input_h))
        crop = crop.resize((cfg.input_w, cfg.input_h), Image.BILINEAR)
        arr = np.asarray(crop, np.float32) / 255.0
        if br.flip:
            arr = arr[:, ::-1]
        if self.training and rng.random() < cfg.photometric_p:
            arr = photometric(arr, rng)
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))

        # targets
        anns = self.anns_by_img.get(im["id"], [])
        tg = self._targets(anns, br, key, c)

        K = np.array(im["K"], np.float32).reshape(3, 3)
        geo = im.get("geo", [0.0, 0.0, 0.0])
        geo_alt = im.get("geo_alt", [0.0, 0.0, 0.0])
        pitch, pv = float(geo[1]), float(geo[2])
        alt, roll, av = float(geo_alt[0]), float(geo_alt[1]), float(geo_alt[2])
        telemetry = np.array([np.sin(pitch) * pv, np.cos(pitch) * pv, np.radians(roll) * av,
                              _std_log(alt, 3.0, 1.0) * av, pv, av], np.float32)
        cam_h = float(c["cam_height"][fi])
        cam_feats = np.array([_std_log(K[0, 0], 8.0, 2.0), _std_log(geo[0], 8.0, 2.0),
                              1.0 if geo[0] > 0 else 0.0, _std_log(max(cam_h, 1e-6), -1.0, 1.0)],
                             np.float32)
        return {
            "image": image,
            "bridge": torch.from_numpy(br.as_tensor_args()),
            "K": torch.from_numpy(K),
            "extrinsic": torch.from_numpy(np.array(c["extrinsics"][fi], np.float32)),
            "cam_height": torch.tensor(cam_h, dtype=torch.float32),
            "seg_key": f"{key[0]}||{key[1]}",
            "image_id": im["id"],
            "telemetry": torch.from_numpy(telemetry),
            "cam_feats": torch.from_numpy(cam_feats),
            "targets": tg,
        }

    def _targets(self, anns, br: ViewBridge, key, c):
        cfg = self.cfg
        n = len(anns)
        if n == 0:
            z2 = lambda *s: torch.zeros(*s)
            return {"cls": torch.zeros(0, dtype=torch.long), "boxes": z2(0, 4),
                    "contact": z2(0, 2), "contact_valid": z2(0), "center_cam": z2(0, 3),
                    "dims": z2(0, 3), "valid3d": z2(0), "axis_embed": z2(0, 2),
                    "axis_weight": z2(0), "sign_target": z2(0), "sign_valid": z2(0),
                    "names": []}
        full_boxes = np.array([a["bbox"] for a in anns], np.float64)
        vb = boxes_full_to_view(full_boxes, br)
        keep = (vb[:, 2] > 2) & (vb[:, 3] > 2)
        anns = [a for a, k in zip(anns, keep) if k]
        vb = vb[keep]
        n = len(anns)
        if n == 0:
            return self._targets([], br, key, c)

        cls = torch.tensor([self.cat_map[a["category_id"]] for a in anns], dtype=torch.long)
        boxes = torch.tensor(np.stack([vb[:, 0] + vb[:, 2] / 2, vb[:, 1] + vb[:, 3] / 2,
                                       vb[:, 2], vb[:, 3]], 1)
                             / np.array([br.w_in, br.h_in, br.w_in, br.h_in]),
                             dtype=torch.float32)
        if cfg.contact_source == "box2d_bottom":
            cu = (vb[:, 0] + vb[:, 2] / 2) / br.w_in
            cv = (vb[:, 1] + vb[:, 3]) / br.h_in
            contact = torch.tensor(np.stack([cu, cv], 1), dtype=torch.float32)
        else:  # stamped 3D-projected contact (shin-biased; pairs with the species offset)
            uv_full = np.array([a.get("contact_uv", [-1, -1]) for a in anns], np.float64)
            u, v = br.full_to_view(uv_full[:, 0], uv_full[:, 1])
            contact = torch.tensor(np.stack([u / br.w_in, v / br.h_in], 1), dtype=torch.float32)
        valid3d = torch.tensor([float(a.get("valid3D", True)) for a in anns])
        contact_valid = valid3d.clone()
        center = torch.tensor(np.array([a["center_cam"] for a in anns]), dtype=torch.float32)
        dims = torch.tensor(np.array([a["dimensions"] for a in anns]), dtype=torch.float32)

        R = torch.tensor(np.array([a["R_cam"] for a in anns]), dtype=torch.float32)
        psi, embed = axis_angle_mod_pi(R[:, :, 0], center, -R[:, :, 1])
        if br.flip:                                  # mirror: psi -> -psi => sin2psi flips
            embed = torch.stack([-embed[:, 0], embed[:, 1]], 1)
        axis_weight = torch.zeros(n)                       # filled by criterion from cfg table
        names = [a["category_name"] for a in anns]

        alpha = torch.tensor([float(a.get("heading_alpha", 0.0)) for a in anns])
        hvalid = torch.tensor([float(a.get("heading_valid", 0.0)) for a in anns])
        sign_target = (torch.cos(alpha - psi) < 0).float()   # flip-invariant (both angles mirror)
        sign_valid = hvalid * valid3d
        return {"cls": cls, "boxes": boxes, "contact": contact, "contact_valid": contact_valid,
                "center_cam": center, "dims": dims, "valid3d": valid3d, "axis_embed": embed,
                "axis_weight": axis_weight, "sign_target": sign_target, "sign_valid": sign_valid,
                "names": names}


def collate(batch):
    out = {k: [] for k in batch[0]}
    for s in batch:
        for k, v in s.items():
            out[k].append(v)
    out["image"] = torch.stack(out["image"])
    for k in ("bridge", "K", "extrinsic", "telemetry", "cam_feats"):
        out[k] = torch.stack(out[k])
    out["cam_height"] = torch.stack(out["cam_height"])
    return out
