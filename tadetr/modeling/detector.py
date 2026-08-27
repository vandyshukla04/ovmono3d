"""TA-DETR detector: backbone -> decoder -> heads -> GEOMETRIC composition.  [M3]

Depth is NOT a head (spec 2.3): per query,
    contact (view coords) --bridge--> full-res label pixel --ray + terrain-->
    foot3d = ray(t*) + dh * n_surface
    center3d = foot3d + n_surface * (dims_h / (2 s_seg)) + R_grid^T @ center_offset
    z_label  = s_seg * (E @ center3d)_z
Deviation from spec, deliberate: the H/2 lift from foot to center is part of the DETERMINISTIC
composition (spec folds it into the learned center_offset). With all residual heads zero-init the
network therefore reproduces geometric_lift EXACTLY at init (preflight P8), and A1 (residuals off)
is architecturally identical to A0 + learned contact.

The fp32 island: composition runs under autocast-disabled fp32 regardless of the training AMP
dtype (root-finding is precision-sensitive; V100 runs the whole model fp32 anyway).

The augmentation bridge: predicted contact is mapped view->full-res (undoing flip/crop) BEFORE
ray-casting -- the terrain describes the real camera. The axis angle mirrors with the flip
(psi_real = -psi_view when flipped), matching the mirrored supervision.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import TADETRConfig
from ..geometry.lift import geometric_lift
from .backbone import TADETRBackbone
from .heads import TADETRHeads
from .herd_scale import HerdScale
from .transformer import TADETRDecoder


class TADETR(nn.Module):
    def __init__(self, cfg: TADETRConfig, dims_median: torch.Tensor, n_classes: int):
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        self.backbone = TADETRBackbone(m.d_model, m.backbone_blocks,
                                       cfg.train.backbone_weights)
        self.decoder = TADETRDecoder(m.d_model, m.dec_layers, m.n_queries, m.n_heads,
                                     m.msda_points, m.herd_bias, m.herd_attention)
        self.heads = TADETRHeads(m.d_model, n_classes, m.dh_bounds, m.center_offset_bound)
        self.herd = HerdScale(m.d_model, n_classes, dims_median)
        self.img_diag_px = math.hypot(cfg.data.input_w, cfg.data.input_h)

    def forward(self, batch: dict, fields: dict, stage: int = 3,
                ref_boxes: torch.Tensor | None = None) -> list:
        """batch: collated dataset dict; fields: {seg_key: TerrainField on device (fp32)}.
        Returns one output dict per decoder layer (last = final)."""
        m = self.cfg.model
        feat = self.backbone(batch["image"])
        hs_list, _ = self.decoder(feat, batch["telemetry"], batch["cam_feats"],
                                  self.heads.cls, ref_boxes=ref_boxes,
                                  use_telemetry=m.use_telemetry_token)
        outputs = []
        for hs in hs_list:
            out = self.heads(hs)
            out["img_diag_px"] = self.img_diag_px
            herd = self.herd(hs, out["logits"],
                             out["dim_residual"] if m.use_dim_residual
                             else torch.zeros_like(out["dim_residual"]),
                             enabled=m.use_herd_scale and stage >= 3,
                             sample=self.training and stage >= 3)
            out["herd"] = herd
            out["dims"] = herd["dims"]
            self._compose(out, batch, fields)
            outputs.append(out)
        return outputs

    def _compose(self, out: dict, batch: dict, fields: dict) -> None:
        m = self.cfg.model
        B, Q, _ = out["contact"].shape
        dev = out["contact"].device
        with torch.autocast(device_type=dev.type, enabled=False):
            contact = out["contact"].float()
            br = batch["bridge"].float()                       # (B,5) sx,sy,ox,oy,flip
            w_in, h_in = self.cfg.data.input_w, self.cfg.data.input_h
            u_v = contact[..., 0] * w_in
            v_v = contact[..., 1] * h_in
            flip = br[:, 4:5]
            u_v = flip * (w_in - 1 - u_v) + (1 - flip) * u_v   # undo mirror
            u_full = br[:, 2:3] + br[:, 0:1] * u_v
            v_full = br[:, 3:4] + br[:, 1:2] * v_v

            z_label = torch.zeros(B, Q, device=dev)
            sigma_z = torch.ones(B, Q, device=dev)
            center_cam = torch.zeros(B, Q, 3, device=dev)
            n_cam = torch.zeros(B, Q, 3, device=dev)
            fallback = torch.zeros(B, Q, dtype=torch.bool, device=dev)

            for i in range(B):
                f = fields[batch["seg_key"][i]]
                K = batch["K"][i].float()
                E = batch["extrinsic"][i].float()
                ch = batch["cam_height"][i].float().expand(Q)
                uv = torch.stack([u_full[i], v_full[i]], dim=-1)
                dh = out["dh_frac"][i].float() * ch if m.use_height_residual \
                    else torch.zeros_like(ch)
                lift = geometric_lift(uv, K.expand(Q, 3, 3), E.expand(Q, 3, 4), f, ch,
                                      dh=dh, sigma_px=out["contact_sigma"][i].float())
                s = f.s_seg
                p_t = f.world_to_tangent(lift["p_world"])
                n_w = f.normal_world(p_t[:, 0], p_t[:, 1])
                h_dims = out["dims"][i, :, 1].float()
                cw = lift["p_world"] + n_w * (h_dims / (2 * s))[:, None]
                if m.use_center_offset:
                    off_t = out["center_offset"][i].float() * batch["cam_height"][i].float()
                    cw = cw + off_t @ f.R_grid                  # tangent -> world (R^T @ o)
                pc = s * ((E[:, :3] @ cw.T).T + E[:, 3])
                if m.direct_z_head:                             # A5 control: regress z, no terrain
                    d_cam = torch.linalg.solve(
                        K.expand(Q, 3, 3),
                        torch.cat([uv, torch.ones(Q, 1, device=dev)], 1)[..., None])[..., 0]
                    zq = torch.exp(out["z_direct"][i].float())  # label gauge, median ~ 1
                    pc = d_cam / d_cam[:, 2:3] * zq[:, None]    # center on the contact ray
                z_label[i] = pc[:, 2]
                sigma_z[i] = zq * 0.05 if m.direct_z_head else lift["sigma_z"] * s
                center_cam[i] = pc
                n_cam[i] = (E[:, :3] @ n_w.T).T
                fallback[i] = lift["fallback"]
        out["z_label"] = z_label
        out["sigma_z"] = sigma_z
        out["center_cam"] = center_cam
        out["surface_up_cam"] = n_cam
        out["lift_fallback"] = fallback
