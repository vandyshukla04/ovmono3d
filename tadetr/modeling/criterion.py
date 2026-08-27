"""TA-DETR criterion (spec section 3, amended heads/losses table).  [M3]

Standard DETR deep supervision: every decoder layer's outputs are matched (2D-only Hungarian) and
scored. Every masked loss follows the project idiom -- if a mask is empty the KEY IS OMITTED,
never a NaN (the T7 lesson). Curriculum stages gate loss groups by epoch (train loop passes
`stage`): 1 = class+box+contact, 2 = +depth/center/dims (herd frozen), 3 = everything.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LossCfg
from .matcher import box_cxcywh_to_xyxy, generalized_box_iou, hungarian_match


def sigmoid_focal_loss(logits, tgt_onehot, alpha=0.25, gamma=2.0):
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, tgt_onehot, reduction="none")
    pt = p * tgt_onehot + (1 - p) * (1 - tgt_onehot)
    w = (alpha * tgt_onehot + (1 - alpha) * (1 - tgt_onehot)) * (1 - pt) ** gamma
    return (w * ce).sum(-1)


class TADETRCriterion(nn.Module):
    def __init__(self, cfg: LossCfg, n_classes: int):
        super().__init__()
        self.cfg = cfg
        self.C = n_classes

    def forward(self, outputs_per_layer: list, targets: list, stage: int) -> dict:
        losses = {}
        for li, out in enumerate(outputs_per_layer):
            pre = f"l{li}_" if li < len(outputs_per_layer) - 1 else ""
            l = self._layer_losses(out, targets, stage)
            for k, v in l.items():
                losses[pre + k] = v
        return losses

    def _layer_losses(self, out: dict, targets: list, stage: int) -> dict:
        cfg = self.cfg
        match = hungarian_match(out["logits"], out["boxes"], out["contact"], targets,
                                cfg.match_class, cfg.match_contact, cfg.match_giou)
        B, Q, _ = out["logits"].shape
        device = out["logits"].device
        L = {}

        # class: focal over all queries; matched -> gt class, rest -> background
        tgt_cls = torch.full((B, Q), self.C, dtype=torch.long, device=device)
        for i, (qi, ti) in enumerate(match):
            if len(qi):
                tgt_cls[i, qi] = targets[i]["cls"][ti].to(device)
        onehot = F.one_hot(tgt_cls, self.C + 1).to(out["logits"].dtype)
        n_matched = max(sum(len(qi) for qi, _ in match), 1)
        L["loss_class"] = cfg.w_class * sigmoid_focal_loss(
            out["logits"], onehot).sum() / n_matched

        # gather matched pairs across the batch
        def gather(field_out, field_tgt):
            po, pt = [], []
            for i, (qi, ti) in enumerate(match):
                if len(qi):
                    po.append(field_out[i][qi])
                    pt.append(targets[i][field_tgt][ti].to(device))
            if not po:
                return None, None
            return torch.cat(po), torch.cat(pt)

        pb, tb = gather(out["boxes"], "boxes")
        if pb is not None:
            L["loss_box_l1"] = cfg.w_box_l1 * F.l1_loss(pb, tb)
            giou = generalized_box_iou(box_cxcywh_to_xyxy(pb), box_cxcywh_to_xyxy(tb))
            L["loss_box_giou"] = cfg.w_box_giou * (1 - giou.diagonal()).mean()

        pc, tc = gather(out["contact"], "contact")
        cv = torch.cat([targets[i]["contact_valid"][ti].to(device)
                        for i, (qi, ti) in enumerate(match) if len(qi)]) if pc is not None else None
        if pc is not None and cv is not None and cv.any():
            m = cv > 0
            L["loss_contact"] = cfg.w_contact_l1 * F.l1_loss(pc[m], tc[m])
            sig = torch.cat([out["contact_sigma"][i][qi]
                             for i, (qi, ti) in enumerate(match) if len(qi)])[m]
            err = (pc[m] - tc[m]).norm(dim=-1) * out["img_diag_px"]
            L["loss_contact_nll"] = cfg.w_contact_nll * (
                err / sig.clamp_min(1e-3) + torch.log(sig.clamp_min(1e-3))).mean()

        if stage >= 2:
            pz, tcen = gather(out["z_label"][..., None], "center_cam")
            v3 = torch.cat([targets[i]["valid3d"][ti].to(device)
                            for i, (qi, ti) in enumerate(match) if len(qi)]) \
                if pz is not None else None
            if pz is not None and (v3 > 0).any():
                m = v3 > 0
                z = pz[:, 0]
                L["loss_z"] = cfg.w_z_l1 * F.l1_loss(z[m], tcen[m][:, 2])
                psig = torch.cat([out["sigma_z"][i][qi] for i, (qi, ti) in enumerate(match)
                                  if len(qi)])
                L["loss_z_nll"] = cfg.w_z_nll * (
                    (z[m] - tcen[m][:, 2]).abs() / psig[m].clamp_min(1e-4)
                    + torch.log(psig[m].clamp_min(1e-4))).mean()
                pcen, _ = gather(out["center_cam"], "center_cam")
                L["loss_center_xy"] = cfg.w_center_xy * F.l1_loss(
                    pcen[m][:, :2], tcen[m][:, :2])
                pd, td = gather(out["dims"], "dims")
                L["loss_dims"] = cfg.w_dims * F.l1_loss(
                    torch.log(pd[m].clamp_min(1e-6)), torch.log(td[m].clamp_min(1e-6)))
                pdh, _ = gather(out["dim_residual"], "dims")
                L["loss_dims_prior"] = cfg.w_dims_prior * pdh[m].pow(2).mean()

            # axis (dense, species-weighted) + sign (sparse motion labels)
            pa, ta = gather(out["axis"], "axis_embed")
            if pa is not None:
                wsp = torch.cat([
                    torch.tensor([cfg.axis_species_weight.get(nm, 0.0)
                                  for nm in [targets[i]["names"][j] for j in ti.tolist()]],
                                 device=device)
                    * targets[i]["valid3d"][ti].to(device)
                    for i, (qi, ti) in enumerate(match) if len(qi)])
                if (wsp > 0).any():
                    pan = pa / pa.norm(dim=-1, keepdim=True).clamp_min(1e-3)
                    L["loss_axis"] = cfg.w_axis * (
                        (pan - ta).abs().sum(-1) * wsp).sum() / wsp.sum().clamp_min(1e-6)
                psg, tsg = gather(out["sign_logit"][..., None], "sign_target")
                sv = torch.cat([targets[i]["sign_valid"][ti].to(device)
                                for i, (qi, ti) in enumerate(match) if len(qi)])
                if (sv > 0).any():
                    m = sv > 0
                    L["loss_sign"] = cfg.w_sign * F.binary_cross_entropy_with_logits(
                        psg[m][:, 0], tsg[m])

        if stage >= 3 and out.get("herd") is not None:
            h = out["herd"]
            kl = 0.5 * (h["mu"].pow(2) / (0.15 ** 2) + h["logvar"].exp() / (0.15 ** 2)
                        - 1 - h["logvar"] + 2 * torch.log(torch.tensor(0.15, device=device)))
            kl = torch.clamp(kl - cfg.kl_free_bits, min=0.0)
            mask = h["has_species"].to(kl.dtype)
            if mask.any():
                L["loss_kl"] = cfg.w_kl * (kl * mask).sum() / mask.sum().clamp_min(1.0)
                L["loss_lscale_prior"] = cfg.w_lscale_prior * (
                    (h["lscale"].pow(2) / (0.15 ** 2)) * mask).sum() / mask.sum().clamp_min(1.0)
        return L
