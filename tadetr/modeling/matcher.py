"""Hungarian matcher — 2D-ONLY by construction (spec section 3): class + contact + GIoU(box2d).
Never a 3D quantity, so matching stays stable while the 3D composition is untrained.  [M3]
"""
from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(N,4),(M,4) xyxy -> (N,M) GIoU."""
    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(0).prod(-1)
    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / union.clamp_min(1e-9)
    lt_h = torch.min(a[:, None, :2], b[None, :, :2])
    rb_h = torch.max(a[:, None, 2:], b[None, :, 2:])
    hull = (rb_h - lt_h).clamp(0).prod(-1)
    return iou - (hull - union) / hull.clamp_min(1e-9)


@torch.no_grad()
def hungarian_match(logits: torch.Tensor, boxes: torch.Tensor, contact: torch.Tensor,
                    targets: list, w_class: float, w_contact: float, w_giou: float):
    """Per image: logits (Q,C+1), boxes (Q,4 cxcywh), contact (Q,2); targets = list of dicts.
    Returns list of (query_idx, target_idx) LongTensor pairs."""
    out = []
    for i, tg in enumerate(targets):
        n = len(tg["cls"])
        if n == 0:
            out.append((torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)))
            continue
        prob = logits[i].softmax(-1)
        cost_cls = -prob[:, tg["cls"]]
        cost_ct = torch.cdist(contact[i], tg["contact"], p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(boxes[i]),
                                         box_cxcywh_to_xyxy(tg["boxes"]))
        C = (w_class * cost_cls + w_contact * cost_ct + w_giou * cost_giou).cpu().numpy()
        qi, ti = linear_sum_assignment(C)
        out.append((torch.as_tensor(qi, dtype=torch.long),
                    torch.as_tensor(ti, dtype=torch.long)))
    return out
