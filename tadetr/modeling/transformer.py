"""TA-DETR decoder (spec 2.2): deformable-DETR decoder, 6 layers, 300 queries, single stride-14
level, with (a) telemetry + camera tokens appended to the self-attention KV set -- NOT to the
deformable memory, whose sampling can only reach grid tokens (spec bug, fixed in design) -- and
(b) one herd cross-instance sub-layer per layer: query-query attention with an additive
same-class bias computed straight-through from the PREVIOUS layer's class logits.  [M3]
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .ops_msda import MSDeformAttn


def pos_embed_2d(h: int, w: int, d: int, device, dtype=torch.float32) -> torch.Tensor:
    """(h*w, d) sine position embedding."""
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                            indexing="ij")
    dim_t = torch.arange(d // 4, device=device, dtype=torch.float32)
    dim_t = 10000 ** (2 * (dim_t // 2) / (d // 2))
    px = xs.flatten()[:, None] / dim_t
    py = ys.flatten()[:, None] / dim_t
    pe = torch.cat([px.sin(), px.cos(), py.sin(), py.cos()], dim=1)
    return pe.to(dtype)


class DecoderLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, n_points: int, herd_bias: float,
                 herd_attention: bool):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.cross = MSDeformAttn(d_model=d, n_levels=1, n_heads=n_heads, n_points=n_points)
        self.n2 = nn.LayerNorm(d)
        self.herd_attention = herd_attention
        self.herd_bias = herd_bias
        if herd_attention:
            self.herd = nn.MultiheadAttention(d, n_heads, batch_first=True)
            self.n3 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        self.n4 = nn.LayerNorm(d)

    def forward(self, q, qpos, mem, mem_shapes, mem_start, refs, extra_tokens, prev_logits):
        # self-attention over queries + telemetry/camera tokens as extra KV
        kv = q + qpos
        if extra_tokens is not None:
            kv = torch.cat([kv, extra_tokens], dim=1)
        a, _ = self.self_attn(q + qpos, kv, torch.cat([q, extra_tokens], dim=1)
                              if extra_tokens is not None else q)
        q = self.n1(q + a)
        # deformable cross-attention into the feature grid
        a = self.cross(q + qpos, refs, mem, mem_shapes, mem_start, None)
        q = self.n2(q + a)
        # herd cross-instance sub-layer with same-class additive bias (straight-through argmax)
        if self.herd_attention:
            if prev_logits is not None:
                cls = prev_logits[..., :-1].argmax(-1)                    # (B,Q), no grad
                same = (cls[:, :, None] == cls[:, None, :]).to(q.dtype)
                bias = self.herd_bias * same                              # additive attn bias
                bias = bias.repeat_interleave(self.herd.num_heads, dim=0)
            else:
                bias = None
            a, _ = self.herd(q + qpos, q + qpos, q, attn_mask=None if bias is None
                             else -bias.max() + bias)  # shift so max bias = 0 (mask is additive)
            q = self.n3(q + a)
        q = self.n4(q + self.ffn(q))
        return q


class TADETRDecoder(nn.Module):
    def __init__(self, d: int, n_layers: int, n_queries: int, n_heads: int, n_points: int,
                 herd_bias: float, herd_attention: bool):
        super().__init__()
        self.query_embed = nn.Embedding(n_queries, d)
        self.query_pos = nn.Embedding(n_queries, d)
        self.ref_head = nn.Linear(d, 2)
        self.layers = nn.ModuleList(DecoderLayer(d, n_heads, n_points, herd_bias, herd_attention)
                                    for _ in range(n_layers))
        self.tel_mlp = nn.Sequential(nn.Linear(6, d), nn.ReLU(), nn.Linear(d, d))
        self.cam_mlp = nn.Sequential(nn.Linear(4, d), nn.ReLU(), nn.Linear(d, d))
        nn.init.uniform_(self.query_pos.weight, -1, 1)

    def forward(self, feat: torch.Tensor, telemetry: torch.Tensor, cam_feats: torch.Tensor,
                class_head: nn.Module, ref_boxes: torch.Tensor | None = None,
                use_telemetry: bool = True):
        """feat (B,d,h,w); telemetry (B,6); cam_feats (B,4);
        ref_boxes (B,Q',4 cxcywh in [0,1]) optionally overrides learned queries (oracle-2D
        box-conditioned parity mode: one query per provided box, reference at box bottom-center).
        Returns list of per-layer query states (B,Q,d) and reference points (B,Q,2)."""
        B, d, h, w = feat.shape
        mem = feat.flatten(2).transpose(1, 2)                       # (B, hw, d)
        mem = mem + pos_embed_2d(h, w, d, feat.device, feat.dtype)[None]
        shapes = torch.as_tensor([[h, w]], device=feat.device)
        start = torch.zeros(1, dtype=torch.long, device=feat.device)

        if ref_boxes is None:
            q = self.query_embed.weight[None].expand(B, -1, -1)
            qpos = self.query_pos.weight[None].expand(B, -1, -1)
            refs = self.ref_head(qpos).sigmoid()                    # (B,Q,2)
        else:
            Qn = ref_boxes.shape[1]
            q = self.query_embed.weight[:1].expand(B, Qn, -1).contiguous()
            qpos = self.query_pos.weight[:1].expand(B, Qn, -1).contiguous()
            refs = torch.stack([ref_boxes[..., 0],
                                (ref_boxes[..., 1] + ref_boxes[..., 3] / 2).clamp(0, 1)], -1)

        toks = [self.cam_mlp(cam_feats)[:, None]]
        if use_telemetry:
            toks.insert(0, self.tel_mlp(telemetry)[:, None])
        extra = torch.cat(toks, dim=1)

        hs_list, prev_logits = [], None
        refs_in = refs[:, :, None, :]                               # (B,Q,1,2) single level
        for layer in self.layers:
            q = layer(q, qpos, mem, shapes, start, refs_in, extra, prev_logits)
            hs_list.append(q)
            prev_logits = class_head(q).detach()
        return hs_list, refs
