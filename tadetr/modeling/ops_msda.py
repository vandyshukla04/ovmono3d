# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
#
# Vendored for TA-DETR from
#   /home/shuklva/DetAny3D/detect_anything/modeling/ops/functions.py
#     (ms_deform_attn_core_pytorch, copied verbatim)
#   /home/shuklva/DetAny3D/detect_anything/modeling/ops/modules.py
#     (MSDeformAttn)
#
# Deviations from the source:
#   * NO mmcv, NO compiled/CUDA extension, NO MSDeformAttnFunction
#     autograd.Function: ``MSDeformAttn.forward`` ALWAYS uses the pure
#     PyTorch grid_sample core below (autograd handles the backward).
#   * The unused ``im2col_step`` attribute and the "power of 2 head dim"
#     CUDA-efficiency warning were dropped (CUDA path is gone).
#   * Everything else -- constructor signature, ``_reset_parameters``
#     (the load-bearing sampling-offset init), the forward contract --
#     is preserved.

from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

__all__ = ["MSDeformAttn", "ms_deform_attn_core_pytorch"]


def ms_deform_attn_core_pytorch(value, value_spatial_shapes,
                                sampling_locations, attention_weights):
    """Pure-PyTorch multi-scale deformable attention core (grid_sample based).

    :param value                (N, sum_l H_l*W_l, n_heads, d_per_head)
    :param value_spatial_shapes (n_levels, 2) tensor of (H_l, W_l)
    :param sampling_locations   (N, Len_q, n_heads, n_levels, n_points, 2) in [0, 1]
    :param attention_weights    (N, Len_q, n_heads, n_levels, n_points)
    :return                     (N, Len_q, n_heads * d_per_head)
    """
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear',
                                          padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0):
        """Multi-Scale Deformable Attention Module (pure-PyTorch forward).

        TA-DETR uses n_levels=1 with a single 73x41 (W x H, i.e.
        spatial_shapes = [[41, 73]]) feature map.

        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        :param ratio        value/output projection width ratio (keep 1.0)

        Forward contract (identical to Deformable-DETR / DetAny3D):

        :param query                    (N, Len_q, d_model)
        :param reference_points         (N, Len_q, n_levels, 2), normalized
                                        to [0, 1] over the padded input --
                                        top-left (0, 0), bottom-right (1, 1);
                                        or (N, Len_q, n_levels, 4) with an
                                        additional (w, h) forming reference
                                        boxes.
        :param input_flatten            (N, sum_l H_l*W_l, d_model)
        :param input_spatial_shapes     (n_levels, 2) LongTensor of
                                        (H_l, W_l), e.g. [[41, 73]].
        :param input_level_start_index  (n_levels,) LongTensor,
                                        [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
                                        (kept for API compatibility; the
                                        PyTorch core does not need it).
        :param input_padding_mask       (N, sum_l H_l*W_l) bool, True for
                                        padding elements, or None.

        :return output                  (N, Len_q, d_model)
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, '
                             'but got {} and {}'.format(d_model, n_heads))

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, int(d_model * ratio))
        self.output_proj = nn.Linear(int(d_model * ratio), d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(
            self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None):
        """See the class docstring for the full input contract."""
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] *
                input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        value = value.view(N, Len_in, self.n_heads,
                           int(self.ratio * self.d_model) // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).\
            view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'
                .format(reference_points.shape[-1]))
        # Pure-PyTorch core -- no CUDA extension, autograd provides backward.
        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output
