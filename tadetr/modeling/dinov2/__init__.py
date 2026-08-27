# Vendored DINOv2 ViT-L/14 backbone for TA-DETR.
#
# Self-contained: no imports from detect_anything, mmcv, or torch.hub.
# xformers is OPTIONAL (plain-attention fallback on CPU or when absent).
#
# Sources (DetAny3D vendored copy of Meta's DINOv2, Apache/Meta license):
#   /home/shuklva/DetAny3D/detect_anything/modeling/backbones/dinov2.py
#   /home/shuklva/DetAny3D/detect_anything/modeling/backbones/metadinov2/
#
# Usage
# -----
#     from tadetr.modeling.dinov2 import build_vit_large
#
#     model = build_vit_large(patch_size=14)            # random init
#     model = build_vit_large(
#         patch_size=14,
#         checkpoint_path="~/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth",
#     )
#     feats = model.get_intermediate_layers(img, n=(12, 18, 24))
#     # -> tuple of three (B, 1024, H/14, W/14) feature maps
#
# INDEXING CONVENTION: ``n`` in ``get_intermediate_layers`` (and
# ``output_idx`` in the constructor / plain ``forward``) is **1-indexed
# block numbers**: ViT-L has blocks 1..24, and n=(12, 18, 24) taps the
# 12th, 18th and 24th blocks -- exactly the TA-DETR taps {12, 18, 24}.
# (The underlying loop takes block i of the 0-indexed ModuleList when
# i + 1 is in ``n``.)
#
# Inputs may be rectangular; each side must be a multiple of the patch
# size (14), e.g. 1022 x 574 -> a 73 x 41 patch grid.  The position
# embedding is bicubically interpolated to the rectangular grid.

import torch

from .vision_transformer import DinoVisionTransformer, vit_large

__all__ = ["build_vit_large", "DinoVisionTransformer", "vit_large"]


def build_vit_large(
    patch_size=14,
    img_size=518,
    init_values=1.0,
    ffn_layer="mlp",
    block_chunks=0,
    num_register_tokens=0,
    output_idx=(12, 18, 24),
    drop_path_rate=0.0,
    use_norm=False,
    export=False,
    interpolate_offset=0.0,
    checkpoint_path=None,
    **kwargs
):
    """Build a DINOv2 ViT-L/14 (embed_dim=1024, depth=24, heads=16).

    Defaults match the official ``dinov2_vitl14`` hub config
    (img_size=518 -> 37x37 base pos-embed grid, layerscale init 1.0,
    MLP FFN, no register tokens, flat block list).

    Args:
        patch_size: patch size (14 for DINOv2).
        img_size: base image size used only to size the stored pos-embed;
            actual inputs of any (multiple-of-patch) H x W are fine.
        init_values: layer-scale init (1.0 matches the pretrained ckpt).
        ffn_layer: "mlp" for ViT-L/14.
        block_chunks: keep 0 -- required by ``get_intermediate_layers``
            and matches the flat ``blocks.N.*`` checkpoint keys.
        num_register_tokens: 0 for the plain vitl14 checkpoint.
        output_idx: 1-indexed block numbers returned by plain
            ``forward`` (TA-DETR taps (12, 18, 24)).
        drop_path_rate: stochastic depth (0.0 for inference).
        use_norm: apply final LayerNorm to ``forward`` outputs.
        export: use plain ``Attention`` instead of ``MemEffAttention``
            (irrelevant numerically; MemEffAttention already falls back
            to plain attention on CPU or when xformers is missing).
        interpolate_offset: pos-embed interpolation kludge; 0.0 uses
            exact output sizes (recommended).
        checkpoint_path: optional local .pth to load (state_dict, e.g.
            dinov2_vitl14_pretrain.pth). Loaded with strict=False;
            the mismatch info is printed.

    Returns:
        DinoVisionTransformer
    """
    model = vit_large(
        patch_size=patch_size,
        img_size=img_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        output_idx=list(output_idx),
        drop_path_rate=drop_path_rate,
        use_norm=use_norm,
        export=export,
        interpolate_offset=interpolate_offset,
        **kwargs
    )
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        info = model.load_state_dict(state_dict, strict=False)
        print("loaded {} with: {}".format(checkpoint_path, info))
    return model
