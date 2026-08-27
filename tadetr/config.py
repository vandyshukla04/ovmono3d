"""TA-DETR configuration — every ablation (A0-A7), curriculum stage, and site difference is a flag
here; the code branches on config, never on hard-coded constants.  [M3]

Load order: dataclass defaults -> yaml overlay (configs/tadetr/*.yaml) -> CLI dotted overrides.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataCfg:
    train_json: str = "datasets/Omni3D/WildBox_train.json"
    val_json: str = "datasets/Omni3D/WildBox_val.json"
    image_root: str = "datasets"                 # file_path in the jsons is relative to this
    terrain_dir: str = "datasets/tadetr/terrain"
    category_meta: str = "configs/wildbox/category_meta_wildlife6.json"
    input_w: int = 1022                          # 73*14 (dinov2/14); A7 dinov3/16 -> 1024
    input_h: int = 574                           # 41*14;             A7 dinov3/16 -> 576
    # contact target convention -- THE M2 finding: the 2D box bottom images the FEET (full-res),
    # the stamped 3D-projected contact_uv sits at SHIN height (518-res legs lost). box2d_bottom is
    # the empirically better pair (NHD-z 5.317 vs 6.001) and needs NO species offset in the lift.
    contact_source: str = "box2d_bottom"         # or "contact_uv_3d" (+species offset in lift)
    frames_per_segment: int = 4                  # spec 4.2: batch-by-segment
    segments_per_batch: int = 2                  # batch = 2 seg x 4 frames = 8
    epoch_frame_stride: int = 2                  # 10fps near-duplicate frames; stride-2 epochs
    num_workers: int = 4
    # augmentation (intrinsics rule is ABSOLUTE: K, contact, camera token co-transform)
    hflip_p: float = 0.5
    rrc_scale_min: float = 0.7                   # random resized crop, spec 4.3
    rrc_p: float = 0.5
    photometric_p: float = 0.5
    terrain_dropout_p: float = 0.3               # replace height field with its plane (edge-tier
                                                 # robustness; one model serves all deploy tiers)


@dataclass
class ModelCfg:
    backbone: str = "dinov2_vitl14"              # A7: "dinov3" (patch 16)
    backbone_blocks: tuple = (12, 18, 24)        # 1-INDEXED (the vendored dinov2 convention)
    d_model: int = 256
    dec_layers: int = 6
    n_queries: int = 300
    n_heads: int = 8
    msda_points: int = 4
    herd_bias: float = 2.0                       # same-class attention bias (spec 2.2)
    herd_attention: bool = True                  # extra cross-instance sub-layer
    # heads / composition
    dh_bounds: tuple = (-0.05, 0.15)             # height residual, units of segment cam_height
    center_offset_bound: float = 0.1             # |offset| <= bound * cam_height (~2 animal sizes)
    use_height_residual: bool = True             # A1: False (delta_h = 0)
    use_center_offset: bool = True               # A1: False
    use_dim_residual: bool = True                # A1: False (dims = class medians)
    use_herd_scale: bool = True                  # A3: False (lscale = 0)
    use_rot_refine: bool = False                 # on after M3 (spec)
    use_telemetry_token: bool = True             # A6: False (zeroed)
    direct_z_head: bool = False                  # A5 control: regress z, ignore terrain
    lora_rank: int = 0                           # optional post-M3


@dataclass
class LossCfg:
    w_class: float = 2.0
    w_box_l1: float = 2.0
    w_box_giou: float = 1.0
    w_contact_l1: float = 5.0
    w_contact_nll: float = 0.5
    w_z_l1: float = 2.0
    w_z_nll: float = 0.5
    w_center_xy: float = 1.0
    w_dims: float = 1.0
    w_dims_prior: float = 0.1
    w_lscale_prior: float = 0.1
    w_kl: float = 0.05
    kl_free_bits: float = 0.5
    w_axis: float = 1.0
    w_sign: float = 0.5
    w_rot_refine: float = 0.5
    # species weights for the dense axis term (measured: PCA long-axis validity per species)
    axis_species_weight: dict = field(default_factory=lambda: {
        "elephant": 1.0, "rhino": 1.0, "plains_zebra": 0.5, "grevys_zebra": 0.5,
        "giraffe": 0.0, "gazelle": 0.0})
    # matcher (2D-only, spec section 3)
    match_class: float = 2.0
    match_contact: float = 5.0
    match_giou: float = 2.0


@dataclass
class TrainCfg:
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 15                             # A1/M3; full A4 run raises this (spec: 50)
    warmup_iters: int = 500
    clip_grad: float = 0.1
    amp: str = "bf16"                            # A40; "off" for V100 (NO bf16 on Volta,
                                                 # fp16 FORBIDDEN: silently zeroed a ViT-L forward)
    accum_steps: int = 2                         # effective batch 16
    seed: int = 0
    # curriculum (spec 4.5): epoch thresholds where loss groups switch on
    stage2_epoch: int = 5                        # + depth/center/dims (scale module frozen mu=0)
    stage3_epoch: int = 15                       # + everything (herd sampling, rot_refine if on)
    ckpt_every_iters: int = 1000
    log_every: int = 50
    out_dir: str = "runs/tadetr_a1"
    backbone_weights: str = ""                   # site yaml sets the dinov2 pth path


@dataclass
class TADETRConfig:
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    @staticmethod
    def load(yaml_path: str | None = None, overrides: list | None = None) -> "TADETRConfig":
        """yaml_path may be comma-separated: later files overlay earlier (arm.yaml,site.yaml)."""
        cfg = TADETRConfig()
        for path in (yaml_path.split(",") if yaml_path else []):
            import yaml
            layers = yaml.safe_load(Path(path).read_text()) or {}
            for section, kv in layers.items():
                sub = getattr(cfg, section)
                for k, v in (kv or {}).items():
                    if not hasattr(sub, k):
                        raise KeyError(f"unknown config key {section}.{k}")
                    setattr(sub, k, tuple(v) if isinstance(getattr(sub, k), tuple) else v)
        for ov in overrides or []:                    # "train.lr=1e-4"
            path, _, val = ov.partition("=")
            section, _, key = path.partition(".")
            sub = getattr(cfg, section)
            cur = getattr(sub, key)                   # raises on unknown key -- loud
            setattr(sub, key, type(cur)(json.loads(val)) if not isinstance(cur, str) else val)
        return cfg

    def dump(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=1, default=str)
