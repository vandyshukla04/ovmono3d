# TA-DETR — Terrain-Anchored Detection with Herd Scale Coupling
## Build specification v0.1 (hand this to the coding agent)

**Working name:** TA-DETR (Terrain-Anchored DETR).
**One-line goal:** A monocular 3D detector for aerial wildlife video that never regresses depth. It predicts a 2D ground-contact point per animal, computes depth by differentiable ray–terrain intersection, and resolves metric scale jointly across all same-species animals in the scene.

**Non-goals for v0.1:** no tracking, no temporal fusion, no onboard/edge optimization, no new species beyond WildBox classes, no backbone training from scratch.

---

## 0. Data contracts (build these first, everything depends on them)

### 0.1 Inputs available
- WildBox dataset: KITTI/Omni3D-format 3D boxes, per-segment scale-normalised camera frame, instance IDs per segment, video-level train/val/test splits, official eval code.
- Per-segment VGGT outputs (already produced by the VGGT-based WildLIFT pipeline; labels live in this frame): per-frame pointmaps `P_t [H, W, 3]` (confirm camera vs world frame and record), per-point confidence `C_t [H, W]`, camera poses `T_t ∈ SE(3)`, recovered intrinsics `K_t [3,3]`. (these outputs for the corresponding wildbox data can be found here mnt\d\3DBOX\Data\WildBox\data)
- Consistency check (blocking, run once): each segment's poses/pointmaps must share one global frame. If VGGT was run in chunks, store the chunk-alignment transforms and assert boundary residuals < 1 percent of median camera height. Add per-segment mean reprojection residual to the feasibility census; flag high-rotation segments.
- Grounded-SAM 2D masks per frame (same detections used in the WildBox baselines — do NOT regenerate).
- Gimbal telemetry per segment where available: pitch, roll, altitude (DJI SRT). May be missing; all telemetry-consuming code must handle `None`.

### 0.2 Terrain cache (precompute once per segment, offline job)
File: `terrain/{segment_id}.npz`
- `H_grid  [G, G] float32` — terrain height field over the segment's ground footprint. Build: accumulate all non-animal pointmap points across the whole segment (animal pixels removed via dilated masks, dilation 15 px), grid the XY plane at resolution `G = 256`, per-cell robust height = confidence-weighted median of points in cell (weights = VGGT per-point confidence, drop points below the 20th confidence percentile), fill empty cells by nearest-neighbor then apply Laplacian smoothing (lambda = 0.5, 20 iterations). Store also:
- `grid_origin [2]`, `grid_scale [1]` — XY-to-cell mapping.
- `H_var [G, G] float32` — per-cell height variance, inflated by (1 / mean cell confidence); used by the uncertainty head and for masking unreliable terrain.
- `plane [4] float32` — best-fit plane (RANSAC, inlier threshold = 2 percent of median camera height) as fallback where `H_var` is high or cells empty.
- Unit test: for every ground-truth box in the segment, signed distance from box bottom-center to terrain, normalised by box height. Assert median |distance| < 0.3 across the train split; log per-segment histograms. (Owner has verified labels sit on terrain; this test guards regressions.)

### 0.3 Sample format the dataloader emits
```
image        [3, 1024, 1024]   resized, intrinsics rescaled accordingly
K            [3, 3]
T_cam2world  [4, 4]
terrain      dict: H_grid, grid_origin, grid_scale, H_var, plane   (torch tensors, on GPU)
telemetry    [4]  = [sin(pitch), cos(pitch), roll, log(altitude_m)]  or zeros + valid flag
targets      list of {cls, box3d(10: c[3], d[3], R as 6d rot), box2d[4], contact_uv[2], instance_id}
segment_id, frame_id, species_present [S]  (multi-hot)
```
`contact_uv` targets: project GT box bottom-face center into the image with `K, T`. Precompute in the dataset builder, not at train time.

---

## 1. Stage 1 — training-free baseline as a frozen module (week 1)

Implement `geometric_lift(mask_or_box2d, K, T, terrain) -> {center3d, sigma_z}` exactly as in the training-free paper:
1. Contact pixel = centroid of the lowest 10 percent of mask pixels (fallback: box bottom-center + species offset table).
2. Ray `p(t) = o + t d` from camera center through contact pixel.
3. Root of `f(t) = p_z(t) − H(p_xy(t))` by: coarse march (64 samples between t_min = 0.2·altitude and t_max = 5·altitude), then 8 secant iterations. Bilinear interpolation on `H_grid`.
4. `sigma_z = (z / f_focal) · sigma_px / sin(theta_g)` where `theta_g` = angle between ray and local terrain tangent plane (normal from `H_grid` central differences), `sigma_px = 2.0` default.

This module is (a) ablation A0, (b) the initialization/sanity oracle, (c) reused inside the network as the intersection layer (Section 2.3). Write it once, in PyTorch, batched, differentiable (no `.item()`, no numpy in the forward path).

**Gate 1:** with GT 2D boxes on WildBox val, NHD-z of `geometric_lift` must land at or below the published fine-tuned band (5.97–11.48). If not, stop and debug terrain/frames before any training.

---

## 2. Stage 2 — the model

### 2.1 Backbone + encoder
- Backbone: DINOv2 ViT-L/14, frozen. Extract patch tokens from layers {12, 18, 24}, project each to `d = 256`, fuse by simple FPN-style top-down sum at stride 14.
- Optional (flag): LoRA rank 16 on the last 8 blocks, enabled only after milestone M3.
- Telemetry token: MLP(4 -> 256), appended to encoder memory. Camera token: MLP(flattened normalized K + camera height above terrain -> 256), appended likewise. If telemetry invalid, use a learned `no_telemetry` embedding.

### 2.2 Decoder and queries
- Standard deformable-DETR decoder, 6 layers, `d = 256`, 300 object queries, 8 heads.
- Insert one extra cross-instance attention sub-layer per decoder layer (after cross-attention, before FFN): queries attend to each other with an additive species-gate bias `b_ij = +2.0 if argmax cls_i == argmax cls_j else 0` (computed from the previous layer's class logits, straight-through). This is the herd-coupling pathway.

### 2.3 Heads (per query, after final decoder layer)
| Head | Output | Notes |
|---|---|---|
| class | S+1 logits | WildBox 6 classes + no-object |
| contact | (u, v) in [0,1]^2 | sigmoid; supervised by `contact_uv` |
| contact_sigma | sigma_px > 0 | softplus + 0.5 |
| height_residual | delta_h | contact height above terrain, in units of median camera height; tanh-bounded to [−0.05, 0.15] |
| center_offset | o_xyz [3] | box center minus contact point, in the terrain-tangent frame; bounded |
| dim_residual | dhat [3] | per-instance log-dimension residual (see 2.5) |
| yaw | (sin, cos) | rotation about local terrain normal |
| rot_refine | 6d rotation delta | small SO(3) correction on top of terrain-normal + yaw; identity-init |

**Depth is NOT a head.** Center comes from:
```
ray      = unproject(contact_uv, K, T)
t*       = intersect(ray, H_grid)            # module from Stage 1, differentiable
foot3d   = ray(t*) + delta_h * n_terrain
center3d = foot3d + R_tangent @ center_offset
```
Gradients flow into `contact_uv` and `delta_h` through the unrolled secant iterations (8 steps, no implicit-function shortcut in v0.1 — measure speed first, optimize later).

### 2.4 Analytic uncertainty passthrough
Per detection, compute `sigma_z_analytic` from the Stage-1 formula using the predicted `contact_sigma`. Final reported depth sigma = `sigma_z_analytic * exp(s_corr)` where `s_corr` is a single learned scalar per class (calibration correction). This keeps uncertainty interpretable: geometry sets the shape, learning sets one gain.

### 2.5 Herd latent scale module
Per (image, species s) with at least one query assigned to s:
```
g_s   = attention-pool over queries of species s (learned query vector per species)
mu_s, logvar_s = MLP(g_s)                       # posterior over log-scale
lscale_s ~ N(mu_s, exp(logvar_s))               # reparameterized sample at train, mu at eval
dims_i = exp(lscale_s) * exp(dhat_i) * D_bar_s  # D_bar_s = per-class mean dims from train-split stats (buffer, not learned)
```
Priors as losses (Section 3): `lscale_s ~ N(0, 0.15^2)` and `dhat_i ~ N(0, Sigma_s)` with `Sigma_s` = per-class log-dim covariance from train-split stats (buffer). Interpretation: the scene shares one scale per species; individuals deviate within allometric variance. Prediction to verify later (paper figure): posterior std of `lscale_s` shrinks ~ 1/sqrt(N_s).

---

## 3. Matching and losses

Hungarian matching cost per (query, target): `2·CE(cls) + 5·L1(contact_uv) + 2·(1 − GIoU(projected 2D box, GT 2D box))`. Match on 2D quantities only — never on 3D — so matching stays stable while 3D heads are untrained.

Total loss (lambda in parentheses):
1. Focal class loss (2.0)
2. Contact L1 (5.0) + contact NLL with `contact_sigma` (0.5)
3. Depth: L1 on z in the segment-normalised frame (2.0) + Laplacian NLL with reported sigma (0.5)
4. Center L1 on x, y (1.0)
5. Dimensions: L1 on log-dims (1.0) + prior NLL on `dhat_i` (0.1) + prior NLL on `lscale_s` (0.1) + KL of scale posterior to prior (0.05)
6. Rotation: yaw L1 on (sin,cos) (1.0) + geodesic loss on refined R (0.5), rot_refine enabled only after milestone M3
7. Auxiliary decoder losses at every layer (standard DETR deep supervision)

Do not add a 3D IoU loss in v0.1.

---

## 4. Training recipe

- Optimizer AdamW, lr 2e-4 (heads/decoder), 2e-5 (LoRA when enabled), weight decay 1e-4, cosine schedule, 50 epochs, effective batch 16.
- **Batch by segment**: each batch element is one frame, but sample 4 frames per segment per batch so the scale module sees multi-frame consistency pressure within a step (scale is per-image in v0.1; per-segment scale via EMA across frames of the same segment is v0.2).
- Augmentations — the intrinsics rule is absolute: any crop/resize/zoom must transform `K`, `contact_uv` targets, and the camera token consistently. Allowed: random resized crop (scale 0.7–1.0), horizontal flip (flip yaw and telemetry roll sign), photometric jitter. Forbidden: vertical flip, rotation, any aug that breaks the gravity/terrain relationship.
- Precision: bf16 autocast; the intersection layer runs in fp32 (root-finding is precision-sensitive).
- Curriculum: epochs 1–5 train contact + class only (freeze 3D heads); epochs 6–15 add depth/center/dims with scale module frozen at mu = 0; epoch 16+ everything. Mirrors the WildBox finding that curriculum init helps.
- Logging: per-epoch NHD decomposition (x, y, z, dims, rot) on val, calibration plot (predicted sigma_z bins vs empirical |error|), scale-posterior std vs N_s scatter.

Compute budget: fits one A100-40GB or two RTX 4090s (frozen ViT-L, 1024 px, batch 16 with grad accumulation 4). Full run ≈ 20–30 h. Ablation grid ≈ 6 runs.

---

## 5. Evaluation

- Primary: official WildBox eval code, untouched. Report AP-BEV@0.50 macro, AP3D macro, full NHD decomposition. Compare rows: published zero-shot (0.00), published fine-tuned OVMono3D-LIFT (8.68 / 13.17), published DetAny3D-FT (1.99 / 4.15), A0 (training-free), full TA-DETR.
- Detection input parity: evaluate both with own detections and with the same Grounded-SAM detections as the baselines (swap-in mode) to isolate 3D-stage gains.
- Calibration: reliability diagram of sigma_z; expected calibration error on depth.
- Provenance stratification: separate metrics on human-corrected vs auto-accepted label frames (annotation metadata available from WildLIFT-A logs).
- Independent-terrain arm (circularity defense — labels derive from VGGT, so VGGT terrain is the shared-backbone condition): rerun eval with terrain rebuilt from COLMAP on masked backgrounds (script `terrain_colmap.py`, similarity-aligned per segment to the label frame via camera trajectories). Optional third arm: CUT3R terrain (learned but different backbone). Report deltas across arms.

---

## 6. Ablation matrix (each is a config flag, build all from day one)

| ID | Config | Question answered |
|---|---|---|
| A0 | geometric_lift on GT and on Grounded-SAM 2D | floor: geometry alone |
| A1 | TA-DETR, contact head only, delta_h = 0, offsets = 0, dims = class means | does learned contact beat mask heuristic? |
| A2 | + height_residual + center_offset | value of residual learning |
| A3 | + dim_residual, scale module OFF (lscale = 0) | per-instance dims without herd coupling |
| A4 | + herd scale module (full model) | value of herd coupling; check 1/sqrt(N) prediction |
| A5 | full model, terrain replaced by direct depth head (regress z) | the control: is terrain the source of the win? |
| A6 | full model, telemetry token zeroed | value of telemetry |

Paper story = A0 -> A4 monotone improvement, A5 clearly worse, A4−A3 gap grows with group size.

---

## 7. Milestones and gates

| # | Deliverable | Time | Gate |
|---|---|---|---|
| M1 | Data contracts + terrain cache + unit tests green | 1 wk | terrain-consistency test passes on train split |
| M2 | Stage-1 module + Gate 1 numbers | 1 wk | NHD-z ≤ fine-tuned band with GT 2D |
| M3 | A1 trains, matches/beats A0 on val | 2 wk | if A1 < A0, debug matching/intersection gradients before proceeding |
| M4 | A2–A3 complete | 2 wk | monotone improvement |
| M5 | A4 + calibration + herd figure | 2 wk | scale-posterior shrinks with N_s |
| M6 | A5–A6, independent-terrain arm, provenance split, writing | 3 wk | — |

Total ≈ 11 weeks with one person + agent; M1–M2 overlap with the training-free paper's experiments (same code).

## 8. Known risks and pre-decided fallbacks

- Intersection gradients unstable near grazing rays -> clamp sin(theta_g) ≥ 0.05 in the backward; detections below that threshold fall back to plane intersection.
- Sparse background in close-up segments -> `H_var` mask triggers plane fallback per cell; log the fraction of plane-fallback detections and report it.
- Scale posterior collapse (logvar -> −inf) -> KL term floor plus free-bits (0.5 nats).
- Matching instability early -> curriculum already freezes 3D heads; if still unstable, add DN-DETR-style denoising queries (flagged, off by default).
- VGGT chunk misalignment within a segment -> caught by the blocking consistency check in 0.1; remedy is re-running global alignment for that segment, never per-chunk terrain.
- VGGT pose degradation on rapid-rotation segments -> census flags them; if Gate 1 outliers concentrate there, exclude and report as an operating-envelope limit rather than debugging indefinitely.