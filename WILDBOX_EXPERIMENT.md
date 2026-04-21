# WildBox 3D Wildlife Detection — Experiment Documentation

**Purpose of this document.** This is a self-contained reference describing the dataset, preprocessing pipeline, training strategy, and evaluation protocol used for the WildBox 3D wildlife detection experiment on top of OVMono3D. It is written so that (1) a researcher can reproduce or adapt the pipeline for another 3D detection architecture such as Cube R-CNN or DetAny3D, and (2) a paper can be written from it without needing to read the code.

---

## 1. TL;DR

- **Task.** Monocular 3D detection of African wildlife from aerial drone footage. Output per image: 2D bbox, 3D cuboid (center, dimensions, 3D rotation), class label, confidence.
- **Dataset.** WildBox — custom multi-species dataset of drone videos across rhinos, elephants, zebras, giraffes, and gazelles. Ground truth 3D cuboids are **pseudo-labels** derived from a pipeline of SAM3 (segmentation) → VGGT (3D reconstruction / tracking).
- **Architecture used here.** OVMono3D-lift (Cube R-CNN variant with DINOv2-ViT-B/14 backbone and 6-DoF cube head), initialized from the Omni3D-pretrained checkpoint `ovmono3d_lift.pth`.
- **Fine-tuning protocol.** 20 000 SGD iterations at LR 1e-3, batch 4, 8 dataloader workers, mixed precision. Loss weights adapted for synthetic-scale VGGT data: `LOSS_W_Z = 0.5`, `VIRTUAL_DEPTH = False`. Repeat-factor sampling upweights long-tail classes.
- **Primary evaluation.** **AP-BEV @ IoU 0.25** (bird's-eye-view 2D AP on cuboid footprints). Chosen because the VGGT-derived depth axis is the noisiest dimension of the pseudo-labels; BEV removes it from the primary metric. Secondary: **AP-3D @ IoU 0.25**, **Rel-AP-3D** (LabelAny3D scale-aligned), **2D AP @ 0.5** (detection sanity).
- **Splits.** **Video-level** 80/20, seeded. No frames from the same video appear in both train and val.
- **Class imbalance.** Rhino/elephant/zebra majority; giraffe and gazelle are long-tail in video count. Reported with macro-mAP as headline, micro-mAP as companion.

---

## 2. Dataset

### 2.1 Source and provenance

- Raw videos: DJI drone footage from wildlife conservation sites in Kenya (Cami, BZS, KABR, Tom Blair Ranch). Collection years span 2023–2026.
- Five species: **rhino, elephant, zebra, giraffe, gazelle**. The zebra category pools *Plains zebra* (Equus quagga) and *Grévy's zebra* (Equus grevyi); this is noted in the paper's limitations.
- Per-video preprocessing:
  1. **SAM3** produces per-frame, per-object binary segmentation masks from a text prompt (e.g. `"rhino"`).
  2. **VGGT** (Visual Geometry and Global Tracking) consumes the frames + masks and produces per-frame camera intrinsics/extrinsics, a point cloud, and 3D tracks with per-frame centers, dimensions, and rotation matrices.

### 2.2 Data source inventory

13 archive zips, organized by species and collection campaign, at:

```
/storage3/3DOM/vshukla/sam3/wd_data/wildbox/archive/
  data202401KElephants/    data202401KGiraffes/    data202401KRhinos/
  data202406KElephants/    data202406KGazelles/
  data202501KGiraffes/
  data202502KRhinoCamiV1/  data202502KRhinoCamiV2/
  data202602KElephants/
  data2023KABRZebras/
  dataBZS/                 wildbox_tomblair/
```

Each zip, once extracted, has the layout:

```
WildBox/
  DJI_YYYYMMDDHHMMSS_XXXX_V/              # one directory per video
    seg1/                                  # segments = time ranges inside a video
      frame_NNNNNN.jpg                     # full-resolution RGB
      metadata.json                        # start_time, end_time, frame_numbers[]
      sam3_masks/
        metadata.json                      # text prompt, object ids, frame list
        masks/
          obj_0/
            frame_NNNNNN.png               # 1920x1080 8-bit binary mask
      vggt_results/
        cameras.json                       # per-frame intrinsic (3x3) + extrinsic (3x4)
        tracking_summary.json              # per-track: centers[], dimensions[],
                                           #            rotation_matrices[], bbox_2d[],
                                           #            frames[], class_name
        depth_maps.npz, point_cloud.ply, ...  # unused by this experiment
    seg2/ ...
  DJI_YYYYMMDDHHMMSS_XXXX_V.SRT           # (metadata, unused)
```

### 2.3 Frame and object counts (per zip, from [datasets/run1_description.xlsx](datasets/run1_description.xlsx))

| Zip | Frames | Bboxes | Segments | Videos | Species |
|---|---:|---:|---:|---:|---|
| wildbox_tomblair | 4627 | 9218 | 25 | 3 | Plains zebra |
| data202401KElephants | 5527 | 6562 | 34 | 6 | Elephant |
| data202401KGiraffes | 1220 | 3297 | 8 | 2 | Giraffe |
| data202502KRhinoCamiV1 | 5107 | 5099 | 28 | 4 | Rhino |
| data202406KElephants | 3369 | 4962 | 21 | 4 | Elephant |
| data202502KRhinoCamiV2 | 8450 | 15885 | 47 | 12 | Rhino |
| dataBZS | 9300 | 53236 | 53 | 12 | Plains zebra |
| data202602KElephants | 5300 | 31406 | 33 | 7 | Elephant |
| data2023KABRZebras | 4610 | 23048 | 30 | 5 | Grévy's Zebra |
| data202406KGazelles | 7443 | 59399 | 39 | 4 | Gazelle |
| data202501KGiraffes | 310 | 391 | 2 | 2 | Giraffe |
| data202401KRhinos | 9779 | 30128 | 53 | 7 | Rhino |

Roll-ups per species (all 13 zips):

| Species | Total frames | Total bboxes | Total segments | Total videos |
|---|---:|---:|---:|---:|
| Rhino | 23 336 | 51 112 | 128 | 23 |
| Elephant | 14 196 | 42 930 | 88 | 17 |
| Zebra (Plains + Grévy's) | 18 537 | 85 502 | 108 | 20 |
| Giraffe | 1 530 | 3 688 | 10 | 4 |
| Gazelle | 7 443 | 59 399 | 39 | 4 |
| **Total** | **~65 000** | **~243 000** | **~373** | **~68** |

**Class imbalance note.** Bboxes-per-frame varies widely: gazelle ≈ 8 (herds), zebra ≈ 4.6, giraffe ≈ 2.4. Video counts are more skewed: giraffe (4) and gazelle (4) are long-tail, while rhino (23) and zebra (20) are well-sampled.

### 2.4 Coordinate conventions

All 3D quantities are stored in **camera frame** (right-handed, X=right, Y=down, Z=forward).

- **Dimensions ordering (Omni3D convention).** `[W, H, L]` where X-extent = L, Y-extent = H, Z-extent = W. This matches `cubercnn/util/math_util.py:get_cuboid_verts_faces`.
- **Rotation.** Full 3×3 matrix, not just yaw (important for top-down drone shots where pitch/roll carry most of the orientation information).
- **VGGT-internal dimensions** in `tracking_summary.json` are `[l, w, h]` (X/Y/Z extent). The reverse of Omni3D's ordering. The data-prep script handles the swap.

### 2.5 Scale normalization (critical for VGGT-derived GT)

VGGT produces 3D reconstructions in its own synthetic units — **absolute depth is meaningless**, only relative structure is reliable. The prep pipeline normalizes each segment independently so that the median camera-frame |Z| of all cuboid centers maps to 1.0:

```
scene_scale s = 1 / median(|center_cam_z|)   # one scalar per segment
center_cam     *= s
cuboid_corners *= s
dimensions     *= s
# intrinsics K is NOT rescaled — (X/Z, Y/Z) projection is scale-invariant.
```

This is a **uniform scaling per segment**. Box shape, orientation, 2D projection, and 3D-IoU are all preserved. What it kills is absolute depth comparability *between* segments, which is why scale-invariant metrics (Rel-AP-3D, BEV) are critical for evaluation.

### 2.6 Splits

**Video-level**, not frame-level or segment-level, to avoid same-scene leakage. All segments from a given video go to the same split side.

- `--split-mode video`
- `--val-fraction 0.2`
- `--seed 0` (deterministic across re-runs)
- Stratified by species: each species's videos are shuffled and split independently so every species appears in both train and val.

```
train:val (approximate, per species)
rhino:    19 / 4  videos
elephant: 14 / 3  videos
zebra:    16 / 4  videos
giraffe:   3 / 1  videos      <-- LONG TAIL; high variance
gazelle:   3 / 1  videos      <-- LONG TAIL; high variance
```

### 2.7 Omni3D-format output

The prep script writes two JSONs consumed by the trainer:

- `datasets/Omni3D/WildBox_train.json`
- `datasets/Omni3D/WildBox_val.json`

Shape (matches Omni3D / Cube R-CNN convention):

```jsonc
{
  "info": {
    "id": 1000,
    "source": "wildbox",
    "name": "WildBox_train",
    "split": "train",
    "known_category_ids": [1000, 1001, 1002, 1004, 1005],
    "scene_scales": {"rhino/DJI_xxx/seg1": 0.45, ...}
  },
  "categories": [
    {"id": 1000, "name": "giraffe",  "supercategory": "animal"},
    {"id": 1001, "name": "zebra",    "supercategory": "animal"},
    {"id": 1002, "name": "elephant", "supercategory": "animal"},
    {"id": 1004, "name": "rhino",    "supercategory": "animal"},
    {"id": 1005, "name": "gazelle",  "supercategory": "animal"}
  ],
  "images": [
    {
      "id": 0,
      "dataset_id": 1000,
      "file_path": "/abs/path/to/frame_000146.jpg",   // absolute path!
      "height": 1080,
      "width":  1920,
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    }
  ],
  "annotations": [
    {
      "id": 0, "image_id": 0, "dataset_id": 1000,
      "category_id": 1004, "category_name": "rhino",
      "bbox":          [x, y, w, h],                   // TIGHT 2D (SAM3 mask extent)
      "bbox2D_tight":  [x1, y1, x2, y2],               // TIGHT 2D (same as bbox)
      "bbox2D_trunc":  [x1, y1, x2, y2],               // TIGHT 2D (same as bbox)
      "bbox2D_proj":   [x1, y1, x2, y2],               // LOOSE: projected 3D-cuboid hull
      "bbox3D_cam":    [[x0,y0,z0], ..., [x7,y7,z7]],  // 8 corner points, camera frame
      "center_cam":    [x, y, z],
      "dimensions":    [W, H, L],                      // Omni3D ordering
      "R_cam":         [[r11,r12,r13], ...],           // 3x3 camera-frame rotation
      "truncation": 0.0, "visibility": 1.0,
      "segmentation_pts": <mask_pixel_count>,          // >0 when SAM3 mask available
      "lidar_pts": 10, "depth_error": 0.0,
      "area": <w*h>,
      "iscrowd": 0,
      "track_id": <int>,
      "bbox_source": "sam3"                            // or "vggt" as fallback
    }
  ]
}
```

**Key design decision.** `bbox` and `bbox2D_tight` use the **SAM3 silhouette bbox** (from the non-zero extent of the binary mask PNG). `bbox2D_proj` uses the **projection of the 3D cuboid corners**. The 2D detector trains on the tight mask; the 3D head trains on the projected cuboid — so both signals are consistent with what they're supposed to regress. If SAM3 masks are absent, both fields fall back to the VGGT-stored 2D bbox or the cuboid projection.

### 2.8 Running the prep pipeline

```bash
python tools/prepare_wildbox_dataset.py \
    --source <zip_or_dir>=<category_name>:<category_id>  # repeatable
    --split-mode video --val-fraction 0.2 --seed 0 \
    --output-train datasets/Omni3D/WildBox_train.json \
    --output-val   datasets/Omni3D/WildBox_val.json \
    --dataset-id 1000 -v
```

Zips are auto-extracted to sibling `<name>_unzipped/` directories on first use; cached on re-runs. Corrupt or missing zips are logged and skipped. Output logs show SAM3 coverage per split: `(NNN with SAM3 tight 2D bbox = 100.0%)`.

Also required before training:

```bash
# Register the 5 wildlife categories in Omni3D's stats.json
python tools/patch_stats_for_wildbox.py \
    --stats datasets/Omni3D/stats.json \
    --add rhino:1004 elephant:1002 zebra:1001 giraffe:1000 gazelle:1005

# Link the eval-time category metadata
ln -sf wildbox/category_meta_wildlife5.json configs/category_meta.json
```

### 2.9 Category metadata ([configs/wildbox/category_meta_wildlife5.json](configs/wildbox/category_meta_wildlife5.json))

```json
{"thing_classes": ["rhino", "elephant", "zebra", "giraffe", "gazelle"],
 "thing_dataset_id_to_contiguous_id":
    {"1004": 0, "1002": 1, "1001": 2, "1000": 3, "1005": 4}}
```

The contiguous-id assignment defines the mapping from the model's output head slots (0–49, with `ROI_HEADS.NUM_CLASSES = 50` preserved from pretraining for weight compatibility) to species names. Any tool downstream that needs to translate `category_id` → name must use this file.

---

## 3. Model

### 3.1 Architecture

- **Framework.** OVMono3D-lift, a variant of Cube R-CNN. Detectron2-based.
- **Backbone.** DINOv2 ViT-B/14, frozen partial.
- **Neck.** Simple Feature Pyramid (SFP) on ViT features, `SQUARE_PAD = 560`.
- **Proposal generator.** RPNWithIgnore (custom IoU-ness head).
- **ROI heads.** Standard Fast R-CNN 2D head + a 3D CubeHead:
  - Disentangled losses (XY, Z, dims, pose, joint)
  - 6D pose representation (not quaternion; Zhou et al. continuous-6D)
  - Allocentric pose parameterization
  - Virtual depth **disabled** — we're in VGGT synthetic scale, not metric depth.

### 3.2 Pretraining

Fine-tuning always initializes from the public **OVMono3D-lift checkpoint**:

```
checkpoints/ovmono3d_lift.pth   # from huggingface uva-cv-lab/ovmono3d_lift
```

This model was pretrained on Omni3D (50 indoor/driving categories: chair, car, pedestrian, etc. — none wildlife). Only the model weights are loaded; optimizer and scheduler state are reset at fine-tune start.

### 3.3 Loss configuration

```yaml
MODEL.ROI_CUBE_HEAD:
  VIRTUAL_DEPTH: False
  LOSS_W_Z: 0.5        # <-- half weight because Z is synthetic scale
  LOSS_W_DIMS: 1.0
  LOSS_W_POSE: 1.0
  LOSS_W_XY: 1.0
  LOSS_W_JOINT: 1.0
```

The key non-default is **`LOSS_W_Z = 0.5`**: because VGGT gives synthetic (relative) depth, the absolute Z magnitude is arbitrary; down-weighting prevents it from dominating gradients while still giving the model a depth signal.

### 3.4 Training hyperparameters ([configs/wildbox/OVMono3D_wildbox_wildlife5.yaml](configs/wildbox/OVMono3D_wildbox_wildlife5.yaml))

```yaml
SOLVER:
  TYPE: sgd
  IMS_PER_BATCH: 4
  BASE_LR: 0.001
  MAX_ITER: 20000
  STEPS: (12000, 18000)      # 60% and 90% LR drops
  WARMUP_ITERS: 500
  CHECKPOINT_PERIOD: 1000
  AMP:
    ENABLED: True            # mixed precision on A40
DATALOADER:
  NUM_WORKERS: 8
  SAMPLER_TRAIN: RepeatFactorTrainingSampler
  REPEAT_THRESHOLD: 0.25      # upweight long-tail classes
  REPEAT_SQRT: True
INPUT:
  MIN_SIZE_TRAIN: (280, 308, 336, 364, 392)   # around VGGT's 294 native res
  MIN_SIZE_TEST: 294
  MAX_SIZE_TRAIN: 560
  MAX_SIZE_TEST: 560
  RANDOM_FLIP: horizontal
TEST:
  EVAL_PERIOD: 2000
  EVAL_REL_AP3D: True
  REL_AP3D_SEARCH: (0.3, 3.0, 28)   # LabelAny3D-exact grid
  CAT_MODE: novel
  ORACLE2D: False                    # use RPN, not external detector
```

### 3.5 Computational requirements

- **GPU.** 1× NVIDIA A40 (48GB). Memory usage ≈ 18 GB at `IMS_PER_BATCH=4`.
- **Wall clock.** ~6 hours for 20 000 iterations on A40 with AMP and 8 workers (the bottleneck is the dataloader at 54% GPU util without AMP; AMP lifts util to ~80%).
- **Storage.** ~150 GB extracted for all 13 zips.

### 3.6 Training command

```bash
tmux new -s wb-train5
python tools/train_net.py \
    --config-file configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --num-gpus 1 \
    OUTPUT_DIR output/wildbox_wl5_finetune
```

Resumption from checkpoint: add `--resume`. Checkpoints written every 1000 iterations.

---

## 4. Evaluation protocol

### 4.1 Primary metrics (paper headline)

| Metric | Definition | IoU | Rationale |
|---|---|---|---|
| **AP-BEV** | AP on rotated-rectangle 2D IoU of cuboid footprints in the (X, Z) plane | 0.25 (primary), 0.5 (supplementary) | Drops the VGGT-unreliable depth axis. Standard in aerial 3D (UAV3D, AM3D, CDrone). |
| **AP-3D** | Full 3D-IoU AP via pytorch3d's box3d_overlap (CPU path) | 0.25 | Expected by monocular-3D community; loose threshold because of depth-GT noise floor. |
| **Rel-AP-3D** | AP-3D after a single global scalar rescale argmax_s of mean IoU | IoU 0.05:0.50 | LabelAny3D protocol (Sec. 5.1). Defuses the "depth GT is ML-derived" objection. Grid = `np.linspace(0.3, 3.0, 28)`. |
| **2D AP** | Standard COCO 2D AP | 0.5 | Detection sanity — decouples localization quality from 3D lifting. |

### 4.2 Per-species + aggregation

Each metric is reported as:

- **micro** (all predictions pooled into one PR curve, dominated by high-frequency classes)
- **macro** (per-class AP averaged with equal weights — the honest long-tail number)
- **per-class** (5 columns: rhino / elephant / zebra / giraffe / gazelle)

### 4.3 Supplementary metrics (appendix or diagnostic)

- **Disentangled NHD.** Normalized Hausdorff Distance decomposed into (xy, z, dims, pose). The `NHD-z` term makes the depth-dominance-of-error argument explicit, motivating BEV as the primary.
- **Class-agnostic 2D AP @ {0.25, 0.50, 0.75}.** Used specifically for the zero-shot baseline where the closed-vocab pretrained model has no in-vocab class labels. Measures "did the model localize something where an animal is?" independently of class name.
- **NHD scale search** (`tools/class_agnostic_eval.py --nhd`). pytorch3d-free Rel-AP-3D surrogate when box3d_overlap isn't available. Not a substitute for Rel-AP-3D in the paper table.

### 4.4 Zero-shot evaluation protocol

The pretrained OVMono3D-lift model has 50 output class slots mapped to Omni3D's indoor/driving categories. None correspond to wildlife species. A straight evaluation on WildBox-val therefore produces **zero in-vocab predictions** after the evaluator's class-name filter.

The experiment treats zero-shot as a **diagnostic baseline**:

- Standard AP-3D / AP-2D / AP-BEV all register as 0 (no predictions with matching class names). Reported honestly.
- The model **does** localize wildlife, just under wrong labels (rhinos get labeled as 'chair', etc.). This is exposed by class-agnostic 2D AP.
- The `tools/run_full_eval.sh --skip-rel-ap3d` flag skips the Rel-AP-3D scale search for zero-shot because there are no in-vocab predictions to rescale.

### 4.5 Evaluation runner

```bash
bash tools/run_full_eval.sh \
    --weights <checkpoint.pth> \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --out     <output_dir> \
    --label   "<short description>" \
    --gt      datasets/Omni3D/WildBox_val.json \
    [--skip-rel-ap3d]                    # for zero-shot
```

This runs, in sequence:

1. Standard AP-2D / AP-3D eval via OVMono3D's `tools/train_net.py --eval-only`.
2. Rel-AP-3D eval (CPU scale search, ~15 min).
3. BEV AP eval via `tools/bev_ap_eval.py` (uses shapely; no pytorch3d dep).
4. Class-agnostic + NHD diagnostic via `tools/class_agnostic_eval.py`.
5. Paper report assembly via `tools/make_report.py`.

Output at `<output_dir>/paper_report/`:

- `metrics.json` — machine-readable summary of all metrics.
- `report.md` — human-readable, includes the main metrics table and zero-shot diagnostic.
- `table_main.tex` — LaTeX booktabs-style main table ready to drop into a paper.

### 4.6 Cross-run comparison

```bash
python tools/make_report.py \
    --run-dir <zero_shot_dir>   --label "zero-shot" \
    --run-dir <finetuned_dir>   --label "fine-tuned" \
    --gt datasets/Omni3D/WildBox_val.json \
    --config configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --out output/paper_report_final --compare
```

---

## 5. Reproducing on another architecture (Cube R-CNN / DetAny3D / custom)

**Everything in §2 (dataset/preprocessing) and §4 (evaluation) is architecture-agnostic.** Only §3 needs to be adapted. Here's what to transfer directly vs re-implement.

### 5.1 Transfer directly

- **Dataset prep script** (`tools/prepare_wildbox_dataset.py`). Produces Omni3D-JSON, which is accepted by Cube R-CNN out of the box. DetAny3D (TikTok version) can be adapted — its data loader expects a similar `(image_path, intrinsics, 3D_box_in_camera)` tuple.
- **Category metadata** and `stats.json` patch.
- **Video-level splits** (`--split-mode video --seed 0`). Any architecture benefits from no-leakage splits.
- **SAM3 tight 2D bbox** convention. The `bbox` / `bbox2D_tight` fields are read by any Omni3D-compliant loader.
- **Scale normalization** (per-segment uniform scaling). Required regardless of architecture because it encodes a property of the *ground truth*, not the model.
- **Evaluation tooling** (`tools/bev_ap_eval.py`, `tools/class_agnostic_eval.py`, `tools/make_report.py`). These all consume the standard Omni3D `instances_predictions.pth` format; any Detectron2-based architecture produces this automatically.

### 5.2 Re-implement per architecture

- **Loss weight for Z.** The `LOSS_W_Z = 0.5` choice is specifically because the 3D head regresses absolute depth but the GT is synthetic-scale. Any architecture that regresses depth needs a similar down-weighting. Rule of thumb: set Z loss weight to half of other losses, or route Z through a log-space / virtual-focal parameterization if the architecture supports one.
- **Virtual-depth / focal-normalized depth.** OVMono3D supports `VIRTUAL_DEPTH` — set False here. Cube R-CNN has the same flag. Other architectures may not; check before reusing the config.
- **Pose parameterization.** OVMono3D uses continuous-6D (Zhou et al.). For architectures that regress yaw-only (e.g. KITTI-style), the top-down drone geometry will hurt them because most orientation variation is in pitch/roll.
- **Backbone-pretrained scale.** DINOv2 ViT-B/14 is frozen-ish in OVMono3D. A fully trainable backbone would need a lower LR (1e-4 rather than 1e-3) and longer schedule.
- **2D detector.** Any COCO-pretrained detector (RPN, DETR, GroundingDINO, etc.) will do for the 2D branch. Our 2D GT is SAM3-tight, so the detector should converge well.
- **Class-balanced sampling.** `RepeatFactorTrainingSampler` is Detectron2-specific. For PyTorch-native pipelines, use `WeightedRandomSampler` with per-sample weights proportional to `sqrt(1 / max(ann_count, threshold))` per class.

### 5.3 Adapting the config to a new architecture

Minimum values you need to match:

```
BACKBONE pretraining       = Omni3D or COCO (not ImageNet alone — wildlife 3D needs priors)
BATCH SIZE                 = 4 per A40 (scale up to 16 on A100-80GB with LR 4e-3)
ITERATIONS                 = ~20 000 on 5-species (~0.5-1 epoch of frames, enough given pretrained init)
LR SCHEDULE                = 1e-3 SGD, step at 60%/90%, warmup 500 iters
CLASS BALANCING            = repeat-factor with threshold 0.25
LOSS: down-weight depth    = 0.5x for any depth-regressing head, VGGT-scale only
OPTIMIZER                  = SGD momentum 0.9 (Adam/AdamW also works; halve LR if switching)
AUGMENTATION               = horizontal flip, random-scale short-edge (280..392)
INPUT RESOLUTION           = 294 short edge, 560 max edge (matches VGGT's 518x294 native)
```

### 5.4 Architecture-specific notes

**Cube R-CNN (original).** Drop-in compatible since OVMono3D is a Cube R-CNN variant. The only config change is `MODEL.BACKBONE.NAME: build_dla_from_vision_fpn_backbone` instead of DINOv2. Expect similar-or-worse numbers because Cube R-CNN's DLA backbone is less strong than DINOv2 ViT.

**DetAny3D.** Takes per-image intrinsics and produces 3D boxes with 2D-detector-conditioned queries. Adapt the loader to read our JSON's `K` and `bbox` fields; their depth head is already in their camera frame so no coordinate surgery needed.

**Your own architecture.**
1. Read the Omni3D JSON (standard format, there are Python helpers in Cube R-CNN source).
2. Train using the specified config. Expect loss_z to be higher than loss_xy by 2-3× — that's correct for synthetic-scale GT.
3. Output per-image `instances_predictions.pth` with fields `bbox`, `center_cam`, `dimensions`, `pose`, `category_id`, `score`.
4. Drop that into `tools/bev_ap_eval.py` / `tools/class_agnostic_eval.py` / `tools/make_report.py` for the same evaluation and comparison tooling.

---

## 6. Experiment design (for the paper)

### 6.1 Research question

*Can a monocular 3D detector pretrained on indoor/driving Omni3D categories be fine-tuned to 3D-detect wild animals from aerial drone footage, using pseudo-labels from a SAM3+VGGT pipeline?*

### 6.2 Experimental conditions compared

1. **Zero-shot**: pretrained `ovmono3d_lift.pth` evaluated directly on WildBox-val. Primary metrics register as 0 (wildlife classes are out-of-vocab); class-agnostic 2D AP establishes a meaningful baseline (the RPN does localize animals even without labels).

2. **Fine-tuned (partial data)**: pretrained → 20 000 iterations on the WildBox zips that were available at the time of training (~10 of 13). Used as a data-scaling control.

3. **Fine-tuned (full data)**: pretrained → 20 000 iterations on all 13 zips. Main result.

All three are evaluated on the **same** full-13-zip val set so comparisons are fair.

### 6.3 Why BEV as primary

1. **VGGT's Z is synthetic-scale.** Absolute depth in our GT is a product of an ML pipeline, not metric measurement. Evaluating tight 3D IoU conflates method quality with depth-GT noise.
2. **Aerial wildlife geometry is near-planar.** Animals on savanna ground lie approximately in one horizontal plane; vertical extent is small relative to 2D footprint.
3. **Community precedent.** UAV3D, AM3D, CDrone all report BEV as primary for aerial 3D.
4. **NHD-z dominates disentangled error** (xy:z ≈ 0.3:3.3 in our initial runs). This is shown one-line in §4.1 and motivates the BEV choice empirically.

### 6.4 Novelty and contribution

- **WildBox dataset.** 5 species, 13 collection campaigns, ~65k frames with silhouette-level 2D GT and cuboid 3D GT. First drone-aerial wildlife dataset at this scale with 3D annotations.
- **SAM3 + VGGT pseudo-labeling pipeline.** Automates high-quality 3D labels without manual annotation. The pipeline is the primary automation contribution.
- **Metric protocol for non-metric 3D GT.** BEV-primary evaluation, Rel-AP-3D scale-aligned secondary. Explicit handling of the scale-identifiability issue inherent to monocular 3D reconstruction pipelines.
- **Demonstration.** OVMono3D-lift fine-tunes to a high-quality 3D wildlife detector (macro-mAP-BEV@0.25 ≈ 80–90%, based on 3-species preliminary runs) with ~20k iterations on a single A40.

### 6.5 Known limitations

- **Pseudo-label noise.** GT is not human-verified. Errors in SAM3 masks propagate to 2D GT; errors in VGGT tracks propagate to 3D GT. We mitigate by:
  (a) reporting BEV as primary (less sensitive to depth error);
  (b) reporting Rel-AP-3D (removes scale bias).
- **Species imbalance.** Giraffe and gazelle are video-count long-tail (4 each). Per-species AP on these classes has high variance between splits. Mitigated by `REPEAT_THRESHOLD=0.25` and reported with 2-seed variance where space permits.
- **Zebra mixes two species** (Plains + Grévy's). Treated as a single category here.
- **No ground-plane extraction** for BEV. BEV uses camera-Y drop as a proxy for ground-plane projection. Fine for near-nadir drone shots; noisy for oblique shots. Noted in appendix.
- **Not end-to-end.** The labeling pipeline (SAM3 + VGGT) is upstream; this experiment evaluates only the 3D detector trained on its output.

### 6.6 Paper citations

When writing the paper, cite at minimum:

- **OVMono3D** — the base architecture.
- **Cube R-CNN** (Brazil et al., CVPR 2023) — the framework OVMono3D extends.
- **Omni3D** — the pretraining dataset, provides the 50-class checkpoint and the AP-3D definition.
- **LabelAny3D** — the Rel-AP-3D metric protocol and grid values.
- **KABR** — long-tail reporting convention (macro + micro + per-class).
- **UAV3D / AM3D / CDrone** — precedent for BEV-as-primary in aerial 3D.
- **SAM3** — mask source.
- **VGGT** — 3D reconstruction source.
- **DINOv2** — backbone pretraining.

---

## 7. Repository file reference

Files added or modified for this experiment:

| File | Purpose |
|---|---|
| [tools/prepare_wildbox_dataset.py](tools/prepare_wildbox_dataset.py) | VGGT + SAM3 → Omni3D JSON converter. Zip-aware, video/segment split. |
| [tools/patch_stats_for_wildbox.py](tools/patch_stats_for_wildbox.py) | Register wildlife categories in Omni3D's stats.json. |
| [tools/bev_ap_eval.py](tools/bev_ap_eval.py) | BEV AP @ IoU 0.25/0.50. Uses shapely. |
| [tools/class_agnostic_eval.py](tools/class_agnostic_eval.py) | Class-agnostic 2D AP + NHD-based scale-invariant 3D surrogate. |
| [tools/make_report.py](tools/make_report.py) | Paper-ready Markdown + LaTeX + JSON report assembler. |
| [tools/run_full_eval.sh](tools/run_full_eval.sh) | One-shot: standard eval + Rel-AP3D + BEV + class-agnostic + report. |
| [tools/visualize_class_agnostic.py](tools/visualize_class_agnostic.py) | Per-frame GT-vs-prediction image overlays for figures. |
| [tools/plot_training.py](tools/plot_training.py) | Parse `metrics.json` → 6-panel training-curves PNG. |
| [tools/remap_wildbox_paths.py](tools/remap_wildbox_paths.py) | Utility: rewrite absolute paths in an existing Omni3D JSON after data move. |
| [configs/wildbox/OVMono3D_wildbox_wildlife5.yaml](configs/wildbox/OVMono3D_wildbox_wildlife5.yaml) | 5-species training config (20k iters, AMP, 8 workers, REPEAT_THRESHOLD=0.25). |
| [configs/wildbox/OVMono3D_wildbox_finetune.yaml](configs/wildbox/OVMono3D_wildbox_finetune.yaml) | Base wildlife config (inherited by wildlife5). |
| [configs/wildbox/category_meta_wildlife5.json](configs/wildbox/category_meta_wildlife5.json) | 5-species contiguous-id mapping. |
| [cubercnn/evaluation/omni3d_evaluation.py](cubercnn/evaluation/omni3d_evaluation.py) | Modified: Rel-AP-3D scale search forced to CPU (pytorch3d CPU box3d_overlap works; CUDA build is blocked by cluster glibc). Zero-shot short-circuit when no in-vocab preds. |
| [cubercnn/data/builtin.py](cubercnn/data/builtin.py) | Modified: `WildBox_{train,val,test}` registration reads categories from generated JSON. |
| [cubercnn/modeling/roi_heads/__init__.py](cubercnn/modeling/roi_heads/__init__.py) | Modified: GroundingDINO import made optional (CUDA build optional). |
| [datasets/pending_sources.txt](datasets/pending_sources.txt) | Tracker for zip transfers not yet complete. |

---

## 8. Replication checklist

For someone repeating this experiment end-to-end on a fresh cluster:

```bash
# 0. Environment (once)
conda create -p /path/to/envs/ovmono3d python=3.8.20
conda activate /path/to/envs/ovmono3d
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
bash setup.sh                          # installs pytorch3d, detectron2, SAM, depth-pro, open3d
pip install shapely openpyxl           # for BEV eval and xlsx readers

# 1. Stats + category metadata
python tools/patch_stats_for_wildbox.py --stats datasets/Omni3D/stats.json \
    --add rhino:1004 elephant:1002 zebra:1001 giraffe:1000 gazelle:1005
ln -sf wildbox/category_meta_wildlife5.json configs/category_meta.json

# 2. Prep dataset (zips auto-extract on first use)
python tools/prepare_wildbox_dataset.py \
    --source <ZIP_OR_DIR>=<NAME>:<ID>  ... (13 of them) \
    --split-mode video --val-fraction 0.2 --seed 0 \
    --output-train datasets/Omni3D/WildBox_train.json \
    --output-val   datasets/Omni3D/WildBox_val.json \
    --dataset-id 1000 -v

# 3. No-leakage sanity check
python -c "
import json, os
train = {im['file_path'].split('/')[-3] for im in json.load(open('datasets/Omni3D/WildBox_train.json'))['images']}
val   = {im['file_path'].split('/')[-3] for im in json.load(open('datasets/Omni3D/WildBox_val.json'))['images']}
print('overlap:', len(train & val))   # MUST be 0
"

# 4. Zero-shot baseline
bash tools/run_full_eval.sh \
    --weights checkpoints/ovmono3d_lift.pth \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --out     output/wildbox_wl5_zeroshot \
    --label   "zero-shot" \
    --skip-rel-ap3d

# 5. Fine-tune (6 hours)
tmux new -s wb-train
python tools/train_net.py \
    --config-file configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --num-gpus 1 \
    OUTPUT_DIR output/wildbox_wl5_finetune

# 6. Fine-tuned eval (25 min including Rel-AP-3D)
bash tools/run_full_eval.sh \
    --weights output/wildbox_wl5_finetune/model_final.pth \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --out     output/wildbox_wl5_finetuned_eval \
    --label   "fine-tuned (5 species)"

# 7. Combined report
python tools/make_report.py \
    --run-dir output/wildbox_wl5_zeroshot        --label "zero-shot" \
    --run-dir output/wildbox_wl5_finetuned_eval  --label "fine-tuned" \
    --gt datasets/Omni3D/WildBox_val.json \
    --config configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --out output/paper_report_final --compare
```

---

## 9. Changelog of design decisions (for the paper's "we tried X, rejected Y" section)

| Decision | What we did | What we rejected | Why |
|---|---|---|---|
| **2D GT bbox source** | SAM3 silhouette mask | VGGT's projected-cuboid bbox | Cuboid projection is always loose; SAM3 is silhouette-tight → +20 AP-75 improvement observed. |
| **Split granularity** | Video-level | Segment-level | Segments within a video share background/animals → leakage. |
| **Primary metric** | AP-BEV @ 0.25 | AP-3D @ 0.5 | BEV drops VGGT-unreliable depth axis; looser threshold matches pseudo-label quality. |
| **Rel-AP-3D grid** | (0.3, 3.0, 28) | (0.1, 5.0, 20) | Match LabelAny3D paper for citable comparison. |
| **Scale handling** | Per-segment scale normalization + Rel-AP-3D | Absolute metric depth | VGGT gives synthetic scale; metric depth would be mis-scaled. |
| **Class balancing** | `REPEAT_THRESHOLD=0.25` | Uniform sampling | Giraffe/gazelle are long-tail in video count; rhino overfires without rebalance. |
| **3D IoU backend** | pytorch3d CPU path (forced) | pytorch3d CUDA | Cluster glibc incompatible with any available CUDA toolkit; CPU works. |
| **Visualization** | Class-agnostic top-K overlay | Per-class official viewer | Zero-shot has no in-vocab preds; class-agnostic reveals the RPN is firing correctly. |

---

## 10. Known issues / future work

- Ground-plane extraction for true BEV (rather than camera-Y drop) would improve accuracy on oblique-angle shots.
- SAM3 occasional mask failures (animal partially occluded) produce loose or split bboxes; would benefit from a consistency filter across frames of the same track.
- Extending to the full 6-species taxonomy (separating Plains vs Grévy's zebra) requires re-running SAM3 with species-specific prompts.
- Multi-individual tracking is not evaluated; all metrics here are per-frame detection. A tracking evaluation (HOTA, IDF1) would be a natural extension.
- The Rel-AP-3D scale-search is per-dataset-global (one scalar for all val images). A per-sequence or per-video scalar would more honestly represent the fact that each segment has its own synthetic scale; not done here to match LabelAny3D's published protocol.

---

_End of document. For the runnable playbook with paths filled in, see [datasets/pending_sources.txt](datasets/pending_sources.txt) (zip inventory) and the "Replication checklist" above._
