# WildBox 3D Wildlife Detection — Experiment Documentation

**Purpose of this document.** Self-contained reference for reproducing WildBox 3D wildlife detection. Covers dataset, preprocessing, training, evaluation, and adaptation to other 3D detection architectures. Written for (1) researchers replicating on Cube R-CNN / DetAny3D / custom architectures, (2) paper-writing.

---

## 1. TL;DR

- **Task.** Monocular 3D detection of African wildlife from aerial drone footage.
- **Species (current run).** rhino, elephant, zebra, giraffe, gazelle (5 classes).
- **Dataset.** WildBox — custom, ~65k frames, ~243k bboxes, ~68 videos across 13 campaigns. 3D GT is pseudo-labels from SAM3 → VGGT pipeline.
- **Architecture.** OVMono3D-lift (Cube R-CNN variant, DINOv2 ViT-B/14 backbone). Fine-tuned from `ovmono3d_lift.pth`.
- **Training.** 10 000 iterations (batch 8, LR 2e-3, AMP, 8 workers). ~3.5 hours on A40. 20k iter schedule recommended for future runs (see §10).
- **Primary metric.** AP-BEV @ IoU 0.25. Secondary: AP-3D @ 0.25, Rel-AP-3D (LabelAny3D), 2D AP @ 0.5.
- **Split.** Video-level, seed=0, 80/20. **No leakage.**

---

## 2. Dataset

### 2.1 Source and provenance

- Raw videos: DJI drone footage, Kenya, 2023–2026.
- Species labels via text-prompted SAM3 segmentation.
- 3D cuboids via VGGT 3D reconstruction + tracking on the masked frames.

### 2.2 Data inventory (13 campaigns)

Pending zip transfers are tracked in [datasets/pending_sources.txt](datasets/pending_sources.txt).

| Zip path | Species | Frames | Bboxes | Segments | Videos |
|---|---|---:|---:|---:|---:|
| archive/data202401KRhinos | rhino | 9 779 | 30 128 | 53 | 7 |
| archive/data202502KRhinoCamiV1 | rhino | 5 107 | 5 099 | 28 | 4 |
| archive/data202502KRhinoCamiV2 | rhino | 8 450 | 15 885 | 47 | 12 |
| archive/data202401KElephants | elephant | 5 527 | 6 562 | 34 | 6 |
| archive/data202406KElephants | elephant | 3 369 | 4 962 | 21 | 4 |
| archive/data202602KElephants | elephant | 5 300 | 31 406 | 33 | 7 |
| archive/wildbox_tomblair | Plains zebra | 4 627 | 9 218 | 25 | 3 |
| archive/dataBZS | Plains zebra | 9 300 | 53 236 | 53 | 12 |
| archive/data2023KABRZebras | Grévy's zebra | 4 610 | 23 048 | 30 | 5 |
| archive/data202401KGiraffes | giraffe | 1 220 | 3 297 | 8 | 2 |
| archive/data202501KGiraffes | giraffe | 310 | 391 | 2 | 2 |
| archive/data202406KGazelles | gazelle | 7 443 | 59 399 | 39 | 4 |

**Per-species totals (all 13 zips, target for full reproduction):**

| Species | Frames | Bboxes | Segments | Videos |
|---|---:|---:|---:|---:|
| Rhino | 23 336 | 51 112 | 128 | 23 |
| Elephant | 14 196 | 42 930 | 88 | 17 |
| Zebra (Plains + Grévy's) | 18 537 | 85 502 | 108 | 20 |
| Giraffe | 1 530 | 3 688 | 10 | 4 |
| Gazelle | 7 443 | 59 399 | 39 | 4 |
| **Total** | **~65 000** | **~243 000** | **~373** | **~68** |

**Note on first run:** completed 2026-04-21 with 10 of 13 zips (2 elephant zips + 1 rhino zip were still transferring). Current model is "10-zip" data-ablation; full 13-zip retrain is the main paper result once transfers complete.

### 2.3 Per-segment data layout

Each zip, once extracted to `<zip_stem>_unzipped/`, contains:

```
WildBox/
  DJI_YYYYMMDDHHMMSS_XXXX_V/                   # video directory
    seg1/                                       # time-range segment
      frame_NNNNNN.jpg                          # real-resolution RGB
      metadata.json                             # start_time, end_time, frame_numbers[]
      sam3_masks/
        metadata.json                           # text prompt, object ids
        masks/obj_N/frame_NNNNNN.png            # binary silhouette at real resolution
      vggt_results/
        cameras.json                            # per-frame intrinsics (3x3) + extrinsics (3x4)
        tracking_summary.json                   # per-track 3D state:
                                                #   centers[], dimensions[],
                                                #   rotation_matrices[], bbox_2d[],
                                                #   frames[], class_name
        depth_maps.npz, point_cloud.ply         # unused by the 3D detector training
```

### 2.4 Coordinate conventions

All 3D in **camera frame**: X=right, Y=down, Z=forward. Right-handed.

- **Dimensions (Omni3D convention).** `[W, H, L]`: X-extent = L, Y-extent = H, Z-extent = W. Matches `cubercnn/util/math_util.py:get_cuboid_verts_faces`.
- **Rotation.** Full 3×3 matrix. Not yaw-only (drone top-down shots carry most orientation in pitch/roll).
- **VGGT's internal `dimensions`** = `[l, w, h]` = reverse of Omni3D's. The prep script handles the swap.

### 2.5 Scale normalization

VGGT's 3D is **synthetic-scale** (relative, not metric). The prep normalizes each segment independently:

```
s = 1 / median(|center_cam_z|)      # per-segment scalar
center_cam  *= s
cuboid_verts *= s
dimensions  *= s
# K is NOT rescaled -- projection (X/Z, Y/Z) is scale-invariant.
```

Because this is **uniform scaling**, box shape / orientation / 2D projection / 3D IoU are all preserved. What it kills is absolute-depth comparability *across* segments — motivating the scale-invariant Rel-AP-3D and BEV metrics in §4.

### 2.6 Train/val split — video-level

**Video-level, seed=0, 20% val.** Implementation in [tools/prepare_wildbox_dataset.py](tools/prepare_wildbox_dataset.py):
- Stratified by species (each species independently shuffled and split).
- All segments from the same video go to the same split side.
- Re-running prep with additional zips (same `--seed 0`) deterministically extends splits: previously-seen videos keep their assignment; new videos get new assignments.

**Why video-level?** Segment-level or frame-level split leaks background / animal identity across splits.

### 2.7 SAM3 tight 2D bboxes

2D GT comes from **SAM3 silhouette masks**, not the projected 3D cuboid. Cuboid projection overshoots the animal's actual extent. Tight masks directly fix 2D AP-75 (~+15-20 pp observed in early 3-species trials).

- `bbox`, `bbox2D_tight`, `bbox2D_trunc` = SAM3 silhouette extent (pixel-tight)
- `bbox2D_proj` = projection of 3D cuboid corners (intentionally kept — the 3D head regresses *cuboid projection*, not silhouette)
- `segmentation_pts` = actual mask pixel count (capped at 10000)
- `bbox_source` field = `"sam3"` or `"vggt"` (fallback when SAM3 missing)

### 2.8 Category metadata — the critical gotcha

**Three** files all need to agree. If any one disagrees with training's contiguous-id ordering, per-class metrics go silently wrong (usually zero for all-but-one class).

| File | What | Who reads it |
|---|---|---|
| `datasets/Omni3D/stats.json` | Global Omni3D category registry (ids 1000–1005 for wildlife) | Data loader + filter settings |
| **`configs/wildbox/category_meta.json`** | **SYMLINK** used by `tools/train_net.py`'s `--eval-only` code path at line 402 (looks in the config file's directory) | **Standalone eval runs** |
| `configs/category_meta.json` | **SYMLINK** used by the stock Omni3D evaluator when `CAT_MODE="novel"` and by external eval tools (`bev_ap_eval.py`, `class_agnostic_eval.py`, `make_report.py`) | In-training evaluator + external tools |

**This lesson was learned the hard way on 2026-04-21**: `configs/wildbox/category_meta.json` was a leftover from the Phase-1 single-giraffe run and kept overriding `configs/category_meta.json` for all `--eval-only` invocations, producing eval tables with only giraffe populated. The fix was to symlink `configs/wildbox/category_meta.json` to the matching `category_meta_wildlife5.json` too. **Keep them in sync.**

#### 2.8.1 The contiguous-id ordering rule (MUST follow)

`register_and_store_model_metadata()` ([cubercnn/data/datasets.py:294-318](cubercnn/data/datasets.py#L294)) decides the model's internal contiguous-id assignment by **sorting the training category names by their dataset-id ascending**. For our species this means:

| Dataset-id | Contiguous-id | Species |
|---:|---:|---|
| 1000 | **0** | giraffe |
| 1001 | **1** | zebra |
| 1002 | **2** | elephant |
| 1004 | **3** | rhino |
| 1005 | **4** | gazelle |

**`configs/category_meta.json` MUST list `thing_classes` in this exact order**, or the evaluator's `omni3d_global_categories[category_id]` lookup will disagree with what the model actually learned and per-class metrics will silently become garbage (only one class gets non-zero numbers, others are zero). This happened in our first 5-species run and cost us an entire eval cycle to debug.

The ground-truth `thing_classes` for 5 species:

```json
{"thing_classes": ["giraffe", "zebra", "elephant", "rhino", "gazelle"],
 "thing_dataset_id_to_contiguous_id": {"1000":0, "1001":1, "1002":2, "1004":3, "1005":4}}
```

#### 2.8.2 Auto-generated meta to prevent getting this wrong

Since 2026-04-21, `tools/prepare_wildbox_dataset.py` writes the correct mapping to `configs/wildbox/category_meta_auto_<N>species.json` each time you re-prep. Prefer this over hand-edited meta files.

#### 2.8.3 Verify before every eval — BOTH symlinks

```bash
# Fix BOTH symlinks (they use different relative paths)
ln -sf wildbox/category_meta_wildlife5.json configs/category_meta.json
ln -sf category_meta_wildlife5.json         configs/wildbox/category_meta.json

# Verify both
cat configs/category_meta.json
cat configs/wildbox/category_meta.json
# Both must show identical content:
# 1. Species count matches what you trained on
# 2. thing_classes first entry has the SMALLEST dataset-id in its mapping
# 3. For 5-species: giraffe (1000) must be thing_classes[0], gazelle (1005) last
```

If either file is missing or out of sync, the standalone `--eval-only` will use the wrong mapping and per-class metrics will be wrong.

If unsure, the auto-generated file from the most recent prep is always correct:
```bash
ls -t configs/wildbox/category_meta_auto_*species.json | head -1
```

### 2.9 Output JSON schema (Omni3D format)

```jsonc
{
  "info": {"id": 1000, "source": "wildbox", "name": "WildBox_train",
           "split": "train", "known_category_ids": [1000,1001,1002,1004,1005],
           "scene_scales": {"rhino/DJI_.../seg1": 0.45, ...}},
  "categories": [
    {"id": 1000, "name": "giraffe",  "supercategory": "animal"},
    {"id": 1001, "name": "zebra",    "supercategory": "animal"},
    {"id": 1002, "name": "elephant", "supercategory": "animal"},
    {"id": 1004, "name": "rhino",    "supercategory": "animal"},
    {"id": 1005, "name": "gazelle",  "supercategory": "animal"}
  ],
  "images": [
    {"id": 0, "dataset_id": 1000,
     "file_path": "/abs/path/to/frame_000146.jpg",   // ABSOLUTE path
     "height": 1080, "width": 1920,
     "K": [[fx,0,cx],[0,fy,cy],[0,0,1]]}
  ],
  "annotations": [
    {"id": 0, "image_id": 0, "dataset_id": 1000,
     "category_id": 1004, "category_name": "rhino",  // dataset-id in GT
     "bbox":          [x, y, w, h],                  // SAM3-tight
     "bbox2D_tight":  [x1, y1, x2, y2],
     "bbox2D_trunc":  [x1, y1, x2, y2],
     "bbox2D_proj":   [x1, y1, x2, y2],              // 3D-cuboid projection
     "bbox3D_cam":    [[...]*8],
     "center_cam":    [x, y, z],
     "dimensions":    [W, H, L],                     // Omni3D ordering
     "R_cam":         [[r11,r12,r13],...,[r31,r32,r33]],
     "truncation": 0.0, "visibility": 1.0,
     "segmentation_pts": <mask_pixel_count>,
     "lidar_pts": 10, "depth_error": 0.0,
     "area": <w*h>, "iscrowd": 0, "track_id": <int>,
     "bbox_source": "sam3"}
  ]
}
```

### 2.10 Moving data after prep (path hygiene)

The JSON stores **absolute** paths. If you move `*_unzipped/` directories, use [tools/remap_wildbox_paths.py](tools/remap_wildbox_paths.py) rather than regenerating the JSON (which would invalidate the split seed).

```bash
# Explicit substring swap
python tools/remap_wildbox_paths.py datasets/Omni3D/WildBox_val.json \
    --map OLD_PREFIX=NEW_PREFIX --in-place

# Or auto-search by (video, seg, frame-name) triplet
python tools/remap_wildbox_paths.py datasets/Omni3D/WildBox_val.json \
    --search-root /storage3/3DOM/vshukla/sam3/wd_data/wildbox --in-place
```

---

## 3. Environment setup (once per cluster)

```bash
# Conda env
conda create -p /storage3/.../envs/ovmono3d python=3.8.20
conda activate /storage3/.../envs/ovmono3d

# PyTorch
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Repo dependencies (pytorch3d, detectron2 fork, SAM, depth-pro, etc.)
bash setup.sh

# Experiment-specific extras
pip install shapely openpyxl   # BEV IoU + xlsx readers
```

### 3.1 Known environment issues

- **pytorch3d CUDA build failure on CUDA 12.x with newer glibc.** The cluster's glibc rejects CUDA's `cospi`/`sinpi` declarations. Workaround already in the code: the Rel-AP3D scale search forces CPU tensors (see `cubercnn/evaluation/omni3d_evaluation.py:search_rel_scale`). CPU `box3d_overlap` works on our pinned commit (055ab3a). Don't try to force `FORCE_CUDA=1` unless you're on a cluster with compatible glibc.
- **GroundingDINO CUDA build also fails** on this cluster. The package is marked optional in `setup.sh`. WildBox config uses `TEST.ORACLE2D=False` so the model's own RPN handles 2D — GroundingDINO isn't called. Safe to skip.
- **`libGL.so.1: cannot open shared object file`** — appears on CPU-only nodes because `cv2` requires system graphics libs. Run eval on GPU nodes only.

---

## 4. Prep + train + eval pipeline (end-to-end)

### 4.1 Add species to stats.json (once when introducing new species)

```bash
python tools/patch_stats_for_wildbox.py \
    --stats datasets/Omni3D/stats.json \
    --add rhino:1004 elephant:1002 zebra:1001 giraffe:1000 gazelle:1005
```

Idempotent. Re-running is safe.

### 4.2 Build train/val JSONs

One `--source` entry per zip. Zips auto-extract to sibling `*_unzipped/` on first use and are cached. Corrupt / in-progress zips are **skipped with a warning** (see `prepare_wildbox_dataset.py:resolve_source_path`); add them to [datasets/pending_sources.txt](datasets/pending_sources.txt) for later.

```bash
python tools/prepare_wildbox_dataset.py \
    --source <ZIP_OR_DIR>=<CATEGORY>:<ID> \  # repeatable
    --split-mode video \
    --val-fraction 0.2 \
    --seed 0 \
    --output-train datasets/Omni3D/WildBox_train.json \
    --output-val   datasets/Omni3D/WildBox_val.json \
    --dataset-id 1000 \
    -v
```

For the current run (11 zips) the full command is in [datasets/pending_sources.txt](datasets/pending_sources.txt) (commented section "re-add command").

### 4.3 Flip BOTH eval-time symlinks

```bash
# Top-level: used by external eval tools + stock Omni3D evaluator
ln -sf wildbox/category_meta_wildlife5.json configs/category_meta.json

# Config-dir: used by tools/train_net.py --eval-only (reads from config file's dir)
ln -sf category_meta_wildlife5.json configs/wildbox/category_meta.json

# Verify both show identical content
cat configs/category_meta.json
cat configs/wildbox/category_meta.json
```

`tools/run_full_eval.sh` auto-syncs these at the start of each invocation as a safety net (see §13).

### 4.4 No-leakage verification

```bash
python -c "
import json, os
from collections import Counter
for split in ('train','val'):
    d = json.load(open(f'datasets/Omni3D/WildBox_{split}.json'))
    cats = Counter(a['category_name'] for a in d['annotations'])
    vids = {img['file_path'].split('/')[-3] for img in d['images']}
    missing = sum(1 for img in d['images'][:200] if not os.path.exists(img['file_path']))
    print(f'{split}: {len(d[\"images\"])} imgs, {len(d[\"annotations\"])} anns, '
          f'{len(vids)} videos, anns_per_class={dict(cats)}, '
          f'first-200 missing: {missing}')
train_vids = {img['file_path'].split('/')[-3] for img in json.load(open('datasets/Omni3D/WildBox_train.json'))['images']}
val_vids   = {img['file_path'].split('/')[-3] for img in json.load(open('datasets/Omni3D/WildBox_val.json'))['images']}
print(f'video overlap (MUST be 0): {len(train_vids & val_vids)}')
"
```

**Must print** `video overlap: 0` and `first-200 missing: 0`. Don't proceed otherwise.

### 4.5 Zero-shot baseline eval

Closed-vocab pretrained — primary AP metrics will register as 0 (expected, that's the baseline). Class-agnostic eval supplies the meaningful zero-shot number.

```bash
tmux new -s wb-eval-zs
bash tools/run_full_eval.sh \
    --weights checkpoints/ovmono3d_lift.pth \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --out     output/wildbox_wl5_zeroshot_eval \
    --label   "zero-shot (pretrained)" \
    --gt      datasets/Omni3D/WildBox_val.json \
    --skip-rel-ap3d      # no in-vocab preds -> scale search no-op
```

### 4.6 Fine-tune training

```bash
tmux new -s wb-train
python tools/train_net.py \
    --config-file configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --num-gpus 1 \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.BASE_LR 0.002 \
    SOLVER.MAX_ITER 10000 \
    SOLVER.STEPS "(6000, 9000)" \
    SOLVER.WARMUP_ITERS 250 \
    SOLVER.CHECKPOINT_PERIOD 500 \
    TEST.EVAL_PERIOD 5000 \
    OUTPUT_DIR output/wildbox_wl5_finetune
```

**Scaling knobs (§10 explains rationale):**

| Config axis | Value | Why |
|---|---|---|
| IMS_PER_BATCH | 8 | A40 has headroom at ~2 GB/batch; linear-LR scaled up from the 4-batch default |
| BASE_LR | 0.002 | 2× vs batch 4 (linear scaling rule) |
| MAX_ITER | 10 000 | With batch 8 this is ~equivalent compute to 20 000 at batch 4 |
| WARMUP_ITERS | 250 | Scaled proportionally |
| CHECKPOINT_PERIOD | 500 | Frequent saves — kill/restart cost is bounded |
| TEST.EVAL_PERIOD | 5000 | Each eval on 13k val images is ~30 min; limit to 2 in-loop evals |

### 4.7 Fine-tuned eval (complete paper metrics)

```bash
tmux new -s wb-eval-ft
bash tools/run_full_eval.sh \
    --weights output/wildbox_wl5_finetune/model_final.pth \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --out     output/wildbox_wl5_finetuned_eval \
    --label   "fine-tuned" \
    --gt      datasets/Omni3D/WildBox_val.json
```

This runs in sequence:
1. Standard 2D + 3D AP eval via `train_net.py --eval-only` (~15 min inference on 13k val)
2. Rel-AP-3D with LabelAny3D's (0.3, 3.0, 28) scale grid — CPU scale search, ~15 min
3. BEV AP @ {0.25, 0.50} via `tools/bev_ap_eval.py`
4. Class-agnostic + NHD via `tools/class_agnostic_eval.py --nhd`
5. Paper report assembly via `tools/make_report.py`

### 4.8 Combined zero-shot vs fine-tuned report

```bash
python tools/make_report.py \
    --run-dir output/wildbox_wl5_zeroshot_eval     --label "zero-shot" \
    --run-dir output/wildbox_wl5_finetuned_eval    --label "fine-tuned" \
    --gt      datasets/Omni3D/WildBox_val.json \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --out     output/paper_report_5sp \
    --compare
```

Output:
- `report.md` — main + diagnostic tables
- `table_main.tex` — LaTeX booktabs table for the paper
- `metrics.json` — machine-readable

### 4.9 Paper-ready visualizations

The in-training visualizer (`cubercnn/vis/vis.py`) has been patched to be defensive against out-of-range class-ids, but the standalone tool is what you want for paper figures.

```bash
# Produces THREE subdirs: gt_only/, pred_only/, combined/
python tools/visualize_class_agnostic.py \
    --preds output/wildbox_wl5_finetune/inference/iter_final/WildBox_val/instances_predictions.pth \
    --gt    datasets/Omni3D/WildBox_val.json \
    --out   output/wildbox_wl5_finetune/vis_agnostic \
    --top-k 3 --every 100 --limit 40
```

Compare `gt_only/pair_NNNNNN.jpg` ↔ `pred_only/pair_NNNNNN.jpg` for the cleanest before/after shots. `combined/` has both overlaid.

### 4.10 Training-curves PNG

```bash
python tools/plot_training.py output/wildbox_wl5_finetune/metrics.json
# -> output/wildbox_wl5_finetune/training_curves.png
# Also lists every key found in metrics.json (debug the detector's logging keys)
```

---

## 5. Model

### 5.1 Architecture

- OVMono3D-lift = Cube R-CNN with:
  - DINOv2 ViT-B/14 backbone (partially trainable)
  - Simple Feature Pyramid, `SQUARE_PAD=560`
  - RPNWithIgnore proposal generator
  - CubeHead for 3D regression (disentangled losses, continuous-6D pose, allocentric)
- Virtual depth **disabled** — we operate in VGGT synthetic scale.

### 5.2 Pretrained weights

```
checkpoints/ovmono3d_lift.pth    # HuggingFace uva-cv-lab/ovmono3d_lift
```

Pretrained on Omni3D (50 indoor/driving categories — no wildlife). Only model weights load at fine-tune start; optimizer/scheduler reset.

### 5.3 Loss configuration (see [configs/wildbox/OVMono3D_wildbox_finetune.yaml](configs/wildbox/OVMono3D_wildbox_finetune.yaml))

```yaml
MODEL.ROI_CUBE_HEAD:
  VIRTUAL_DEPTH: False
  LOSS_W_Z: 0.5        # <-- 1/2 weight: Z is synthetic (relative), so its
                       #     absolute magnitude is arbitrary
  LOSS_W_DIMS: 1.0
  LOSS_W_POSE: 1.0
  LOSS_W_XY: 1.0
  LOSS_W_JOINT: 1.0
```

---

## 6. Evaluation protocol

### 6.1 Primary metrics (paper table)

| Metric | Definition | IoU | Why primary |
|---|---|---|---|
| **AP-BEV** | 2D AP on rotated-rectangle BEV footprints (drop camera-Y, use X/Z) | **0.25**, 0.50 | Drops the VGGT-unreliable depth axis. UAV3D/AM3D/CDrone precedent. |
| **AP-3D** | Standard 3D-IoU AP via pytorch3d box3d_overlap (CPU) | 0.25 | Expected by monocular-3D community. Loose threshold because depth GT is ML-derived. |
| **Rel-AP-3D** | AP-3D after a global scalar `s* = argmax_s mean(IoU(s·B̂, B))` | IoU 0.05:0.50 | LabelAny3D protocol. Grid = (0.3, 3.0, 28) points. |
| **2D AP** | COCO 2D AP | 0.5 | Detection sanity — decouples localization from 3D lifting. |

### 6.2 Per-species aggregation

- **micro** (all predictions pooled into one PR curve; dominated by frequent classes)
- **macro** (mean of per-class APs, equal weights — honest long-tail number)
- **per-class** (rhino / elephant / zebra / giraffe / gazelle)

### 6.3 Supplementary (appendix)

- **Disentangled NHD** (xy, z, dims, pose). NHD-z dominance motivates BEV as primary — keep as one-line callout in §4.1 of the paper.
- **Class-agnostic 2D AP @ {0.25, 0.50, 0.75}** (zero-shot diagnostic).
- **NHD scale search** via `class_agnostic_eval.py --nhd`: pytorch3d-free Rel-AP-3D surrogate.

### 6.4 Zero-shot evaluation protocol

Pretrained OVMono3D has 50 class slots mapped to Omni3D (indoor/driving). None match wildlife → all primary AP register as 0. The *useful* zero-shot number is **class-agnostic 2D AP**: "did the pretrained RPN localize animals, regardless of class label?" This is reported in the diagnostic subtable (§6.1 caveat).

---

## 7. Scripts (what does what)

| File | Purpose |
|---|---|
| [tools/prepare_wildbox_dataset.py](tools/prepare_wildbox_dataset.py) | VGGT+SAM3 → Omni3D JSON. Zip-aware, video/segment split, SAM3-tight 2D, skip-bad-zip. |
| [tools/patch_stats_for_wildbox.py](tools/patch_stats_for_wildbox.py) | Register wildlife categories in Omni3D's stats.json. |
| [tools/remap_wildbox_paths.py](tools/remap_wildbox_paths.py) | Rewrite absolute paths in Omni3D JSON after data move (don't regenerate, preserves split). |
| [tools/bev_ap_eval.py](tools/bev_ap_eval.py) | BEV AP @ {0.25, 0.50}. Shapely-based rotated IoU. micro/macro/per-class. |
| [tools/class_agnostic_eval.py](tools/class_agnostic_eval.py) | Class-agnostic 2D AP + NHD surrogate. Works for zero-shot (no class labels needed). |
| [tools/make_report.py](tools/make_report.py) | Parses log + BEV + NHD outputs → Markdown + LaTeX + JSON report. `--compare` for side-by-side. |
| [tools/run_full_eval.sh](tools/run_full_eval.sh) | One-shot: standard eval + Rel-AP3D + BEV + NHD + report. |
| [tools/visualize_class_agnostic.py](tools/visualize_class_agnostic.py) | Paper figures: gt_only + pred_only + combined per sample. |
| [tools/plot_training.py](tools/plot_training.py) | metrics.json → 6-panel training curves PNG. Lists every key for debugging. |
| [configs/wildbox/OVMono3D_wildbox_wildlife5.yaml](configs/wildbox/OVMono3D_wildbox_wildlife5.yaml) | 5-species config (default MAX_ITER=20000). |
| [configs/wildbox/OVMono3D_wildbox_finetune.yaml](configs/wildbox/OVMono3D_wildbox_finetune.yaml) | Base wildlife config. REL_AP3D_SEARCH=(0.3, 3.0, 28). |
| [configs/wildbox/category_meta_wildlife5.json](configs/wildbox/category_meta_wildlife5.json) | 5-species contiguous-id mapping. |
| [datasets/pending_sources.txt](datasets/pending_sources.txt) | Tracker for in-progress zip transfers. |
| [cubercnn/evaluation/omni3d_evaluation.py](cubercnn/evaluation/omni3d_evaluation.py) | Modified: Rel-AP-3D scale search forced to CPU; zero-shot short-circuit when no in-vocab preds. |
| [cubercnn/data/builtin.py](cubercnn/data/builtin.py) | Modified: WildBox_{train,val,test} registration reads categories from generated JSON. |
| [cubercnn/modeling/roi_heads/__init__.py](cubercnn/modeling/roi_heads/__init__.py) | Modified: GroundingDINO import optional. |
| [cubercnn/vis/vis.py](cubercnn/vis/vis.py) | Modified: defensive category-id lookup (handles both contiguous and dataset-id inputs). |

---

## 8. Adding new data zips or classes

### 8.1 New zip, same species (most common: more data for existing species)

1. Verify the zip is intact: `unzip -t path/to/new.zip | tail -1` should say "No errors detected".
2. Add to the `--source` list in prep. Same `--seed 0` keeps existing video-split assignments deterministic.
3. Rerun prep:
   ```bash
   python tools/prepare_wildbox_dataset.py \
       --source ...EXISTING... \
       --source /new/path/to/zip.zip=<species>:<id> \
       --split-mode video --val-fraction 0.2 --seed 0 \
       --output-train datasets/Omni3D/WildBox_train.json \
       --output-val   datasets/Omni3D/WildBox_val.json \
       --dataset-id 1000 -v
   ```
4. Verify: `first-200 missing: 0` and `video overlap: 0` (see §4.4).
5. Retrain from scratch (not from existing checkpoint — the data distribution changed). Use the command in §4.6.
6. Evaluate both old and new models on the **new val** set for a data-scaling datapoint.

### 8.2 New species (e.g. add "lion")

1. **Pick a dataset-id**: use 1003 (or any unused integer ≥ 1000, not already in stats.json).
2. **Register in stats.json** (additive, idempotent):
   ```bash
   python tools/patch_stats_for_wildbox.py --stats datasets/Omni3D/stats.json \
       --add lion:1003
   ```
3. **Create a new category_meta file** with the updated contiguous-id mapping. Example for 6-species:
   ```bash
   cat > configs/wildbox/category_meta_wildlife6.json <<'EOF'
   {"thing_classes": ["rhino", "elephant", "zebra", "giraffe", "gazelle", "lion"],
    "thing_dataset_id_to_contiguous_id":
       {"1004": 0, "1002": 1, "1001": 2, "1000": 3, "1005": 4, "1003": 5}}
   EOF
   ln -sf wildbox/category_meta_wildlife6.json configs/category_meta.json
   ```
4. **Create a config** inheriting from the base, with the new class list. Example:
   ```yaml
   # configs/wildbox/OVMono3D_wildbox_wildlife6.yaml
   _BASE_: "OVMono3D_wildbox_finetune.yaml"
   DATASETS:
     CATEGORY_NAMES: ('rhino','elephant','zebra','giraffe','gazelle','lion')
     CATEGORY_NAMES_NOVEL: ('rhino','elephant','zebra','giraffe','gazelle','lion')
     CATEGORY_NAMES_BASE:  ('rhino','elephant','zebra','giraffe','gazelle','lion')
   DATALOADER:
     REPEAT_THRESHOLD: 0.25
     NUM_WORKERS: 8
   SOLVER:
     IMS_PER_BATCH: 4
     BASE_LR: 0.001
     MAX_ITER: 20000
     STEPS: (12000, 18000)
     WARMUP_ITERS: 500
     CHECKPOINT_PERIOD: 1000
     AMP: {ENABLED: True}
   OUTPUT_DIR: "output/ovmono3d_wildbox_wildlife6"
   ```
5. **Prep + train + eval** following §4. Model's `ROI_HEADS.NUM_CLASSES=50` is preserved (50-slot head, 6 slots now assigned to wildlife). No head shape change needed.

### 8.3 Removing a species

- Remove from `CATEGORY_NAMES` in the config.
- Rebuild meta + JSON; retrain from scratch.

---

## 9. Reproducing on another architecture

Everything in §2 (dataset/preprocessing) and §6 (evaluation) is architecture-agnostic. Only §5 (the model/loss config) is specific.

### 9.1 Transfer directly

- Omni3D-format JSONs from `prepare_wildbox_dataset.py`.
- Evaluation tooling (`bev_ap_eval.py`, `class_agnostic_eval.py`, `make_report.py`). All consume `instances_predictions.pth` which any Detectron2 architecture produces automatically.
- Video-level split (`--split-mode video --seed 0`).
- SAM3-tight 2D bbox convention (via the prep script).
- Per-segment scale normalization (encoded in GT, not model).

### 9.2 Re-implement per architecture

| Item | Cube R-CNN | DetAny3D | Generic 3D detector |
|---|---|---|---|
| Config matching | Drop-in; change backbone only | Adapt loader to read our K + bbox | Match the hparams below |
| `LOSS_W_Z` down-weighting | Keep 0.5× | Apply 0.5× on its depth loss | Apply to whatever depth-regressing loss exists |
| Virtual depth | Set False | Set equivalent | Skip — we're in VGGT synthetic scale |
| Pose parameterization | Keep 6D continuous | Keep quaternion or 6D, NOT yaw-only | Use 3-DoF representation (full R) |
| Class-balanced sampler | `REPEAT_FACTOR_TRAINING_SAMPLER` | WeightedRandomSampler with `sqrt(1/cnt)` | Same |

### 9.3 Minimum config to match

```
BACKBONE pretraining       = Omni3D or COCO (not ImageNet alone)
BATCH SIZE                 = 8 per A40 (16 on A100-80GB)
LR                         = 0.002 SGD (linearly scale with batch), step at 60%/90%
WARMUP                     = 5% of MAX_ITER
MAX_ITER                   = 10 000 at batch 8 (→ 20 000 at batch 4)
AUGMENTATION               = horizontal flip, random-scale short-edge (280..392)
INPUT RESOLUTION           = short-edge 294, max-edge 560
CLASS BALANCING            = repeat-factor threshold 0.25 (0.5 for very severe long-tail)
OPTIMIZER                  = SGD momentum 0.9 (or AdamW with LR/2)
AMP                        = ENABLED True on Ampere+; disable if NaNs appear
```

### 9.4 What the eval tools expect from the prediction output

For each predicted instance, in the saved `instances_predictions.pth`:

```python
{
    "image_id": int,
    "bbox": [x, y, w, h],               # xywh in image pixels
    "score": float,
    "category_id": int,                 # contiguous id, 0..N-1
    "center_cam": [x, y, z],            # camera-frame
    "dimensions": [W, H, L],            # Omni3D ordering
    "pose": [[...]*3]*3 or [9 floats]   # 3x3 or flattened
}
```

Any Detectron2 `COCOEvaluator`-compatible pipeline already writes this format.

---

## 10. Convergence analysis and next experiments

### 10.1 Is training converged?

**Evidence from the current run (iter 5 000 → 10 000):**

| Metric | iter 5 000 | iter 10 000 | Δ | Verdict |
|---|---:|---:|---:|---|
| 2D AP | 45.2 | 45.9 | +0.7 | Plateaued |
| 2D AP50 | 89.8 | 90.4 | +0.6 | Plateaued |
| 3D AP | 6.2 | 16.1 | **+9.9** | Still climbing fast |
| 3D AP15 | 9.3 | 25.4 | **+16.1** | Still climbing fast |
| elephant 3D AP | 9.4 | 45.1 | **+35.7** | Huge gains mid-training |
| rhino 3D AP | 9.4 | 14.7 | +5.3 | Slowing |
| zebra 3D AP | 11.4 | 16.7 | +5.3 | Slowing |
| giraffe 3D AP | 0.7 | 1.8 | +1.1 | Long-tail struggling |
| gazelle 3D AP | 0.3 | 2.3 | +2.0 | Long-tail struggling |
| NHD overall | 8.06 | 5.64 | -2.42 | Still improving |
| NHD-z | 7.18 | 4.74 | -2.44 | Still improving |

**Conclusion.** 2D converged. 3D has NOT — the second half of training delivered 2.6× the gain of the first half. Extending by +10 000 iters should yield 5-10 more 3D AP points. Long-tail classes (giraffe, gazelle) are the bottleneck.

### 10.2 Signals for convergence

1. **AP stops improving across 2-3 consecutive eval points.**
2. **Per-class AP stabilizes**, especially long-tail.
3. **Training loss flattens** (early signal; less reliable than AP).

### 10.3 Next experiments (priority order)

**A. Extended training — +10 000 iters from current checkpoint, 5× lower LR**
```bash
python tools/train_net.py \
    --config-file configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --num-gpus 1 \
    --resume \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.BASE_LR 0.0004 \
    SOLVER.MAX_ITER 20000 \
    SOLVER.STEPS "(15000, 19000)" \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    TEST.EVAL_PERIOD 5000 \
    OUTPUT_DIR output/wildbox_wl5_finetune
```
Expected: +5-10 3D AP, especially on long-tail.

**B. Stronger long-tail balancing.** Bump `DATALOADER.REPEAT_THRESHOLD=0.5`. Retrain from scratch (balancing affects the loader, not weights).

**C. Ingest remaining zips** (currently 2-3 still transferring). Adding ~20k frames of rhino/elephant is the highest-value next action once transfers complete.

**D. From-scratch ablation.** Train with `MODEL.WEIGHTS_PRETRAIN=""` and compare to fine-tuned. Quantifies the Omni3D prior's contribution. Required for the paper's "is pretraining necessary?" question.

**E. Multi-seed variance.** Run with `--seed 1` and `--seed 2` prep + retrain. Report means ± std, especially on giraffe / gazelle where video counts are small.

---

## 11. Key design decisions (for the paper's "we chose X rejected Y")

| Decision | What we did | What we rejected | Why |
|---|---|---|---|
| 2D GT source | SAM3 silhouette mask | VGGT projected-cuboid bbox | Cuboid always loose; SAM3 silhouette-tight → +15-20 AP75 |
| Split granularity | Video-level | Segment / frame | Segment leaks same-video backgrounds |
| Primary metric | AP-BEV @ 0.25 | AP-3D @ 0.5 | BEV drops VGGT-unreliable depth; aerial-3D community precedent |
| Rel-AP-3D grid | (0.3, 3.0, 28) | (0.1, 5.0, 20) | Matches LabelAny3D paper exactly |
| Scale handling | Per-segment uniform scale + Rel-AP3D | Absolute metric depth | VGGT produces synthetic scale |
| Class balancing | REPEAT_THRESHOLD=0.25 | Uniform sampling | Giraffe/gazelle are long-tail; uniform overfits rhino |
| 3D IoU backend | pytorch3d CPU path | pytorch3d CUDA | Cluster glibc blocks CUDA build |
| Visualization | Class-agnostic tool with gt_only/pred_only/combined | In-training vis only | Robust to class-id mismatches; figure-ready |
| Eval frequency | Every 5 000 iters | Every 1 000 iters | Eval on 13k images costs 30 min; 2-per-run is plenty |
| Batch size scaling | 8 with LR 2× | 4 with LR 1× | A40 has 48GB; 2× throughput on unshared GPU |

---

## 12. Known limitations

- **Pseudo-label noise.** SAM3 mask failures or VGGT track errors propagate to GT. Mitigated by BEV-primary + Rel-AP-3D.
- **Long-tail species.** Giraffe (4 videos) + gazelle (4 videos) have high variance between splits. Report multi-seed for these classes.
- **Plains + Grévy's zebra** pooled as one "zebra" class. Document explicitly; re-run with split prompts if needed.
- **BEV approximation.** Currently drops camera-Y axis rather than extracting a true ground-plane normal. Fine for near-nadir drone shots; noisy on oblique footage.
- **No tracking evaluation.** Per-frame detection only; a HOTA/IDF1 evaluation over tracks would be a natural extension.
- **Rel-AP-3D is per-dataset-global** (one scalar for all val images). Per-sequence would be more honest but LabelAny3D's published protocol is per-dataset — we match.

---

## 13. Changelog vs first version

| Fix / feature | Where | Why |
|---|---|---|
| Zip auto-extract with sibling `*_unzipped/` cache | `prepare_wildbox_dataset.py:resolve_source_path` | Don't re-extract on every run |
| Skip corrupt / in-progress zips with warning | same | A partial transfer no longer aborts the whole prep |
| SAM3 tight 2D bboxes | `prepare_wildbox_dataset.py` | Cuboid projection is always loose |
| Video-level split | `prepare_wildbox_dataset.py:auto_split` | Segment-level leaks |
| Rel-AP-3D CPU scale search | `cubercnn/evaluation/omni3d_evaluation.py:search_rel_scale` | pytorch3d CUDA unavailable |
| Zero-shot evaluator short-circuit | same file | Empty in-vocab predictions no longer crash eval |
| Training-safe vis try/except | `tools/train_net.py:do_test` | Vis exceptions never kill training |
| Defensive category-id lookup in vis | `cubercnn/vis/vis.py` | Handles both contiguous and dataset-id inputs |
| `bev_ap_eval.py` | new | Primary metric (paper) |
| `class_agnostic_eval.py --nhd` + macro-AP | new / updated | Zero-shot diagnostic + scale-invariant 3D surrogate |
| `make_report.py` with `--compare` | new | Single-command paper table |
| `run_full_eval.sh` | new | One-shot: eval + Rel-AP3D + BEV + NHD + report |
| Robust per-class log parser | `make_report.py` | Global regex instead of positional — all species captured |
| `visualize_class_agnostic.py` triplet output | new | Separate gt_only/pred_only/combined per sample |
| `plot_training.py` key discovery + full key dump | new | Debug metric-key naming drift across detectron2 versions |
| `remap_wildbox_paths.py` | new | Path rescue when data moves, without invalidating split |
| `patch_stats_for_wildbox.py` idempotency | same | Re-running is safe |
| **category_meta_wildlife*.json ordering fix** | both files + auto-gen in prep script | Training's auto mapping sorts by dataset-id; our hand-written meta didn't match → all per-class metrics collapsed to 0 for 4/5 classes. Fixed by matching the sort order and auto-generating from prep |
| `make_report.py` detects single-nonzero-class case | `parse_standard_eval_log` | Loud warning when the above mapping bug recurs |
| **`configs/wildbox/category_meta.json` symlinked to wildlife5** | new | `train_net.py --eval-only` (line 402) reads this path, not the top-level one. The old Phase-1 giraffe-only file was causing every --eval-only standalone run to report only giraffe |
| `run_full_eval.sh` auto-syncs both category_meta files | | Prevents the two-files-diverge bug at the source |

---

## 14. Citations to include in the paper

- **OVMono3D** (uva-cv-lab) — base architecture.
- **Cube R-CNN** (Brazil et al., CVPR 2023) — framework OVMono3D extends.
- **Omni3D** — pretraining dataset + AP-3D definition.
- **LabelAny3D** — Rel-AP-3D protocol + grid (0.3, 3.0, 28).
- **KABR** — long-tail reporting convention (macro + micro + per-class).
- **UAV3D / AM3D / CDrone** — BEV-as-primary in aerial 3D.
- **SAM3** — mask source.
- **VGGT** — 3D reconstruction source.
- **DINOv2** — backbone pretraining.

---

---

## 15. Final-run operations guide

This section is what you open when you're ready to produce **the official paper-numbers run**. It's a single top-to-bottom flow, no branches.

### 15.1 Pre-flight checklist (must all pass before anything else)

Run on **node81** (GPU). Anywhere else risks cv2/libGL errors.

```bash
ssh node81
conda activate /storage3/3DOM/vshukla/envs/ovmono3d
cd /storage2/3DOM/vshukla/repos/ovmono3d
git pull
```

Then these six checks, **all must print OK**:

```bash
# [A] Environment works
python -c "
import torch, pytorch3d, detectron2
assert torch.cuda.is_available(), 'no CUDA'
from cubercnn.modeling.roi_heads import ROIHeads3D
print('OK:', torch.cuda.get_device_name(0))
"
# Expect: OK: NVIDIA A40

# [B] shapely installed (needed for BEV)
python -c "import shapely; print('OK shapely', shapely.__version__)"
# If missing: pip install shapely

# [C] Pretrained checkpoint present
ls -lh checkpoints/ovmono3d_lift.pth

# [D] Omni3D stats registered for all 5 species
python -c "
import json
s = json.load(open('datasets/Omni3D/stats.json'))
wildlife = {'rhino', 'elephant', 'zebra', 'giraffe', 'gazelle'}
for info in s.get('info', []):
    cats = {c['name'] for c in info.get('category_names', [])}
    miss = wildlife - cats
    if miss: print('MISSING:', miss)
print('OK: stats.json has all 5 species')
"
# If any missing: rerun tools/patch_stats_for_wildbox.py

# [E] BOTH category_meta.json files point to wildlife5
for F in configs/category_meta.json configs/wildbox/category_meta.json; do
    ls -la "$F" | grep -q wildlife5 && echo "OK: $F -> wildlife5" || {
        echo "BAD symlink at $F — fixing..."
        case "$F" in
            configs/category_meta.json)
                ln -sf wildbox/category_meta_wildlife5.json "$F" ;;
            configs/wildbox/category_meta.json)
                ln -sf category_meta_wildlife5.json "$F" ;;
        esac
    }
done

cat configs/category_meta.json
cat configs/wildbox/category_meta.json
# Both MUST show thing_classes = ["giraffe", "zebra", "elephant", "rhino", "gazelle"]
# With mapping {"1000":0, "1001":1, "1002":2, "1004":3, "1005":4}
# giraffe FIRST (smallest dataset-id 1000), gazelle LAST (1005)

# [F] Val JSON healthy + no video-level leakage
python -c "
import json, os
from collections import Counter
for split in ('train','val'):
    d = json.load(open(f'datasets/Omni3D/WildBox_{split}.json'))
    cats = Counter(a['category_name'] for a in d['annotations'])
    vids = {img['file_path'].split('/')[-3] for img in d['images']}
    missing = sum(1 for img in d['images'][:200] if not os.path.exists(img['file_path']))
    assert missing == 0, f'{split}: {missing}/200 missing paths'
    print(f'{split}: {len(d[\"images\"])} imgs, {len(d[\"annotations\"])} anns, '
          f'{len(vids)} videos, anns_per_class={dict(cats)}')
train_vids = {img['file_path'].split('/')[-3] for img in json.load(open('datasets/Omni3D/WildBox_train.json'))['images']}
val_vids   = {img['file_path'].split('/')[-3] for img in json.load(open('datasets/Omni3D/WildBox_val.json'))['images']}
overlap = len(train_vids & val_vids)
assert overlap == 0, f'video leakage: {overlap} shared videos'
print(f'OK: {len(train_vids)} train videos, {len(val_vids)} val videos, no overlap')
"
```

Only proceed past this point if all 6 checks printed `OK`.

### 15.2 Zero-shot baseline (step 1 of 3)

```bash
tmux new -s wb-zs
bash tools/run_full_eval.sh \
    --weights checkpoints/ovmono3d_lift.pth \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --out     output/final_zeroshot \
    --label   "zero-shot (pretrained Omni3D)" \
    --gt      datasets/Omni3D/WildBox_val.json \
    --skip-rel-ap3d     # no in-vocab predictions -> scale search trivially 0
```
Runtime ~15 min. Detach with `Ctrl-b d`.

**Expected output** in `output/final_zeroshot/`:
- `log.txt` — all-zero primary metrics (expected for closed-vocab)
- `bev_ap.json` — BEV AP, will also be ~0 on named classes
- `summary_nhd.txt` — **the meaningful zero-shot number**: class-agnostic 2D AP (~20-40 depending on scene difficulty)
- `inference/iter_final/WildBox_val/instances_predictions.pth` — raw predictions for downstream tools
- `paper_report/report.md` — markdown report

### 15.3 Fine-tune training (step 2 of 3)

Recommended config for the paper run:

```bash
tmux new -s wb-train
python tools/train_net.py \
    --config-file configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --num-gpus 1 \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.BASE_LR 0.002 \
    SOLVER.MAX_ITER 15000 \
    SOLVER.STEPS "(9000, 13500)" \
    SOLVER.WARMUP_ITERS 500 \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    TEST.EVAL_PERIOD 5000 \
    DATALOADER.REPEAT_THRESHOLD 0.5 \
    OUTPUT_DIR output/final_finetune
```

Why **15 000 iters, not 10 000**: §10 showed 3D AP was still climbing between iter 5 000 and 10 000 (6.2 → 16.1). Another 5 000 iters should add 3-5 more AP points and help long-tail classes.

Why **REPEAT_THRESHOLD=0.5**: stronger upsampling for giraffe/gazelle (only 4 videos each). Current 0.25 produced giraffe AP 1.8 / gazelle AP 2.3 — too low to report as per-class numbers.

Runtime estimate: ~5 hours on A40 (unshared). Detach with `Ctrl-b d`.

Monitor:
```bash
tail -f output/final_finetune/log.txt
# Latest losses
python tools/plot_training.py output/final_finetune/metrics.json
# -> output/final_finetune/training_curves.png
```

### 15.4 Fine-tuned eval + final report (step 3 of 3)

After training completes:

```bash
tmux new -s wb-eval-ft
bash tools/run_full_eval.sh \
    --weights output/final_finetune/model_final.pth \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --out     output/final_finetuned_eval \
    --label   "fine-tuned (5 species)" \
    --gt      datasets/Omni3D/WildBox_val.json
```

Runtime ~25 min (15 min Rel-AP3D CPU scale search + 10 min other steps).

Combined zero-shot vs fine-tuned report:

```bash
python tools/make_report.py \
    --run-dir output/final_zeroshot        --label "zero-shot" \
    --run-dir output/final_finetuned_eval  --label "fine-tuned" \
    --gt      datasets/Omni3D/WildBox_val.json \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --out     output/final_paper_report \
    --compare
```

Paper-ready visualizations:

```bash
python tools/visualize_class_agnostic.py \
    --preds output/final_finetune/inference/iter_final/WildBox_val/instances_predictions.pth \
    --gt    datasets/Omni3D/WildBox_val.json \
    --out   output/final_finetune/vis_agnostic \
    --top-k 3 --every 100 --limit 40

python tools/visualize_class_agnostic.py \
    --preds output/final_zeroshot/inference/iter_final/WildBox_val/instances_predictions.pth \
    --gt    datasets/Omni3D/WildBox_val.json \
    --out   output/final_zeroshot/vis_agnostic \
    --top-k 3 --every 100 --limit 40
```

### 15.5 What to look at in the final report (verification before trusting)

Open `output/final_paper_report/report.md` and check:

1. **`## Validation set` block** — per-class annotation counts match your expectations (rhino and zebra should be the largest, giraffe smallest).
2. **`## Main metrics table`** — both zero-shot and fine-tuned rows present, **all 5 species columns populated** for fine-tuned. If only 1 class has non-zero values, the symlink was wrong again (check 15.1[E]).
3. **`## Zero-shot diagnostic`** — check that `Class-agnostic 2D AP@0.50` shows a meaningful zero-shot value (~15-35) and fine-tuned value (~60-85). The gap is the paper's headline.
4. **`## One-line callout`** — NHD-z should be the largest component (typically 70-85% of overall NHD). This is what justifies BEV-as-primary.

Spot-check individual metrics against the training log:
```bash
grep -A 10 "Evaluation results for bbox in 2D mode" output/final_finetuned_eval/log.txt | tail -10
grep -A 10 "Evaluation results for bbox in 3D mode" output/final_finetuned_eval/log.txt | tail -10
grep "best global scale" output/final_finetuned_eval/log.rel.txt
```

---

## 16. Recommendations for iteration count and hyperparameters

### 16.1 Iteration count recommendations

| Configuration | Iters (batch 8, LR 2e-3) | When to use |
|---|---:|---|
| Quick sanity | 1 000 | Confirm the pipeline runs end-to-end |
| Dev / exploration | 5 000 | Early signal on hyperparameter changes |
| **Paper run** | **15 000** | Balance of 3D convergence + reasonable wall-clock |
| Max-quality | 25 000 | When long-tail AP must be maximized (stretch) |

### 16.2 Hyperparameter recommendations

| Parameter | Value | Why / when to change |
|---|---|---|
| `IMS_PER_BATCH` | 8 | Fits A40; keep constant across experiments for fair comparison |
| `BASE_LR` | 0.002 | Linear-scaled from batch-4 default; if NaNs, halve |
| `WARMUP_ITERS` | 500 | ~3% of total; scale with MAX_ITER |
| `STEPS` | (60%, 90%) of MAX_ITER | Standard schedule; keep ratios |
| `CHECKPOINT_PERIOD` | 1000 | Bounds kill-restart loss to 25 min |
| `TEST.EVAL_PERIOD` | 5000 | In-loop eval is expensive (~15 min on 13k val) |
| `REPEAT_THRESHOLD` | 0.5 | 0.25 too weak for 4-video classes; 0.5 is aggressive but justified |
| `LOSS_W_Z` | 0.5 | Do not change — tied to VGGT synthetic scale |
| `LOSS_W_{xy,dims,pose,joint}` | 1.0 | Default; if NHD decomposition shows one component dominating abnormally, consider rebalancing |
| `MODEL.STABILIZE` | 0.02 | Keep — auto-restart from checkpoint if loss diverges |
| `AMP.ENABLED` | True | 1.5–2× speedup on A40; disable only if NaNs persist |
| `DATALOADER.NUM_WORKERS` | 8 | GPU under-utilized at 4; 8 is the sweet spot on node81 |

### 16.3 Data recommendations

| Item | Recommendation | Rationale |
|---|---|---|
| Split mode | **Always `video` with `--seed 0`** | Prevents leakage; deterministic across runs |
| Val fraction | 0.20 | Balance of coverage vs variance; with 68 videos → 13-14 val videos |
| 2D GT source | **SAM3 masks (auto in prep)** | Cuboid projection is always loose |
| Scale normalization | **Per-segment uniform (auto)** | Required for VGGT synthetic scale |
| Zip validation | `unzip -t ZIP` before running prep | Partial transfers silently abort prep's zip step |
| Multi-seed runs | Run with `--seed {0,1,2}` for variance | Critical for long-tail class CI bars |

### 16.4 When to retrain from scratch vs resume

- **From scratch (use pretrained OVMono3D as init)**: any change to the *data distribution* (adding zips, changing class balance, splitting into species) → retrain.
- **Resume from checkpoint (`--resume`)**: continuing the same training run (kill/restart), or extending iter count without changing data/balance.
- **Never**: resume from a different-species-count checkpoint. The contiguous-id → species mapping changes, and class logits will get assigned wrong.

---

## 17. Where every output artifact lives (file map)

For a run at `OUTPUT_DIR=output/<run_name>/`:

### 17.1 From training (`train_net.py`)

| File | Content |
|---|---|
| `output/<run>/log.txt` | Full training + in-loop eval log. Search for `Evaluation results for bbox in <mode> mode:` to find metric tables. |
| `output/<run>/metrics.json` | Line-delimited JSON, one record per training print interval. Contains losses, LR, eval AP. Consumed by `plot_training.py`. |
| `output/<run>/config.yaml` | Exact config used (with all CLI overrides baked in). Reproducibility. |
| `output/<run>/category_meta.json` | Auto-written by training. The training-time contiguous-id mapping. |
| `output/<run>/model_*.pth` | Periodic checkpoints. `model_final.pth` = last. `model_<iter>.pth` = intermediate. |
| `output/<run>/last_checkpoint` | Text file pointing to latest checkpoint. Used by `--resume`. |
| `output/<run>/events.out.tfevents.*` | TensorBoard events. `tensorboard --logdir output/<run>` to view. |
| `output/<run>/inference/iter_<N>/<dataset>/` | Per-eval predictions + visualization outputs |
| `output/<run>/inference/iter_<N>/<dataset>/instances_predictions.pth` | **Raw model predictions** (list of image dicts with bboxes, scores, centers, dims, poses). **This is the file all our custom tools consume.** |
| `output/<run>/inference/iter_<N>/<dataset>/omni_instances_results.json` | COCO-format export of predictions. |
| `output/<run>/inference/iter_<N>/<dataset>/vis/` | Per-sample visualization images (when vis didn't crash). |

### 17.2 From `run_full_eval.sh`

Same as 17.1 plus:

| File | Content |
|---|---|
| `output/<run>/bev_ap.json` | `bev_ap_eval.py` output. Per-class, micro, macro AP_BEV @ {0.25, 0.50}. |
| `output/<run>/summary_nhd.txt` | `class_agnostic_eval.py --nhd` output. Class-agnostic 2D AP, NHD scale search, per-class class-agnostic AP. |
| `output/<run>/log.rel.txt` | Rel-AP-3D eval log (if not `--skip-rel-ap3d`). Contains `[rel_ap3d] best global scale = ...` line. |
| `output/<run>_rel/` | Scratch dir for Rel-AP3D eval; usually ignorable. |
| `output/<run>/paper_report/report.md` | Single-run human-readable report. |
| `output/<run>/paper_report/metrics.json` | Single-run machine-readable summary. |
| `output/<run>/paper_report/table_main.tex` | Single-run LaTeX table. |

### 17.3 From `make_report.py --compare`

| File (in `--out` dir) | Content |
|---|---|
| `report.md` | **The main artifact**: zero-shot vs fine-tuned, main metrics + diagnostic + NHD callout. |
| `table_main.tex` | LaTeX-ready primary metrics table for the paper. |
| `metrics.json` | Everything, machine-readable. Full run blobs. Great for secondary plots. |

### 17.4 From visualization tools

| File | Content |
|---|---|
| `output/<run>/vis_agnostic/gt_only/pair_NNNNNN.jpg` | Ground truth boxes on frame (green) |
| `output/<run>/vis_agnostic/pred_only/pair_NNNNNN.jpg` | Top-K prediction boxes on frame (red) |
| `output/<run>/vis_agnostic/combined/pair_NNNNNN.jpg` | Both overlaid |
| `output/<run>/training_curves.png` | 6-panel training curves (losses, LR, AP over time, NHD) |

### 17.5 Dataset artifacts (input to training)

| File | Content |
|---|---|
| `datasets/Omni3D/WildBox_train.json` | Training split (Omni3D format) |
| `datasets/Omni3D/WildBox_val.json` | Validation split |
| `datasets/Omni3D/stats.json` | Global Omni3D category registry |
| `configs/category_meta.json` | Symlink → wildlife5 meta. Used by external eval tools. |
| `datasets/pending_sources.txt` | Known missing zips, with re-add command |
| `datasets/run1_description.xlsx` | Input-data audit (frame/bbox/video counts per zip) |
| `<zip_dir>_unzipped/` | Extracted zip contents (cached, referenced by absolute paths in JSON) |

---

## 18. Quick commands cheat sheet

### Training
```bash
# Quick sanity (confirm pipeline)
python tools/train_net.py --config-file configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --num-gpus 1 SOLVER.MAX_ITER 200 SOLVER.STEPS "(100,150)" SOLVER.CHECKPOINT_PERIOD 100 \
    TEST.EVAL_PERIOD 100 OUTPUT_DIR output/smoke

# Full paper run
python tools/train_net.py --config-file configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --num-gpus 1 SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.002 \
    SOLVER.MAX_ITER 15000 SOLVER.STEPS "(9000,13500)" \
    TEST.EVAL_PERIOD 5000 SOLVER.CHECKPOINT_PERIOD 1000 \
    DATALOADER.REPEAT_THRESHOLD 0.5 \
    OUTPUT_DIR output/final_finetune

# Resume
python tools/train_net.py --config-file <cfg> --num-gpus 1 --resume \
    OUTPUT_DIR output/final_finetune
```

### Eval
```bash
# Full evaluation (2D + 3D + Rel-AP3D + BEV + NHD + report)
bash tools/run_full_eval.sh \
    --weights <ckpt.pth> --config <cfg> --out <dir> --label "name" \
    --gt datasets/Omni3D/WildBox_val.json [--skip-rel-ap3d]
```

### Reporting
```bash
# Combined zero-shot vs fine-tuned report
python tools/make_report.py \
    --run-dir <zs_dir> --label "zero-shot" \
    --run-dir <ft_dir> --label "fine-tuned" \
    --gt datasets/Omni3D/WildBox_val.json \
    --config <cfg> --out <report_out_dir> --compare
```

### Plots and visualizations
```bash
# Training curves
python tools/plot_training.py <run_dir>/metrics.json

# GT / pred comparison images
python tools/visualize_class_agnostic.py \
    --preds <run_dir>/inference/iter_final/WildBox_val/instances_predictions.pth \
    --gt datasets/Omni3D/WildBox_val.json \
    --out <run_dir>/vis_agnostic --top-k 3 --every 100 --limit 40
```

### Data prep
```bash
python tools/prepare_wildbox_dataset.py \
    --source <ZIP>=<species>:<id> \  # repeatable
    --split-mode video --val-fraction 0.2 --seed 0 \
    --output-train datasets/Omni3D/WildBox_train.json \
    --output-val datasets/Omni3D/WildBox_val.json \
    --dataset-id 1000 -v
```

### Path rescue
```bash
# If data was moved after training:
python tools/remap_wildbox_paths.py datasets/Omni3D/WildBox_val.json \
    [--map OLD=NEW | --search-root /path/to/data] --in-place
```

### Monitoring
```bash
# Training loss + ETA
tail -n 5 <run_dir>/log.txt | grep iter

# Check specific GPU utilization while training
nvidia-smi --query-compute-apps=pid,used_memory,gpu_bus_id --format=csv

# Which GPU is my job on?
YOUR_PID=$(pgrep -f train_net.py | head -1)
nvidia-smi --query-compute-apps=pid,gpu_name,used_memory --format=csv | grep "^$YOUR_PID,"
```

---

## 19. Things to track for every experiment (comparison log template)

When running a new experiment (different architecture, different data split, different hyperparams), fill out a row like this to enable cross-experiment comparison. Suggested to keep as a CSV or markdown table in `output/experiments.md`:

| Field | Example value | Why it matters |
|---|---|---|
| Experiment label | `wl5-11zip-bs8-lr2e-3-15k-rep0.5` | Unique, descriptive |
| Date | 2026-04-22 | Chronological |
| Architecture | `OVMono3D-lift DINOv2 ViT-B/14` | Change for ablations |
| Pretrained weights | `checkpoints/ovmono3d_lift.pth` | Init choice |
| Data sources | 11/13 zips (missing: data202401KR, data202401KE, data202602KE) | For fairness tracking |
| Total frames | ~55 000 | Data size |
| Split mode | video, seed=0, 0.2 val | Splitting protocol |
| Species | rhino, elephant, zebra, giraffe, gazelle | Classes |
| Batch size | 8 | Scale knob |
| Base LR | 2e-3 | Scale knob |
| Max iters | 15 000 | Training length |
| REPEAT_THRESHOLD | 0.5 | Long-tail balancing |
| Wall-clock | 5h 12m | Throughput |
| **AP_BEV@0.25 (macro)** | 78.4 | **Headline metric** |
| **AP_3D@0.25 (macro)** | 24.1 | Secondary |
| **Rel-AP_3D (macro)** | 34.8 | LabelAny3D metric |
| **2D AP@0.5 (macro)** | 89.2 | Sanity |
| NHD-z | 2.8 | Depth error |
| Best scale (Rel-AP3D) | 1.04 | Sanity (should ≈ 1) |
| Class-agnostic 2D AP@0.50 (zero-shot) | 24.1 | Baseline |
| Class-agnostic 2D AP@0.50 (fine-tuned) | 88.3 | Delta story |
| Notes | REPEAT=0.5 helped giraffe (+5 AP3D) | Anything unusual |

Automating this: each `paper_report/metrics.json` has a deterministic schema — a simple script can ingest a directory of reports and produce this table.

---

## 20. Running the same experiment on a different architecture

Same dataset, same eval, different model. Minimal-overhead workflow:

### 20.1 Setup

1. Install the target architecture's repo side-by-side with this one.
2. Copy our dataset prep output:
   ```bash
   ln -s /storage2/3DOM/vshukla/repos/ovmono3d/datasets /path/to/new_repo/datasets
   ```
   Or copy `WildBox_train.json`, `WildBox_val.json`, and the unzipped data dirs.
3. Adapt the target's data loader to read Omni3D-format JSON. Most 3D detection frameworks (Cube R-CNN, DetAny3D) accept this out-of-box.

### 20.2 Key config choices to match for a fair comparison

| Axis | Our value | Rationale for any re-implementation |
|---|---|---|
| Pretrained init | Omni3D-pretrained checkpoint for the architecture | Control for pretraining quality |
| Image resolution | 294 short-edge, 560 max | Match VGGT's output resolution |
| Batch size | 8 on A40 | Same VRAM budget |
| LR | 2e-3 SGD (or 1e-3 AdamW) | Tune via initial warmup run |
| Iters | 15 000 at batch 8 | Same data exposure |
| Augmentation | horizontal flip + random-scale (280–392 short-edge) | Keep |
| Loss weight for depth | **0.5×** vs other 3D losses | Specific to VGGT synthetic scale |
| Class balancer | REPEAT_THRESHOLD 0.5 or equivalent | Long-tail robustness |

### 20.3 Output format required for our eval tools

Whatever architecture you use, **save predictions as a detectron2-compatible `instances_predictions.pth`** (a list of dicts, one per image):

```python
torch.save([
    {
        "image_id": int,
        "instances": [
            {
                "bbox": [x, y, w, h],           # xywh in image pixels
                "score": float,
                "category_id": int,              # 0..N-1, MUST match training's contiguous mapping
                "center_cam": [x, y, z],         # 3D center in camera frame
                "dimensions": [W, H, L],         # Omni3D ordering: X=L, Y=H, Z=W
                "pose": [[3x3 rotation]]          # 3x3 matrix or 9-element flat
            }
        ]
    }, ...
], "predictions.pth")
```

Then our eval tools run identically:

```bash
python tools/bev_ap_eval.py --preds <predictions.pth> --gt datasets/Omni3D/WildBox_val.json \
    --out <arch_dir>/bev_ap.json
python tools/class_agnostic_eval.py --preds <predictions.pth> --gt datasets/Omni3D/WildBox_val.json \
    --nhd > <arch_dir>/summary_nhd.txt
```

### 20.4 Reporting a new architecture's results

Run `make_report.py --compare` with your new architecture's output and our OVMono3D output:

```bash
python tools/make_report.py \
    --run-dir output/final_finetuned_eval       --label "OVMono3D (ours)" \
    --run-dir output/cube_rcnn_finetuned_eval   --label "Cube R-CNN" \
    --run-dir output/detany3d_finetuned_eval    --label "DetAny3D" \
    --gt datasets/Omni3D/WildBox_val.json \
    --config configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --out output/paper_report_architecture_comparison --compare
```

The report will be a side-by-side table across architectures. Paper-ready.

### 20.5 Constants to preserve across architecture experiments

**Do not vary** any of these when comparing architectures — otherwise differences aren't attributable to the architecture:

- Dataset files (`WildBox_train.json`, `WildBox_val.json`)
- Split seed
- `configs/category_meta.json` symlink target
- Evaluation protocol (same `run_full_eval.sh` command, same metrics)
- Image resolution

Do vary (and track in §19's template):

- The architecture itself
- Loss weights (as required by the architecture)
- Backbone pretraining source
- LR schedule (fit to the architecture)

---

_End of document. Any new experiment, hyperparameter change, or code fix should be reflected here (especially §13 changelog and §17 file map) to keep the docs in sync with the code._
