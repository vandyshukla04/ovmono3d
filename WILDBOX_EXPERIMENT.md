# WildBox 3D Wildlife Detection — Experiment Documentation

**Purpose of this document.** Self-contained reference for reproducing WildBox 3D wildlife detection. Covers dataset, preprocessing, training, evaluation, and adaptation to other 3D detection architectures. Written for (1) researchers replicating on Cube R-CNN / DetAny3D / custom architectures, (2) paper-writing.

**Related doc:** [QUICK_START_SMOKE_TEST.md](QUICK_START_SMOKE_TEST.md) — a ~2-hour end-to-end sanity test with one zip per species + 2000 training iters. Run this after pulling new code or changing configs, before committing to a multi-hour full run.

---

## 1. TL;DR

- **Task.** Monocular 3D detection of African wildlife from aerial drone footage.
- **Species (paper).** 6 classes: giraffe, grevys_zebra, elephant, plains_zebra, rhino, gazelle (contiguous ids 0-5; dataset ids 1000, 1001, 1002, 1003, 1004, 1005).
- **Dataset.** WildBox — 15 drone campaign zips, **59 758 frames, 237 505 3D-cuboid instances, 64 videos**. Paper-ready stats auto-regenerated every prep (`tools/dataset_stats.py`). 3D GT is pseudo-labels from SAM3-tight 2D + VGGT reconstruction.
- **Architecture.** OVMono3D-lift (Cube R-CNN variant, DINOv2 ViT-B/14 backbone). Fine-tuned from `ovmono3d_lift.pth`.
- **Training.** 15 000 iterations (batch 8, LR 2e-3, AMP, REPEAT_THRESHOLD=0.5). 3 seeds for mean±std on rare classes (giraffe, gazelle, grevys_zebra — all ≤5 train videos). ~5 h per seed, ~15 h total on A40.
- **Evaluation.** Three-row comparison for every paper table:
  1. Zero-shot with model's own RPN (`TEST.ORACLE2D=False`) — closed-vocab baseline
  2. Zero-shot paper-protocol (`TEST.ORACLE2D=True`, GroundingDINO oracle 2D boxes) — OVMono3D paper convention
  3. Fine-tuned (multi-seed wildlife6)
- **Primary metric.** **AP-BEV @ IoU 0.50** (KITTI-convention tight IoU). Supplementary: AP-BEV@0.25, AP-3D@0.25, Rel-AP-3D (LabelAny3D), 2D AP @ 0.5, class-agnostic 2D AP, disentangled NHD.
- **Split.** Video-level, seed=0, 80/20 — 51 train / 13 val videos, zero video-level leakage.
- **Start here for ops.** [FINAL_RUN.md](FINAL_RUN.md) is the 13-step linear pipeline. [QUICK_START_SMOKE_TEST.md](QUICK_START_SMOKE_TEST.md) is the 2-h smoke.

---

## 2. Dataset

### 2.1 Source and provenance

- Raw videos: DJI drone footage, Kenya, 2023–2026.
- Species labels via text-prompted SAM3 segmentation.
- 3D cuboids via VGGT 3D reconstruction + tracking on the masked frames.

### 2.2 Data inventory (15 campaign zips, 6 species)

The canonical snapshot lives in `datasets/Omni3D/dataset_stats/dataset_stats.md` — regenerated every prep via `tools/dataset_stats.py`. The static inventory below lists the source zips; actual per-class counts in train/val splits come from the auto-generated stats file.

| Zip path | Species (class) | Notes |
|---|---|---|
| archive/data202401KGiraffes | giraffe | main giraffe source |
| archive/data202501KGiraffes | giraffe | small supplementary |
| archive/data2023KABRZebras | grevys_zebra | KABR reserve — the *only* Grévy's zebra source |
| archive/wildbox_tomblair | plains_zebra | |
| archive/dataBZS | plains_zebra | |
| archive/data202307KZebras | plains_zebra | |
| archive/202401KZebras | plains_zebra | |
| archive/data202401KElephants | elephant | |
| archive/data202406KElephants | elephant | |
| archive/data202501KElephants | elephant | |
| archive/data202602KElephants | elephant | |
| archive/data202401KRhinos | rhino | |
| archive/data202502KRhinoCamiV1 | rhino | |
| archive/data202502KRhinoCamiV2 | rhino | |
| archive/data202406KGazelles | gazelle | only gazelle source |

**Current aggregate (from dataset_stats.md after prepare on 2026-04-24):**

| Split | Videos | Segments | Frames | Boxes |
|---|---:|---:|---:|---:|
| Train | 51 | 263 | 45 979 | 170 554 |
| Val | 13 | 82 | 13 779 | 66 951 |
| **Total** | **64** | **345** | **59 758** | **237 505** |

**Per-species train/val video counts (the rare classes determine multi-seed policy):**

| Species | train vids | val vids | Rare? (≤5 vids) |
|---|---:|---:|---|
| elephant | 14 | 3 | |
| gazelle | 3 | 1 | **YES** |
| giraffe | 3 | 1 | **YES** — val is thin (110 boxes) |
| grevys_zebra | 4 | 1 | **YES** |
| plains_zebra | 12 | 3 | |
| rhino | 15 | 4 | |

Rare-class classes get 3-seed mean±std in the paper; stable classes get single-seed.

**Zebra species provenance** (paper text): per user confirmation, `data2023KABRZebras` contains exclusively Grévy's zebra (*Equus grevyi*); the other four zebra zips (tomblair, BZS, 202307K, 202401K) are plains zebra (*Equus quagga*). Paper dataset section must state this.

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
- **Pretrained checkpoint's `iteration` field silently skipped training** (caught 2026-04-24, commit f894ab6). `ovmono3d_lift.pth` stores `iteration=115999` from Omni3D pre-training. [tools/train_net.py:183](tools/train_net.py#L183) previously read this field unconditionally (regardless of `resume` flag) and set `start_iter=116000`. Any fine-tune with `MAX_ITER<116000` silently exited the training loop with zero iterations — no error, no stack trace, just a `model_final.pth` byte-identical to the pretrained.

  **How to spot (red flags in the log — the first two of which we MISSED on the bad run):**
  1. The startup log says `[DatasetMapper] Augmentations used in inference: [ResizeShortestEdge(...)]` but NEVER says `Augmentations used in training: [...RandomFlip(), multi-scale resize...]`. Real training always logs both, since it uses separate augmentation pipelines for train vs periodic eval. Single "inference" augmentations = the training loop was never entered.
  2. The log jumps from `Environment info` directly to `Start inference on N batches` within 1-2 minutes of launch. Real training spends hours between those events, emitting `iter: 100 total_loss: 1.234 ...` lines every 20 iterations.
  3. `grep -cE "iter: [0-9]+" log.txt` returns 0.
  4. The single log line `Starting training from iteration 116000 (resume=False)` — definitive.
  5. `model_final.pth` is byte-identical to `ovmono3d_lift.pth` (same 603,232,016-byte file).

  Fix in the code now respects `resume=False` → `start_iter=0`; the backup belt-and-suspenders is to strip iteration from the pretrained:
  ```bash
  python -c "
  import torch
  c = torch.load('checkpoints/ovmono3d_lift.pth', weights_only=False, map_location='cpu')
  c.pop('iteration', None); c.pop('optimizer', None); c.pop('scheduler', None)
  torch.save(c, 'checkpoints/ovmono3d_lift.pth')
  "
  ```
  **5-species wildlife5 runs predating this fix happened to start from iter 0** because they didn't CLI-override `MODEL.WEIGHTS` — only the later multi-seed path did, which routed through the bug. Any run launched via `tools/run_multi_seed.sh` pre-f894ab6 is invalid (zero-trained).
- **GroundingDINO CUDA build also fails** on this cluster (same cospi/sinpi glibc mismatch). **Use `pip install groundingdino-py==0.4.0`** — the PyPI package has a proper GPU Python fallback (~0.7 s/img on A40). Do **not** use the official `IDEA-Research/GroundingDINO` GitHub clone directly (its `ms_deform_attn.py` crashes with `NameError: name '_C' is not defined` when the CUDA op can't load — no fallback). Do **not** use `pip install groundingdino==0.1.0` either — that version's fallback runs on **CPU** (~41 s/img), 60× slower than `groundingdino-py`. See §6.4.3 for the precompute command.
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

### 6.4 Zero-shot evaluation protocol — IMPORTANT

There are **two** possible zero-shot protocols for OVMono3D. We currently run the stricter one by default; the paper's Table 1 uses the other one.

#### 6.4.1 The paper's zero-shot protocol (what we should match for Table 1)

OVMono3D paper uses **`TEST.ORACLE2D=True`** with **precomputed GroundingDINO detections** saved at `datasets/Omni3D/gdino_<dataset>_oracle_2d.json`. At eval time:

1. For each val image, look up pre-computed 2D boxes from GroundingDINO (text-prompted per species — "rhino", "elephant", etc.)
2. These 2D boxes replace the model's own RPN proposals
3. The model's 3D cube-head lifts these GDino 2D boxes to 3D cuboids
4. Standard AP is computed on these 3D predictions

This measures: *"given open-vocab 2D localization from GDino (which was never trained on WildBox), how well does OVMono3D's 3D lift onto unseen wildlife classes?"*

#### 6.4.2 What we actually run by default (`TEST.ORACLE2D=False`)

We have `ORACLE2D: False` in [configs/wildbox/OVMono3D_wildbox_finetune.yaml](configs/wildbox/OVMono3D_wildbox_finetune.yaml#L46). This means:

1. 2D boxes come from the **model's own RPN + 2D classifier** (Omni3D-pretrained 50 classes)
2. No GDino text prompts at test time

For zero-shot, this is **strictly harder** than the paper's protocol because none of the 50 pretrained class slots correspond to wildlife — every 2D-classified prediction is dropped by the evaluator's class filter, giving near-zero standard AP.

**Why we started with this:** the original base config's `ORACLE2D=True` path requires a `gdino_WildBox_val_oracle_2d.json` file, which OVMono3D only ships for the original Omni3D datasets. Setting `ORACLE2D=False` was the unblock-eval path during initial development.

#### 6.4.3 How to run the paper's protocol

To match the paper's zero-shot Table 1 numbers, we need the GDino oracle files for WildBox. **Runtime for the full 13 361 val images is ~2.5 h on A40** once the right GDino package is installed.

**Step 1 — install the correct GDino package.** This is the #1 gotcha and the one you must get right:

```bash
pip uninstall -y groundingdino groundingdino-py       # wipe any prior install
pip install --no-cache-dir groundingdino-py==0.4.0    # PyPI — proper GPU fallback
```

`groundingdino-py` is the maintained PyPI fork whose `ms_deform_attn.py` has a **GPU Python fallback** that runs `F.grid_sample` on CUDA tensors when the custom C++ op can't load. Measured on A40: **~0.7 s/image** steady-state, batch=1. See §3.1 for the alternatives (github clone, `groundingdino==0.1.0`) and why they don't work.

**Step 2 — download the GDino SwinB checkpoint** (if not already present):

```bash
mkdir -p checkpoints
wget -O checkpoints/groundingdino_swinb_cogcoor.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
# Config (if not already in the repo):
test -f configs/GroundingDINO_SwinB_cfg.py || \
  wget -O configs/GroundingDINO_SwinB_cfg.py \
  https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py
```

**Step 3 — smoke-test on 3 images first** (should take ~5 s after BERT warmup):

```bash
python tools/precompute_gdino_oracle.py \
    --gt       datasets/Omni3D/WildBox_val.json \
    --out      /tmp/smoke_3.json \
    --species  rhino elephant zebra giraffe gazelle \
    --device   cuda \
    --box-threshold 0.15 --text-threshold 0.10 \
    --limit    3
```

The smoke should log `per-img=0.7-1.0s` and produce non-zero `kept=N` boxes. If `per-img > 5 s/img`, you're on the `groundingdino==0.1.0` CPU-fallback path — re-check step 1.

**Step 4 — launch the full run in the background** (~2.5 h):

```bash
mkdir -p logs
nohup python tools/precompute_gdino_oracle.py \
    --gt       datasets/Omni3D/WildBox_val.json \
    --out      datasets/Omni3D/gdino_WildBox_val_oracle_2d.json \
    --species  rhino elephant zebra giraffe gazelle \
    --device   cuda \
    --box-threshold 0.15 --text-threshold 0.10 \
    --log-every 100 \
    > logs/gdino_oracle.log 2>&1 &
disown
tail -f logs/gdino_oracle.log       # Ctrl-C to detach; job keeps running
```

**Step 5 — run paper-protocol zero-shot eval** (uses the precomputed oracle JSON):

```bash
bash tools/run_full_eval.sh \
    --weights checkpoints/ovmono3d_lift.pth \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife5_oracle2d.yaml \
    --out     output/wildbox_wl5_zeroshot_oracle2d \
    --label   "zero-shot (paper protocol, GDino oracle)" \
    --gt      datasets/Omni3D/WildBox_val.json \
    --skip-rel-ap3d
```

The oracle JSON is reusable — precompute once, every subsequent zero-shot eval reads it in seconds.

**Threshold tuning (why 0.15/0.10 and not GDino's default 0.25/0.25).** Drone wildlife is small and distant; GDino's web-imagery-tuned defaults suppress most animals at altitude. 0.15 box / 0.10 text keeps noisy detections that the 3D cube head can still refine. If the full run comes back with <1 box per image on average, drop to 0.10/0.05 and re-run; if recall looks fine, stay at 0.15/0.10 for paper consistency.

**Preprocessing must use `load_image`, not a hand-rolled normalize.** GroundingDINO expects images resized to min-side 800 / max-side 1333 before normalization (`RandomResize([800], max_size=1333)`). Feeding native drone resolution (typ. 1920×1080) raw silently produces boxes with **median IoU ≈ 0.10** against GT — almost nothing matches and downstream 3D AP is zero. Using `from groundingdino.util.inference import load_image` applies the transform correctly and jumps median IoU to **~0.97** on the same images. [tools/precompute_gdino_oracle.py](tools/precompute_gdino_oracle.py)'s `preprocess_image` already does this — do not bypass it with a custom preprocessor when porting to another architecture's 2D wrapper.

**Oracle JSON schema** (produced by `tools/precompute_gdino_oracle.py`, consumed by the evaluator via `DATASETS.ORACLE2D_FILES`):

```jsonc
// datasets/Omni3D/gdino_WildBox_val_oracle_2d.json — one entry per image
[
  {
    "image_id": <int>,                          // matches GT JSON
    "K": [[...3x3 intrinsics...]],              // copied from GT
    "file_path": "...",                         // copied from GT
    "height": H, "width": W,
    "instances": [
      {
        "bbox": [x, y, w, h],                   // xywh in pixels
        "score": <float>,                       // GDino confidence
        "category_id": <dataset_id>,            // e.g., 1004 for rhino
        "category_name": "rhino"
      },
      ...
    ]
  },
  ...
]
```

This format is **architecture-agnostic** — see §20.6 for how to reuse it with other models.

#### 6.4.4 Which protocol to report

For a paper following OVMono3D/LabelAny3D convention, **the headline zero-shot column uses `ORACLE2D=True` (GDino-based)**. Our current `ORACLE2D=False` numbers are a *different, stricter* experiment ("closed-vocab RPN transfer") — useful as a supplementary baseline but not the main number.

**Action**: run the GDino precompute, then rerun zero-shot eval with `ORACLE2D=True`. See §21.4 for the task entry.

#### 6.4.5 The useful zero-shot number either way

Class-agnostic 2D AP (via [tools/class_agnostic_eval.py](tools/class_agnostic_eval.py)) is meaningful under **both** protocols — it ignores class labels and measures pure localization transfer. Report it in both conditions as a sanity/consistency check.

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

**See §20 — full cross-architecture protocol** (portable oracle-JSON contract, consumer-side eval wiring, side-by-side reporting table, 5-step adoption checklist).

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

Features and tools added relative to upstream OVMono3D. Bugs caught during development are in §21.2 (don't duplicate here).

| Feature / tool | Where | Purpose |
|---|---|---|
| Zip-aware data prep with caching + skip-corrupt | `prepare_wildbox_dataset.py` | Don't re-extract on re-prep; partial transfers don't abort |
| SAM3 tight 2D bboxes | `prepare_wildbox_dataset.py` | Cuboid projection is always loose — SAM3 masks give paper-quality 2D |
| Video-level split | `prepare_wildbox_dataset.py:auto_split` | Segment-level leaks; video-level doesn't |
| Auto-generated `category_meta_wildlifeN.json` | `prepare_wildbox_dataset.py` | Sort order matches training's internal mapping |
| Dataset inventory auto-generator | `tools/dataset_stats.py` | Paper-ready per-species stats + size distribution PNG, regenerated every prep |
| GroundingDINO oracle precompute | `tools/precompute_gdino_oracle.py` + `OVMono3D_wildbox_wildlife5_oracle2d.yaml` | Paper-protocol zero-shot (`TEST.ORACLE2D=True`) |
| BEV AP (primary paper metric) | `tools/bev_ap_eval.py` | Shapely rotated-rectangle IoU; micro/macro/per-class |
| Class-agnostic 2D AP + NHD 3D surrogate | `tools/class_agnostic_eval.py` | Meaningful zero-shot number when standard AP is 0 |
| `make_report.py` with `--compare` | new | Single-command multi-row paper table; parses log.txt + log.rel.txt + bev_ap.json |
| `run_full_eval.sh` | new | One-shot: standard eval + Rel-AP3D + BEV + class-agnostic + report |
| OVMono3D-faithful novel-view visualizer | `tools/visualize_class_agnostic.py` | Separate 2D/3D/novel-view panels, per-id colors, ground grid toggle |
| Training-curves plotter with key discovery | `tools/plot_training.py` | Robust to metric-key naming drift across detectron2 versions |
| `remap_wildbox_paths.py` | new | Path rescue when data moves, preserves split |
| Multi-seed launcher + aggregator | `tools/run_multi_seed.sh` + `tools/aggregate_seed_ap.py` | 3-seed mean±std for rare classes |
| Rel-AP3D boundary checker | `tools/check_rel_ap3d_boundary.py` | Confirms scale-search grid is adequate (reviewer #2) |
| Per-block Rel-AP3D scale search + progress prints | `cubercnn/evaluation/omni3d_evaluation.py:search_rel_scale` | Old flat-list approach was O(N×M) per scale — never finished on 13k val images |
| dataset_id → contiguous_id normalization in evaluators | bev_ap_eval + class_agnostic_eval + omni3d_evaluation | Handles both fine-tuned (contiguous) and ORACLE2D=True (dataset-id) prediction formats |
| Training-safe vis try/except | `tools/train_net.py:do_test` | Vis exceptions never kill training |
| Zero-shot evaluator short-circuit | `omni3d_evaluation.py` | "zero in-vocab predictions" logs a warning instead of crashing |
| `resume=False` forces `start_iter=0` | `tools/train_net.py:183` (commit f894ab6) | Pretrained checkpoint's `iteration=115999` no longer silently skips training |

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

## 15. Final-run operations guide

**See [FINAL_RUN.md](FINAL_RUN.md)** — it's the canonical linear pipeline from prereqs to paper-ready tables and figures, organized as 13 copy-pasteable steps with expected outputs at each.

The pre-flight checklist (`tmux` + env + checkpoint + stats.json + symlinks + split integrity) is FINAL_RUN.md §0-§5. The three-row paper report (ORACLE2D=False zero-shot / GDino-oracle zero-shot / fine-tuned multi-seed) is §8-§10. The single gotcha worth re-flagging here: if `grep -E "Starting training from iteration" log.txt` says anything other than `iteration 0 (resume=False)`, training was silently skipped — see §3.1.

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

**Primary references** — use these, not duplicates of their content here:
- **[FINAL_RUN.md](FINAL_RUN.md)** — the canonical 13-step pipeline (data prep → training → eval → report → figures). Every command is a single copy-paste block.
- **[QUICK_START_SMOKE_TEST.md](QUICK_START_SMOKE_TEST.md)** — smallest-zip-per-species end-to-end smoke in ~2 hours.
- **[§4](#4-prep--train--eval-pipeline-end-to-end)** — same pipeline with per-step rationale.

Only the utilities that aren't in those three docs live here:

```bash
# Resume a stopped training run (checkpoint is at OUTPUT_DIR/last_checkpoint)
python tools/train_net.py --config-file <cfg> --num-gpus 1 --resume \
    OUTPUT_DIR output/your_run

# Remap paths after moving the data directory (no re-prep required)
python tools/remap_wildbox_paths.py datasets/Omni3D/WildBox_val.json \
    [--map OLD=NEW | --search-root /path/to/new/data] --in-place

# Which GPU is your training job on?
YOUR_PID=$(pgrep -f train_net.py | head -1)
nvidia-smi --query-compute-apps=pid,gpu_name,used_memory --format=csv | grep "^$YOUR_PID,"

# Tail training loss + ETA from anywhere
tail -n 5 output/<run_dir>/log.txt | grep iter
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
| Data sources | 11/12 zips (missing: data202401KR, data202401KE, data202602KE) | For fairness tracking |
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
- **Zero-shot 2D detector** — use the same `gdino_WildBox_val_oracle_2d.json` across every architecture (see §20.6). Otherwise zero-shot numbers aren't comparable.

Do vary (and track in §19's template):

- The architecture itself
- Loss weights (as required by the architecture)
- Backbone pretraining source
- LR schedule (fit to the architecture)

### 20.6 Paper-protocol zero-shot across architectures (the portable contract)

The OVMono3D paper reports zero-shot 3D by pairing **one open-vocab 2D detector (GroundingDINO)** with each method's 3D head. For cross-architecture comparison, this is the single most important thing to hold fixed. The interchange format is simple: precompute GDino boxes once, hand the resulting JSON to every architecture.

**What "paper protocol zero-shot" means concretely:**

1. Take a pretrained (Omni3D-trained, **not** WildBox-finetuned) checkpoint of architecture X.
2. At eval time, **replace X's 2D proposals with GDino's text-prompted detections** for the WildBox species.
3. Let X's 3D head lift those 2D boxes to 3D.
4. Score with our standard eval stack (2D AP, 3D AP, Rel-AP3D, BEV AP, NHD).

This measures *"how well does X's 3D lift onto an unseen visual domain, given that open-vocab 2D localization is solved for us?"* — the exact question the paper asks.

**Interchange JSON** — produce once via [tools/precompute_gdino_oracle.py](tools/precompute_gdino_oracle.py) (§6.4.3). Every architecture consumes the same file:

```
datasets/Omni3D/gdino_WildBox_val_oracle_2d.json
```

Schema is in §6.4.3 — one list entry per image with `image_id`, `K`, `file_path`, `height`, `width`, and a list of `instances` (2D xywh box, score, dataset-space `category_id`, `category_name`). No 3D info.

**How each architecture should consume it at eval time.** Wire a hook in the architecture's test-time inference that, for each image:

- **Instead of** running the architecture's own 2D stage (RPN, anchor head, DETR queries, whatever), **load the oracle boxes for that `image_id`**.
- Map `category_id` from dataset-space to the architecture's contiguous-id space (same mapping you use during training — see §2.8).
- Feed the loaded boxes into the architecture's 3D head / ROI pool / cuboid regressor.

In OVMono3D (Cube R-CNN variant) this is already a built-in flag: `TEST.ORACLE2D=True` + `DATASETS.ORACLE2D_FILES` — see [configs/wildbox/OVMono3D_wildbox_wildlife5_oracle2d.yaml](configs/wildbox/OVMono3D_wildbox_wildlife5_oracle2d.yaml). For a different architecture you may need to add a few lines to its eval loop that load the JSON, rescale boxes to the architecture's preprocessing resolution, and hand them to whatever module normally receives 2D proposals.

**Reporting table for cross-arch zero-shot comparison.** Every row uses the same oracle file:

| Arch | Backbone | 3D head type | 2D source | AP_BEV | Rel-AP3D | NHD |
|---|---|---|---|---:|---:|---:|
| OVMono3D (ours) | DINOv2 ViT-B/14 | Cube R-CNN | GDino oracle | ... | ... | ... |
| Cube R-CNN | ResNet-50 | Cube R-CNN | GDino oracle | ... | ... | ... |
| DetAny3D | ? | DINO-decoder | GDino oracle | ... | ... | ... |

**Corresponding fine-tuned comparison** (the complementary half of the paper story) uses each architecture's own 2D stage, no oracle — because at this point the 2D head is no longer the limitation; the architecture has seen WildBox:

| Arch | Backbone | 3D head type | 2D source | AP_BEV | Rel-AP3D | NHD |
|---|---|---|---|---:|---:|---:|
| OVMono3D (ours) | DINOv2 ViT-B/14 | Cube R-CNN | *own RPN* (finetuned) | ... | ... | ... |
| Cube R-CNN | ResNet-50 | Cube R-CNN | *own RPN* (finetuned) | ... | ... | ... |

**Additional report column — `ORACLE2D=False` zero-shot (closed-vocab RPN transfer).** Our repo defaults to this stricter baseline (see §6.4.2) because it doesn't require the GDino oracle. Keep it in the paper supplement as an *extra* zero-shot row — measures whether the pretrained RPN alone generalizes to wildlife, with no open-vocab help. Class-agnostic 2D AP here is the most honest number; standard AP is near-zero because pretraining's class slots don't overlap with WildBox.

**Checklist for adding architecture X to the comparison:**

1. Run the GDino precompute (§6.4.3) exactly once — output goes to `datasets/Omni3D/gdino_WildBox_val_oracle_2d.json`. Every downstream arch reads this file; do **not** re-precompute per architecture.
2. Implement the "load oracle boxes instead of RPN proposals" hook in X's eval loop (the amount of code depends on X — for DETR-style it may be a 10-line replacement of the decoder's query input; for two-stage it's replacing the RPN proposal list).
3. Confirm X's contiguous-id mapping matches ours ({giraffe:0, zebra:1, elephant:2, rhino:3, gazelle:4} — sorted by dataset-id ascending per §2.8.1).
4. Run zero-shot eval → produce `instances_predictions.pth` in the schema from §20.3.
5. Point our `tools/bev_ap_eval.py`, `tools/class_agnostic_eval.py`, `tools/make_report.py` at X's prediction file. Report row appears alongside ours automatically via `--compare`.

That's the whole contract. If X's zero-shot row is much worse than ours and it's **not** using the oracle, the gap is confounded — X's own 2D stage may just be weaker than GDino on this domain. Paper claims about the 3D head need oracle input on both sides.

---

---

## 21. Current state (as of 2026-04-24)

Single-paragraph snapshot for resume after compaction: **6-species dataset live (plains_zebra + grevys_zebra split from KABR), 15 zips total, 64 videos, 60k frames, 237k instances, video-level 80/20 split. Multi-seed training (3 seeds × 15k iters × REPEAT_THRESHOLD=0.5) running now after `f894ab6` fixed the silent-skip-training bug.** Results TBD ~18:30 tonight.

### 21.1 What's in place (code + data)

- **Dataset prep** (`tools/prepare_wildbox_dataset.py`): 15-zip 6-species corpus, SAM3-tight 2D, video-level split. All 15 archive dirs under `/storage3/3DOM/vshukla/sam3/wd_data/wildbox/archive/`; unzipped dirs cached alongside.
- **Dataset stats auto-generator** (`tools/dataset_stats.py`): produces `dataset_stats.md` + `.json` + `size_distribution.png` every prep.
- **6-species config** ([OVMono3D_wildbox_wildlife6.yaml](configs/wildbox/OVMono3D_wildbox_wildlife6.yaml)): `MODEL.ROI_HEADS.NUM_CLASSES: 6`, `REPEAT_THRESHOLD: 0.5`, `MAX_ITER: 15000`, `REL_AP3D_SEARCH: [0.05, 3.0, 32]`. Symlinks `configs/category_meta.json` and `configs/wildbox/category_meta.json` both point to `category_meta_wildlife6.json`.
- **Paper-protocol zero-shot pipeline**: GDino oracle JSON at `datasets/Omni3D/gdino_WildBox_val_oracle_2d.json` (2.5h precompute, ~0.7 s/img on A40), evaluator wired via [OVMono3D_wildbox_wildlife5_oracle2d.yaml](configs/wildbox/OVMono3D_wildbox_wildlife5_oracle2d.yaml). Will need regeneration when val set changes (current oracle file reflects 6222 val images from the 11-zip 5-species prep, not the 13779-image 15-zip 6-species val).
- **Multi-seed infrastructure**: `tools/run_multi_seed.sh` (3 seeds sequential, resumable), `tools/aggregate_seed_ap.py` (mean±std for rare classes).
- **Eval infrastructure**: `tools/run_full_eval.sh` covers standard AP + Rel-AP3D + BEV + class-agnostic + NHD; `tools/make_report.py --compare` assembles multi-row tables.

### 21.2 Bugs caught and fixed during development

All fixed in the current codebase (don't re-introduce):

| # | Symptom | Root cause | Fix | Commit |
|---|---|---|---|---|
| 1 | Per-class metrics all 0 except giraffe during standalone eval | `train_net.py --eval-only` reads `category_meta.json` from the **config file's dir**, not `configs/`. Stale Phase-1 single-class file was winning | Symlink both `configs/category_meta.json` AND `configs/wildbox/category_meta.json` to `category_meta_wildlife6.json`. `run_full_eval.sh` auto-syncs both at start | pre-history |
| 2 | Rel-AP3D stuck forever | `search_rel_scale` built one giant `box3d_overlap(N×M)` matrix per scale × 28 scales | Per-(image, category) block IoU with progress prints | pre-history |
| 3 | BEV AP showed `0 preds, 0 GT` per class for ORACLE2D=True | Eval filtered by contiguous-id but oracle preds carry dataset-ids 1000-1005 | Normalize dataset-id → contiguous-id in `pred_bev_by_img` | d8547ed |
| 4 | Per-class AP macro=0, micro=88 under oracle eval | Same dataset-id/contiguous-id mismatch in `class_agnostic_eval.py` and main COCO evaluator | Same normalization pattern in both | e7de1e6 |
| 5 | Make_report's Rel-AP_3D row always `-` | Parser only read `log.txt`, not `log.rel.txt` | Concat both before regex | 1da8843 |
| 6 | GDino oracle boxes had median IoU 0.10 vs GT | `preprocess_image` skipped GDino's `RandomResize([800], max_size=1333)` | Use `groundingdino.util.inference.load_image` directly | e5bf1c9 |
| 7 | Multi-seed training produced 0 iter lines, `model_final.pth` byte-identical to pretrained (silent "no-training" failure) | `train_net.py:183` read checkpoint's `iteration` field unconditionally regardless of `resume=False`. Pretrained has `iteration=115999`; `start_iter>MAX_ITER` → training loop exited immediately | Respect `resume` flag when reading iteration | f894ab6 |

Bug #7 is the silent killer to remember. See §3.1 for the four red flags that should trigger a "training-was-skipped" diagnosis within 2 minutes of launch.

### 21.3 Post-retrain checklist (morning of 2026-04-25)

When the three-seed wildlife6 run completes:

```bash
# 1) Sanity — all 3 seeds actually trained (should be NON-empty counts)
for S in 0 1 2; do
    D=output/wl6_rt0.5_multiseed/seed$S
    printf "seed%d: " $S
    grep -cE "iter: [0-9]+" $D/log.txt
done
# Expect: seed0: ~750, seed1: ~750, seed2: ~750 (at default d2 log every 20 iters)

# 2) Per-seed model_final.pth sizes should DIFFER from the 603,232,016-byte pretrained
stat -c "%s %n" output/wl6_rt0.5_multiseed/seed*/model_final.pth checkpoints/ovmono3d_lift.pth

# 3) Aggregate mean±std
python tools/aggregate_seed_ap.py \
    --run-dirs output/wl6_rt0.5_multiseed/seed{0,1,2}/eval \
    --rare-classes giraffe gazelle grevys_zebra \
    --classes giraffe grevys_zebra elephant plains_zebra rhino gazelle \
    --out output/wl6_rt0.5_multiseed/mean_std_report

# 4) Three-row paper report (once oracle JSON is regenerated for the 13779-image val):
#    See FINAL_RUN.md §9 for the regen command + §10 for the assembly.
```

### 21.4 Known pending work (paper v2)

- **Zebra species audit** — current split assumes "KABR = Grévy's, all others = plains" based on user knowledge. For paper v1 this is adequate. For paper v2, verify per-video with a domain expert.
- **Giraffe val is thin** — 1 val video, 110 boxes. Per-class giraffe metrics will have high variance regardless of training-seed variance. Flag in paper limitations section; no fix available without more giraffe videos.
- **Train/val size mismatch on giraffe** — train median bbox 38k px² vs val median 112k px² (val has closer animals). Known hazard of video-level splits with ≤4 videos per class. Stabilizes as we add more giraffe data.
- **pytorch3d CUDA build** remains blocked by cluster glibc on CUDA 12.2+. `search_rel_scale` forces CPU tensors — permanent state.

---

_End of document. Any new experiment, hyperparameter change, or code fix should be reflected here (especially §13 changelog and §17 file map) to keep the docs in sync with the code._
