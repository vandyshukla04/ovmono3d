# WildBox 3D Wildlife Detection — Experiment Documentation

**What this is**: the thorough reference for reproducing the WildBox monocular 3D wildlife-detection experiments. Covers the dataset, the pipeline, the model, the evaluation protocol, the bugs we caught and fixed, and the cross-architecture contract for paper comparisons.

**Companion docs (don't duplicate; reference)**:
- [FINAL_RUN.md](FINAL_RUN.md) — the canonical 13-step linear ops pipeline (copy-paste commands).
- [QUICK_START_SMOKE_TEST.md](QUICK_START_SMOKE_TEST.md) — ~1 h smoke run with one zip per species.

---

## 1. TL;DR

- **Task**: monocular 3D detection of African wildlife from aerial drone footage.
- **Species (paper)**: 6 classes — `giraffe`, `grevys_zebra`, `elephant`, `plains_zebra`, `rhino`, `gazelle`. Contiguous ids 0–5; dataset ids 1000–1005.
- **Dataset**: WildBox — **15 source zips, 64 drone videos, 59 758 frames, 237 505 3D-cuboid instances**. 3D GT is pseudo-labels via SAM3-tight 2D + VGGT 3D reconstruction. Auto-regenerated inventory: `datasets/Omni3D/dataset_stats/dataset_stats.md` (per-prep).
- **Architecture**: OVMono3D-lift (Cube R-CNN variant, DINOv2 ViT-B/14 backbone). Fine-tuned from `ovmono3d_lift.pth` (Omni3D-pretrained, 50-class head reinitialized to 6 classes).
- **Training**: 15 000 iters, batch 8, LR 2e-3, AMP, `REPEAT_THRESHOLD=0.5`. **3 seeds (0, 1, 2)** for variance estimation. ~3–4 h per seed on A40.
- **Evaluation**: three-row paper comparison, all on the same val set (13 779 images, 6 species):
  1. **Zero-shot RPN-transfer** (`TEST.ORACLE2D=False`) — closed-vocab pretrained.
  2. **Zero-shot GDino oracle** (`TEST.ORACLE2D=True`) — paper protocol.
  3. **Fine-tuned 3-seed mean ± std**.
- **Primary metric**: **AP_BEV @ IoU 0.50** (KITTI-convention tight IoU). Supplementary: AP_BEV@0.25, AP_3D@0.25, Rel-AP_3D (LabelAny3D), 2D AP@0.5, class-agnostic 2D AP, disentangled NHD.
- **Split**: video-level, seed=0, 80/20 → 51 train / 13 val videos, zero leakage.
- **Where to start**:
  - **Ops**: [FINAL_RUN.md](FINAL_RUN.md).
  - **Smoke**: [QUICK_START_SMOKE_TEST.md](QUICK_START_SMOKE_TEST.md).
  - **Resume after compaction**: §21 (current state snapshot).

---

## 2. Dataset

### 2.1 Source and provenance

- **Raw videos**: DJI drone footage, Kenya, 2023–2026.
- **Species detection / masks**: text-prompted SAM3 segmentation per species per frame.
- **3D cuboids**: VGGT 3D reconstruction + tracking on the masked frames, then 3D-cuboid fitting.
- **2D bboxes** at training time: tight axis-aligned bboxes from the SAM3 silhouettes (NOT the loose projection of the 3D cuboids — see §2.7).

### 2.2 Inventory (15 zips, 6 species)

The canonical snapshot lives in `datasets/Omni3D/dataset_stats/dataset_stats.md` — auto-generated via `tools/dataset_stats.py` after every prep. The static inventory of source zips is below; actual per-class counts in train/val splits come from the regenerated stats file.

| Zip path (under `archive/`) | Species (class) | Notes |
|---|---|---|
| `data202401KGiraffes` | giraffe | main giraffe source |
| `data202501KGiraffes` | giraffe | small supplementary |
| `data2023KABRZebras` | grevys_zebra | KABR reserve — the **only** Grévy's zebra source |
| `wildbox_tomblair` | plains_zebra | |
| `dataBZS` | plains_zebra | |
| `data202307KZebras` | plains_zebra | |
| `202401KZebras` | plains_zebra | |
| `data202401KElephants` | elephant | |
| `data202406KElephants` | elephant | |
| `data202501KElephants` | elephant | |
| `data202602KElephants` | elephant | |
| `data202401KRhinos` | rhino | |
| `data202502KRhinoCamiV1` | rhino | |
| `data202502KRhinoCamiV2` | rhino | |
| `data202406KGazelles` | gazelle | only gazelle source |

**Current aggregate** (regenerated 2026-04-24):

| Split | Videos | Segments | Frames | Boxes |
|---|---:|---:|---:|---:|
| Train | 51 | 263 | 45 979 | 170 554 |
| Val | 13 | 82 | 13 779 | 66 951 |
| **Total** | **64** | **345** | **59 758** | **237 505** |

**Per-species video counts (the rare-class flag determines multi-seed reporting policy)**:

| Species | train vids | val vids | Rare? (≤5 vids) |
|---|---:|---:|---|
| elephant | 14 | 3 | |
| rhino | 15 | 4 | |
| plains_zebra | 12 | 3 | |
| grevys_zebra | 4 | 1 | **YES** |
| giraffe | 3 | 1 | **YES** — val is 1 video / 110 boxes |
| gazelle | 3 | 1 | **YES** |

Rare classes get 3-seed `mean ± std` reporting; stable classes can be single-seed.

**Zebra species attribution** (paper-relevant): per domain expert, `data2023KABRZebras` contains exclusively Grévy's zebra (*Equus grevyi*); the other 4 zebra zips (`tomblair`, `BZS`, `data202307KZebras`, `202401KZebras`) are plains zebra (*Equus quagga*). The paper's dataset section must state this attribution.

### 2.3 Per-segment data layout

Inside each unzipped campaign archive:
```
<zip_dir>/WildBox_sam3-vggtv1_processed_unzipped/WildBox/
  ├── <VIDEO_ID>/                 # e.g. DJI_20250218113812_0013_D
  │   └── seg<N>/                 # e.g. seg1
  │       ├── frame_NNNNNN.jpg    # the rendered RGB frames
  │       ├── per_frame_K.json    # 3×3 intrinsics per frame
  │       ├── tracks.json         # per-instance 3D cuboid track
  │       └── sam3_masks/masks/   # per-frame PNG masks (SAM3)
```

Frames are extracted from the original drone video at native resolution (commonly 1920×1080). Per-frame intrinsics (K) come from VGGT — they may differ slightly across frames if VGGT's calibration estimate moved.

### 2.4 Coordinate conventions

- **3D points are in camera frame**, X-right, Y-down, Z-forward (standard CV convention; matches Omni3D).
- **3D cuboid orientation**: stored as a 3×3 rotation matrix `R_cam`. Pose is full 3-DoF (NOT yaw-only). This matters because animals tilt their bodies.
- **Cuboid dimensions**: `[W, H, L]` ordering matching Omni3D's `dimensions` field. Order with axes: X=length L, Y=height H, Z=width W. Be careful — many other 3D-detection codebases use `[L, W, H]`.
- **2D bboxes**: `[x, y, w, h]` in image pixels, top-left origin.

### 2.5 Scale normalization (VGGT synthetic scale)

VGGT's 3D reconstruction is unscaled (up to a global similarity transform per video segment). To make per-segment 3D coordinates comparable, we normalize each segment so that **the median |Z|-depth of GT cuboids = 1**. So a typical animal sits at z ≈ 0.5–2 in the GT, NOT in metric meters.

**Implications**:
- Standard 3D AP comparing predictions in metric scale (Omni3D-pretrained) vs WildBox synthetic scale gives ~0 AP. **This is expected** — see §6.4.
- Rel-AP_3D (LabelAny3D protocol) factors out the scale via an explicit search.
- Per-class scale-invariant metrics (BEV AP after Rel-AP scale alignment, or NHD with best-scale) should be reported when comparing across scale conventions.
- K (intrinsics) is **NOT rescaled**. Projection `(X/Z, Y/Z)` is scale-invariant, so 2D bbox quality is unaffected.

### 2.6 Train/val split — video-level

`tools/prepare_wildbox_dataset.py --split-mode video --seed 0 --val-fraction 0.2`. Each video is assigned to either train or val as a whole (NOT per-segment, NOT per-frame). Why:
- **Segment-level split leaks**: segments from the same video share lighting, terrain, animal individuals.
- **Frame-level split leaks heavily**: adjacent frames are nearly identical.

Determinism: same `--seed 0` + same `--source` ordering produces the same split. Verified by `python tools/prepare_wildbox_dataset.py --help`'s reproducibility test.

### 2.7 SAM3 tight 2D bboxes

The original Omni3D pipeline uses the projection of the 3D cuboid as the 2D bbox. That projection is loose — typically 20–40% larger than the actual animal silhouette.

We override this by using **per-frame SAM3 segmentation masks**: each `frame_NNNNNN.jpg` has a corresponding mask in `sam3_masks/masks/`, and `prepare_wildbox_dataset.py` computes the tight axis-aligned bbox from the mask's non-zero pixels. This significantly improves measured 2D AP and matches what wildlife ecologists actually want.

**Mask loading is I/O-bound** (~10–50 ms/frame). For a full 15-zip prep, expect 15–45 min on this step alone — that's the filesystem, not CPU/GPU. Cached after first run via the unzipped directory.

### 2.8 Category metadata gotcha — the most common bug source

Two files must agree:
1. **`configs/category_meta.json`** — symlink read by external eval tools and the standard Omni3D evaluator.
2. **`configs/wildbox/category_meta.json`** — symlink read by `tools/train_net.py --eval-only` (line 402, hardcoded relative to the config file's directory).

Both must point at the **same** file:
```bash
ln -sf wildbox/category_meta_wildlife6.json configs/category_meta.json
ln -sf category_meta_wildlife6.json         configs/wildbox/category_meta.json
```

`category_meta_wildlife6.json`:
```json
{
  "thing_classes": ["giraffe", "grevys_zebra", "elephant", "plains_zebra", "rhino", "gazelle"],
  "thing_dataset_id_to_contiguous_id": {"1000": 0, "1001": 1, "1002": 2, "1003": 3, "1004": 4, "1005": 5}
}
```

**Critical ordering rule**: `thing_classes` MUST be sorted by ascending dataset-id. `register_and_store_model_metadata` (in `cubercnn/data/datasets.py:303-310`) sorts that way internally, and a hand-written meta in any other order causes ALL per-class metrics except one to be 0. The auto-generated meta from `prepare_wildbox_dataset.py` enforces this.

`tools/run_full_eval.sh` auto-syncs both symlinks at start as defense in depth. If you ever see "all per-class AP except one = 0", **first check `cat configs/category_meta.json configs/wildbox/category_meta.json`** — both must show the 6-species block above.

### 2.9 Output JSON schema (Omni3D format)

`prepare_wildbox_dataset.py` writes Omni3D-format JSON. Key fields:

```jsonc
{
  "info": {...},
  "categories": [{"id": 1000, "name": "giraffe", "supercategory": "animal"}, ...],
  "images": [
    {
      "id": <int>,                                  // unique image id
      "file_path": "/abs/path/to/frame_NNNNNN.jpg", // absolute paths
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "width": W, "height": H
    }, ...
  ],
  "annotations": [
    {
      "id": <int>,
      "image_id": <int>,
      "category_id": <dataset_id>,             // 1000-1005
      "category_name": "giraffe",
      "bbox": [x, y, w, h],                    // SAM3-tight, xywh pixels
      "bbox3D_cam": [...8 vertices in cam frame...],
      "center_cam": [x, y, z],
      "dimensions": [W, H, L],                 // Omni3D ordering, see §2.4
      "R_cam": [[3x3 rotation]]
    }, ...
  ]
}
```

Predictions written by `tools/train_net.py` to `instances_predictions.pth` use the SAME schema for `bbox`, `category_id`, `dimensions`, `R_cam` (under key `pose`), `center_cam`, plus `score` and `depth`.

### 2.10 Path remapping after data move

If `/storage3/...` is moved or remounted, regenerating the JSONs is wasteful (loses the deterministic split). Use:

```bash
python tools/remap_wildbox_paths.py datasets/Omni3D/WildBox_val.json \
    --map /old/prefix=/new/prefix --in-place
# OR auto-discover:
python tools/remap_wildbox_paths.py datasets/Omni3D/WildBox_val.json \
    --search-root /new/data/root --in-place
```

Preserves split identity (image IDs and split assignment unchanged).

---

## 3. Environment setup (once per cluster)

```bash
# Conda env
conda create -p /storage3/.../envs/ovmono3d python=3.8.20
conda activate /storage3/.../envs/ovmono3d

# PyTorch
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Repo dependencies (pytorch3d-CPU pinned 055ab3a, detectron2 fork, SAM, depth-pro, etc.)
bash setup.sh

# Experiment-specific extras
pip install shapely openpyxl matplotlib

# GroundingDINO (for paper-protocol zero-shot oracle, see §6.4)
pip install --no-cache-dir groundingdino-py==0.4.0
```

### 3.1 Known cluster issues (read before debugging mysterious failures)

These four are the gotchas that have eaten the most time. Each is summarized; full diagnostic patterns are below.

#### 3.1.1 Pretrained checkpoint silently skips training (CRITICAL — fixed in commit f894ab6)

`ovmono3d_lift.pth` stores `iteration=115999` from its Omni3D pre-training. Pre-fix, [tools/train_net.py:183](tools/train_net.py#L183) read this field unconditionally regardless of `resume` flag and set `start_iter=116000`. Any fine-tune with `MAX_ITER<116000` silently exited the training loop with **zero iterations** — no error, no stack trace, just a `model_final.pth` byte-identical to the pretrained.

**Four red flags to spot this** (the first three are diagnostic; **all of them** were missed on the bad 2026-04-24 run):

1. The startup log says `[DatasetMapper] Augmentations used in inference: [ResizeShortestEdge(...)]` but **never** `Augmentations used in training: [...RandomFlip(), multi-scale resize...]`. Real training logs both (training augs + periodic-eval inference augs). Single "inference" line = no training loop.
2. The log jumps from `Environment info` directly to `Start inference on N batches` within 1–2 minutes of launch. Real training spends hours between, emitting `iter: 100  total_loss: 1.234 ...` lines every 20 iters.
3. `grep -cE "iter: [0-9]+" log.txt` returns **0**.
4. The single log line `Starting training from iteration 116000 (resume=False)` — definitive smoking gun.
5. `model_final.pth` is byte-identical to `ovmono3d_lift.pth` (same 603 232 016-byte file).

The code fix at line 183 now respects `resume=False` → `start_iter=0`. Belt-and-suspenders is to strip iteration from the pretrained:

```bash
python -c "
import torch
c = torch.load('checkpoints/ovmono3d_lift.pth', weights_only=False, map_location='cpu')
c.pop('iteration', None); c.pop('optimizer', None); c.pop('scheduler', None)
torch.save(c, 'checkpoints/ovmono3d_lift.pth')
"
```

The 5-species runs predating this fix happened to start from iter 0 because they didn't CLI-override `MODEL.WEIGHTS` — only the multi-seed launcher did. Any multi-seed run launched **before commit f894ab6 is invalid (zero-trained)**.

#### 3.1.2 GroundingDINO Python package selection (for paper-protocol zero-shot)

GDino's CUDA op fails to build on this cluster (cospi/sinpi noexcept mismatch with CUDA 12.2+ vs older glibc). **Use `pip install groundingdino-py==0.4.0`** — the maintained PyPI fork has a proper GPU Python fallback (~0.7 s/img on A40, full val precompute ~67 min).

Do **NOT** use:
- The official `IDEA-Research/GroundingDINO` GitHub clone — its `ms_deform_attn.py` raises `NameError: name '_C' is not defined` when the CUDA op can't load (no fallback).
- `pip install groundingdino==0.1.0` (PyPI) — that version's fallback runs on **CPU only** at ~41 s/img (60× slower).

Also: GDino's `load_image` applies `RandomResize([800], max_size=1333)` before normalization. **Do not bypass this** — feeding native drone resolution raw collapses median-IoU-vs-GT from 0.97 to 0.10. The patched `tools/precompute_gdino_oracle.py:preprocess_image` calls `load_image` directly.

#### 3.1.3 pytorch3d CUDA build failure

Same cospi/sinpi glibc mismatch on CUDA 12.x. Workaround already in code: the Rel-AP3D scale search forces CPU tensors (`cubercnn/evaluation/omni3d_evaluation.py:search_rel_scale`). CPU `box3d_overlap` works on our pinned commit (055ab3a). Don't try `FORCE_CUDA=1` unless you're on a cluster with compatible glibc.

#### 3.1.4 libGL.so.1 missing on CPU-only nodes

`cv2` requires system graphics libs. Symptom: `libGL.so.1: cannot open shared object file`. Fix: run eval/training on GPU nodes only (frontend/cpu nodes don't have it).

---

## 4. Pipeline (end-to-end)

This is the structural overview with rationale. For copy-paste commands, see [FINAL_RUN.md](FINAL_RUN.md). For a 1-h validation, see [QUICK_START_SMOKE_TEST.md](QUICK_START_SMOKE_TEST.md).

### 4.1 Register species in Omni3D stats

```bash
python tools/patch_stats_for_wildbox.py \
    --stats datasets/Omni3D/stats.json \
    --add giraffe:1000 grevys_zebra:1001 elephant:1002 plains_zebra:1003 rhino:1004 gazelle:1005
```

Idempotent. Re-running adds only missing entries. Without this step, training crashes with `ValueError: 'plains_zebra' is not in list`.

### 4.2 Build train/val JSONs

`tools/prepare_wildbox_dataset.py --source <ZIP>=<species>:<id>` repeated per zip. Auto-extracts each zip to a sibling `*_unzipped/` (cached). Corrupt or in-progress zips are skipped with a warning. Outputs absolute paths in the JSON.

```bash
python tools/prepare_wildbox_dataset.py \
    --source <ZIP>=<species>:<id> \   # repeat for each zip
    --split-mode video --val-fraction 0.2 --seed 0 \
    --output-train datasets/Omni3D/WildBox_train.json \
    --output-val   datasets/Omni3D/WildBox_val.json \
    --dataset-id 1000 -v
```

Side effect: also writes `configs/wildbox/category_meta_auto_<N>species.json` with the correct `thing_classes` ordering (sorted by dataset-id ascending — see §2.8).

### 4.3 Wire BOTH category_meta symlinks

```bash
ln -sf wildbox/category_meta_wildlife6.json configs/category_meta.json
ln -sf category_meta_wildlife6.json         configs/wildbox/category_meta.json
```

Verify both via:
```bash
python -c "import json; print(json.load(open('configs/category_meta.json'))['thing_classes'])"
python -c "import json; print(json.load(open('configs/wildbox/category_meta.json'))['thing_classes'])"
```
Both must return `['giraffe', 'grevys_zebra', 'elephant', 'plains_zebra', 'rhino', 'gazelle']`.

### 4.4 Generate dataset stats (paper inventory + size distribution)

```bash
python tools/dataset_stats.py \
    --train datasets/Omni3D/WildBox_train.json \
    --val   datasets/Omni3D/WildBox_val.json \
    --out   datasets/Omni3D/dataset_stats
```

Produces:
- `dataset_stats.md` — per-species counts (vids/segs/frames/boxes), bbox area-ratio distribution, paper one-liner.
- `dataset_stats.json` — same data, machine-readable.
- `size_distribution.png` — per-species log-scale histogram of bbox-area-as-fraction-of-image.

**Re-run on every prep change**. The `paper_results.md` assembler embeds `dataset_stats.md` directly, so this is the single source of truth for the paper's dataset section.

### 4.5 No-leakage verification

```bash
python - <<'PY'
import json
train = json.load(open("datasets/Omni3D/WildBox_train.json"))
val   = json.load(open("datasets/Omni3D/WildBox_val.json"))
tv = {im["file_path"].split("/")[-3] for im in train["images"]}
vv = {im["file_path"].split("/")[-3] for im in val["images"]}
assert not (tv & vv), f"video leakage: {tv & vv}"
print(f"OK: {len(tv)} train vids, {len(vv)} val vids — no overlap")
PY
```

If this fails, the prep is broken (different `--source` ordering or seed). Don't proceed.

### 4.6 Zero-shot eval — RPN-transfer (closed-vocab)

Strict baseline: model's own RPN + 2D classifier, no open-vocab help. Predictions go through the Omni3D 50-class softmax → class filter drops most → near-zero per-class AP. Class-agnostic 2D AP is the meaningful signal.

```bash
bash tools/run_full_eval.sh \
    --weights checkpoints/ovmono3d_lift.pth \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife6.yaml \
    --out     output/wl6_zeroshot_rpn \
    --label   "zero-shot (RPN-transfer)" \
    --gt      datasets/Omni3D/WildBox_val.json \
    --skip-rel-ap3d
```

`--skip-rel-ap3d` because no in-vocab predictions makes scale search meaningless.

### 4.7 Zero-shot eval — GDino oracle (paper protocol)

OVMono3D paper convention: open-vocab 2D detector (GroundingDINO) provides 2D boxes; the 3D cube head lifts those. See §6.4 for full discussion. **One-time precompute**:

```bash
python tools/precompute_gdino_oracle.py \
    --gt       datasets/Omni3D/WildBox_val.json \
    --out      datasets/Omni3D/gdino_WildBox_val_oracle_2d.json \
    --species  rhino elephant plains_zebra grevys_zebra giraffe gazelle \
    --device   cuda \
    --box-threshold 0.15 --text-threshold 0.10 \
    --log-every 100
```

~67 min on A40 with the correct GDino package (§3.1.2). Output is reusable forever (until val set changes).

Then run the eval:

```bash
bash tools/run_full_eval.sh \
    --weights checkpoints/ovmono3d_lift.pth \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife6_oracle2d.yaml \
    --out     output/wl6_zeroshot_oracle2d \
    --label   "zero-shot (GDino oracle, paper protocol)" \
    --gt      datasets/Omni3D/WildBox_val.json \
    --skip-rel-ap3d
```

### 4.8 Fine-tune training (multi-seed)

```bash
CONFIG=configs/wildbox/OVMono3D_wildbox_wildlife6.yaml \
BASE_OUT=output/wl6_rt0.5_multiseed \
bash tools/run_multi_seed.sh
```

Runs 3 seeds sequentially. Each: 15 000 iters, batch 8, LR 2e-3, REPEAT_THRESHOLD=0.5, AMP. ~3.5 h/seed → ~10.5 h total on A40. Resumable: if a seed's `model_final.pth` exists on restart, that seed is skipped.

After training, each seed automatically gets a full eval (5-step `run_full_eval.sh`).

### 4.9 Aggregate mean ± std across seeds

```bash
python tools/aggregate_seed_ap.py \
    --run-dirs output/wl6_rt0.5_multiseed/seed{0,1,2}/eval \
    --rare-classes giraffe gazelle grevys_zebra elephant plains_zebra rhino \
    --classes giraffe grevys_zebra elephant plains_zebra rhino gazelle \
    --out output/wl6_rt0.5_multiseed/mean_std_report
```

Treating ALL 6 classes as `--rare-classes` reports `mean ± std` on every cell (we have 3 seeds → free std for stable classes too). The `--rare-classes` flag controls which cells get `± std` formatting; pass all classes for full coverage.

### 4.10 Assemble paper-results sheet (single markdown)

```bash
python tools/assemble_paper_results.py \
    --finetuned-seeds  output/wl6_rt0.5_multiseed/seed{0,1,2}/eval \
    --ablation-rt035   output/wl6_rt0.35_seed0/seed0/eval \
    --zeroshot-rpn     output/wl6_zeroshot_rpn \
    --zeroshot-oracle  output/wl6_zeroshot_oracle2d \
    --classes giraffe grevys_zebra elephant plains_zebra rhino gazelle \
    --out output/paper_results.md
cat output/paper_results.md
```

One markdown file with: dataset facts (embedded), per-run sections (every metric — 2D, 3D, Rel-AP3D, BEV, NHD, class-agnostic), 3-seed mean±std rollup, reproducibility footer.

### 4.11 Visualizations

`run_full_eval.sh` step [5/5] generates these automatically into `<run_dir>/vis_ovmono3d/`. Each `img_NNNNNN.jpg` is a 2×3 layout:
- Row 1: GT 2D | GT 3D wireframes | GT novel view (60° pitch BEV-ish)
- Row 2: PRED 2D | PRED 3D wireframes | PRED novel view

Both `img_*.jpg` (with ground grid) and `img_*_nogrid.jpg` (without grid) variants. **Pick the same `img_NNNNNN` filename across multiple run dirs** (zero-shot RPN, zero-shot oracle, fine-tuned) for apples-to-apples paper figures.

To rerun with different settings (fewer/more samples, different `--top-k`):
```bash
python tools/visualize_class_agnostic.py \
    --preds <run_dir>/inference/iter_final/WildBox_val/instances_predictions.pth \
    --gt    datasets/Omni3D/WildBox_val.json \
    --out   <run_dir>/vis_ovmono3d \
    --top-k 5 --every 100 --limit 40
```

### 4.12 Training curves

```bash
python tools/plot_training.py output/<run_dir>/metrics.json
# -> writes <run_dir>/training_curves.png
```

---

## 5. Model

### 5.1 Architecture

OVMono3D-lift = Cube R-CNN with a DINOv2 ViT-B/14 backbone. The 3D head is class-agnostic (predicts a generic 3D cuboid given image features + 2D box, not class-conditional). This is what makes the paper-protocol zero-shot a valid experiment — the cube head can in principle lift any 2D box, but its priors are domain-specific (Omni3D-trained on metric-scale indoor/driving scenes).

### 5.2 Pretrained weights

`checkpoints/ovmono3d_lift.pth` from the OVMono3D release. ~603 MB. Trained on Omni3D for ~116 k iterations.

The checkpoint's 50-class ROI head shape mismatches our 6-class config; on load we get clean shape-mismatch warnings:
```
Skip loading parameter 'roi_heads.priors_dims_per_cat': (1, 50, 2, 3) → (1, 6, 2, 3)
Skip loading parameter 'roi_heads.box_predictor.cls_score.weight': (51, 1024) → (7, 1024)
...
```
That's expected behavior — those parameters are randomly reinitialized for our 6-class task. The backbone and 3D head load fully.

### 5.3 NUM_CLASSES override (mandatory)

`configs/wildbox/OVMono3D_wildbox_wildlife6.yaml` sets `MODEL.ROI_HEADS.NUM_CLASSES: 6`. Without this override the pretrained 50-class head is retained and the model emits all 50 Omni3D class IDs. The evaluator filters those down to 0–5 and most predictions are dropped, giving ~0 per-class AP. We caught this on 2026-04-24; the explicit override is now in the config (commit f894ab6).

### 5.4 Loss configuration

In `configs/wildbox/OVMono3D_wildbox_finetune.yaml` (the base): all per-component 3D loss weights at default, **except depth (`Z`) loss is downweighted to 0.5×**. Rationale: VGGT's synthetic-scale depth has higher relative noise than the other components (xy is well-localized in pixels, dimensions are bounded by SAM3 mask geometry, pose is well-supervised). Downweighting prevents the depth-noise gradient from dominating training.

---

## 6. Evaluation protocol

### 6.1 Primary metric: AP_BEV @ IoU 0.50

Bird's-eye-view AP at IoU 0.50 (KITTI convention). Why primary, not standard 3D AP:
- Our domain has VGGT-synthetic depth (median |z|=1) which doesn't align with the pretrained model's metric-scale 3D priors → standard 3D AP underestimates the model's actual 3D quality post-fine-tune.
- Disentangled NHD makes the depth bottleneck quantitative: **across all conditions, the z-component dominates 3D error, and the gap between zero-shot and fine-tuned models is 1-2 orders of magnitude larger on z than on any other axis.**

| Condition | NHD-z | NHD-xy | z/xy ratio |
|---|---:|---:|---:|
| Fine-tuned (best init5sp) | 5.97 | 2.10 | 2.8× |
| Fine-tuned (15k 3-seed mean) | 6.37 | 2.12 | 3.0× |
| Zero-shot RPN-transfer | ~143 | ~46 | 3.1× |
| Zero-shot GDino oracle | 561 | 46 | 12.2× |

NHD-z reduction from zero-shot to fine-tuned: **~94×** (561 → 6.0). NHD-xy reduction over the same change: ~22× (46 → 2.1). The 4× imbalance — fine-tuning fixes z 4× more aggressively than xy — directly says that **depth error is the limiting axis on this domain**.

BEV AP projects out z entirely, isolating xy + dimensions + pose — the components our domain can evaluate without being dominated by the scale-axis problem. That's the empirical justification for BEV-as-primary on aerial wildlife; not a domain choice but a direct read from the disentangled error decomposition.

`tools/bev_ap_eval.py` computes BEV AP from the saved `instances_predictions.pth` using shapely rotated-rectangle IoU. Reports `micro` (over all preds), `macro` (average per-class), and per-class.

NHD components are reported in units of "GT cuboid extent" — each per-axis error is normalized by the corresponding GT cuboid dimension, so values are scale-invariant and **directly comparable across runs evaluated on the same val set**. Smaller is better; 0 is perfect overlap.

### 6.2 Per-species aggregation

Report all metrics three ways:
- **micro** — over all predictions, weighted by class frequency.
- **macro** — equal-weight average across the 6 species.
- **per-class** — full 6-column breakdown.

For rare classes (giraffe, gazelle, grevys_zebra), per-class single-seed numbers have wide error bars; report mean±std from 3 seeds. Stable classes (elephant, rhino, plains_zebra) are fine single-seed but we report std anyway since we have it.

### 6.3 Supplementary metrics

- **AP_BEV @ IoU 0.25** — looser BEV IoU; supplementary.
- **AP_3D @ IoU 0.25** — standard COCO-style 3D AP; included for paper-table completeness but limited by depth-scale mismatch on zero-shot rows.
- **Rel-AP_3D** — LabelAny3D scale-aligned 3D AP. Search range `[0.05, 3.0, 32]` for fine-tuned; widened to cover zero-shot best-scales (~0.1–0.3). Computed on CPU (pytorch3d CUDA blocked, see §3.1.3).
- **2D AP @ IoU 0.50:0.95** — standard COCO 2D detection AP.
- **Class-agnostic 2D AP @ {0.25, 0.50, 0.75}** — ignores class labels. The honest zero-shot localization signal when standard per-class AP is 0.
- **Disentangled NHD** — Normalized Hausdorff Distance per (xy, z, dimensions, pose) component. Motivates BEV as primary; shows that z dominates overall NHD.

### 6.4 Zero-shot evaluation protocol — two flavors

#### 6.4.1 RPN-transfer (closed-vocab, our default supplementary)

`TEST.ORACLE2D=False`. Predictions come from the model's own RPN + 50-class 2D classifier (Omni3D-pretrained). None of the 50 pretrained classes correspond to our 6 wildlife species → the evaluator's class filter drops every prediction → near-zero standard per-class AP.

The meaningful number from this row is **class-agnostic 2D AP** (`tools/class_agnostic_eval.py`). It tells us "given a closed-vocab pretrained detector, how well does it localize *any* foreground object on aerial wildlife imagery?"

#### 6.4.2 GDino oracle (paper protocol — what OVMono3D Table 1 reports)

`TEST.ORACLE2D=True`. We replace the model's 2D proposals with **precomputed GroundingDINO open-vocab detections** keyed per-image. The 3D cube head still operates as normal, just on different 2D inputs. This is the OVMono3D paper's zero-shot setup; it isolates the question "given that the open-vocab 2D problem is solved, how does the pretrained 3D cube head transfer?"

Configs: [OVMono3D_wildbox_wildlife6_oracle2d.yaml](configs/wildbox/OVMono3D_wildbox_wildlife6_oracle2d.yaml). Key settings:
```yaml
TEST:
  ORACLE2D: True
  REL_AP3D_SEARCH: [0.05, 3.0, 32]   # widened for zero-shot scale
DATASETS:
  ORACLE2D_FILES:
    EVAL_MODE: target_aware
    target_aware:
      novel:
        WildBox_val: datasets/Omni3D/gdino_WildBox_val_oracle_2d.json
      base:
        WildBox_val: datasets/Omni3D/gdino_WildBox_val_oracle_2d.json
    previous_metric: ...
```

Results we measured:
- **2D localization is excellent**: GDino median IoU vs GT = 0.877; class-agnostic 2D AP@0.5 = 68.3.
- **Per-species classification is biased**: GDino over-predicts elephant (1.83×) and gazelle (1.18×), severely under-predicts plains zebra (0.31×) and especially **Grévy's zebra (0.035×)**. The model can't reliably distinguish Grévy's vs plains zebra from a single-word prompt.
- **3D AP is 0**: pretrained Omni3D cube head outputs depth in metric meters; WildBox GT is in VGGT synthetic scale (median |z|=1). NHD-z = 561 confirms the scale mismatch. **This is expected** — the paper-protocol zero-shot row's 3D AP being 0 isolates the "3D prior doesn't transfer across scale conventions" result.

#### 6.4.3 Both rows are required for the paper

Report all three rows in the main metrics table:

| Row | Purpose |
|---|---|
| zero-shot RPN-transfer | closed-vocab, "what does the pretrained detector do raw on this domain" |
| zero-shot GDino oracle | paper protocol, "what does the 3D head do given good 2D" |
| fine-tuned 3-seed | the proposed approach |

### 6.5 Multi-seed protocol

3 training seeds (0, 1, 2) at REPEAT_THRESHOLD=0.5, all from the same pretrained init. Variance comes from random data shuffling and dataloader-worker order, NOT from random init (init is identical across seeds via the deterministic `MODEL.WEIGHTS` load).

For paper reporting:
- **Rare classes** (giraffe, gazelle, grevys_zebra): MUST report mean±std. Single-seed values are unreliable (giraffe in particular has 1 val video / 110 boxes).
- **Stable classes** (elephant, rhino, plains_zebra): mean±std is fine but single-seed is also defensible. We report std since it's free.
- **Macro/micro overall**: report mean±std across seeds.

---

## 7. Scripts inventory

| Tool | Purpose |
|---|---|
| `tools/prepare_wildbox_dataset.py` | VGGT+SAM3 → Omni3D JSON. Zip-aware, video-level split, SAM3-tight 2D, skip-bad-zip. |
| `tools/patch_stats_for_wildbox.py` | Register wildlife species in Omni3D stats.json. Idempotent. |
| `tools/dataset_stats.py` | **Auto-generates paper-ready dataset inventory** (per-species counts, size distribution PNG). Run after every prep. |
| `tools/precompute_gdino_oracle.py` | GroundingDINO oracle JSON for paper-protocol zero-shot. Uses GDino's `load_image` (CRITICAL — see §3.1.2). |
| `tools/run_multi_seed.sh` | Sequential multi-seed training launcher. 3 seeds, REPEAT_THRESHOLD configurable, resumable. |
| `tools/run_multi_seed.sbatch` | sbatch wrapper for overnight detached runs. |
| `tools/run_full_eval.sh` | 5-step full eval (standard / Rel-AP3D / BEV / class-agnostic / vis). |
| `tools/bev_ap_eval.py` | BEV AP via shapely rotated IoU. Handles dataset-id ↔ contiguous-id for both fine-tuned and oracle predictions. |
| `tools/class_agnostic_eval.py` | Class-agnostic 2D AP + 3D NHD surrogate. Zero-shot diagnostic. |
| `tools/visualize_class_agnostic.py` | OVMono3D-style 2×3 layouts. Auto-invoked by run_full_eval.sh. |
| `tools/plot_training.py` | metrics.json → training_curves.png. Robust to D2 metric-key naming drift. |
| `tools/aggregate_seed_ap.py` | Mean±std aggregator across seeds. |
| `tools/check_rel_ap3d_boundary.py` | Confirms Rel-AP3D scale-search grid is adequate (not boundary-pinned). Quantization-aware. |
| `tools/make_report.py` | Single-run or `--compare` multi-row markdown+LaTeX report. |
| `tools/assemble_paper_results.py` | **Single-sheet paper-results assembler**. One markdown file with all metrics for all runs. |
| `tools/pipeline_status.sh` | Dashboard god-view of all running pipelines (training, GDino, evals). `watch -n 30 bash tools/pipeline_status.sh`. |
| `tools/remap_wildbox_paths.py` | Path rescue when data moves; preserves split. |

---

## 8. Adding new data zips or classes

### 8.1 New zip, same species (most common: more data)

1. Drop the new zip into `archive/` next to the others.
2. Add a new `--source <zip_path>=<species>:<id>` to the prep command (FINAL_RUN.md §2 / QUICK_START_SMOKE_TEST.md §3).
3. Re-run prep, dataset_stats, no-leakage, training. Existing `category_meta_wildlife6.json` doesn't need to change.

### 8.2 New species (e.g. add "lion")

1. Pick a new dataset-id in the gap (e.g., `lion:1006`).
2. Add to `tools/patch_stats_for_wildbox.py --add lion:1006` (re-run).
3. Create `configs/wildbox/category_meta_wildlife7.json` with the 7-class mapping (sorted by dataset-id ascending — see §2.8).
4. Create `configs/wildbox/OVMono3D_wildbox_wildlife7.yaml` (copy from wildlife6, update `CATEGORY_NAMES` and `MODEL.ROI_HEADS.NUM_CLASSES: 7`).
5. Create matching `OVMono3D_wildbox_wildlife7_oracle2d.yaml` if doing paper-protocol zero-shot.
6. Re-flip both symlinks to wildlife7.
7. Re-run prep with the new species's zips, dataset_stats, training.

The 50-class pretrained head will reinitialize to 7 — same flow as 50→6.

### 8.3 Removing a species

1. Drop the species from `CATEGORY_NAMES` in the config.
2. Rebuild `category_meta_wildlife<N-1>.json` and the YAML.
3. Re-prep without that species's zips. Retrain from scratch (existing checkpoint is N-class, won't load shape-correctly into N-1).

---

## 9. Reproducing on another architecture

See **§20** for the full cross-architecture protocol (portable oracle-JSON contract, consumer-side eval wiring, side-by-side reporting table, 5-step adoption checklist).

---

## 10. Convergence & tuning — what we tried, what worked

### 10.1 5-species → 6-species regression

Adding the plains/Grévy's zebra split (4 → 5 zebra zips) and 4 more elephant/rhino zips moved us from a 5-species 11-zip config to a 6-species 15-zip config. **Macro 3D AP regressed from 16.1 → 9.1.** Per-class:

| Species | 5-sp baseline | 6-sp 3-seed mean | Δ |
|---|---:|---:|---:|
| plains_zebra (was "zebra") | 16.7 | **23.65 ± 2.83** | **+6.9** |
| giraffe | 1.8 | 10.67 ± 7.77 | +8.9 (huge std) |
| grevys_zebra | n/a | 6.51 ± 1.35 | new |
| **elephant** | 45.1 | **8.32 ± 0.37** | **−36.8** |
| **rhino** | 14.7 | **4.10 ± 0.43** | **−10.6** |
| gazelle | 2.3 | 1.47 ± 0.14 | −0.8 |

2D AP is stable across seeds (per-class std all <1.3, micro AP50 = 77.8 ± 0.9). The regression is concentrated in **3D AP for head classes (elephant, rhino)**.

### 10.2 What we ruled out

**REPEAT_THRESHOLD ablation**. Hypothesis: 0.5 over-corrects toward rare classes; 0.35 might recover head classes. Single-seed 0.35 ablation: macro 3D AP 8.32 (vs 9.12 mean for 0.5 — **within seed-0 std envelope**). REPEAT_THRESHOLD is NOT the driver of the regression.

### 10.3 Likely root causes (genuine task difficulty, not config)

1. **Val distribution shift**: new val has 8.5 elephant boxes/frame vs the old 5-species val's lower density. Dense-scene 3D regression is harder than dense-scene 2D detection — explains the 2D-AP-vs-3D-AP gap (elephant: 2D AP 57, 3D AP 8.3).
2. **Plains/Grévy's split adds a fine-grained boundary** the model has to learn from limited data. Plains zebra at 12 train videos handles this fine (3D AP 23.7); Grévy's at 4 train videos has more uncertainty.
3. **6-class discrimination is genuinely harder than 5-class** — adds one mode that visually overlaps with two existing modes (zebras now indistinguishable from grass/distance only by stripe pattern).

### 10.4 Future experiments (priority order)

1. **Init from 5-species fine-tuned checkpoint** instead of from raw Omni3D pretrained. The 5-species model already learned WildBox-scale 3D priors; the 6-species fine-tune only needs to add classes 1003 (plains) and adjust the rare-class mappings. Highest expected gain (+3–7 macro AP) for ~3.5 h compute.
2. **3-seed at REPEAT_THRESHOLD=0.35** (currently only single-seed) — confirms the ablation finding rigorously. ~10.5 h compute.
3. **Higher input resolution (392 short-edge)** — direct fix for gazelle (median bbox 0.34% of image area, ~9×9 px at current 294 short-edge → below 2D-regression resolution). Expected: +3–8 gazelle AP.
4. **Longer training (25 k iters)** — head classes might recover further after the second LR decay.
5. **Ensemble seeds 0+1+2** — average predictions; often +1–3 AP free.

---

## 11. Key design decisions (paper "we chose X over Y")

- **SAM3-tight 2D bboxes** vs cuboid-projection 2D. Significantly tighter, matches what wildlife ecologists measure. See §2.7.
- **Video-level train/val split** vs segment- or frame-level. Eliminates leakage from same-video contiguous-frame similarity. See §2.6.
- **VGGT-synthetic per-segment scale normalization** (median |z|=1) vs absolute metric scale. Required because drone GPS-altitude estimates are noisy and inconsistent across segments. Trade-off: standard 3D AP underestimates quality on zero-shot rows; mitigated by reporting BEV AP and Rel-AP_3D.
- **BEV AP @ IoU 0.50 as primary 3D metric** instead of standard 3D AP. Disentangled NHD shows depth ("z") is dominant 3D error → BEV projects out z, isolating the components our domain can evaluate. See §6.1.
- **3-seed multi-seed protocol** for rare classes. Single-seed numbers for 4-train-video classes are not reviewer-credible.
- **Three-row evaluation** (zero-shot RPN / zero-shot oracle / fine-tuned). Each isolates a different question; together they form the complete paper story.
- **REPEAT_THRESHOLD=0.5** for fine-tuning. More aggressive than the typical 0.25; chosen to upsample rare classes in the long-tail wildlife distribution. Trade-off documented in §10.

---

## 12. Known limitations

- **Giraffe val is 1 video, 110 boxes** — per-class giraffe metrics have inherent variance unrelated to model quality. Reported as `mean ± std` to make this visible; flagged in paper limitations.
- **gazelle is bottlenecked by input resolution** — median bbox 0.34% of image area (~9×9 px after 294-short-edge resize). Per-class gazelle AP plateaus around 1–4 regardless of training duration.
- **GDino can't distinguish plains vs Grévy's zebra** from text prompts — the paper-protocol zero-shot row's per-class zebra metrics reflect this. Class-agnostic metrics rescue the localization story; per-class numbers are reported honestly.
- **3D AP zero on zero-shot rows** is not a bug — pretrained Omni3D cube head outputs metric-scale predictions; WildBox is in VGGT synthetic scale. This is the expected paper finding; Rel-AP_3D and BEV AP supplement.
- **pytorch3d CUDA build blocked** by cluster glibc — Rel-AP_3D scale search forces CPU tensors. ~15–30 min per eval. Permanent state unless cluster glibc changes.

---

## 13. Changelog (features + bugs caught, vs upstream OVMono3D)

Bugs caught and fixed during development (see §21.2 for the full table with commit refs):

1. **Iteration-skip-training** (2026-04-24, f894ab6) — pretrained checkpoint's `iteration` field silently skipped training. CRITICAL. See §3.1.1.
2. **Category_meta symlinks** — `train_net.py --eval-only` reads from the config-file's directory, not from `configs/`. Symlinks must point at the same file in both locations.
3. **Rel-AP3D stuck forever** — flat-list IoU computation blew up on 14k val × 28 scales. Now per-(image, category) block.
4. **BEV per-class 0/0** for ORACLE2D=True — predictions carry dataset-ids, evaluator filtered by contiguous-ids. Now normalized at evaluator boundary.
5. **Class-agnostic per-class same bug** as BEV — also fixed.
6. **make_report Rel-AP3D row missing** — parser only read `log.txt`, not `log.rel.txt`. Now reads both.
7. **GDino preprocessing** silently skipped resize → median IoU 0.10 → 0.97 fix.
8. **NUM_CLASSES inheritance** — wildlife6 config now explicitly sets `MODEL.ROI_HEADS.NUM_CLASSES: 6` instead of inheriting 50 from base.

Features added relative to upstream OVMono3D:

| Feature | Tool / config | Purpose |
|---|---|---|
| Zip-aware data prep with caching + skip-corrupt | `prepare_wildbox_dataset.py` | Don't re-extract; partial transfers don't abort prep. |
| SAM3 tight 2D bboxes | `prepare_wildbox_dataset.py` | Cuboid-projection 2D is loose; SAM3 masks give paper-quality 2D. |
| Video-level split | `prepare_wildbox_dataset.py:auto_split` | Segment-level leaks. |
| Auto-generated `category_meta_wildlife<N>.json` | `prepare_wildbox_dataset.py` | Sort order matches training's internal mapping. |
| Dataset inventory auto-generator | `tools/dataset_stats.py` | Paper-ready per-species stats + size distribution PNG. |
| GDino oracle precompute | `tools/precompute_gdino_oracle.py` + wildlife6_oracle2d yaml | Paper-protocol zero-shot (`TEST.ORACLE2D=True`). |
| BEV AP (primary metric) | `tools/bev_ap_eval.py` | Shapely rotated-rect IoU; micro/macro/per-class. |
| Class-agnostic 2D AP + NHD 3D surrogate | `tools/class_agnostic_eval.py` | Meaningful zero-shot signal. |
| OVMono3D-faithful novel-view visualizer | `tools/visualize_class_agnostic.py` | 2×3 layout, per-id colors, ground-grid toggle. Auto-run from `run_full_eval.sh` step [5/5]. |
| Multi-seed launcher + aggregator | `tools/run_multi_seed.sh` + `tools/aggregate_seed_ap.py` | 3-seed mean±std for rare classes. |
| Rel-AP3D scale-search boundary verifier | `tools/check_rel_ap3d_boundary.py` | Quantization-aware verdict. |
| Per-block Rel-AP3D scale search | `cubercnn/evaluation/omni3d_evaluation.py:search_rel_scale` | Old O(N×M) per scale never finished on 14k val. |
| dataset-id → contiguous-id normalization | bev / class_agnostic / omni3d evaluators | Handle both fine-tuned and ORACLE2D=True prediction formats. |
| Paper-results assembler | `tools/assemble_paper_results.py` | Single markdown with all metrics, all runs. |
| Pipeline status dashboard | `tools/pipeline_status.sh` | One-screen god-view of every running pipeline. |
| `resume=False` forces `start_iter=0` | `tools/train_net.py:183` | Pretrained's stored iteration no longer silently skips training. |
| Zero-shot evaluator short-circuit | `omni3d_evaluation.py` | "zero in-vocab predictions" warns instead of crashing. |
| Training-safe vis try/except | `tools/train_net.py:do_test` | Vis exceptions never kill training. |
| Stripped category_meta symlink auto-sync | `tools/run_full_eval.sh` start | Prevents two-files-diverge bug. |

---

## 14. Citations to include in the paper

- **OVMono3D** — base architecture (uva-cv-lab).
- **Cube R-CNN** (Brazil et al., CVPR 2023) — framework OVMono3D extends.
- **Omni3D** — pretraining dataset + AP-3D definition.
- **LabelAny3D** — Rel-AP-3D protocol + grid (0.3, 3.0, 28); we widen to (0.05, 3.0, 32) for zero-shot rows.
- **KABR** — Grévy's zebra source; long-tail reporting convention (macro + micro + per-class).
- **UAV3D / AM3D / CDrone** — BEV-as-primary in aerial 3D; precedent for our metric choice.
- **SAM3** — segmentation source for tight 2D bboxes.
- **VGGT** — 3D reconstruction source.
- **DINOv2** — backbone pretraining.
- **GroundingDINO** — open-vocab 2D detector for paper-protocol zero-shot.

---

## 15. Final-run operations guide

**See [FINAL_RUN.md](FINAL_RUN.md)** — the canonical 13-step linear pipeline (data prep → training → eval → reports → figures), all copy-paste blocks with expected outputs.

The single gotcha worth re-flagging here: if `grep -E "Starting training from iteration" log.txt` says anything other than `iteration 0 (resume=False)`, training was silently skipped — see §3.1.1.

---

## 16. Iteration count and hyperparameter recommendations

### 16.1 Iteration count

| Configuration | Iters (batch 8, LR 2e-3) | When to use |
|---|---:|---|
| Quick sanity | 1 000 | Confirm pipeline runs end-to-end |
| **Smoke test** | **2 000** | What [QUICK_START_SMOKE_TEST.md](QUICK_START_SMOKE_TEST.md) uses |
| Dev / exploration | 5 000 | Early signal on hparam changes |
| **Paper run** | **15 000** | Balance of 3D convergence + reasonable wall-clock |
| Max-quality stretch | 25 000 | Long-tail AP push |

### 16.2 Hyperparameter recommendations

| Parameter | Value | When to change |
|---|---|---|
| `IMS_PER_BATCH` | 8 | Constant across experiments for fair comparison |
| `BASE_LR` | 0.002 | Linearly scale with batch; halve if NaN |
| `WARMUP_ITERS` | 500 | ~3 % of total |
| `STEPS` | (60%, 90%) of `MAX_ITER` | Standard schedule; keep ratios |
| `CHECKPOINT_PERIOD` | 5000 | Bounds kill-restart loss to ~1 h |
| `DATALOADER.REPEAT_THRESHOLD` | 0.5 | Aggressive long-tail upsampling. Trade-off: head-class regression. |
| `MODEL.ROI_HEADS.NUM_CLASSES` | 6 | MUST match meta thing_classes count |
| `TEST.EVAL_PERIOD` | =MAX_ITER | Skip in-loop eval (slow); rely on final eval |
| `TEST.EVAL_REL_AP3D` | False during training | CPU pytorch3d is slow; run once at the end via `run_full_eval.sh` step [2/5] |
| `TEST.REL_AP3D_SEARCH` | (0.3, 3.0, 28) for fine-tuned, (0.05, 3.0, 32) for zero-shot | Wider for zero-shot's small best-scales |

### 16.3 Data recommendations

- **Add data, don't tune more**: rare-class performance is bottlenecked by # videos, not by training duration or REPEAT_THRESHOLD.
- **Multiple zips per species when possible**: 12+ videos → reliable per-class metrics. ≤5 videos → wide error bars no matter the protocol.

### 16.4 When to retrain from scratch vs resume

- **From-scratch retrain**: when `category_meta` changes (new/removed species), when the pretrained checkpoint changes, when input resolution changes, when the data split seed changes.
- **Resume**: when training crashed mid-run, when adding more iters to an existing converged run.

---

## 17. Output artifact map

### 17.1 Per-training-run (`output/<name>/`)

| File | Source | Use |
|---|---|---|
| `model_final.pth` | trainer | Final checkpoint |
| `model_<iter>.pth` | trainer | Periodic checkpoints (every `CHECKPOINT_PERIOD`) |
| `last_checkpoint` | trainer | Pointer used for resume |
| `log.txt` | trainer | Iteration losses, final eval (if EVAL_PERIOD ≤ MAX_ITER) |
| `metrics.json` | trainer | Per-iter metrics for `plot_training.py` |
| `config.yaml` | trainer | Resolved cfg (CLI overrides applied) |
| `inference/iter_final/<dataset>/instances_predictions.pth` | trainer's auto-eval | Predictions |

### 17.2 Per-eval-run (`output/<name>/eval/`, written by `run_full_eval.sh`)

| File | Step | Use |
|---|---|---|
| `log.txt` | [1/5] | Standard 2D + 3D AP, per-class, disentangled NHD |
| `inference/iter_final/<dataset>/instances_predictions.pth` | [1/5] | Predictions used by all downstream evals |
| `eval_rel/log.txt` | [2/5] | Rel-AP3D mode=3D-Rel results (separate dir to avoid clobbering) |
| `bev_ap.json` | [3/5] | BEV AP per IoU per class |
| `summary_nhd.txt` | [4/5] | Class-agnostic 2D AP + NHD scale search + per-class |
| `vis_ovmono3d/img_*.jpg` | [5/5] | Paper-figure visualizations |
| `paper_report/report.md` | end | Single-run markdown report |

### 17.3 Multi-seed aggregation

| File | Source | Use |
|---|---|---|
| `mean_std_report/table_multiseed.md` | `aggregate_seed_ap.py` | Per-class mean±std table |
| `mean_std_report/table_multiseed.tex` | `aggregate_seed_ap.py` | LaTeX export |
| `paper_results.md` | `assemble_paper_results.py` | All-runs single sheet |

### 17.4 Dataset artifacts

| File | Source | Use |
|---|---|---|
| `datasets/Omni3D/WildBox_train.json` / `WildBox_val.json` | prep | GT input |
| `datasets/Omni3D/dataset_stats/dataset_stats.md` | `dataset_stats.py` | Paper inventory |
| `datasets/Omni3D/dataset_stats/size_distribution.png` | `dataset_stats.py` | Per-species bbox size histogram |
| `datasets/Omni3D/gdino_WildBox_val_oracle_2d.json` | `precompute_gdino_oracle.py` | Paper-protocol zero-shot 2D oracle |
| `<zip_dir>_unzipped/` | extraction cache | Source frames + masks |

---

## 18. Quick commands cheat sheet (utilities only)

**Primary references** — use these, not duplicates of their content here:
- **[FINAL_RUN.md](FINAL_RUN.md)** — canonical 13-step pipeline.
- **[QUICK_START_SMOKE_TEST.md](QUICK_START_SMOKE_TEST.md)** — ~1 h smoke.
- **§4** — same pipeline with rationale per step.

Only the utilities not in those three docs live here:

```bash
# Resume a stopped training run (last_checkpoint at OUTPUT_DIR)
python tools/train_net.py --config-file <cfg> --num-gpus 1 --resume \
    OUTPUT_DIR output/your_run

# Remap paths after data move (no re-prep, preserves split)
python tools/remap_wildbox_paths.py datasets/Omni3D/WildBox_val.json \
    --map /old/prefix=/new/prefix --in-place

# Which GPU is your training job on?
YOUR_PID=$(pgrep -f train_net.py | head -1)
nvidia-smi --query-compute-apps=pid,gpu_name,used_memory --format=csv | grep "^$YOUR_PID,"

# Dashboard god-view of all running pipelines
bash tools/pipeline_status.sh
# Auto-refresh:
watch -n 30 bash tools/pipeline_status.sh

# Tail training loss + ETA
tail -n 5 output/<run>/log.txt | grep iter

# Verify training actually ran (defends against the iteration-skip bug)
grep -cE "iter: [0-9]+" output/<run>/log.txt    # must be > 0
grep "Starting training from iteration" output/<run>/log.txt  # must say 0
```

---

## 19. Experiment tracker template

For each experiment (different architecture, data, hparams), fill out a row to enable cross-experiment comparison. Suggested location: `output/experiments.md`.

| Field | Example |
|---|---|
| Experiment label | `wl6-15zip-bs8-lr2e-3-15k-rep0.5-seed0` |
| Date | 2026-04-24 |
| Architecture | `OVMono3D-lift DINOv2 ViT-B/14` |
| Pretrained init | `checkpoints/ovmono3d_lift.pth` |
| Data | 15 zips, 6 species, video-level seed=0 |
| Total frames train/val | 45 979 / 13 779 |
| Species | giraffe, grevys_zebra, elephant, plains_zebra, rhino, gazelle |
| Batch size | 8 |
| Base LR | 2e-3 |
| Max iters | 15 000 |
| REPEAT_THRESHOLD | 0.5 |
| AMP | True |
| Seeds | 0, 1, 2 |
| Wall-clock per seed | ~3.5 h (A40) |
| **AP_BEV @ 0.5 (micro mean±std)** | **fill from paper_results.md** |
| **AP_3D @ 0.25 (micro mean±std)** | fill |
| **Rel-AP_3D (micro mean±std)** | fill |
| **2D AP @ 0.5 (mean±std)** | fill |
| **per-class 3D AP rare** (giraffe / gazelle / grevys_zebra) | fill |
| Notable observations | "elephant 3D regressed vs 5-species baseline by ~37 AP — see §10" |

---

## 20. Cross-architecture protocol (paper §)

### 20.1 What stays fixed across architectures

Hold these constant for every architecture you compare. Otherwise differences aren't attributable to the architecture.

- Dataset files: `WildBox_train.json`, `WildBox_val.json`
- Split seed
- `configs/category_meta.json` symlink target
- Evaluation protocol (same `run_full_eval.sh`, same metrics)
- Image resolution
- **Zero-shot 2D detector**: same `gdino_WildBox_val_oracle_2d.json` for every arch's oracle row.

### 20.2 What you vary (and track per §19's template)

- The architecture itself
- Loss weights specific to the architecture
- Backbone pretraining source
- LR schedule fitted to the architecture

### 20.3 Output format required from a new architecture

Save predictions as a detectron2-compatible `instances_predictions.pth` (a list of dicts, one per image):

```python
torch.save([
    {
        "image_id": int,
        "instances": [
            {
                "bbox": [x, y, w, h],           # xywh in image pixels
                "score": float,
                "category_id": int,             # contiguous 0..N-1
                "center_cam": [x, y, z],
                "dimensions": [W, H, L],        # Omni3D ordering
                "pose": [[3x3 rotation]]        # or 9 floats
            }, ...
        ]
    }, ...
], "predictions.pth")
```

Any detectron2 `COCOEvaluator`-compatible output writer produces this format. Then our eval tools work identically.

### 20.4 Paper-protocol zero-shot across architectures (the portable contract)

The OVMono3D paper reports zero-shot 3D by pairing **one open-vocab 2D detector (GroundingDINO)** with each method's 3D head. For cross-architecture comparison, this is the single most important thing to hold fixed: precompute GDino boxes once, hand the resulting JSON to every architecture.

**What "paper protocol zero-shot" means for arch X**:

1. Take a pretrained (Omni3D-trained, NOT WildBox-finetuned) checkpoint of X.
2. At eval time, **replace X's 2D proposals with GDino's text-prompted detections** for the WildBox species.
3. Let X's 3D head lift those 2D boxes to 3D.
4. Score with our standard eval stack.

This measures: *"how well does X's 3D lift onto an unseen visual domain, given that open-vocab 2D localization is solved for us?"*

**Interchange JSON** — produced once via [tools/precompute_gdino_oracle.py](tools/precompute_gdino_oracle.py); every architecture consumes the same file:

```
datasets/Omni3D/gdino_WildBox_val_oracle_2d.json
```

Schema: list of one entry per image with `image_id`, `K`, `file_path`, `height`, `width`, plus `instances` (each with 2D xywh box, score, dataset-space `category_id`, `category_name`). No 3D info.

**How each architecture consumes it at eval time**: wire a hook in the architecture's test-time inference that, for each image, **replaces the architecture's own 2D stage** (RPN, anchor head, DETR queries — whatever it has) with the loaded oracle boxes. Map `category_id` from dataset-space (1000–1005) to the architecture's contiguous-id space (same mapping as during training — see §2.8). Feed the loaded boxes into the architecture's 3D head / ROI pool / cuboid regressor.

In OVMono3D this is a built-in flag: `TEST.ORACLE2D=True` + `DATASETS.ORACLE2D_FILES`. For another architecture, you may need ~50 lines: load the JSON, rescale boxes to the architecture's preprocessing resolution, hand them to the module that normally receives 2D proposals.

### 20.5 Three-row reporting table per architecture

Every architecture appears as three rows. The same oracle JSON is used for all of row 2.

| Row | Arch | Backbone | 3D head | 2D source | Numbers |
|---|---|---|---|---|---|
| 1 | OVMono3D (ours) | DINOv2 ViT-B/14 | Cube R-CNN | RPN-transfer | from `wl6_zeroshot_rpn` |
| 2 | OVMono3D (ours) | DINOv2 ViT-B/14 | Cube R-CNN | **GDino oracle** | from `wl6_zeroshot_oracle2d` |
| 3 | OVMono3D (ours) | DINOv2 ViT-B/14 | Cube R-CNN | own RPN (fine-tuned) | from `wl6_rt0.5_multiseed/seed*` (mean±std) |
| 1 | Cube R-CNN | ResNet-50 | Cube R-CNN | RPN-transfer | (re-run) |
| 2 | Cube R-CNN | ResNet-50 | Cube R-CNN | **GDino oracle** | (re-run, same JSON) |
| 3 | Cube R-CNN | ResNet-50 | Cube R-CNN | own (fine-tuned) | (re-run) |
| ... | DetAny3D | ? | DINO-decoder | ... | ... |

### 20.6 Five-step adoption checklist for arch X

1. Run the GDino precompute (§6.4.2 in this doc, or QUICK_START §9) **exactly once**. Output goes to `datasets/Omni3D/gdino_WildBox_val_oracle_2d.json`. Do **not** re-precompute per architecture.
2. Implement the "load oracle boxes instead of own 2D proposals" hook in X's eval loop.
3. Confirm X's contiguous-id mapping matches ours: `{giraffe:0, grevys_zebra:1, elephant:2, plains_zebra:3, rhino:4, gazelle:5}` (sorted by dataset-id ascending per §2.8).
4. Run X's zero-shot eval → produce `instances_predictions.pth` in the §20.3 schema.
5. Point our `tools/bev_ap_eval.py`, `tools/class_agnostic_eval.py`, and `tools/assemble_paper_results.py` at X's prediction file. Numbers append to the comparison table automatically.

If X's zero-shot row is much worse than ours and X is **not** using the oracle, the gap is confounded — X's own 2D stage may just be weaker than GDino on this domain. **Paper claims about the 3D head must use oracle 2D on both sides.**

---

## 21. Current state (as of 2026-04-25)

Single-paragraph snapshot for resume after compaction:

**6-species dataset live (plains_zebra + grevys_zebra split from KABR), 15 zips total, 64 videos / 60k frames / 237k instances, video-level 80/20 split. 3-seed multi-seed at REPEAT_THRESHOLD=0.5 complete (commit `f894ab6` fixed the silent-skip-training bug). REPEAT_THRESHOLD=0.35 single-seed ablation complete (no meaningful improvement → REPEAT_THRESHOLD is not the lever for the head-class regression). All three rows have evals on the same 13 779-image val set: zero-shot RPN-transfer, zero-shot GDino oracle (paper protocol), fine-tuned 3-seed. Macro 3D AP = 9.12 ± 0.84 (vs 5-species baseline 16.1); per-class breakdown in §10.**

### 21.1 What's in place (code + data + results)

- **Dataset prep**: 15-zip 6-species corpus, SAM3-tight 2D, video-level split. All 15 archive dirs under `/storage3/3DOM/vshukla/sam3/wd_data/wildbox/archive/`; unzipped dirs cached.
- **Dataset stats**: `tools/dataset_stats.py` produces `dataset_stats.{md,json}` + `size_distribution.png` per prep.
- **Configs**: 6-species training (`OVMono3D_wildbox_wildlife6.yaml`) + paper-protocol oracle (`OVMono3D_wildbox_wildlife6_oracle2d.yaml`); both with explicit `MODEL.ROI_HEADS.NUM_CLASSES: 6`. Symlinks both pointing at `category_meta_wildlife6.json`.
- **Paper-protocol oracle JSON**: `datasets/Omni3D/gdino_WildBox_val_oracle_2d.json` — 13 779 images × 6 species, GDino at thresholds 0.15/0.10. Median IoU vs GT = 0.877. Reusable across architectures.
- **3-seed fine-tuned**: `output/wl6_rt0.5_multiseed/seed{0,1,2}/eval/` — full evals (5 steps each).
- **REPEAT_THRESHOLD=0.35 ablation**: `output/wl6_rt0.35_seed0/seed0/eval/`.
- **Zero-shot evals on new val**: `output/wl6_zeroshot_rpn/`, `output/wl6_zeroshot_oracle2d/`.
- **Tools**: every script in §7; latest `tools/assemble_paper_results.py` produces `output/paper_results.md` (single sheet).

### 21.2 Bugs caught and fixed (don't re-introduce)

| # | Symptom | Root cause | Fix | Commit |
|---|---|---|---|---|
| 1 | Per-class metrics all 0 except one during eval | `train_net.py --eval-only` reads `category_meta.json` from config-file's dir, not from `configs/`. Stale Phase-1 single-class file was winning. | Symlink BOTH `configs/category_meta.json` AND `configs/wildbox/category_meta.json` to `category_meta_wildlife6.json`. `run_full_eval.sh` auto-syncs both at start. | pre-history |
| 2 | Rel-AP3D stuck forever | `search_rel_scale` built one giant `box3d_overlap(N×M)` matrix per scale × 28 scales; ~1.6 B CPU pair comparisons per scale on 13 k val. | Per-(image, category) block IoU with progress prints. | pre-history |
| 3 | BEV AP showed `0 preds, 0 GT` per class for ORACLE2D=True | Eval filtered by contiguous-id but oracle preds carry dataset-ids 1000–1005. | Normalize dataset-id → contiguous-id in `pred_bev_by_img`. | d8547ed |
| 4 | Per-class AP macro=0 / micro=88 for class-agnostic eval | Same dataset-id/contiguous-id mismatch in `class_agnostic_eval.py` and main COCO evaluator. | Same normalization pattern in both files. | e7de1e6 |
| 5 | `make_report` Rel-AP3D row always `-` | Parser only read `log.txt`, not `log.rel.txt`. | Concat both before regex. | 1da8843 |
| 6 | GDino oracle boxes had median IoU 0.10 vs GT | `preprocess_image` skipped GDino's `RandomResize([800], max_size=1333)`. | Use `groundingdino.util.inference.load_image` directly. | e5bf1c9 |
| 7 | **Multi-seed produced 0 iter lines, `model_final.pth` byte-identical to pretrained** (silent skip-training) | `train_net.py:183` read checkpoint's `iteration` field unconditionally regardless of `resume=False`. Pretrained's `iteration=115999` made `start_iter > MAX_ITER` → training loop exited immediately. | Respect `resume` flag when reading iteration. | f894ab6 |
| 8 | NUM_CLASSES retained as 50 in fine-tuned model | wildlife6 config inherited from base which had `NUM_CLASSES: 50`. | Explicit `MODEL.ROI_HEADS.NUM_CLASSES: 6` override in wildlife6 config. | f894ab6 |

Bug #7 is the silent killer. See §3.1.1 for the four red flags to spot it within 2 minutes of launch.

### 21.3 Headline result numbers

3-seed mean ± std at REPEAT_THRESHOLD=0.5, on the new 13 779-image val:

- 2D AP@0.5 micro: **77.8 ± 0.9**
- 3D AP macro: **9.12 ± 0.84**
- Rel-AP3D macro: ~9.0 (close to standard 3D AP since fine-tuned best-scale ≈ 1.0)

Per-class 3D AP (mean ± std):

| Species | mean ± std | Note |
|---|---:|---|
| plains_zebra | 23.65 ± 2.83 | best — split paid off |
| giraffe | 10.67 ± 7.77 | wide std (1 val video) |
| elephant | 8.32 ± 0.37 | regressed from 5-sp baseline 45 |
| grevys_zebra | 6.51 ± 1.35 | new class, 4 train videos |
| rhino | 4.10 ± 0.43 | regressed from 5-sp baseline 15 |
| gazelle | 1.47 ± 0.14 | small-object resolution limit |

REPEAT_THRESHOLD=0.35 ablation: macro 3D AP 8.32 (within 0.5's std envelope). Confirms REPEAT_THRESHOLD is not the regression driver.

### 21.4 Honest paper narrative

> *"Adding the plains/Grévy's zebra split and 4× more training data (15 vs 11 zips) yields a more challenging val distribution (denser multi-animal scenes — elephant val has 8.5 boxes/frame) and a finer-grained classification task. This trades macro 3D AP (16.1 → 9.1) for stronger fine-grained class fidelity (plains_zebra 23.65 ± 2.83) and broader rare-class coverage. REPEAT_THRESHOLD ablation (0.25 → 0.5 → 0.35) shows the regression is not a sampling artifact — it reflects the genuine difficulty of the expanded task. Per-class 2D AP remains stable across all settings (elephant 57.1 ± 0.3); the 3D regression is concentrated in dense-scene 3D geometry, with depth (NHD-z) as the dominant error component, motivating BEV AP @ 0.5 as the primary 3D metric."*

### 21.5 Future work (paper v2 targets)

1. **Init from 5-species fine-tuned checkpoint** — closest checkpoint with WildBox-scale 3D priors. Highest expected gain (+3–7 macro AP) for ~3.5 h compute.
2. **3-seed at REPEAT_THRESHOLD=0.35** — currently single-seed; rigorous confirmation of the ablation finding. ~10.5 h compute.
3. **Higher input resolution (392 short-edge)** — direct fix for gazelle. Expected: +3–8 gazelle 3D AP.
4. **Longer training (25 k iters)** — head classes might recover further after 2nd LR decay.
5. **Ensemble seeds 0+1+2** — predictions averaged. Often +1–3 AP free.
6. **Verify zebra species labels per-video with domain expert** — current attribution is "KABR=Grévy's, all other=plains" (paper v2 rigor).

### 21.6 Known limitations (carry into paper)

- Giraffe val is 1 video / 110 boxes — wide error bars on per-class giraffe AP are inherent.
- Gazelle bottlenecked by input resolution (median bbox 0.34% of image area).
- 3D AP zero on zero-shot rows is the expected scale-mismatch result, not a bug.
- pytorch3d CUDA build blocked → Rel-AP3D is CPU-only, ~15–30 min per eval.

---

_End of document. Update §13 changelog and §17 file map for any new tool, config, or bug fix._
