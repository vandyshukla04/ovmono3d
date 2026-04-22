# Quick-start smoke test — end-to-end in ~2 hours

**Purpose:** prove this branch works start-to-finish before committing to a
multi-hour full training run. Uses **one zip per species** (smallest
available) and a short 2000-iteration fine-tune. Not a production run —
numbers will be low, but every pipeline step is exercised.

**When to use:** after pulling new code, changing configs, or rewriting
tools that affect the pipeline. Run this to catch breakage before scaling.

**Cluster:** all GPU steps on node81 (A40); CPU steps work anywhere. Env:
`/storage3/3DOM/vshukla/envs/ovmono3d`.

---

## 0. Setup (copy once at session start)

```bash
ssh node81
conda activate /storage3/3DOM/vshukla/envs/ovmono3d
cd /storage2/3DOM/vshukla/repos/ovmono3d
git pull
export SMOKE_OUT=output/smoke_$(date +%Y%m%d_%H%M%S)
mkdir -p "$SMOKE_OUT"
echo "SMOKE_OUT=$SMOKE_OUT"
```

## 1. Smallest available zip per species (picked for speed)

| Species | Zip | Frames | Videos |
|---|---|---:|---:|
| rhino | `data202502KRhinoCamiV1` | 5 107 | 4 |
| elephant | `data202406KElephants` | 3 369 | 4 |
| zebra | `data2023KABRZebras` | 4 610 | 5 |
| giraffe | `data202401KGiraffes` | 1 220 | 2 |
| gazelle | `data202406KGazelles` | 7 443 | 4 |
| **Total** | | **~21 749** | **~19** |

## 2. Data prep (single-zip-per-species, video-level split, SAM3 tight bboxes)

```bash
# Make sure wildlife categories are registered in Omni3D stats (idempotent)
python tools/patch_stats_for_wildbox.py \
    --stats datasets/Omni3D/stats.json \
    --add rhino:1004 elephant:1002 zebra:1001 giraffe:1000 gazelle:1005

# Build train/val JSONs (auto-extracts each zip on first run, caches after)
python tools/prepare_wildbox_dataset.py \
    --source /storage3/3DOM/vshukla/sam3/wd_data/wildbox/archive/data202502KRhinoCamiV1/WildBox_sam3-vggtv1_processed.zip=rhino:1004 \
    --source /storage3/3DOM/vshukla/sam3/wd_data/wildbox/archive/data202406KElephants/WildBox_sam3-vggtv1_processed.zip=elephant:1002 \
    --source /storage3/3DOM/vshukla/sam3/wd_data/wildbox/archive/data2023KABRZebras/WildBox_sam3-vggtv1_processed.zip=zebra:1001 \
    --source /storage3/3DOM/vshukla/sam3/wd_data/wildbox/archive/data202401KGiraffes/WildBox_sam3-vggtv1_processed.zip=giraffe:1000 \
    --source /storage3/3DOM/vshukla/sam3/wd_data/wildbox/archive/data202406KGazelles/WildBox_sam3-vggtv1_processed.zip=gazelle:1005 \
    --split-mode video --val-fraction 0.2 --seed 0 \
    --output-train datasets/Omni3D/WildBox_train.json \
    --output-val   datasets/Omni3D/WildBox_val.json \
    --dataset-id 1000 -v
```

**First run takes 5–20 min** (extracts the 5 zips). Subsequent runs reuse
the `*_unzipped/` sibling dirs and take <1 min.

## 3. Wire up BOTH category_meta symlinks + verify

```bash
ln -sf wildbox/category_meta_wildlife5.json configs/category_meta.json
ln -sf category_meta_wildlife5.json         configs/wildbox/category_meta.json

# BOTH must show giraffe first (matching training's sorted-by-dataset-id order)
cat configs/category_meta.json
cat configs/wildbox/category_meta.json

# No-leakage + file-exists sanity check
python -c "
import json, os
from collections import Counter
for split in ('train','val'):
    d = json.load(open(f'datasets/Omni3D/WildBox_{split}.json'))
    cats = Counter(a['category_name'] for a in d['annotations'])
    vids = {img['file_path'].split('/')[-3] for img in d['images']}
    missing = sum(1 for img in d['images'][:200] if not os.path.exists(img['file_path']))
    assert missing == 0, f'{split}: {missing} paths do not resolve'
    print(f'{split}: {len(d[\"images\"])} imgs, {len(d[\"annotations\"])} anns, '
          f'{len(vids)} videos, anns_per_class={dict(cats)}')
train_v = {im['file_path'].split('/')[-3] for im in json.load(open('datasets/Omni3D/WildBox_train.json'))['images']}
val_v   = {im['file_path'].split('/')[-3] for im in json.load(open('datasets/Omni3D/WildBox_val.json'))['images']}
ov = len(train_v & val_v)
assert ov == 0, f'video leakage: {ov} shared videos'
print(f'OK: {len(train_v)} train vids, {len(val_v)} val vids, no overlap')
"
```

**Abort if any line fails.** Video overlap must be 0.

## 4. Zero-shot eval (~5 min, `--skip-rel-ap3d` because no in-vocab preds)

```bash
bash tools/run_full_eval.sh \
    --weights checkpoints/ovmono3d_lift.pth \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --out     "$SMOKE_OUT/zeroshot" \
    --label   "zero-shot smoke" \
    --gt      datasets/Omni3D/WildBox_val.json \
    --skip-rel-ap3d
```

**Expected:** near-zero standard AP (closed-vocab pretraining has no
wildlife classes). Class-agnostic 2D AP in `summary_nhd.txt` should show
`AP@0.50` in the range 5–25 (proves the RPN transfers).

## 5. Short fine-tune (~1 hour, 2000 iters)

```bash
tmux new -s smoke-train
python tools/train_net.py \
    --config-file configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --num-gpus 1 \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.BASE_LR 0.002 \
    SOLVER.MAX_ITER 2000 \
    SOLVER.STEPS "(1200, 1800)" \
    SOLVER.WARMUP_ITERS 100 \
    SOLVER.CHECKPOINT_PERIOD 500 \
    TEST.EVAL_PERIOD 10000 \
    OUTPUT_DIR "$SMOKE_OUT/finetune"
# Ctrl-b d  to detach, tmux attach -t smoke-train to come back
```

`TEST.EVAL_PERIOD 10000 > MAX_ITER` means **no in-loop evals** — saves
time on the smoke. We eval once at the end in step 6.

**Expected:** loss drops from ~1.5 to ~0.4 over 2000 iters. Any NaN =
AMP instability → rerun with `SOLVER.AMP.ENABLED False`.

## 6. Fine-tuned eval with Rel-AP3D (~20 min)

```bash
bash tools/run_full_eval.sh \
    --weights "$SMOKE_OUT/finetune/model_final.pth" \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --out     "$SMOKE_OUT/finetuned_eval" \
    --label   "fine-tuned smoke" \
    --gt      datasets/Omni3D/WildBox_val.json
```

**Expected:** 2D AP50 in the 40–70 range, 3D AP 5–15, Rel-AP3D similar
to AP3D (best-scale ≈ 1.0). Per-class numbers show all 5 species
populated (if only one class is non-zero, symlink bug recurred).

## 7. Combined zero-shot vs fine-tuned paper report

```bash
python tools/make_report.py \
    --run-dir "$SMOKE_OUT/zeroshot"       --label "zero-shot" \
    --run-dir "$SMOKE_OUT/finetuned_eval" --label "fine-tuned" \
    --gt      datasets/Omni3D/WildBox_val.json \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
    --out     "$SMOKE_OUT/paper_report" \
    --compare

cat "$SMOKE_OUT/paper_report/report.md"
```

**Verify:**
- `Per-class annotation counts` lists all 5 species.
- Fine-tuned row has non-zero values in every per-class column of 2D AP.
- NHD-z line is the largest component of overall NHD.

## 8. Paper-figure visualizations (~1 min, 2D + 3D + novel view, with/without grid)

```bash
# Fine-tuned
python tools/visualize_class_agnostic.py \
    --preds "$SMOKE_OUT/finetune/inference/iter_final/WildBox_val/instances_predictions.pth" \
    --gt    datasets/Omni3D/WildBox_val.json \
    --out   "$SMOKE_OUT/vis_finetuned" \
    --top-k 5 --every 50 --limit 20

# Zero-shot (useful comparison)
python tools/visualize_class_agnostic.py \
    --preds "$SMOKE_OUT/zeroshot/inference/iter_final/WildBox_val/instances_predictions.pth" \
    --gt    datasets/Omni3D/WildBox_val.json \
    --out   "$SMOKE_OUT/vis_zeroshot" \
    --top-k 5 --every 50 --limit 20
```

Outputs:
```
$SMOKE_OUT/vis_finetuned/
    img_NNNNNN.jpg         ← 2×3 grid, novel view WITH ground grid
    img_NNNNNN_nogrid.jpg  ← same layout, novel view WITHOUT grid
```

Each file shows, top-to-bottom:
- Row 1 (GT):   2D boxes  |  3D wireframes  |  novel view (60° pitch)
- Row 2 (PRED): same three columns

## 9. Training curves (~5 seconds)

```bash
python tools/plot_training.py "$SMOKE_OUT/finetune/metrics.json"
ls "$SMOKE_OUT/finetune/training_curves.png"
```

## 10. Final file inventory

```bash
tree -L 2 "$SMOKE_OUT"
# or
find "$SMOKE_OUT" -maxdepth 3 -type f | sort
```

**Must exist:**
- `$SMOKE_OUT/finetune/model_final.pth`
- `$SMOKE_OUT/finetune/inference/iter_final/WildBox_val/instances_predictions.pth`
- `$SMOKE_OUT/finetune/metrics.json`
- `$SMOKE_OUT/finetune/training_curves.png`
- `$SMOKE_OUT/zeroshot/summary_nhd.txt`
- `$SMOKE_OUT/finetuned_eval/summary_nhd.txt`
- `$SMOKE_OUT/finetuned_eval/bev_ap.json`
- `$SMOKE_OUT/finetuned_eval/log.txt` (contains all 5 species per-class AP)
- `$SMOKE_OUT/finetuned_eval/log.rel.txt` (contains Rel-AP3D)
- `$SMOKE_OUT/paper_report/report.md`
- `$SMOKE_OUT/paper_report/table_main.tex`
- `$SMOKE_OUT/vis_finetuned/img_NNNNNN.jpg` × ≥20
- `$SMOKE_OUT/vis_finetuned/img_NNNNNN_nogrid.jpg` × ≥20

## Timing summary

| Step | Wall clock |
|---|---:|
| 2 — prep (first run, includes extractions) | 10–20 min |
| 2 — prep (cached) | <1 min |
| 4 — zero-shot eval | ~5 min |
| 5 — short fine-tune (2000 iters) | ~55 min |
| 6 — fine-tuned eval + Rel-AP3D | ~20 min |
| 7-9 — report + vis + plots | ~2 min |
| **Total** | **~1h 40m** (cached prep), **~2h** (first run) |

---

## If any step fails

Check these in order:

1. **Symlinks**: `cat configs/category_meta.json configs/wildbox/category_meta.json` — both must show `"thing_classes": ["giraffe", "zebra", "elephant", "rhino", "gazelle"]`.
2. **Env**: `python -c "import torch, pytorch3d, detectron2; from cubercnn.modeling.roi_heads import ROIHeads3D"`.
3. **Branch**: `git log --oneline -1` — should be at the latest commit on `wildbox_ovmono3d`.
4. **Data paths**: `ls /storage3/3DOM/vshukla/sam3/wd_data/wildbox/archive/*/WildBox_sam3-vggtv*.zip` — all 5 zips must be non-empty.
5. **Disk space**: `df -h /storage3` — need ~50 GB free for the 5 extracted zips.

See [WILDBOX_EXPERIMENT.md §2.8](WILDBOX_EXPERIMENT.md) for the category-
meta gotcha, [§15.1](WILDBOX_EXPERIMENT.md) for the full pre-flight list,
and [§21](WILDBOX_EXPERIMENT.md) for the current-state / resume context.

## What this proves (on a green run)

- Data prep with zip extraction ✓
- SAM3 tight 2D bbox pipeline ✓
- Video-level splits with deterministic seed ✓
- Category-meta consistency (both symlinks + stats.json) ✓
- Zero-shot pipeline + class-agnostic eval ✓
- Training loop + AMP + RepeatFactorSampler ✓
- Standard eval (2D AP, 3D AP) with correct per-class mapping ✓
- Rel-AP3D per-block scale search ✓
- BEV AP with dataset-id → contiguous-id mapping ✓
- Paper report generation (markdown + LaTeX) ✓
- OVMono3D-style novel-view visualization (with + without ground grid) ✓

If this smoke test passes end-to-end, the branch is ready for a full run
(see WILDBOX_EXPERIMENT.md §15 for the scaled-up playbook with all 13
zips and 15 000 training iterations).
