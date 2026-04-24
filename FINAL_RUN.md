# WildBox final paper run — step-by-step

Linear pipeline from raw zips → trained model → paper-ready report + figures. Every step is a single copy-paste block. For the "why" behind any command, see [WILDBOX_EXPERIMENT.md](WILDBOX_EXPERIMENT.md); this file is the "what" only.

**Cluster:** `node81` (A40), env at `/storage3/3DOM/vshukla/envs/ovmono3d`, repo at `/storage2/3DOM/vshukla/repos/ovmono3d`.

**Run time budget:**
- Data prep: ~15 min (if zips already extracted) or ~45 min (first-time extraction)
- Training × 3 seeds: ~10.5 h sequential (overnight)
- Eval × 3 seeds: ~2 h sequential
- Reports + figures: ~15 min

**Before you start:** cluster has internet? sbatch available? If yes, use sbatch (step 4b). Else srun + tmux (step 4a).

---

## 0. Prerequisites (one-time; skip if env already works)

```bash
ssh node81
conda activate /storage3/3DOM/vshukla/envs/ovmono3d
cd /storage2/3DOM/vshukla/repos/ovmono3d
git pull

# Must all succeed
python -c "import torch, pytorch3d, detectron2; print('torch:', torch.cuda.is_available())"
pip install shapely openpyxl matplotlib   # if not already installed
test -f checkpoints/ovmono3d_lift.pth && echo "OK: pretrained checkpoint present"
```

## 1. Register the 6 species in Omni3D stats

```bash
python tools/patch_stats_for_wildbox.py \
    --stats datasets/Omni3D/stats.json \
    --add giraffe:1000 grevys_zebra:1001 elephant:1002 plains_zebra:1003 rhino:1004 gazelle:1005
```

Idempotent. Safe to re-run.

## 2. Build train/val JSONs — all 15 zips, 6 species, video-level split

```bash
export ARCHIVE=/storage3/3DOM/vshukla/sam3/wd_data/wildbox/archive

python tools/prepare_wildbox_dataset.py \
    \
    --source ${ARCHIVE}/data202401KGiraffes/WildBox_sam3-vggtv1_processed.zip=giraffe:1000 \
    --source ${ARCHIVE}/data202501KGiraffes/WildBox_sam3-vggtv1_processed.zip=giraffe:1000 \
    \
    --source ${ARCHIVE}/data2023KABRZebras/WildBox_sam3-vggtv1_processed.zip=grevys_zebra:1001 \
    \
    --source ${ARCHIVE}/data202401KElephants/WildBox_sam3-vggtv1_processed.zip=elephant:1002 \
    --source ${ARCHIVE}/data202406KElephants/WildBox_sam3-vggtv1_processed.zip=elephant:1002 \
    --source ${ARCHIVE}/data202501KElephants/WildBox_sam3-vggtv1_processed.zip=elephant:1002 \
    --source ${ARCHIVE}/data202602KElephants/WildBox_sam3-vggtv1_processed.zip=elephant:1002 \
    \
    --source ${ARCHIVE}/data202307KZebras/WildBox_sam3-vggtv1_processed.zip=plains_zebra:1003 \
    --source ${ARCHIVE}/202401KZebras/WildBox_sam3-vggtv1_processed.zip=plains_zebra:1003 \
    --source ${ARCHIVE}/dataBZS/WildBox_sam3-vggtv1_processed.zip=plains_zebra:1003 \
    --source ${ARCHIVE}/wildbox_tomblair/WildBox_sam3-vggtv1_processed.zip=plains_zebra:1003 \
    \
    --source ${ARCHIVE}/data202401KRhinos/WildBox_sam3-vggtv1_processed.zip=rhino:1004 \
    --source ${ARCHIVE}/data202502KRhinoCamiV1/WildBox_sam3-vggtv1_processed.zip=rhino:1004 \
    --source ${ARCHIVE}/data202502KRhinoCamiV2/WildBox_sam3-vggtv1_processed.zip=rhino:1004 \
    \
    --source ${ARCHIVE}/data202406KGazelles/WildBox_sam3-vggtv1_processed.zip=gazelle:1005 \
    \
    --split-mode video --val-fraction 0.2 --seed 0 \
    --output-train datasets/Omni3D/WildBox_train.json \
    --output-val   datasets/Omni3D/WildBox_val.json \
    --dataset-id 1000 -v
```

**If any `--source` path doesn't exist** (maybe the zip is under a different internal filename), find it:
```bash
find ${ARCHIVE}/<campaign_dir>/ -name '*.zip' -o -name '*_unzipped' | head
```
and swap the `--source` line. Corrupt / in-progress zips are skipped with a warning.

## 3. Wire up BOTH category_meta symlinks (6 species)

```bash
ln -sf wildbox/category_meta_wildlife6.json configs/category_meta.json
ln -sf category_meta_wildlife6.json        configs/wildbox/category_meta.json

# Verify both show the SAME 6-class mapping
for p in configs/category_meta.json configs/wildbox/category_meta.json; do
    echo "--- $p ---"
    python -c "import json; d=json.load(open('$p')); print('classes:', d['thing_classes']); print('map:', d['thing_dataset_id_to_contiguous_id'])"
done
```

Expected:
```
classes: ['giraffe', 'grevys_zebra', 'elephant', 'plains_zebra', 'rhino', 'gazelle']
map: {'1000': 0, '1001': 1, '1002': 2, '1003': 3, '1004': 4, '1005': 5}
```

## 4. Dataset distribution report (auto-generated, paper-ready)

```bash
python tools/dataset_stats.py \
    --train datasets/Omni3D/WildBox_train.json \
    --val   datasets/Omni3D/WildBox_val.json \
    --out   datasets/Omni3D/dataset_stats

cat datasets/Omni3D/dataset_stats/dataset_stats.md
```

Produces:
- `datasets/Omni3D/dataset_stats/dataset_stats.md` — per-species counts (vids/segs/frames/boxes), bbox area-ratio distribution, paper one-liner
- `datasets/Omni3D/dataset_stats/dataset_stats.json` — same as above, machine-readable
- `datasets/Omni3D/dataset_stats/size_distribution.png` — per-species log-scale histogram of bbox-area-as-fraction-of-image

**Re-run this every time you change the prep** (new zips, different split seed, etc.) so the paper's numbers match the JSONs you trained on.

## 5. No-leakage + path-exists sanity check

```bash
python - <<'PY'
import json
train = json.load(open("datasets/Omni3D/WildBox_train.json"))
val   = json.load(open("datasets/Omni3D/WildBox_val.json"))
train_v = {im["file_path"].split("/")[-3] for im in train["images"]}
val_v   = {im["file_path"].split("/")[-3] for im in val["images"]}
assert not (train_v & val_v), f"video leakage: {train_v & val_v}"
print(f"OK: {len(train_v)} train vids, {len(val_v)} val vids, no overlap")
print(f"     {len(train['images'])} train frames, {len(val['images'])} val frames")

# File-exists check on first 10 of each
import os
for label, split in [("train", train), ("val", val)]:
    missing = sum(1 for im in split["images"][:100]
                  if not os.path.exists(os.path.join("datasets", im["file_path"])
                                        if not im["file_path"].startswith("/")
                                        else im["file_path"]))
    print(f"  {label}: {missing}/100 sampled images missing")
PY
```

Abort if any line fails.

## 6a. Launch multi-seed training via tmux + srun (if you don't have sbatch)

```bash
mkdir -p logs output/wl5_rt0.5_multiseed
tmux new -s multiseed
srun --gres=gpu:1 --time=24:00:00 --pty bash
# Inside the srun shell:
source /storage3/3DOM/vshukla/envs/ovmono3d/bin/activate
cd /storage2/3DOM/vshukla/repos/ovmono3d
CONFIG=configs/wildbox/OVMono3D_wildbox_wildlife6.yaml \
BASE_OUT=output/wl6_rt0.5_multiseed \
bash tools/run_multi_seed.sh > logs/multiseed.log 2>&1
# Ctrl-b d to detach tmux; srun + job keep running
# Later: tmux attach -t multiseed
```

## 6b. Launch multi-seed training via sbatch (preferred if available)

```bash
mkdir -p logs

# Submit
sbatch --export=ALL,CONFIG=configs/wildbox/OVMono3D_wildbox_wildlife6.yaml,BASE_OUT=output/wl6_rt0.5_multiseed \
    tools/run_multi_seed.sbatch

# Monitor
squeue -u $USER
tail -f logs/multiseed_*.log
```

**What this does:**
- 3 training seeds (0, 1, 2) sequentially
- Each: 10 k iters, batch 8, LR 2e-3, REPEAT_THRESHOLD=0.5, same pretrained init
- Full eval after each seed (standard AP + Rel-AP3D + BEV + class-agnostic + NHD)
- Writes to `output/wl6_rt0.5_multiseed/seed{0,1,2}/`
- Resumable: if a seed's `model_final.pth` exists on restart, it's skipped.

Expected completion: ~10-12 h. Walk away; come back tomorrow.

## 7. (Morning) verify all 3 seeds finished

```bash
for S in 0 1 2; do
    D=output/wl6_rt0.5_multiseed/seed$S
    echo "=== seed $S ==="
    ls -la $D/model_final.pth $D/eval/bev_ap.json $D/eval/log.txt 2>&1 | grep -v "^ls:"
done
```

Three `model_final.pth` + three `bev_ap.json` = all good.

## 8. Aggregate mean ± std across seeds

```bash
python tools/aggregate_seed_ap.py \
    --run-dirs output/wl6_rt0.5_multiseed/seed0/eval \
               output/wl6_rt0.5_multiseed/seed1/eval \
               output/wl6_rt0.5_multiseed/seed2/eval \
    --rare-classes giraffe gazelle grevys_zebra \
    --classes giraffe grevys_zebra elephant plains_zebra rhino gazelle \
    --out output/wl6_rt0.5_multiseed/mean_std_report

cat output/wl6_rt0.5_multiseed/mean_std_report/table_multiseed.md
```

Rare-class cells show `mean ± std`; stable-class cells show seed-0 value.

## 9. Zero-shot rows for the paper's main comparison table

```bash
# Paper-protocol zero-shot with GDino oracle (reuses existing oracle JSON)
bash tools/run_full_eval.sh \
    --weights checkpoints/ovmono3d_lift.pth \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife5_oracle2d.yaml \
    --out     output/wl6_zeroshot_oracle2d \
    --label   "zero-shot (paper protocol)" \
    --gt      datasets/Omni3D/WildBox_val.json \
    --skip-rel-ap3d

# Closed-vocab RPN-transfer zero-shot (stricter, supplementary)
bash tools/run_full_eval.sh \
    --weights checkpoints/ovmono3d_lift.pth \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife6.yaml \
    --out     output/wl6_zeroshot_rpn \
    --label   "zero-shot (RPN-transfer)" \
    --gt      datasets/Omni3D/WildBox_val.json \
    --skip-rel-ap3d
```

**Note:** if you added 4 new zebra zips + 2 new elephant zips + 1 new rhino zip in step 2, the `gdino_WildBox_val_oracle_2d.json` is stale (only covered the old val set). Regenerate it:

```bash
# Only needed if the val images changed from what the old oracle JSON covered
python tools/precompute_gdino_oracle.py \
    --gt       datasets/Omni3D/WildBox_val.json \
    --out      datasets/Omni3D/gdino_WildBox_val_oracle_2d.json \
    --species  rhino elephant plains_zebra grevys_zebra giraffe gazelle \
    --device   cuda \
    --box-threshold 0.15 --text-threshold 0.10
# ~2.5 h on A40
```

## 10. Build the final three-row paper report

```bash
# Seed-0 from the multi-seed run stands in for "fine-tuned (single-seed)" so
# make_report.py reads the standard log. Mean±std for rare classes is in
# table_multiseed.md (step 8) — reference in the paper text.

python tools/make_report.py \
    --run-dir output/wl6_zeroshot_rpn          --label "zero-shot (RPN-transfer)" \
    --run-dir output/wl6_zeroshot_oracle2d     --label "zero-shot (GDino oracle)" \
    --run-dir output/wl6_rt0.5_multiseed/seed0/eval  --label "fine-tuned (seed 0)" \
    --gt      datasets/Omni3D/WildBox_val.json \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife6.yaml \
    --out     output/paper_report_final \
    --compare

cat output/paper_report_final/report.md
```

## 11. Paper-figure visualizations

```bash
for RUN in \
    output/wl6_rt0.5_multiseed/seed0 \
    output/wl6_zeroshot_oracle2d \
    output/wl6_zeroshot_rpn; do
    python tools/visualize_class_agnostic.py \
        --preds ${RUN}/inference/iter_final/WildBox_val/instances_predictions.pth \
        --gt    datasets/Omni3D/WildBox_val.json \
        --out   ${RUN}/vis_agnostic \
        --top-k 5 --every 100 --limit 40
done
```

Each output dir contains `img_NNNNNN.jpg` (2×3 layout with ground grid) and `img_NNNNNN_nogrid.jpg` (same layout, grid off). Paper Figure 3 idea: stack the same `img_NNNNNN` from each of the three dirs as rows — (a) fine-tuned, (b) GDino-oracle zero-shot, (c) RPN-transfer zero-shot — for identical scene, different levels of domain adaptation.

## 12. Training curves (for appendix)

```bash
for S in 0 1 2; do
    python tools/plot_training.py output/wl6_rt0.5_multiseed/seed$S/metrics.json
done
ls output/wl6_rt0.5_multiseed/seed*/training_curves.png
```

## 13. Assemble paper numbers

After all of the above, the paper's Table 1 pulls from:
- Main metrics rows: `output/paper_report_final/report.md`
- Rare-class mean ± std: `output/wl6_rt0.5_multiseed/mean_std_report/table_multiseed.md`
- Dataset section: `datasets/Omni3D/dataset_stats/dataset_stats.md`
- Figures: `output/**/vis_agnostic/img_NNNNNN.jpg` and `size_distribution.png`

Copy the LaTeX tables (`table_main.tex`, `table_multiseed.tex`) directly into the manuscript.

---

## Troubleshooting

- **Symlink mismatch** (per-class metrics all 0 except one): verify both `configs/category_meta.json` and `configs/wildbox/category_meta.json` point to `wildlife6` — see step 3.
- **Video leakage**: step 5 catches it. If present, re-run step 2 with the same `--seed 0` — output is deterministic for a given seed.
- **Training crashes at iteration 0 (NaN)**: set `SOLVER.AMP.ENABLED False` as a CLI override.
- **sbatch job queued too long**: cancel and use step 6a (srun + tmux) instead.
- **After data re-prep, evaluator errors on unknown `category_id`**: re-run step 3 (symlink wiring) — the old symlinks pointed to wildlife5.
- **Training silently skipped, all per-class AP ≈ 0, `model_final.pth` byte-identical to pretrained** (the 2026-04-24 bug): check `grep -E "Starting training from iteration" logs/train_seed0.log`. If it says `iteration 116000` (or any large number), the pretrained checkpoint's stored `iteration` field confused the trainer. Fixed in commit f894ab6. Confirm `git log --oneline -3` shows that commit, then retrain. See WILDBOX_EXPERIMENT.md §3.1 for the belt-and-suspenders command to strip the `iteration` field from the pretrained.
