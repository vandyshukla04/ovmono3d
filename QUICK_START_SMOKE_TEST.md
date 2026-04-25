# Quick-start smoke test — end-to-end in ~1 hour

**Purpose**: prove the WildBox pipeline works start-to-finish before committing to the multi-hour full training run. Uses **one zip per species (smallest available)** and a short 2 000-iteration fine-tune. Numbers will be low — that's fine. Goal is to exercise every step.

**Run this**: after pulling new code, changing configs, or porting to a new cluster. Catches breakage before scaling.

**Pipeline this validates**: 6-species data prep → category-meta wiring → dataset stats → zero-shot RPN eval → fine-tune → 5-step full eval (standard / Rel-AP3D / BEV / class-agnostic / OVMono3D-style vis) → paper-results assembly.

**Cluster**: GPU steps on `node81` (A40); env `/storage3/3DOM/vshukla/envs/ovmono3d`. All commands use a fixed `output/smoke/` path so multiple terminals can run side-by-side without env-var coordination.

---

## 0. Setup (in tmux on node81)

```bash
ssh node81
conda activate /storage3/3DOM/vshukla/envs/ovmono3d
cd /storage2/3DOM/vshukla/repos/ovmono3d
git pull
mkdir -p output/smoke logs

# Fresh slate (optional)
# rm -rf output/smoke && mkdir -p output/smoke
```

**Use tmux** for the training step so it survives ssh disconnect:
```bash
tmux new -s smoke   # all commands below run inside this session
```

---

## 1. Smallest zip per species (6 zips)

| Species | Zip | Notes |
|---|---|---|
| giraffe | `data202401KGiraffes` | smaller of 2 giraffe zips |
| grevys_zebra | `data2023KABRZebras` | only Grévy's-zebra source (KABR reserve) |
| elephant | `data202406KElephants` | smallest elephant zip |
| plains_zebra | `data202307KZebras` | smallest plains zebra zip |
| rhino | `data202502KRhinoCamiV1` | smaller of the rhino zips |
| gazelle | `data202406KGazelles` | only gazelle source |

If any of these doesn't exist on your cluster, `ls /storage3/3DOM/vshukla/sam3/wd_data/wildbox/archive/` and pick another zip of the same species.

---

## 2. Register species in Omni3D stats (idempotent — safe to re-run)

```bash
python tools/patch_stats_for_wildbox.py \
    --stats datasets/Omni3D/stats.json \
    --add giraffe:1000 grevys_zebra:1001 elephant:1002 plains_zebra:1003 rhino:1004 gazelle:1005

# Sanity: all 6 must be in stats.json
python -c "
import json
s = json.load(open('datasets/Omni3D/stats.json'))
wanted = ['giraffe', 'grevys_zebra', 'elephant', 'plains_zebra', 'rhino', 'gazelle']
have = s.get('category_names', [])
missing = [c for c in wanted if c not in have]
print('missing:', missing if missing else 'none ✓')
"
```

---

## 3. Build train/val JSONs (video-level split, SAM3 tight bboxes)

```bash
export ARCHIVE=/storage3/3DOM/vshukla/sam3/wd_data/wildbox/archive

python tools/prepare_wildbox_dataset.py \
    --source ${ARCHIVE}/data202401KGiraffes/WildBox_sam3-vggtv1_processed.zip=giraffe:1000 \
    --source ${ARCHIVE}/data2023KABRZebras/WildBox_sam3-vggtv1_processed.zip=grevys_zebra:1001 \
    --source ${ARCHIVE}/data202406KElephants/WildBox_sam3-vggtv1_processed.zip=elephant:1002 \
    --source ${ARCHIVE}/data202307KZebras/WildBox_sam3-vggtv1_processed.zip=plains_zebra:1003 \
    --source ${ARCHIVE}/data202502KRhinoCamiV1/WildBox_sam3-vggtv1_processed.zip=rhino:1004 \
    --source ${ARCHIVE}/data202406KGazelles/WildBox_sam3-vggtv1_processed.zip=gazelle:1005 \
    --split-mode video --val-fraction 0.2 --seed 0 \
    --output-train output/smoke/WildBox_train.json \
    --output-val   output/smoke/WildBox_val.json \
    --dataset-id 1000 -v

# Use these (smoke-specific) JSONs for the rest of the smoke
export GT=output/smoke/WildBox_val.json
```

**First run takes 5–15 min** (extracts 6 zips). Subsequent runs reuse the cached `*_unzipped/` sibling dirs.

---

## 4. Wire up BOTH category_meta symlinks

```bash
ln -sf wildbox/category_meta_wildlife6.json configs/category_meta.json
ln -sf category_meta_wildlife6.json         configs/wildbox/category_meta.json

# Both must show the 6-species mapping in the same order
for p in configs/category_meta.json configs/wildbox/category_meta.json; do
    echo "=== $p ==="
    python -c "import json; d=json.load(open('$p')); print(d['thing_classes']); print(d['thing_dataset_id_to_contiguous_id'])"
done
# Expect: ['giraffe', 'grevys_zebra', 'elephant', 'plains_zebra', 'rhino', 'gazelle']
#         {'1000': 0, '1001': 1, '1002': 2, '1003': 3, '1004': 4, '1005': 5}
```

---

## 5. Generate dataset stats + no-leakage check

```bash
python tools/dataset_stats.py \
    --train output/smoke/WildBox_train.json \
    --val   output/smoke/WildBox_val.json \
    --out   output/smoke/dataset_stats

cat output/smoke/dataset_stats/dataset_stats.md

# Sanity — no video-level leakage + sampled paths exist
python - <<'PY'
import json, os
train = json.load(open('output/smoke/WildBox_train.json'))
val   = json.load(open('output/smoke/WildBox_val.json'))
tv = {im['file_path'].split('/')[-3] for im in train['images']}
vv = {im['file_path'].split('/')[-3] for im in val['images']}
assert not (tv & vv), f'video leakage: {tv & vv}'
print(f"OK: {len(tv)} train vids, {len(vv)} val vids, no overlap")
print(f"     {len(train['images'])} train frames / {len(val['images'])} val frames")

missing = sum(1 for im in val['images'][::max(1, len(val['images'])//20)][:20]
              if not os.path.exists(im['file_path']
                                    if im['file_path'].startswith('/')
                                    else 'datasets/' + im['file_path']))
print(f"{missing}/20 sampled paths missing")
PY
```

Abort if anything fails. All 6 species should appear in `dataset_stats.md` with non-zero train+val counts.

---

## 6. Zero-shot RPN-transfer eval (~5 min)

```bash
bash tools/run_full_eval.sh \
    --weights checkpoints/ovmono3d_lift.pth \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife6.yaml \
    --out     output/smoke/zeroshot_rpn \
    --label   "smoke zero-shot (RPN-transfer)" \
    --gt      $GT \
    --skip-rel-ap3d
```

Closed-vocab pretrained model on wildlife → expect near-zero per-class AP, but **class-agnostic 2D AP@0.5 ≈ 5–25** in `summary_nhd.txt` (proves the RPN finds animals even with wrong labels).

`--skip-rel-ap3d` because no in-vocab predictions makes scale search meaningless.

---

## 7. Short fine-tune (~30 min, 2 000 iters)

```bash
python tools/train_net.py \
    --config-file configs/wildbox/OVMono3D_wildbox_wildlife6.yaml \
    --num-gpus 1 \
    MODEL.WEIGHTS checkpoints/ovmono3d_lift.pth \
    DATASETS.TRAIN '("WildBox_train",)' \
    DATASETS.TEST  '("WildBox_val",)' \
    SEED 0 \
    DATALOADER.REPEAT_THRESHOLD 0.5 \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.BASE_LR 0.002 \
    SOLVER.MAX_ITER 2000 \
    SOLVER.STEPS "(1200, 1800)" \
    SOLVER.WARMUP_ITERS 100 \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    TEST.EVAL_PERIOD 10000 \
    OUTPUT_DIR output/smoke/finetune
```

### CRITICAL — verify training actually ran (the 2026-04-24 silent-skip bug)

```bash
grep -E "Starting training from iteration" output/smoke/finetune/log.txt
# REQUIRED: "Starting training from iteration 0 (resume=False)"
# If you see "iteration 116000" → the bugfix in commit f894ab6 isn't applied → git pull and retry.

grep -cE "iter: [0-9]+" output/smoke/finetune/log.txt
# REQUIRED: > 80 (D2 logs every ~20 iters; 2000 iters → ~100 lines)
# If 0: training was silently skipped. See WILDBOX_EXPERIMENT.md §3.1 for the four red flags.
```

**Expected loss trajectory**: 1.5–2.0 at iter 100 → 0.6–0.9 by iter 2000. NaN anywhere = AMP instability → re-run with `SOLVER.AMP.ENABLED False`.

---

## 8. Fine-tuned full eval (~10 min, all 5 stages)

```bash
bash tools/run_full_eval.sh \
    --weights output/smoke/finetune/model_final.pth \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife6.yaml \
    --out     output/smoke/finetuned_eval \
    --label   "smoke fine-tuned" \
    --gt      $GT
```

`run_full_eval.sh` runs 5 steps automatically:
1. **Standard 2D + 3D AP** — `log.txt`
2. **Rel-AP3D** (LabelAny3D scale-aligned) — `log.rel.txt`
3. **BEV AP** — `bev_ap.json`
4. **Class-agnostic + NHD surrogate** — `summary_nhd.txt`
5. **OVMono3D-style 2×3 visualizations** — `vis_ovmono3d/img_*.jpg`

**Expected on smoke (2 000 iters, 6 species)**:
- 2D AP50 in 30–60 range
- 3D AP 1–10
- Rel-AP3D similar to 3D AP (best-scale should be 0.8–1.5 if VGGT scale was learned)
- All 6 species populated in per-class table — if only 1 is non-zero, **symlink bug recurred** (re-run step 4)
- NHD-z dominant in disentangled NHD

---

## 9. (Optional) Paper-protocol zero-shot via GDino oracle

Skip this on first smoke; come back after you've invested the one-time precompute. Match the OVMono3D paper's zero-shot Table 1 protocol.

```bash
# (a) Install correct GDino package (one-time)
pip install --no-cache-dir groundingdino-py==0.4.0
# DO NOT use the github clone or groundingdino==0.1.0 — see WILDBOX_EXPERIMENT.md §3.1.

# (b) Download GDino checkpoint (one-time, ~1 GB)
mkdir -p checkpoints
test -f checkpoints/groundingdino_swinb_cogcoor.pth || \
  wget -O checkpoints/groundingdino_swinb_cogcoor.pth \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

# (c) Smoke 3 images first — should take < 30 s
python tools/precompute_gdino_oracle.py \
    --gt       $GT \
    --out      /tmp/gdino_smoke3.json \
    --species  rhino elephant plains_zebra grevys_zebra giraffe gazelle \
    --device   cuda \
    --box-threshold 0.15 --text-threshold 0.10 \
    --limit    3
# If per-img > 5 s/img → wrong package; re-check step (a).

# (d) Full precompute on smoke val (small, ~5-15 min instead of full 2.5h)
nohup python tools/precompute_gdino_oracle.py \
    --gt       $GT \
    --out      output/smoke/gdino_smoke_oracle.json \
    --species  rhino elephant plains_zebra grevys_zebra giraffe gazelle \
    --device   cuda \
    --box-threshold 0.15 --text-threshold 0.10 \
    --log-every 100 \
    > logs/smoke_gdino.log 2>&1 &
disown
# Wait for it; tail -f logs/smoke_gdino.log

# (e) Oracle zero-shot eval (~10 min)
# Note: writes the GDino JSON to the canonical location the config expects, OR
# create a temporary smoke oracle yaml. Easiest: copy the smoke oracle JSON over
# the canonical path for this run only (back up first if you have a real one):
test -f datasets/Omni3D/gdino_WildBox_val_oracle_2d.json && \
  cp datasets/Omni3D/gdino_WildBox_val_oracle_2d.json datasets/Omni3D/gdino_WildBox_val_oracle_2d.json.bak
cp output/smoke/gdino_smoke_oracle.json datasets/Omni3D/gdino_WildBox_val_oracle_2d.json

bash tools/run_full_eval.sh \
    --weights checkpoints/ovmono3d_lift.pth \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife6_oracle2d.yaml \
    --out     output/smoke/zeroshot_oracle \
    --label   "smoke zero-shot (paper protocol, GDino oracle)" \
    --gt      $GT \
    --skip-rel-ap3d

# Restore canonical GDino oracle if it existed
test -f datasets/Omni3D/gdino_WildBox_val_oracle_2d.json.bak && \
  mv datasets/Omni3D/gdino_WildBox_val_oracle_2d.json.bak datasets/Omni3D/gdino_WildBox_val_oracle_2d.json
```

---

## 10. Assemble the smoke paper-results sheet (~30 s)

```bash
python tools/assemble_paper_results.py \
    --finetuned-seeds output/smoke/finetuned_eval \
    --zeroshot-rpn    output/smoke/zeroshot_rpn \
    $([ -d output/smoke/zeroshot_oracle ] && echo "--zeroshot-oracle output/smoke/zeroshot_oracle") \
    --gt              $GT \
    --dataset-stats   output/smoke/dataset_stats/dataset_stats.md \
    --classes         giraffe grevys_zebra elephant plains_zebra rhino gazelle \
    --out             output/smoke/paper_results.md

cat output/smoke/paper_results.md
```

Single markdown with all numbers. **This is what passing the smoke means**: the file exists, has populated metric tables for both runs, includes the dataset facts, and the per-class column has all 6 species.

---

## 11. Training curves (~5 s)

```bash
python tools/plot_training.py output/smoke/finetune/metrics.json
ls output/smoke/finetune/training_curves.png
```

---

## 12. Final inventory

```bash
find output/smoke -maxdepth 4 -type f -name "*.json" -o -name "*.pth" -o -name "*.md" -o -name "*.txt" | sort
```

**Required artifacts (all must exist)**:
- `output/smoke/WildBox_{train,val}.json`
- `output/smoke/dataset_stats/dataset_stats.{md,json,png}`
- `output/smoke/finetune/model_final.pth`
- `output/smoke/finetune/log.txt` (with `iter: NNN` lines, NOT empty)
- `output/smoke/finetune/training_curves.png`
- `output/smoke/zeroshot_rpn/{log.txt,bev_ap.json,summary_nhd.txt,paper_report/report.md,vis_ovmono3d/}`
- `output/smoke/finetuned_eval/{log.txt,log.rel.txt,bev_ap.json,summary_nhd.txt,paper_report/report.md,vis_ovmono3d/}`
- `output/smoke/paper_results.md` (the master sheet)

---

## Timing summary

| Step | Wall clock |
|---|---:|
| 3 — data prep (first run, includes extraction) | 5–15 min |
| 3 — data prep (cached) | <1 min |
| 6 — zero-shot RPN-transfer eval | ~5 min |
| 7 — short fine-tune (2 000 iters) | ~30 min |
| 8 — fine-tuned full eval (5 steps) | ~10 min |
| 10–12 — assembly + curves + inventory | ~1 min |
| **Total (skipping §9 oracle)** | **~50 min** (cached prep), **~1 h** (first run) |
| 9 — paper-protocol oracle (one-time precompute) | +~15 min on smoke val |
| **Total including §9** | **+~25 min** after one-time precompute |

---

## If anything fails

In the order most likely to be wrong:

1. **Symlinks (§4 wrong)** — every per-class AP is 0 except one, OR
   `ValueError: 'plains_zebra' is not in list` from `register_and_store_model_metadata`.
   Fix: re-run §4 commands. Both `configs/category_meta.json` and `configs/wildbox/category_meta.json` must point to `category_meta_wildlife6.json`.

2. **Stats not patched (§2 skipped)** — `ValueError: 'plains_zebra' is not in list` at training start.
   Fix: re-run §2.

3. **Training silently skipped** (§7 fails the `iteration 0` check) — `model_final.pth` is byte-identical to the pretrained, no `iter: NNN` lines in `log.txt`.
   Fix: `git pull` (must be at commit `f894ab6` or later). See [WILDBOX_EXPERIMENT.md §3.1](WILDBOX_EXPERIMENT.md) for the four red flags.

4. **NaN training loss** (§7 trajectory) — AMP instability.
   Fix: add `SOLVER.AMP.ENABLED False` to the §7 command.

5. **Missing zip** (§3 fails) — one of the 6 source zip paths doesn't exist.
   Fix: `ls /storage3/3DOM/vshukla/sam3/wd_data/wildbox/archive/` and pick a substitute zip of the same species.

6. **Disk space** — extracting 6 zips needs ~30 GB free.
   Fix: `df -h /storage3` before starting.

For deeper debugging, see [WILDBOX_EXPERIMENT.md §3.1](WILDBOX_EXPERIMENT.md) (env issues), [§21.2](WILDBOX_EXPERIMENT.md) (bugs we caught with diagnostics).

---

## What this smoke proves on a green run

- 6-species data prep with zip extraction + caching ✓
- SAM3 tight 2D bbox pipeline ✓
- Video-level split with deterministic seed (no leakage) ✓
- Auto-generated dataset_stats.md (paper-ready inventory) ✓
- Category_meta consistency (BOTH symlinks point to wildlife6) ✓
- stats.json registration of all 6 species (including grevys_zebra + plains_zebra) ✓
- NUM_CLASSES=6 head reinitialization from 50-class pretrained ✓
- Iteration-skip-training fix (training actually runs) ✓
- Standard 2D + 3D AP eval with correct class mapping ✓
- Rel-AP3D per-block scale search ✓
- BEV AP with dataset-id → contiguous-id normalization for both fine-tuned and oracle predictions ✓
- Class-agnostic 2D AP + NHD diagnostic ✓
- OVMono3D-style 2×3 visualizations (auto via run_full_eval step 5) ✓
- Single-sheet paper-results assembly ✓

Once green, scale up to the full 15-zip run via [FINAL_RUN.md](FINAL_RUN.md).
