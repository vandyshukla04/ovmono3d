# WildBox — OVMono3D-LIFT eval / training code

Supplementary code for the WildBox monocular 3D wildlife detection benchmark
(anonymous review submission).

This repository extends **OVMono3D** (Yao et al., *Open Vocabulary Monocular
3D Object Detection*, arXiv:2411.16833) with the WildBox dataset configs,
training scripts, and evaluation pipeline used to produce the numbers in
the paper. The architecture itself is unchanged from upstream; for the
upstream README and credits, see [`ORIGINAL_OVMONO3D_README.md`](ORIGINAL_OVMONO3D_README.md).

---

## What's in this repo

| Path | Purpose |
|------|---------|
| `configs/wildbox/` | YAML configs for every reported run (5-species pretrain, 6-species fine-tune, GT-2D oracle, geometric baseline) + category metadata |
| `cubercnn/` | Cube R-CNN model code (mostly upstream from Yao et al.) |
| `datasets/` | Dataset registration utilities |
| `tools/train_net.py` | detectron2 training / evaluation entry point |
| `tools/run_full_eval.sh` | one-shot eval runner: produces 2D + 3D AP, BEV AP, NHD, and Rel-AP3D |
| `tools/run_multi_seed.sh` / `.sbatch` | multi-seed training (used for ± std bars in the paper) |
| `tools/run_init5sp_15k_singleseed.sh` | trains the headline curriculum-init checkpoint |
| `tools/run_gt2d_zs_eval.sh` | zero-shot evaluation with GT 2D box prompts |
| `tools/run_ovmono3d_geo_wildbox.sh` | runs the geometric baseline (LIFT-vs-GEO comparison) |
| `tools/assemble_paper_results.py` | aggregates all eval outputs into the paper tables |
| `tools/bev_ap_eval.py` | BEV AP @ {0.25, 0.5} |
| `tools/class_agnostic_eval.py` | class-agnostic AP + NHD breakdown |
| `tools/rel_ap3d_from_predictions.py` | Rel-AP3D (scale-invariant 3D AP) |
| `tools/aggregate_seed_ap.py` | computes mean ± std across seeds |
| `tools/precompute_gdino_oracle.py` | Grounding-DINO oracle 2D detections for the GT-2D protocol |
| `tools/build_gt2d_oracle_json.py` | builds the GT-2D-prompt eval JSON |
| `tools/prepare_wildbox_dataset.py` | one-time dataset preprocessing |

---

## Setup

```bash
bash setup.sh                 # creates conda env, installs detectron2 + deps
bash download_data.sh         # downloads pretrained weights (SAM, DINOv2, etc.)
```

The WildBox dataset itself is hosted with the paper supplementary materials —
download `WildBox_train_paper.json`, `WildBox_val_paper.json`, the per-video
zips, and extract them into `datasets/wildbox/` (extraction recipe is in the
dataset's `DATASET_README.md`).

---

## Reproducing the paper's headline numbers

The headline result is the **6-species curriculum fine-tune**
(Table 1, "init5sp curriculum, 10k iters" row), trained as:

1. 10 000 iters of 5-species pretraining on `wildlife5`.
2. 10 000 iters of 6-species continuation initialised from (1).

### Single-seed reproduction (sufficient for headline mean)

```bash
bash tools/run_init5sp_15k_singleseed.sh    # trains the curriculum-init ckpt
bash tools/run_full_eval.sh \
    --weights output/wildbox_wl6_init5sp_seed0/model_final.pth \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife6.yaml \
    --out     output/eval_init5sp_seed0 \
    --label   "curriculum init5sp seed 0"
```

The eval produces, in `output/eval_init5sp_seed0/`:

* `paper_report/report.md` – human-readable
* `paper_report/table_main.tex` – the LaTeX row used in the paper
* `paper_report/metrics.json` – machine-readable

### Multi-seed reproduction (for ± std bars)

```bash
bash tools/run_multi_seed.sh                # 3 seeds × init5sp curriculum
bash tools/run_multi_seed.sh CONFIG=configs/wildbox/OVMono3D_wildbox_wildlife6.yaml \
                              SEEDS="0 1 2"
python tools/aggregate_seed_ap.py output/wl6_init5sp_multiseed/seed*/eval/
```

### Pre-trained checkpoint

The headline checkpoint (`ovmono3d_lift_init5sp_seed0.pth`, 574 MB) is
provided alongside the dataset in the same anonymous supplementary upload
(see `CHECKPOINTS_README.md`). To skip training and go straight to eval:

```bash
bash tools/run_full_eval.sh \
    --weights /path/to/ovmono3d_lift_init5sp_seed0.pth \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife6.yaml \
    --out     output/eval_pretrained_headline \
    --label   "pretrained headline"
```

---

## Reproducing the ablation table

Direct (no curriculum) fine-tune at matched compute:

```bash
bash tools/run_multi_seed.sh CONFIG=configs/wildbox/OVMono3D_wildbox_wildlife6.yaml \
                              SEEDS="0 1 2" REPEAT_THRESHOLD=0.5 ITERS=15000
bash tools/run_multi_seed.sh CONFIG=configs/wildbox/OVMono3D_wildbox_wildlife6.yaml \
                              SEEDS="0"     REPEAT_THRESHOLD=0.35 ITERS=15000
```

Plateau check (init5sp at 15k):

```bash
ITERS=15000 bash tools/run_init5sp_15k_singleseed.sh
```

---

## Reproducing the zero-shot table

Vanilla zero-shot (no fine-tune):

```bash
bash tools/run_full_eval.sh \
    --weights output/upstream_omni3d_pretrained.pth \
    --config  configs/wildbox/OVMono3D_wildbox_wildlife6.yaml \
    --out     output/eval_zeroshot
```

GT-2D-prompt zero-shot (perfect-2D ceiling):

```bash
python tools/build_gt2d_oracle_json.py \
    --val datasets/Omni3D/WildBox_val_paper.json \
    --out datasets/Omni3D/WildBox_val_paper_gt2d.json
bash tools/run_gt2d_zs_eval.sh
```

Geometric baseline (LIFT vs GEO comparison):

```bash
bash tools/run_ovmono3d_geo_wildbox.sh
```

---

## Repo origin and licensing

This repo is a fork of [OVMono3D by Yao et al.](https://github.com/UVA-Computer-Vision-Lab/ovmono3d)
The architecture, the Cube R-CNN backbone, and the open-vocabulary 2D detector
integration are all from upstream — see `ORIGINAL_OVMONO3D_README.md` for full
attribution. The contributions in this fork are: WildBox dataset configs, the
6-species curriculum training schedule, the BEV / Rel-AP3D / NHD eval scripts,
and the multi-seed orchestration. License is unchanged from upstream
(see `LICENSE`).
