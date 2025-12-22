# RHINO Dataset Training Guide for OVMono3D

Complete guide for training OVMono3D on custom RHINO wildlife detection data.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Troubleshooting](#troubleshooting)

## Overview

This project extends **OVMono3D** (Open-Vocabulary Monocular 3D Object Detection) to detect rhinos in wildlife camera footage. OVMono3D uses foundation models (DINO v2) to perform 3D bounding box detection from single images.

**Key Features:**
- Single-class detection (rhino, category_id: 98)
- Uses pre-trained OVMono3D weights for transfer learning
- DINO v2 ViT-B/14 backbone with frozen features
- 3D bounding boxes from CUT3R reconstructions
- 12 video sequences (8 train, 2 val, 2 test)

## Project Structure

```
ovmono3d/
├── cubercnn/              # Core 3D detection framework
│   ├── modeling/          # Models (backbones, ROI heads, etc.)
│   ├── data/              # Dataset loading and augmentation
│   └── config/            # Configuration system
│
├── configs/
│   ├── RHINO_train.yaml   # RHINO training configuration
│   └── OVMono3D_*.yaml    # Original OVMono3D configs
│
├── rhino_tools/           # RHINO-specific tools
│   ├── prepare_rhino_dataset.py  # Main data preparation pipeline
│   ├── demo_rhino.py      # Inference script
│   ├── reg_rhino.py       # Dataset registration
│   ├── analyze_dimensions.py  # Dimension analysis
│   ├── legacy/            # Old scripts (archived)
│   └── README.md          # Detailed tool documentation
│
├── datasets/
│   ├── Omni3D/            # JSON annotations
│   │   ├── RHINO_train.json
│   │   ├── RHINO_val.json
│   │   ├── RHINO_test.json
│   │   └── stats.json     # Category registry
│   └── rhino/             # Image data (12 video folders)
│
├── checkpoints/           # Pre-trained model weights
│   └── ovmono3d_lift.pth  # OVMono3D pre-trained checkpoint
│
├── output/                # Training outputs
│   └── rhino_cubercnn_b4_ovmono_ckpt/  # Best training run
│
├── tools/
│   ├── train_net.py       # Training script
│   └── ovmono3d_geo.py    # Inference script
│
└── GroundingDINO/         # Grounding DINO submodule (optional)
```

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <your-repo-url>
cd ovmono3d

# Install dependencies (PyTorch, Detectron2, etc.)
pip install -r requirements.txt

# Download pre-trained weights
wget -P checkpoints/ <ovmono3d_lift.pth URL>
```

### 2. Prepare RHINO Dataset

**One-step preparation** (replaces all manual fix scripts):

```bash
python rhino_tools/prepare_rhino_dataset.py \
    --cutr_videos /path/to/CUT3R/examples/wd_data/rhinos_cami \
    --cutr_results /path/to/CUT3R/results \
    --output_json_dir datasets/Omni3D \
    --output_image_dir datasets/rhino
```

This automatically:
- ✓ Finds matched video-result pairs
- ✓ Generates train/val/test JSONs with correct format
- ✓ Registers rhino (ID: 98) in stats.json
- ✓ Validates all fields (K matrix, dataset_id, category_id)
- ✓ No more manual fix scripts needed!

### 3. Train Model

```bash
python tools/train_net.py \
    --config-file configs/RHINO_train.yaml \
    --num-gpus 1
```

Monitor training:
```bash
# View metrics
cat output/rhino_cubercnn_b4_ovmono_ckpt/metrics.json

# TensorBoard (if configured)
tensorboard --logdir output/rhino_cubercnn_b4_ovmono_ckpt
```

### 4. Run Inference

```bash
python rhino_tools/demo_rhino.py \
    --config-file configs/RHINO_train.yaml \
    --input-folder /path/to/test/images \
    --threshold 0.25 \
    --output output/rhino_inference
```

## Dataset Preparation

### Prerequisites

Your CUT3R output should have this structure:

```
CUT3R/
├── examples/wd_data/rhinos_cami/
│   ├── rhin-30_3/
│   ├── rhin-30_4/
│   └── ...
│
└── results/
    ├── tmp-rhin-30_3-revisit-1/
    │   ├── bounding_boxes/    # *.json (3D boxes)
    │   ├── camera/             # *.npz (K matrices)
    │   └── ...
    └── ...
```

### Running the Pipeline

The unified preparation script handles everything:

```bash
python rhino_tools/prepare_rhino_dataset.py
```

**What it does:**

1. **Finds matched pairs**: Matches video directories with result directories
2. **Generates annotations**: Creates properly formatted JSON files
3. **Registers category**: Adds rhino (ID: 98) to stats.json
4. **Validates format**: Ensures all fields are correct

**Output:**
```
datasets/Omni3D/
├── RHINO_train.json    # 8 videos, ~500-1000 images
├── RHINO_val.json      # 2 videos
├── RHINO_test.json     # 2 videos
└── stats.json          # Updated with rhino category
```

### Data Splits

- **Train (8 videos)**: 30_4, 32_1, 94_1, 35_1, 36_1, 35_2, 57_1, 35_3
- **Val (2 videos)**: 90_1, 105_1
- **Test (2 videos)**: 30_3, 57_2

To customize splits, edit `DEFAULT_SPLITS` in `prepare_rhino_dataset.py`.

### Validation

The pipeline includes automatic validation:
```
✓ info.id
✓ categories
✓ images
✓ annotations
✓ Image: K matrix (3x3)
✓ Image: dataset_id
✓ Annotation: category_id (98)
✓ Annotation: dataset_id
```

## Training

### Configuration

Key settings in [configs/RHINO_train.yaml](configs/RHINO_train.yaml):

```yaml
MODEL:
  WEIGHTS: "checkpoints/ovmono3d_lift.pth"  # Pre-trained weights
  BACKBONE:
    NAME: "build_dinov2_fpn_backbone"
    FREEZE_AT: 2  # Freeze early layers

DATASETS:
  TRAIN: ("RHINO_train",)
  TEST: ("RHINO_val",)
  CATEGORY_NAMES: ['rhino']

SOLVER:
  IMS_PER_BATCH: 4          # Batch size
  BASE_LR: 0.001            # Learning rate
  MAX_ITER: 10000           # Training iterations
  CHECKPOINT_PERIOD: 1000   # Save every 1000 iters

INPUT:
  MIN_SIZE_TRAIN: (476,)    # Input image size
  MAX_SIZE_TRAIN: 644
```

### Training Command

```bash
python tools/train_net.py \
    --config-file configs/RHINO_train.yaml \
    --num-gpus 1 \
    OUTPUT_DIR output/my_rhino_training
```

### Multi-GPU Training

```bash
python tools/train_net.py \
    --config-file configs/RHINO_train.yaml \
    --num-gpus 4 \
    SOLVER.IMS_PER_BATCH 16
```

### Resume Training

```bash
python tools/train_net.py \
    --config-file configs/RHINO_train.yaml \
    --resume \
    MODEL.WEIGHTS output/rhino_cubercnn_b4_ovmono_ckpt/model_final.pth
```

### Training Tips

1. **Use pre-trained weights**: Always start from `ovmono3d_lift.pth`
2. **Monitor validation**: Check AP3D on validation set
3. **Adjust learning rate**: Reduce if loss plateaus
4. **Batch size**: 4-8 works well on single GPU
5. **Data augmentation**: Already included in DatasetMapper3D

## Inference

### Basic Inference

```bash
python rhino_tools/demo_rhino.py \
    --config-file configs/RHINO_train.yaml \
    --input-folder test_images/ \
    --output output/inference \
    --threshold 0.25
```

### Custom Camera Parameters

```bash
python rhino_tools/demo_rhino.py \
    --config-file configs/RHINO_train.yaml \
    --input-folder test_images/ \
    --focal-length 500.0 \
    --principal-point 384.0 216.0 \
    --threshold 0.3
```

### Output Format

The demo script generates:
- **Visualizations**: Images with 3D boxes overlaid
- **JSON detections**: Structured detection results
- **Top-down view**: Bird's eye view of detections

## Troubleshooting

### Common Issues

#### 1. KeyError: 'rhino_train' in dataset_id_to_unknown_cats

**Cause:** Old version of `dataset_mapper.py` that doesn't handle new datasets gracefully.

**Fix:** The fix is already applied in [cubercnn/data/dataset_mapper.py:47](cubercnn/data/dataset_mapper.py#L47):
```python
unknown_categories = self.dataset_id_to_unknown_cats.get(dataset_id, [])
```

#### 2. K matrix indexing error

**Cause:** K matrix stored as flat list instead of 3x3 nested list.

**Fix:** The preparation pipeline now formats K correctly:
```python
K_nested = [
    [float(K[0,0]), float(K[0,1]), float(K[0,2])],
    [float(K[1,0]), float(K[1,1]), float(K[1,2])],
    [float(K[2,0]), float(K[2,1]), float(K[2,2])]
]
```

#### 3. Category ID mismatch

**Cause:** Inconsistent category IDs (98 vs 99 vs 1000).

**Fix:** Pipeline uses `RHINO_CATEGORY_ID = 98` consistently everywhere.

#### 4. Missing dataset_id in annotations

**Cause:** Annotations missing the `dataset_id` field.

**Fix:** Pipeline adds `dataset_id` to both images and annotations automatically.

#### 5. No training progress / Loss not decreasing

**Possible causes:**
- Learning rate too high/low
- Incorrect data format
- Missing pre-trained weights

**Debug steps:**
```bash
# Check data loading
python -c "from detectron2.data import DatasetCatalog; print(DatasetCatalog.list())"

# Verify JSON format
python rhino_tools/prepare_rhino_dataset.py

# Check metrics
cat output/rhino_cubercnn_b4_ovmono_ckpt/metrics.json
```

### Legacy Scripts (No Longer Needed)

The following scripts in `rhino_tools/legacy/` are now **obsolete**:

- ~~`regenerate_json.py`~~ → Use `prepare_rhino_dataset.py`
- ~~`fix_category.py`~~ → Fixed automatically in pipeline
- ~~`fix_k_format.py`~~ → Fixed automatically in pipeline
- ~~`add_dataset_id.py`~~ → Added automatically in pipeline
- ~~`add_rhino_to_stats.py`~~ → Integrated into pipeline

**You should only need to run `prepare_rhino_dataset.py` once!**

## Performance Benchmarks

### Best Training Run

Location: `output/rhino_cubercnn_b4_ovmono_ckpt/`

**Configuration:**
- Pre-trained: ovmono3d_lift.pth
- Backbone: DINO v2 ViT-B/14 (frozen)
- Batch size: 4
- Learning rate: 0.001
- Iterations: 10,000

**Results:** (Check `metrics.json` for actual numbers)

## Advanced Topics

### Custom Backbone

To use a different backbone, modify `configs/RHINO_train.yaml`:

```yaml
MODEL:
  BACKBONE:
    NAME: "build_resnet_fpn_backbone"  # or build_clip_fpn_backbone, etc.
```

### Data Augmentation

Augmentation is handled in `cubercnn/data/dataset_mapper.py`:
- Random resizing
- Horizontal flipping (with 3D pose correction)
- Color jittering

### Adding More Categories

To train on multiple animal species:

1. Update category list in preparation script
2. Modify `RHINO_train.yaml`:
   ```yaml
   DATASETS:
     CATEGORY_NAMES: ['rhino', 'elephant', 'lion']
   ```
3. Update stats.json with new categories

## Citation

```bibtex
@article{brazil2023omni3d,
  title={Omni3D: A Large Benchmark and Model for 3D Object Detection in the Wild},
  author={Brazil, Garrick and others},
  journal={CVPR},
  year={2023}
}
```

## Support

- **Issues**: Create an issue on GitHub
- **Documentation**: See `rhino_tools/README.md` for tool-specific docs
- **Original OVMono3D**: Check the main README.md

---

**Quick Reference Commands:**

```bash
# Prepare dataset
python rhino_tools/prepare_rhino_dataset.py

# Train
python tools/train_net.py --config-file configs/RHINO_train.yaml --num-gpus 1

# Inference
python rhino_tools/demo_rhino.py --config-file configs/RHINO_train.yaml --input-folder images/

# Analyze dimensions
python rhino_tools/analyze_dimensions.py
```
