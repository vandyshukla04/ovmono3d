# RHINO Dataset Tools

This directory contains all tools and utilities for preparing and working with the RHINO wildlife detection dataset for OVMono3D training.

## Quick Start

### 1. Prepare Dataset from CUT3R Output

Run the complete pipeline to generate training-ready JSON annotations:

```bash
python rhino_tools/prepare_rhino_dataset.py \
    --cutr_videos /path/to/CUT3R/examples/wd_data/rhinos_cami \
    --cutr_results /path/to/CUT3R/results \
    --output_json_dir datasets/Omni3D \
    --output_image_dir datasets/rhino
```

This single script will:
- Find matched video-result pairs from CUT3R
- Generate `RHINO_train.json`, `RHINO_val.json`, `RHINO_test.json`
- Register rhino category (ID: 98) in `stats.json`
- Validate the dataset format
- Ensure all fields are correctly formatted (K matrix, dataset_id, category_id, etc.)

### 2. Train Model

```bash
python tools/train_net.py \
    --config-file configs/RHINO_train.yaml \
    --num-gpus 1
```

### 3. Run Inference

```bash
python rhino_tools/demo_rhino.py \
    --config-file configs/RHINO_train.yaml \
    --input-folder /path/to/test/images \
    --threshold 0.25 \
    --output output/rhino_inference
```

## Dataset Structure

```
datasets/
├── Omni3D/
│   ├── RHINO_train.json    # Training annotations (8 videos)
│   ├── RHINO_val.json      # Validation annotations (2 videos)
│   ├── RHINO_test.json     # Test annotations (2 videos)
│   └── stats.json          # Category registry (includes rhino: 98)
│
└── rhino/
    ├── 30_3/               # Video ID
    │   ├── 0000.jpg
    │   ├── 0008.jpg
    │   └── ...
    ├── 30_4/
    ├── 32_1/
    └── ...                 # 12 videos total
```

## Video Splits

**Training (8 videos):**
- 30_4, 32_1, 94_1, 35_1, 36_1, 35_2, 57_1, 35_3

**Validation (2 videos):**
- 90_1, 105_1

**Test (2 videos):**
- 30_3, 57_2

## JSON Format

Each JSON file follows the Omni3D format with proper 3D annotations:

```json
{
  "info": {
    "id": "rhino_train",           // lowercase dataset_id
    "known_category_ids": [98]
  },
  "categories": [
    {"id": 98, "name": "rhino", "supercategory": "animal"}
  ],
  "images": [
    {
      "id": 1,
      "file_path": "rhino/36_1/1720.jpg",
      "dataset_id": "rhino_train",  // Must match info.id
      "K": [                         // 3x3 nested list format
        [470.60, 0.0, 256.0],
        [0.0, 470.60, 144.0],
        [0.0, 0.0, 1.0]
      ],
      "height": 432,
      "width": 768
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 98,             // Must be 98 (rhino)
      "dataset_id": "rhino_train",   // Must match image dataset_id
      "bbox": [x, y, w, h],          // 2D bbox (XYWH)
      "center_cam": [x, y, z],       // 3D center in camera coords
      "dimensions": [w, h, l],       // 3D object dimensions
      "R_cam": [[...], [...], [...]],  // 3x3 rotation matrix
      "bbox3D_cam": [[...], ...],    // 8 corners of 3D box
      ...
    }
  ]
}
```

## Tools in This Directory

### Core Pipeline
- `prepare_rhino_dataset.py` - **Main script**: Complete dataset preparation pipeline

### Utilities
- `demo_rhino.py` - Inference script for rhino detection
- `reg_rhino.py` - Dataset registration helper
- `analyze_dimensions.py` - Analyze rhino dimension statistics

### Legacy Scripts (Archived)
The following scripts are no longer needed (replaced by `prepare_rhino_dataset.py`):
- ~~`regenerate_json.py`~~ - Merged into pipeline
- ~~`fix_category.py`~~ - No longer needed
- ~~`fix_k_format.py`~~ - No longer needed
- ~~`add_dataset_id.py`~~ - No longer needed
- ~~`add_rhino_to_stats.py`~~ - Merged into pipeline

## Common Issues & Solutions

### Issue: "KeyError: 'rhino_train' in dataset_id_to_unknown_cats"

**Solution:** The dataset_id must be lowercase. The pipeline script handles this automatically.

### Issue: "K matrix indexing error"

**Solution:** K must be a 3x3 nested list, not a flat list. The pipeline script formats this correctly.

### Issue: "Category ID mismatch"

**Solution:** Use category_id=98 consistently. The pipeline ensures this.

### Issue: "Missing dataset_id in annotations"

**Solution:** Both image and annotation entries need dataset_id. The pipeline adds this automatically.

## Training Configuration

Key settings in `configs/RHINO_train.yaml`:

```yaml
MODEL:
  WEIGHTS: "checkpoints/ovmono3d_lift.pth"  # Pre-trained OVMono3D
  BACKBONE:
    NAME: "build_dinov2_fpn_backbone"
    FREEZE_AT: 2

DATASETS:
  TRAIN: ("RHINO_train",)
  TEST: ("RHINO_val",)
  CATEGORY_NAMES: ['rhino']  # Single class

SOLVER:
  IMS_PER_BATCH: 4
  BASE_LR: 0.001
  MAX_ITER: 10000

INPUT:
  MIN_SIZE_TRAIN: (476,)
  MAX_SIZE_TRAIN: 644
```

## Performance Tips

1. **Use pre-trained weights**: Start from `ovmono3d_lift.pth` for better convergence
2. **Batch size**: Use 4-8 depending on GPU memory
3. **Learning rate**: 0.001 works well with frozen backbone
4. **Data augmentation**: Already included in DatasetMapper3D
5. **Validation**: Monitor AP3D metrics on validation set

## Debugging

### Check dataset registration
```python
from detectron2.data import DatasetCatalog
print(DatasetCatalog.list())  # Should include RHINO_train, etc.
```

### Verify JSON format
```bash
python rhino_tools/prepare_rhino_dataset.py  # Includes validation
```

### Analyze dimensions
```bash
python rhino_tools/analyze_dimensions.py
```

## Citation

If you use this RHINO dataset or tools, please cite:

```bibtex
@article{brazil2023omni3d,
  title={Omni3D: A Large Benchmark and Model for 3D Object Detection in the Wild},
  author={Brazil, Garrick and Abhinav, Kumar and Pons-Moll, Gerard and Liu, Xiaoming and Straub, Julian},
  journal={CVPR},
  year={2023}
}
```
