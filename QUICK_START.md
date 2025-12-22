# RHINO Training Quick Start

**One-page guide to get started with RHINO dataset training.**

---

## ✅ Verification

Your setup is ready! Run this to verify:

```bash
python rhino_tools/verify_installation.py
```

---

## 📊 Dataset Status

- **Training**: 728 images, 1217 annotations (8 videos)
- **Validation**: 180 images, 180 annotations (2 videos)  
- **Test**: 175 images, 175 annotations (2 videos)
- **Category**: Rhino (ID: 98)
- **Checkpoint**: ovmono3d_lift.pth (575 MB) ✓

---

## 🚀 Three Commands to Success

### 1️⃣ Prepare Dataset (if needed)

```bash
python rhino_tools/prepare_rhino_dataset.py
```

**Only run this if you need to regenerate from CUT3R output!**
Your dataset is already prepared and validated.

### 2️⃣ Train Model

```bash
python tools/train_net.py \
    --config-file configs/RHINO_train.yaml \
    --num-gpus 1
```

Training will start from pre-trained OVMono3D weights.
Monitor progress in `output/rhino_cubercnn_b4_ovmono_ckpt/`

### 3️⃣ Run Inference

```bash
python rhino_tools/demo_rhino.py \
    --config-file configs/RHINO_train.yaml \
    --input-folder /path/to/test/images \
    --threshold 0.25 \
    --output output/my_inference
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `RHINO_TRAINING_GUIDE.md` | Complete training guide |
| `CLEANUP_SUMMARY.md` | What was cleaned up |
| `rhino_tools/README.md` | Tool documentation |
| `configs/RHINO_train.yaml` | Training configuration |
| `output/rhino_cubercnn_b4_ovmono_ckpt/` | Best trained model |

---

## 🆘 Need Help?

**Common issues:** See [RHINO_TRAINING_GUIDE.md](RHINO_TRAINING_GUIDE.md#troubleshooting)

**Tool details:** See [rhino_tools/README.md](rhino_tools/README.md)

**Verify setup:** Run `python rhino_tools/verify_installation.py`

---

## 🎯 What Changed After Cleanup

### Before ❌
- 8 separate scripts scattered in root
- Multi-step manual workflow
- Unclear execution order
- No validation
- Poor documentation

### After ✅  
- Organized `rhino_tools/` directory
- Single pipeline: `prepare_rhino_dataset.py`
- Automatic validation
- Comprehensive docs
- Professional structure

---

**Ready to train? Run step 2️⃣ above!**
