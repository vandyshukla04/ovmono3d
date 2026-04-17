#!/usr/bin/env bash
# OVMono3D environment setup. Run AFTER:
#   conda activate <env>
#   pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
#
# Requirements on the shell:
#   - nvcc on PATH, CUDA_HOME set (needed for pytorch3d + GroundingDINO CUDA kernels).
#     Example:
#       module load cuda/12.1
#       export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
#       export TORCH_CUDA_ARCH_LIST="8.6"   # A40 = sm_86
#   - C++ compiler (gcc >= 9, <= 12 is safe).
set -euo pipefail

# Older ML packages (detectron2, pytorch3d 055ab3a) expect numpy 1.x.
pip install "numpy<2"

# PEP-517 isolated builds can't see our torch install, so disable isolation
# for every source-built package below. Keep the ninja/wheel deps present
# in the active env.
pip install ninja setuptools wheel

pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@055ab3a
pip install --no-build-isolation git+https://github.com/yaojin17/detectron2.git  # slightly modified detectron2 for OVMono3D
pip install cython opencv-python scipy pandas einops open_clip_torch open3d

pip install --no-build-isolation git+https://github.com/apple/ml-depth-pro.git@b2cd0d5
pip install --no-build-isolation git+https://github.com/facebookresearch/segment-anything.git@dca509f
pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git@856dde2

mkdir -p checkpoints
# Skip re-downloading if the 4 checkpoints are already present.
if [ ! -f ./checkpoints/groundingdino_swinb_cogcoor.pth ]; then
    wget -P ./checkpoints/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
fi
if [ ! -f ./checkpoints/depth_pro.pt ]; then
    wget -P checkpoints https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt
fi
if [ ! -f ./checkpoints/sam_vit_h_4b8939.pth ]; then
    wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi
if [ ! -f ./checkpoints/ovmono3d_lift.pth ]; then
    huggingface-cli download uva-cv-lab/ovmono3d_lift ovmono3d_lift.pth --local-dir checkpoints
fi
