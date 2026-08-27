#!/bin/bash
#SBATCH --job-name=tadetr_a1
#SBATCH --partition=gpu-A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/storage2/3DOM/vshukla/runs/tadetr_a1/slurm_%j.log
# TA-DETR A1 on the A40. Resume-safe: resubmitting with the same command picks up last.pth.
source ~/.bashrc
conda activate /storage3/3DOM/vshukla/envs/ovmono3d
cd /storage2/3DOM/vshukla/repos/ovmono3d
mkdir -p /storage2/3DOM/vshukla/runs/tadetr_a1
python tools/tadetr/train_tadetr.py \
  --config configs/tadetr/a1.yaml,configs/tadetr/cluster_paths.yaml \
  --resume \
  train.out_dir='"/storage2/3DOM/vshukla/runs/tadetr_a1"'
