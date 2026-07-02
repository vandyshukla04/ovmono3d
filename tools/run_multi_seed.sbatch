#!/usr/bin/env bash
#SBATCH --job-name=wildbox_multiseed
#SBATCH --output=logs/multiseed_%j.log
#SBATCH --error=logs/multiseed_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Multi-seed training via sbatch. Detaches cleanly — laptop lid can close,
# SSH can drop; job keeps running. Monitor with `squeue -u $USER` and
# `tail -f logs/multiseed_<jobid>.log`.
#
# Submit:   sbatch tools/run_multi_seed.sbatch
# Cancel:   scancel <jobid>
# Status:   squeue -u $USER
#
# Edit the resources above if your cluster needs a partition/qos flag
# (e.g. `#SBATCH --partition=gpu`, `#SBATCH --qos=normal`). Leave
# time=24h; multi-seed is ~10-12h, buffer accounts for val-set eval +
# Rel-AP3D scale search.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Activate env (adjust the path below to point at your local OVMono3D conda env)
source "${OVMONO3D_ENV:-$HOME/envs/ovmono3d}/bin/activate" 2>/dev/null || \
  conda activate "${OVMONO3D_ENV:-ovmono3d}"

# Defer to the bash orchestrator; inherit env overrides from caller
#   sbatch --export=CONFIG=...,SEEDS="0 1 2" tools/run_multi_seed.sbatch
bash tools/run_multi_seed.sh
