#!/usr/bin/env bash
# OVMono3D-LIFT, 5-species-checkpoint init, single seed, 15k iters.
#
# Tests the "10k is enough; init5sp plateaus past that" claim from the paper:
# our existing init5sp 3-seed runs all cap at 10k. Going to 15k either
# confirms the plateau (claim holds) or reveals headroom (claim breaks and
# we revise the §4 narrative). Single seed is enough to detect a real trend
# because the rare-class noise floor was already mapped by the 3-seed 10k
# study.
#
# Run from repo root, with one idle GPU available:
#   tmux new -s init5sp15k
#   CUDA_VISIBLE_DEVICES=N bash tools/run_init5sp_15k_singleseed.sh
#
# Output:
#   output/wl6_init5sp_15k_seed0/seed0/   — checkpoints + train log
#   output/wl6_init5sp_15k_seed0/seed0/eval/  — full eval + paper_report
#
# After it finishes, re-run sanity audit to fold the row in:
#   bash tools/run_sanity_audit_cluster.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Match the existing 3-seed init5sp setup; only MAX_ITER and BASE_OUT change.
# (Verified against output/wl6_init5sp_*/seed*/config.yaml on /mnt/d:
#  WEIGHTS = 5-species fine-tuned model, RT = 0.5, classes = 6.)
export CONFIG="${CONFIG:-configs/wildbox/OVMono3D_wildbox_wildlife6.yaml}"
export WEIGHTS="${WEIGHTS:-output/wildbox_wl5_finetune/model_final.pth}"
export BASE_OUT="${BASE_OUT:-output/wl6_init5sp_15k_seed0}"
export SEEDS="${SEEDS:-0}"
export MAX_ITER="${MAX_ITER:-15000}"
export REPEAT_THRESHOLD="${REPEAT_THRESHOLD:-0.5}"
export GT="${GT:-datasets/Omni3D/WildBox_val.json}"

# Pre-flight: weights file must exist (silent skip-train regression risk
# was the 2026-04-24 incident; explicit check kills another whole class
# of "ran for 10 minutes, model_final.pth byte-identical to start" bugs).
if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: weights not found at $WEIGHTS" >&2
    echo "Set WEIGHTS=<path> if your 5-species checkpoint lives elsewhere." >&2
    exit 1
fi

# Pre-flight GPU check (mirrors run_ovmono3d_geo_wildbox.sh).
if command -v nvidia-smi >/dev/null && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    DEV="${CUDA_VISIBLE_DEVICES%%,*}"
    USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
              -i "$DEV" 2>/dev/null | tr -d ' ' || echo 0)
    if [ "$USED_MB" -gt 2048 ] 2>/dev/null; then
        echo "!!! GPU $DEV has ${USED_MB} MiB in use by another process." >&2
        nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv >&2
        exit 3
    fi
    echo "==> GPU $DEV: ${USED_MB} MiB used at launch (OK)"
fi

echo "==> init5sp single-seed @ 15k iters"
echo "    config:  $CONFIG"
echo "    weights: $WEIGHTS"
echo "    out:     $BASE_OUT"
echo "    iter:    $MAX_ITER"
echo

exec bash tools/run_multi_seed.sh
