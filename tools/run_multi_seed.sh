#!/usr/bin/env bash
# Multi-seed training study for rare-class variance (reviewer #4).
#
# Runs 3 sequential training seeds with REPEAT_THRESHOLD=0.5 (reviewer
# #6), each starting from the same pretrained checkpoint. Seeds differ
# only in random init + data shuffling; data + hyperparameters
# otherwise identical. Each seed gets its own eval + report dir.
#
# Why 3 seeds: for rare classes (giraffe 4 vids, gazelle ~4 vids,
# Grévy's zebra 5 vids), single-seed per-class AP has high variance.
# mean±std across seeds is the ED-track-credible number.
#
# Why REPEAT_THRESHOLD=0.5: more aggressive upsampling of rare-class
# images. Current baseline (0.25) gave giraffe 3D AP=1.81, gazelle=2.27;
# the hope is 0.5 moves those into double-digits without hurting
# elephant/rhino much.
#
# Hardware: 1x A40, ~3.5h per seed at 10k iters → ~10.5h total.
# Launch with nohup and disown for overnight operation; log per-seed.
#
# Usage:
#   bash tools/run_multi_seed.sh                  # all 3 seeds
#   SEEDS="0 1" bash tools/run_multi_seed.sh      # just seeds 0 and 1
#   SKIP_EVAL=1 bash tools/run_multi_seed.sh      # train only; eval later
#
# After all seeds finish, aggregate with:
#   python tools/aggregate_seed_ap.py \
#     --run-dirs output/wl5_rt0.5_multiseed/seed*/eval \
#     --rare-classes giraffe gazelle grevys_zebra \
#     --out output/wl5_rt0.5_multiseed/mean_std_report

set -euo pipefail

# Fail fast if we haven't renamed zebra -> grevys_zebra yet (reviewer #3)
if grep -q '"zebra"' configs/wildbox/category_meta_wildlife5.json 2>/dev/null; then
    echo "ERROR: 'zebra' is still present in category_meta_wildlife5.json."
    echo "Run the rename first: python tools/rename_zebra_to_grevys.py"
    exit 1
fi

CONFIG="${CONFIG:-configs/wildbox/OVMono3D_wildbox_wildlife5.yaml}"
WEIGHTS="${WEIGHTS:-checkpoints/ovmono3d_lift.pth}"
BASE_OUT="${BASE_OUT:-output/wl5_rt0.5_multiseed}"
GT="${GT:-datasets/Omni3D/WildBox_val.json}"
SEEDS="${SEEDS:-0 1 2}"
MAX_ITER="${MAX_ITER:-10000}"
BATCH="${BATCH:-8}"
LR="${LR:-0.002}"
REPEAT_THRESHOLD="${REPEAT_THRESHOLD:-0.5}"

mkdir -p "$BASE_OUT" logs

echo "Multi-seed study"
echo "  config:     $CONFIG"
echo "  weights:    $WEIGHTS"
echo "  base out:   $BASE_OUT"
echo "  seeds:      $SEEDS"
echo "  iters:      $MAX_ITER"
echo "  rt:         $REPEAT_THRESHOLD"
echo "  started:    $(date)"
echo

for SEED in $SEEDS; do
    OUT="${BASE_OUT}/seed${SEED}"
    TRAIN_LOG="logs/train_seed${SEED}.log"
    EVAL_DIR="${OUT}/eval"

    echo "=== seed ${SEED}: train -> ${OUT}  (log: ${TRAIN_LOG}) ==="
    if [ -f "${OUT}/model_final.pth" ]; then
        echo "  model_final.pth exists; skipping training"
    else
        # Compute STEPS as (60%, 90%) of MAX_ITER
        STEP1=$((MAX_ITER * 60 / 100))
        STEP2=$((MAX_ITER * 90 / 100))
        python tools/train_net.py \
            --config-file "$CONFIG" \
            --num-gpus 1 \
            MODEL.WEIGHTS "$WEIGHTS" \
            SEED "$SEED" \
            DATALOADER.REPEAT_THRESHOLD "$REPEAT_THRESHOLD" \
            SOLVER.IMS_PER_BATCH "$BATCH" \
            SOLVER.BASE_LR "$LR" \
            SOLVER.MAX_ITER "$MAX_ITER" \
            SOLVER.STEPS "($STEP1, $STEP2)" \
            SOLVER.WARMUP_ITERS 500 \
            SOLVER.CHECKPOINT_PERIOD 5000 \
            TEST.EVAL_PERIOD "$MAX_ITER" \
            OUTPUT_DIR "$OUT" \
            > "$TRAIN_LOG" 2>&1
    fi

    if [ "${SKIP_EVAL:-0}" = "1" ]; then
        echo "  SKIP_EVAL=1; skipping eval for seed ${SEED}"
        continue
    fi

    echo "=== seed ${SEED}: eval -> ${EVAL_DIR} ==="
    if [ -f "${EVAL_DIR}/bev_ap.json" ]; then
        echo "  eval already exists; skipping"
    else
        bash tools/run_full_eval.sh \
            --weights "${OUT}/model_final.pth" \
            --config "$CONFIG" \
            --out "$EVAL_DIR" \
            --label "seed${SEED} (rt=${REPEAT_THRESHOLD})" \
            --gt "$GT"
    fi

    echo "=== seed ${SEED}: done at $(date) ==="
    echo
done

echo "All seeds complete at $(date)"
echo "Aggregate with:"
echo "  python tools/aggregate_seed_ap.py \\"
echo "    --run-dirs ${BASE_OUT}/seed*/eval \\"
echo "    --rare-classes giraffe gazelle grevys_zebra \\"
echo "    --out ${BASE_OUT}/mean_std_report"
