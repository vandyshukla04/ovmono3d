#!/usr/bin/env bash
# Run OVMono3D-GEO inference + evaluation on WildBox_val for both zero-shot
# protocols, mirroring the wl6_zeroshot_oracle2d / wl6_zeroshot_rpn pair we
# already have for OVMono3D-LIFT. Adds two rows to the main results table
# from a *different model family* (geometric: SAM + Depth Pro + PCA, no
# learned cube head).
#
# GEO is a purely geometric pipeline (SAM + Depth Pro + PCA) that consumes
# 2D boxes from an *external* source — there is no "RPN-transfer" baseline
# for GEO because GEO has no learned RPN of its own. The cross-protocol
# story (oracle vs RPN as the 2D source) is already covered by the two LIFT
# zero-shot rows; for GEO we only run the oracle-2D protocol.
#
# Run from repo root, with cluster GPU available:
#   bash tools/run_ovmono3d_geo_wildbox.sh                   # oracle2d (default)
#   bash tools/run_ovmono3d_geo_wildbox.sh oracle2d          # explicit
#   bash tools/run_ovmono3d_geo_wildbox.sh rpn-from-lift     # opt-in: feed
#                                                              GEO with LIFT's
#                                                              own RPN preds as
#                                                              the 2D source
#                                                              (advanced)
#
# Outputs:
#   output/wl6_geo_oracle2d/        — GEO predictions + paper_report
#   output/wl6_geo_*/log.txt        — full inference + eval log
#
# Prereqs (one-time):
#   pip install depth_pro segment_anything open3d scikit-learn
#   ./checkpoints/sam_vit_h_4b8939.pth                       — SAM weights
#   Depth Pro weights cached at depth_pro's default location
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PROTOCOL="${1:-oracle2d}"

# Cuda fragmentation mitigation — Depth Pro's residual blocks hit OOM on
# 44GB GPUs once the card is shared with a second torch process; expandable
# segments lets allocations grow into freed regions instead of fragmenting.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Pre-flight GPU sanity: refuse to launch if the pinned device already has
# >2 GB used by another process. Saves the user from discovering OOM 30 min
# into a multi-hour run.
if command -v nvidia-smi >/dev/null && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    DEV="${CUDA_VISIBLE_DEVICES%%,*}"
    USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
              -i "$DEV" 2>/dev/null | tr -d ' ' || echo 0)
    if [ "$USED_MB" -gt 2048 ] 2>/dev/null; then
        echo "!!! GPU $DEV already has ${USED_MB} MiB in use by another process." >&2
        echo "    Pick an idle GPU. Current state:" >&2
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
                   --format=csv >&2
        exit 3
    fi
    echo "==> GPU $DEV: ${USED_MB} MiB used at launch (OK)"
fi

GT="datasets/Omni3D/WildBox_val.json"
ORACLE_JSON="datasets/Omni3D/gdino_WildBox_val_oracle_2d.json"
# rpn-from-lift: feed GEO with LIFT's own RPN-transfer predictions as the 2D
# source. Only meaningful if you have a LIFT zeroshot_rpn run on disk.
RPN_FROM_LIFT_JSON="${RPN_FROM_LIFT_JSON:-output/wl6_zeroshot_rpn/inference/iter_final/WildBox_val/instances_predictions.pth}"

# guard: GT + the per-protocol 2D source JSONs must exist
[ -f "$GT" ] || { echo "missing $GT" >&2; exit 1; }

run_one() {
    # $1: tag (oracle2d|rpn)
    # $2: 2D source JSON (oracle or rpn-transfer)
    local TAG="$1"; local SRC_2D="$2"
    local OUT="output/wl6_geo_${TAG}"
    [ -f "$SRC_2D" ] || { echo "missing $SRC_2D" >&2; return 1; }

    mkdir -p "$OUT"
    echo
    echo "================================================================"
    echo "==> GEO inference: $TAG"
    echo "    2D source: $SRC_2D"
    echo "    output:    $OUT"
    echo "================================================================"

    # 1. GEO inference (SAM + Depth Pro + PCA per matched 2D box)
    OVMONO3D_GEO_DATASETS="WildBox_val:${SRC_2D}" \
    OVMONO3D_GEO_OUTPUT_DIR="$OUT" \
        python tools/ovmono3d_geo.py 2>&1 | tee "$OUT/inference.log"

    [ -f "$OUT/WildBox_val.pth" ] || {
        echo "FAIL: $OUT/WildBox_val.pth missing — inference broke" >&2
        return 1
    }

    # Symlink predictions into the canonical layout so downstream tools
    # (sanity_audit, make_report) discover them at the expected path.
    mkdir -p "$OUT/inference/iter_final/WildBox_val"
    ln -sf "$(realpath "$OUT/WildBox_val.pth")" \
        "$OUT/inference/iter_final/WildBox_val/instances_predictions.pth"

    # 2. Standard Omni3D evaluator (BEV + 2D + disentangled NHD)
    echo
    echo "==> GEO eval: $TAG"
    python -c "
from tools.eval_ovmono3d_geo import evaluate_predictions
evaluate_predictions(
    dataset_names=['WildBox_val'],
    prediction_paths={'WildBox_val': '$OUT/WildBox_val.pth'},
    filter_settings={
        'visibility_thres': 0.33333333,
        'truncation_thres': 0.33333333,
        'min_height_thres': 0.0625,
        'max_depth': 1e8,
        'category_names': None,
        'ignore_names': ['dontcare', 'ignore', 'void'],
        'trunc_2D_boxes': True,
        'modal_2D_boxes': False,
        'max_height_thres': 1.5,
    },
    output_dir='$OUT/eval',
    category_path='configs/category_meta.json',
    eval_mode='novel',
    iter_label='final',
)
" 2>&1 | tee "$OUT/eval.log"

    # 3. BEV-AP eval (the paper's primary 3D metric — KITTI-tight IoU)
    python tools/bev_ap_eval.py \
        --preds "$OUT/WildBox_val.pth" \
        --gt    "$GT" \
        --iou-thresholds 0.25 0.5 \
        --out   "$OUT/bev_ap.json" 2>&1 | tee -a "$OUT/eval.log"

    # 4. Class-agnostic 2D AP + NHD summary (matches our LIFT runs' shape)
    python tools/class_agnostic_eval.py \
        --preds "$OUT/WildBox_val.pth" \
        --gt    "$GT" \
        --nhd 2>&1 | tee "$OUT/summary_nhd.txt"

    # 5. paper_report assembly (matches run_full_eval.sh CLI shape)
    python tools/make_report.py \
        --run-dir "$OUT" \
        --label   "OVMono3D-GEO ($TAG)" \
        --gt      "$GT" \
        --out     "$OUT/paper_report" 2>&1 | tee "$OUT/report.log"

    echo "==> done: $OUT"
}

case "$PROTOCOL" in
    oracle2d)
        run_one "oracle2d" "$ORACLE_JSON"
        ;;
    rpn-from-lift)
        # advanced opt-in path; needs a LIFT zeroshot_rpn run on disk
        # converted into oracle-shape JSON. Out of scope for v1.0 paper.
        echo "rpn-from-lift not yet implemented — feeds LIFT RPN preds back" \
             "into GEO as the 2D source. Use 'oracle2d' (default)." >&2
        exit 2
        ;;
    *)
        echo "usage: $0 [oracle2d|rpn-from-lift]" >&2
        echo "  default: oracle2d" >&2
        exit 2
        ;;
esac

echo
echo "==> All requested GEO runs complete."
echo "==> Add to sanity audit by re-running:"
echo "    bash tools/run_sanity_audit_cluster.sh"
echo "    (script auto-discovers output/wl6_geo_* once they're written)"
