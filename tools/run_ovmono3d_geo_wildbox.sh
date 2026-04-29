#!/usr/bin/env bash
# Run OVMono3D-GEO inference + evaluation on WildBox_val for both zero-shot
# protocols, mirroring the wl6_zeroshot_oracle2d / wl6_zeroshot_rpn pair we
# already have for OVMono3D-LIFT. Adds two rows to the main results table
# from a *different model family* (geometric: SAM + Depth Pro + PCA, no
# learned cube head).
#
# Run from repo root, with cluster GPU available:
#   bash tools/run_ovmono3d_geo_wildbox.sh                   # both protocols
#   bash tools/run_ovmono3d_geo_wildbox.sh oracle2d          # GDino-oracle 2D only
#   bash tools/run_ovmono3d_geo_wildbox.sh rpn               # RPN-transfer 2D only
#
# Outputs:
#   output/wl6_geo_oracle2d/        — GEO predictions + paper_report
#   output/wl6_geo_rpn/             — GEO predictions + paper_report
#   output/wl6_geo_*/log.txt        — full inference + eval log
#
# Prereqs (one-time):
#   pip install depth_pro segment_anything open3d scikit-learn
#   ./checkpoints/sam_vit_h_4b8939.pth                       — SAM weights
#   Depth Pro weights cached at depth_pro's default location
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PROTOCOL="${1:-all}"

GT="datasets/Omni3D/WildBox_val.json"
ORACLE_JSON="datasets/Omni3D/gdino_WildBox_val_oracle_2d.json"
RPN_JSON="datasets/Omni3D/rpn_WildBox_val_2d.json"

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
    oracle2d) run_one "oracle2d" "$ORACLE_JSON" ;;
    rpn)      run_one "rpn"      "$RPN_JSON" ;;
    all)
        run_one "oracle2d" "$ORACLE_JSON"
        run_one "rpn"      "$RPN_JSON"
        ;;
    *)
        echo "usage: $0 [oracle2d|rpn|all]" >&2
        exit 2
        ;;
esac

echo
echo "==> All requested GEO runs complete."
echo "==> Add to sanity audit by re-running:"
echo "    bash tools/run_sanity_audit_cluster.sh"
echo "    (script auto-discovers output/wl6_geo_* once they're written)"
