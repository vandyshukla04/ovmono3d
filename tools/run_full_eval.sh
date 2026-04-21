#!/usr/bin/env bash
# One-shot evaluation runner for a single checkpoint.
#
# Usage:
#   bash tools/run_full_eval.sh \
#     --weights output/wildbox_wl5_finetune/model_final.pth \
#     --config  configs/wildbox/OVMono3D_wildbox_wildlife5.yaml \
#     --out     output/wildbox_wl5_finetuned_eval \
#     --label   "fine-tuned 5 species"
#
# Produces, in <out>/:
#   log.txt                           standard eval (2D + 3D)
#   log.rel.txt                       Rel-AP3D run (slow, ~15 min on CPU)
#   bev_ap.json                       BEV AP @ {0.25, 0.5}
#   summary_nhd.txt                   class-agnostic + NHD
#   paper_report/report.md            human-readable
#   paper_report/table_main.tex       LaTeX main table
#   paper_report/metrics.json         machine-readable
set -euo pipefail

# ---- args ----
WEIGHTS=""
CONFIG=""
OUT=""
LABEL="run"
GT="datasets/Omni3D/WildBox_val.json"
SKIP_REL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --weights) WEIGHTS="$2"; shift 2;;
        --config)  CONFIG="$2"; shift 2;;
        --out)     OUT="$2"; shift 2;;
        --label)   LABEL="$2"; shift 2;;
        --gt)      GT="$2"; shift 2;;
        --skip-rel-ap3d) SKIP_REL=1; shift;;
        *) echo "unknown arg: $1"; exit 2;;
    esac
done

if [[ -z "$WEIGHTS" || -z "$CONFIG" || -z "$OUT" ]]; then
    echo "usage: $0 --weights X --config Y --out Z [--label L] [--gt P] [--skip-rel-ap3d]" >&2
    exit 2
fi

mkdir -p "$OUT"

# Keep the two category_meta.json files in sync. The --eval-only code path in
# tools/train_net.py:402 reads the one next to the config file, while
# external tools read the top-level one. If they disagree, per-class metrics
# silently become wrong (the Phase-1 giraffe-only file bug of 2026-04-21).
TOP_META="configs/category_meta.json"
CFG_DIR_META="$(dirname "$CONFIG")/category_meta.json"
if [[ -e "$TOP_META" ]]; then
    TOP_TARGET="$(readlink -f "$TOP_META" 2>/dev/null || echo "$TOP_META")"
    CFG_TARGET="$(readlink -f "$CFG_DIR_META" 2>/dev/null || echo "$CFG_DIR_META")"
    if [[ "$TOP_TARGET" != "$CFG_TARGET" ]]; then
        echo "!! $TOP_META and $CFG_DIR_META resolve to different files:"
        echo "   $TOP_META -> $TOP_TARGET"
        echo "   $CFG_DIR_META -> $CFG_TARGET"
        echo "   Syncing $CFG_DIR_META to match $TOP_META ..."
        # Use a basename symlink so it remains relative to the wildbox dir
        top_basename=$(basename "$TOP_TARGET")
        ln -sf "$top_basename" "$CFG_DIR_META"
    fi
fi

echo "[1/4] Standard eval (2D + 3D, no Rel-AP3D) -> $OUT/log.txt"
python tools/train_net.py --eval-only \
    --config-file "$CONFIG" \
    --num-gpus 1 \
    MODEL.WEIGHTS "$WEIGHTS" \
    TEST.EVAL_REL_AP3D False \
    OUTPUT_DIR "$OUT" 2>&1 | tee "$OUT/log.txt"

PREDS="$OUT/inference/iter_final/WildBox_val/instances_predictions.pth"
if [[ ! -f "$PREDS" ]]; then
    echo "ERROR: instances_predictions.pth missing at $PREDS" >&2
    exit 3
fi

if [[ $SKIP_REL -eq 0 ]]; then
    echo "[2/4] Rel-AP3D eval (LabelAny3D scale search, CPU, ~15 min) -> $OUT/log.rel.txt"
    # Write into a parallel dir so the standard log.txt isn't clobbered.
    REL_OUT="${OUT}_rel"
    mkdir -p "$REL_OUT"
    python tools/train_net.py --eval-only \
        --config-file "$CONFIG" \
        --num-gpus 1 \
        MODEL.WEIGHTS "$WEIGHTS" \
        TEST.EVAL_REL_AP3D True \
        OUTPUT_DIR "$REL_OUT" 2>&1 | tee "$OUT/log.rel.txt"
else
    echo "[2/4] Skipping Rel-AP3D (--skip-rel-ap3d)"
fi

echo "[3/4] BEV AP -> $OUT/bev_ap.json"
python tools/bev_ap_eval.py \
    --preds "$PREDS" \
    --gt    "$GT" \
    --out   "$OUT/bev_ap.json" 2>&1 | tee -a "$OUT/log.txt"

echo "[4/4] Class-agnostic + NHD surrogate -> $OUT/summary_nhd.txt"
python tools/class_agnostic_eval.py \
    --preds "$PREDS" \
    --gt    "$GT" \
    --nhd > "$OUT/summary_nhd.txt"
cat "$OUT/summary_nhd.txt"

# Build single-run report
python tools/make_report.py \
    --run-dir "$OUT" \
    --label   "$LABEL" \
    --gt      "$GT" \
    --config  "$CONFIG" \
    --out     "$OUT/paper_report"

echo
echo "Done. See $OUT/paper_report/report.md"
