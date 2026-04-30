#!/usr/bin/env bash
# Run OVMono3D-LIFT zero-shot evaluation on WildBox_val with GT 2D boxes
# as the oracle source — the "perfect 2D + perfect class" ceiling row.
# Pairs symmetrically with DetAny3D's zs_gt2d_v3 condition.
#
# Run from repo root, with one idle GPU available:
#   bash tools/run_gt2d_zs_eval.sh
#
# What it does:
#   1. Build datasets/Omni3D/gt2d_WildBox_val_oracle_2d.json (idempotent —
#      skips if the file already exists, or set FORCE_BUILD=1 to redo).
#   2. Eval-only run of OVMono3D-LIFT with TEST.ORACLE2D=True pointing at
#      the GT-2D JSON. Uses the same checkpoint as wl6_zeroshot_oracle2d.
#   3. BEV-AP + class-agnostic + paper_report assembly + visualizations,
#      same downstream pipeline as run_full_eval.sh.
#
# Output: output/wl6_zeroshot_gt2d/  (metrics, vis, paper_report)

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

GT="datasets/Omni3D/WildBox_val.json"
GT2D_JSON="datasets/Omni3D/gt2d_WildBox_val_oracle_2d.json"
CONFIG="configs/wildbox/OVMono3D_wildbox_wildlife6_gt2d.yaml"
WEIGHTS="${WEIGHTS:-checkpoints/ovmono3d_lift.pth}"
OUT="output/wl6_zeroshot_gt2d"
LABEL="OVMono3D-LIFT zero-shot (GT 2D ceiling)"

# Pre-flight
[ -f "$GT" ]      || { echo "missing $GT" >&2; exit 1; }
[ -f "$WEIGHTS" ] || { echo "missing $WEIGHTS" >&2; exit 1; }
[ -f "$CONFIG" ]  || { echo "missing $CONFIG" >&2; exit 1; }

# Pre-flight GPU sanity (mirrors the GEO + 15k launchers).
if command -v nvidia-smi >/dev/null && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    DEV="${CUDA_VISIBLE_DEVICES%%,*}"
    USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
              -i "$DEV" 2>/dev/null | tr -d ' ' || echo 0)
    if [ "$USED_MB" -gt 2048 ] 2>/dev/null; then
        echo "!!! GPU $DEV has ${USED_MB} MiB in use by another process." >&2
        nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv >&2
        exit 3
    fi
fi

# 1. Build the GT-2D oracle JSON if missing.
if [ ! -f "$GT2D_JSON" ] || [ "${FORCE_BUILD:-0}" = "1" ]; then
    echo "==> building $GT2D_JSON"
    python tools/build_gt2d_oracle_json.py --gt "$GT" --out "$GT2D_JSON"
else
    echo "==> reusing existing $GT2D_JSON  (set FORCE_BUILD=1 to rebuild)"
fi

mkdir -p "$OUT"

# 2. Eval-only run. We don't reuse run_full_eval.sh because it expects the
# checkpoint to be a fine-tuned WildBox model; here we use the pretrained
# Omni3D weights to get the zero-shot ceiling.
echo
echo "==> eval-only: $LABEL"
echo "    config:  $CONFIG"
echo "    weights: $WEIGHTS"
echo "    out:     $OUT"
echo

python tools/train_net.py \
    --config-file "$CONFIG" \
    --eval-only \
    MODEL.WEIGHTS "$WEIGHTS" \
    OUTPUT_DIR "$OUT" \
    2>&1 | tee "$OUT/log.txt"

# 3. BEV-AP at the canonical paper IoUs.
PREDS="$OUT/inference/iter_final/WildBox_val/instances_predictions.pth"
[ -f "$PREDS" ] || { echo "no predictions written at $PREDS" >&2; exit 4; }

python tools/bev_ap_eval.py \
    --preds "$PREDS" --gt "$GT" \
    --iou-thresholds 0.25 0.5 \
    --out "$OUT/bev_ap.json" 2>&1 | tee -a "$OUT/log.txt"

# 4. Class-agnostic + NHD-best-scale (the diagnostics the appendix uses).
python tools/class_agnostic_eval.py \
    --preds "$PREDS" --gt "$GT" \
    --nhd 2>&1 | tee "$OUT/summary_nhd.txt"

# 5. Paper visualizations (same script LIFT uses elsewhere).
python tools/visualize_class_agnostic.py \
    --preds "$PREDS" --gt "$GT" --out "$OUT/vis_ovmono3d" \
    --top-k 5 --every 100 --limit 40 \
    2>&1 | tee "$OUT/vis.log" \
    || echo "  (vis skipped — see $OUT/vis.log)"

# 6. paper_report assembly.
python tools/make_report.py \
    --run-dir "$OUT" \
    --label   "$LABEL" \
    --gt      "$GT" \
    --config  "$CONFIG" \
    --out     "$OUT/paper_report" 2>&1 | tee "$OUT/report.log"

echo
echo "==> done: $OUT"
echo "==> next: bash tools/run_sanity_audit_cluster.sh"
echo "    (the audit auto-discovers output/wl6_* — new GT-2D row will appear)"
