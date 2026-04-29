#!/usr/bin/env bash
# Re-evaluate a LIFT run on the SAME image-ID subset GEO covers, to give
# a fair apples-to-apples LIFT-vs-GEO comparison row in the appendix.
#
# Why this is needed:
# - GEO ran with STRIDE=4 → 3445/13779 frames
# - LIFT zero-shot ran on the full 13779 frames
# - macro/micro AP recall denominators differ → numbers aren't comparable
# - Fix: take LIFT's predictions, filter to GEO's image_ids, filter the GT
#   JSON to the same image_ids, re-run BEV-AP / class-agnostic / make_report
#   on the filtered set. NHD diagnostics are matched-pair-only so they are
#   already comparable, but rerunning them on the subset avoids any
#   asymmetric "matched in 13k vs 3k" effect.
#
# Run from repo root, after both LIFT zero-shot and GEO have produced
# their instances_predictions.pth files:
#   bash tools/subset_eval_to_geo.sh \
#        --lift-run output/wl6_zeroshot_oracle2d \
#        --geo-run  output/wl6_geo_oracle2d
#
# Output:
#   output/<lift-run>_geomatched/   parallel run dir with subset preds + GT
#   output/<lift-run>_geomatched/paper_report/metrics.json
#
# After this, re-run the sanity audit and the new row will appear in the
# appendix labeled "<lift-run>_geomatched", directly comparable to the GEO
# row. Recommend reporting BOTH the original full-set LIFT row (for
# comparison with the FT runs) AND this geomatched row (for cross-model-
# family comparison with GEO) in the paper.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LIFT_RUN=""
GEO_RUN=""
GT="${GT:-datasets/Omni3D/WildBox_val.json}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lift-run) LIFT_RUN="$2"; shift 2 ;;
        --geo-run)  GEO_RUN="$2"; shift 2 ;;
        --gt)       GT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$LIFT_RUN" || -z "$GEO_RUN" ]]; then
    echo "usage: $0 --lift-run <dir> --geo-run <dir> [--gt <path>]" >&2
    exit 2
fi

LIFT_PREDS="$LIFT_RUN/inference/iter_final/WildBox_val/instances_predictions.pth"
GEO_PREDS="$GEO_RUN/inference/iter_final/WildBox_val/instances_predictions.pth"

[[ -f "$LIFT_PREDS" ]] || { echo "missing $LIFT_PREDS" >&2; exit 1; }
[[ -f "$GEO_PREDS"  ]] || { echo "missing $GEO_PREDS"  >&2; exit 1; }
[[ -f "$GT"         ]] || { echo "missing $GT"         >&2; exit 1; }

OUT="${LIFT_RUN}_geomatched"
mkdir -p "$OUT/inference/iter_final/WildBox_val"

echo "==> LIFT preds:   $LIFT_PREDS"
echo "==> GEO preds:    $GEO_PREDS  (defines subset)"
echo "==> GT:           $GT"
echo "==> out:          $OUT"
echo

# 1. Build the subset by filtering LIFT preds + GT JSON to GEO's image_ids.
python - "$LIFT_PREDS" "$GEO_PREDS" "$GT" "$OUT" <<'PY'
import json, sys, torch
from pathlib import Path

lift_pred_pth, geo_pred_pth, gt_path, out_dir = map(Path, sys.argv[1:5])

print(f"[subset] loading GEO preds → image_id set")
geo = torch.load(geo_pred_pth, weights_only=False)
ids = set()
for im in geo:
    if "image_id" in im:
        ids.add(int(im["image_id"]))
    elif "instances" in im and im["instances"]:
        ids.add(int(im["instances"][0]["image_id"]))
print(f"[subset]   {len(ids)} unique image_ids in GEO output")

print(f"[subset] loading LIFT preds → filter")
lift = torch.load(lift_pred_pth, weights_only=False)
lift_subset = []
for im in lift:
    iid = im.get("image_id")
    if iid is None and im.get("instances"):
        iid = im["instances"][0].get("image_id")
    if iid is not None and int(iid) in ids:
        lift_subset.append(im)
print(f"[subset]   kept {len(lift_subset)}/{len(lift)} LIFT prediction records")

out_pred = out_dir / "inference/iter_final/WildBox_val/instances_predictions.pth"
out_pred.parent.mkdir(parents=True, exist_ok=True)
torch.save(lift_subset, out_pred)
# bev_ap_eval / class_agnostic_eval read this canonical name; symlink for
# any tool that prefers the parent-of-run pattern (matches GEO layout).
parent_pth = out_dir / "WildBox_val.pth"
if parent_pth.exists() or parent_pth.is_symlink():
    parent_pth.unlink()
parent_pth.symlink_to(out_pred.resolve())
print(f"[subset] wrote {out_pred} ({out_pred.stat().st_size/1024/1024:.1f} MB)")

print(f"[subset] filtering GT JSON → only the {len(ids)} subset image_ids")
gt = json.loads(gt_path.read_text())
gt["images"]      = [im for im in gt["images"]      if int(im["id"]) in ids]
gt["annotations"] = [an for an in gt["annotations"] if int(an["image_id"]) in ids]
print(f"[subset]   GT subset: {len(gt['images'])} images, "
      f"{len(gt['annotations'])} annotations")
out_gt = out_dir / "WildBox_val_subset.json"
out_gt.write_text(json.dumps(gt))
print(f"[subset] wrote {out_gt}")
PY

OUT_PRED="$OUT/inference/iter_final/WildBox_val/instances_predictions.pth"
OUT_GT="$OUT/WildBox_val_subset.json"

# 2. Disentangled-NHD eval (the omni3d evaluator path).
python -c "
from tools.eval_ovmono3d_geo import evaluate_predictions
evaluate_predictions(
    dataset_names=['WildBox_val'],
    prediction_paths={'WildBox_val': '$OUT_PRED'},
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
" 2>&1 | tee "$OUT/eval.log" \
  || echo "  (eval_ovmono3d_geo failed — disent NHD will be missing in report)"

# 3. BEV-AP on the subset.
python tools/bev_ap_eval.py \
    --preds "$OUT_PRED" --gt "$OUT_GT" \
    --iou-thresholds 0.25 0.5 \
    --out "$OUT/bev_ap.json" 2>&1 | tee -a "$OUT/eval.log"

# 4. Class-agnostic + NHD-best-scale.
python tools/class_agnostic_eval.py \
    --preds "$OUT_PRED" --gt "$OUT_GT" \
    --nhd 2>&1 | tee "$OUT/summary_nhd.txt"

# 5. paper_report — point at the subset GT so denominators match.
python tools/make_report.py \
    --run-dir "$OUT" \
    --label   "$(basename "$LIFT_RUN") @ GEO-stride-4 subset" \
    --gt      "$OUT_GT" \
    --out     "$OUT/paper_report" 2>&1 | tee "$OUT/report.log"

echo
echo "==> done: $OUT"
echo "==> Re-run sanity audit to fold this into appendix:"
echo "    bash tools/run_sanity_audit_cluster.sh"
