#!/usr/bin/env bash
# Phase-2 sanity audit on the cluster. Discovers all output/wl6_* runs,
# pulls in the GT JSON, and emits the appendix markdown.
#
# Run from repo root:
#   bash tools/run_sanity_audit_cluster.sh
#
# Outputs:
#   output/paper_appendix_sanity.md         — single appendix markdown
#   output/sanity_audit.log                  — full stdout/stderr
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

GT="${GT:-datasets/Omni3D/WildBox_val.json}"
OUT="${OUT:-output/paper_appendix_sanity.md}"
LOG="${LOG:-output/sanity_audit.log}"

# Discover every wl6_* run that has at least one paper_report or
# instances_predictions.pth on disk.
mapfile -t RUNS < <(
  for d in output/wl6_*; do
    [ -d "$d" ] || continue
    # multi-seed dirs nest as run/seedN/inference/iter_*/dataset/preds.pth
    # (depth 5), single-run dirs are run/inference/iter_*/dataset/preds.pth
    # (depth 4). Use depth 6 to safely cover both, plus future variants.
    if find "$d" -maxdepth 6 -name "instances_predictions.pth" -print -quit \
            | grep -q .; then
      echo "$d"
    fi
  done
)

if [ "${#RUNS[@]}" -eq 0 ]; then
  echo "no output/wl6_* runs found with predictions" >&2
  exit 1
fi

echo "==> $(date -Iseconds)"
echo "==> sanity audit on ${#RUNS[@]} runs:"
printf '    %s\n' "${RUNS[@]}"
echo "==> GT:  $GT"
echo "==> out: $OUT"
echo

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

# --deep needs to import scipy + load each instances_predictions.pth, so this
# can take a while per run (NHD pair-matching is the bottleneck). bev_ap_eval
# is sub-shelled per run for the relaxed-IoU section, which dominates wall
# time on zero-shot runs (63k preds vs 67k GT footprints).
python tools/zeroshot_sanity_audit.py \
  --runs "${RUNS[@]}" \
  --gt "$GT" \
  --deep \
  --out "$OUT" \
  2>&1 | tee "$LOG"

echo
echo "==> done: $OUT"
