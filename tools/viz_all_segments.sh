#!/usr/bin/env bash
# Run the canonical annotator-pipeline visualizer on ONE sample frame per
# segment under each given group root. Picks the middle of the first
# track's frame coverage.
#
# Usage:
#     bash tools/viz_all_segments.sh <out_dir> <group_dir> [<group_dir> ...]
set -e
OUT="$1"; shift
mkdir -p "$OUT"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || dirname "$(readlink -f "$0")")"
VIZ="$REPO_ROOT/tools/viz_via_annotator.py"

for ROOT in "$@"; do
    [ -d "$ROOT" ] || { echo "skipping missing $ROOT"; continue; }
    GRP=$(basename "$ROOT")
    echo
    echo "=== $GRP ==="
    while read SEG; do
        FRAME=$(python3 -c "
import json
from pathlib import Path
ts_p = Path('$SEG/tracking_summary.json')
meta_p = Path('$SEG/vggt_metadata.json')
if not ts_p.exists() or not meta_p.exists():
    print(1); raise SystemExit
ts = json.loads(ts_p.read_text())
meta = json.loads(meta_p.read_text())
fnums = meta['frame_numbers']
if not ts.get('tracks'):
    print(1); raise SystemExit
t = next(iter(ts['tracks'].values()))
frames = t.get('frames', [])
if not frames:
    print(1); raise SystemExit
mid_idx = frames[len(frames) // 2]
print(fnums[mid_idx])
" 2>/dev/null || echo 1)
        SEG_REL=$(realpath --relative-to="$ROOT" "$SEG")
        OUT_NAME="$GRP--$(echo $SEG_REL | tr / _)--frame_${FRAME}.jpg"
        echo -n "  $SEG_REL  frame=$FRAME  -> "
        python "$VIZ" "$SEG" "$FRAME" "$OUT/$OUT_NAME" 2>&1 \
            | grep -E "drew|abort" | head -1
    done < <(find "$ROOT" -maxdepth 4 -type d -name "seg*" \
                  -exec test -f {}/cameras.json \; -print | sort)
done

echo
echo "=== ALL DONE ==="
echo "Output dir: $OUT"
echo "$(ls $OUT | wc -l) files written"
