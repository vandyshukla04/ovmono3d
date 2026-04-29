#!/usr/bin/env bash
# One-shot status check for in-flight WildBox cluster jobs.
# Run from repo root: bash tools/status_check.sh
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

echo "==== $(date -Iseconds) ===="
echo

# ---- GPU -----------------------------------------------------------------
echo "## GPUs"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
               --format=csv,noheader | sed 's/^/  /'
    echo
    echo "## GPU processes (compute apps)"
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
               --format=csv,noheader 2>/dev/null | sed 's/^/  /' \
        || echo "  (none / unavailable)"
else
    echo "  nvidia-smi not on PATH"
fi
echo

# ---- Our processes -------------------------------------------------------
echo "## Our python jobs (ovmono3d-related)"
ps -eo pid,etime,pcpu,pmem,args --sort=-pcpu \
   | grep -E "(ovmono3d_geo|zeroshot_sanity_audit|run_sanity_audit_cluster|run_ovmono3d_geo_wildbox|bev_ap_eval|class_agnostic_eval)" \
   | grep -v grep \
   | sed 's/^/  /' \
   || echo "  (no matching processes)"
echo

# ---- Sanity audit log ----------------------------------------------------
echo "## Sanity audit (output/sanity_audit.log)"
if [ -f output/sanity_audit.log ]; then
    AGE=$(($(date +%s) - $(stat -c %Y output/sanity_audit.log)))
    SIZE=$(du -h output/sanity_audit.log | cut -f1)
    echo "  size=$SIZE   last-modified ${AGE}s ago"
    echo "  --- last 8 lines ---"
    tail -n 8 output/sanity_audit.log | sed 's/^/  | /'
else
    echo "  (no log yet)"
fi
if [ -f output/paper_appendix_sanity.md ]; then
    LINES=$(wc -l < output/paper_appendix_sanity.md)
    echo "  → output/paper_appendix_sanity.md exists ($LINES lines)"
fi
echo

# ---- GEO inference log ---------------------------------------------------
echo "## GEO inference (output/wl6_geo_oracle2d/inference.log)"
LOG="output/wl6_geo_oracle2d/inference.log"
if [ -f "$LOG" ]; then
    AGE=$(($(date +%s) - $(stat -c %Y "$LOG")))
    SIZE=$(du -h "$LOG" | cut -f1)
    echo "  size=$SIZE   last-modified ${AGE}s ago"
    echo "  --- last 5 lines (tqdm uses \\r so latest is what counts) ---"
    # tqdm carriage returns: split on \r, keep the last non-empty fragment
    tail -c 4096 "$LOG" | tr '\r' '\n' | grep -v '^$' | tail -n 5 \
        | sed 's/^/  | /'
else
    echo "  (no log yet)"
fi
if [ -f output/wl6_geo_oracle2d/WildBox_val.pth ]; then
    SIZE=$(du -h output/wl6_geo_oracle2d/WildBox_val.pth | cut -f1)
    echo "  → WildBox_val.pth exists ($SIZE) — inference DONE; eval next"
fi
echo

# ---- Disk hygiene --------------------------------------------------------
echo "## Output disk usage (top 8)"
du -sh output/* 2>/dev/null | sort -hr | head -n 8 | sed 's/^/  /'
echo

# ---- Cheatsheet ----------------------------------------------------------
echo "## What to expect"
echo "  - sanity audit DONE when 'wrote output/paper_appendix_sanity.md' in log"
echo "  - GEO inference DONE when WildBox_val.pth file appears (~tens of MB)"
echo "  - GEO eval DONE when output/wl6_geo_oracle2d/paper_report/ exists"
echo "  - re-run sanity audit after GEO finishes to fold in the new row:"
echo "      bash tools/run_sanity_audit_cluster.sh"
