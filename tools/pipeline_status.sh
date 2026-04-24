#!/usr/bin/env bash
# One-screen snapshot of everything running in the WildBox pipeline.
# Designed to be `watch`-able: `watch -n 30 bash tools/pipeline_status.sh`
#
# Covers:
#   1) My python processes (train / eval / GDino precompute)
#   2) My GPU usage on this node
#   3) Latest progress lines from every active log under output/
#   4) Seed completion checkboxes for the multi-seed run
#
# Runs quickly (<3 s). No side effects. Read-only.

set -uo pipefail

BOLD='\033[1m'; DIM='\033[2m'; GREEN='\033[32m'; YELLOW='\033[33m'; RED='\033[31m'; RESET='\033[0m'

hr() { printf "${DIM}---------------------------------------------------------${RESET}\n"; }
header() { printf "\n${BOLD}%s${RESET}\n" "$1"; }

header "[$(date +%H:%M:%S)]  Pipeline snapshot"

# ---- 1) My running python processes ----
header "1. Your active python jobs"
ps -u "$USER" -o pid,etime,%cpu,%mem,cmd 2>/dev/null | \
    awk 'NR==1 || /train_net\.py|multi_seed|precompute_gdino|run_full_eval|class_agnostic|bev_ap_eval|make_report/' | \
    awk '!/grep/ && !/awk/' | \
    head -20 || echo "  (none)"

# ---- 2) My GPU memory on node ----
header "2. Your GPU usage (this node only)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | \
        while IFS=, read -r pid pname mem; do
            pid_trimmed=$(echo "$pid" | tr -d ' ')
            owner=$(ps -p "$pid_trimmed" -o user= 2>/dev/null | tr -d ' ')
            if [ "$owner" = "$USER" ]; then
                printf "  PID %-8s  owner=%-10s  mem=%-10s  %s\n" "$pid_trimmed" "$owner" "$mem" "$pname"
            fi
        done
    echo ""
    echo "  GPU totals (all users on this A40):"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  /'
else
    echo "  nvidia-smi not on PATH (on frontend?). Tail train logs remotely instead."
fi

# ---- 3) Active pipeline logs ----
header "3. Latest line from each active log"

show_log () {
    local label="$1" path="$2" grep_pat="${3:-iter:}"
    if [ ! -f "$path" ]; then
        printf "  ${DIM}%-50s  not found${RESET}\n" "$label"
        return
    fi
    local age_s=$(( $(date +%s) - $(stat -c %Y "$path" 2>/dev/null || echo 0) ))
    local line
    line=$(grep -E "$grep_pat" "$path" 2>/dev/null | tail -1)
    if [ -z "$line" ]; then
        line=$(tail -1 "$path" 2>/dev/null | cut -c1-180)
    fi
    # Color the age: green <5min, yellow <30min, red older/dead
    local color=$GREEN
    (( age_s > 300 ))  && color=$YELLOW
    (( age_s > 1800 )) && color=$RED
    printf "  %-50s  ${color}%dm ago${RESET}\n    ${DIM}%s${RESET}\n" "$label" $((age_s/60)) "$line"
}

# Multi-seed orchestrator
show_log "multi-seed orchestrator (logs/multiseed.log)" "logs/multiseed.log" "seed|done|Aggregate"

# Per-seed training logs
for S in 0 1 2; do
    for cand in "output/wl6_rt0.5_multiseed/seed$S/log.txt" \
                "logs/train_seed${S}.log"; do
        if [ -f "$cand" ]; then
            show_log "seed $S training ($(basename $cand))" "$cand" "iter:|eta:"
            break
        fi
    done
    # Eval pipeline
    show_log "seed $S standard eval"      "output/wl6_rt0.5_multiseed/seed$S/eval/log.txt"      "mode=2D|mode=3D|Inference done"
    show_log "seed $S Rel-AP3D eval"      "output/wl6_rt0.5_multiseed/seed$S/eval_rel/log.txt"  "3D-Rel|best global|Inference done"
done

# Zero-shot runs
show_log "zero-shot RPN"     "output/wl6_zeroshot_rpn/log.txt"       "mode=2D|Inference done|\[5/5\]"
show_log "zero-shot oracle"  "output/wl6_zeroshot_oracle2d/log.txt"  "mode=2D|Inference done|\[5/5\]"

# GDino precompute
for g in logs/gdino_oracle*.log logs/multiseed.log; do
    [ -f "$g" ] && show_log "GDino precompute ($(basename $g))" "$g" "kept=|Wrote|Processing"
done

# ---- 4) Multi-seed completion status ----
header "4. Seeds completion"
for S in 0 1 2; do
    model="output/wl6_rt0.5_multiseed/seed$S/model_final.pth"
    bev="output/wl6_rt0.5_multiseed/seed$S/eval/bev_ap.json"
    rel="output/wl6_rt0.5_multiseed/seed$S/eval_rel/log.txt"
    vis="output/wl6_rt0.5_multiseed/seed$S/eval/vis_ovmono3d"
    printf "  seed %d  " "$S"
    [ -f "$model" ] && printf "${GREEN}✓ model${RESET}    " || printf "${DIM}· model${RESET}    "
    [ -f "$bev" ]   && printf "${GREEN}✓ eval${RESET}     " || printf "${DIM}· eval${RESET}     "
    [ -f "$rel" ]   && printf "${GREEN}✓ rel-ap3d${RESET} " || printf "${DIM}· rel-ap3d${RESET} "
    [ -d "$vis" ]   && printf "${GREEN}✓ vis${RESET}"      || printf "${DIM}· vis${RESET}"
    echo ""
done

# Zero-shot status
header "5. Zero-shot + reports"
for name in wl6_zeroshot_rpn wl6_zeroshot_oracle2d; do
    bev="output/$name/bev_ap.json"
    report="output/$name/paper_report/report.md"
    printf "  %-32s  " "$name"
    [ -f "$bev" ]    && printf "${GREEN}✓ eval${RESET}     " || printf "${DIM}· eval${RESET}     "
    [ -f "$report" ] && printf "${GREEN}✓ report${RESET}"   || printf "${DIM}· report${RESET}"
    echo ""
done

hr
echo "  tip: watch -n 30 bash tools/pipeline_status.sh    # auto-refresh every 30s"
