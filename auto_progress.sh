#!/bin/bash
# Lightweight progress watcher: every 5 min, write to submissions/2026-05-11/progress.txt
# the current best-val for each running experiment + which CSVs have landed.

set -uo pipefail
cd /home/fxiao/misc/156
SUB_DIR="submissions/2026-05-11"
mkdir -p "${SUB_DIR}"

VARIANTS=(
  "aug_randlight aug_rlight"
  "aug_randstd aug_rstd"
  "aug_randstrong aug_rstrong"
  "aug_trivial aug_triv"
  "focal_g1 focal_g1"
  "focal_g2 focal_g2"
  "focal_g3 focal_g3"
  "mse_w30 mse_w30"
  "mse_w50 mse_w50"
  "mse_w70 mse_w70"
  "pseudo_t03 pseudo_t03"
  "pseudo_t05 pseudo_t05"
  "pseudo_t07 pseudo_t07"
)

# Also include the still-running om_mlp / om_mlp4 from before
EXTRA_RUNS=("om_mlp" "om_mlp4")

while true; do
  {
    echo "===== Round 1 progress @ $(date -Iseconds) ====="
    echo ""
    printf "%-22s  %-8s  %-12s  %-15s  %s\n" "experiment" "status" "step" "best_val" "csv_landed"
    echo "----------------------------------------------------------------------------------------"
    # Find each variant's most recent slurm log
    for entry in "${VARIANTS[@]}"; do
      variant=$(echo "$entry" | awk '{print $1}')
      short=$(echo "$entry" | awk '{print $2}')
      log=""
      for f in $(ls -t /home/fxiao/misc/156/slurm_logs/*.out 2>/dev/null); do
        if grep -q "config     : configs/v1_3class_omnirad_b14_s0_${variant}.yaml" "$f" 2>/dev/null; then
          log="$f"
          break
        fi
      done
      if [ -z "$log" ]; then
        printf "%-22s  %-8s  %-12s  %-15s  %s\n" "$variant" "WAIT" "-" "-" "no"
        continue
      fi
      jid=$(basename "$log" .out)
      best=$(grep "new best" "$log" 2>/dev/null | tail -1 | grep -oE "nmse=[0-9.]+" | cut -d= -f2)
      step=$(grep "step " "$log" 2>/dev/null | tail -1 | grep -oE "step *[0-9,]+/[0-9,]+" | head -1 | tr -s ' ')
      state=$(squeue -j "$jid" -h -o "%t" 2>/dev/null | head -1)
      if [ -z "$state" ]; then state="DONE"; fi
      csv_landed="no"
      [ -f "${SUB_DIR}/omnirad_${short}_va.csv" ] && csv_landed="yes"
      printf "%-22s  %-8s  %-12s  %-15s  %s\n" "$variant" "$state" "${step:-?}" "${best:-?}" "$csv_landed"
    done
    echo ""
    echo "===== Other still-running ====="
    for run in "${EXTRA_RUNS[@]}"; do
      log=""
      for f in $(ls -t /home/fxiao/misc/156/slurm_logs/*.out 2>/dev/null); do
        if grep -q "config     : configs/v1_3class_omnirad_b14_s0_${run}.yaml" "$f" 2>/dev/null; then
          log="$f"
          break
        fi
      done
      if [ -z "$log" ]; then continue; fi
      best=$(grep "new best" "$log" 2>/dev/null | tail -1 | grep -oE "nmse=[0-9.]+" | cut -d= -f2)
      step=$(grep "step " "$log" 2>/dev/null | tail -1 | grep -oE "step *[0-9,]+/[0-9,]+" | head -1 | tr -s ' ')
      jid=$(basename "$log" .out)
      state=$(squeue -j "$jid" -h -o "%t" 2>/dev/null | head -1)
      if [ -z "$state" ]; then state="DONE"; fi
      printf "%-22s  %-8s  %-12s  %-15s\n" "$run" "$state" "${step:-?}" "${best:-?}"
    done
    echo ""
    echo "===== CSVs in ${SUB_DIR} ====="
    ls -la "${SUB_DIR}/" 2>/dev/null | grep -v "^total" | grep -v "^d"
  } > "${SUB_DIR}/progress.txt" 2>&1
  sleep 300
done
