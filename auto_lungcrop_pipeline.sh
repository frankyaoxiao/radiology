#!/bin/bash
# Full lung-crop experiment orchestration:
#   1. Wait for lung_bboxes.csv (produced by the lung_seg SLURM job)
#   2. Submit lung-crop training (3x OmniRad + 1x EVA-02), each with infer dependency
#   3. Wait for all prediction CSVs
#   4. Pool + build ensemble candidates using lung-crop preds for pulmonary labels
set -uo pipefail
cd /home/fxiao/misc/156

BBOX=/data/artifacts/frank/misc/labels/lung_bboxes.csv
SUB16=submissions/2026-05-16

echo "[$(date -Iseconds)] waiting for lung bboxes CSV..." >&2
until [ -f "$BBOX" ]; do sleep 60; done
# Wait for file to be fully written (size stable)
prev=0; while true; do cur=$(stat -c%s "$BBOX"); [ "$cur" = "$prev" ] && [ "$cur" -gt 1000 ] && break; prev=$cur; sleep 15; done
echo "[$(date -Iseconds)] bboxes ready: $(wc -l < $BBOX) rows" >&2

# --- Submit training jobs (8h limit for backfill scheduling) ---
declare -a TRAIN_JOBS=()
declare -a PRED_OUTS=()

submit_pair () {
  local cfg=$1 tag=$2 run=$3 out=$4 extra=${5:-}
  local jt=$(sbatch --parsable --partition=dev --time=8:00:00 -J "$tag" slurm/train_single.sh "configs/${cfg}.yaml")
  local jp=$(sbatch --parsable --partition=dev --time=1:00:00 --dependency=afterok:${jt} -J "p${tag}" \
      slurm/infer_and_va.sh "/data/artifacts/frank/misc/runs/${run}/ckpts/ckpt_best.pt" "$out" $extra)
  echo "  submitted ${cfg}: train=$jt pred=$jp -> ${out}_va.csv" >&2
  TRAIN_JOBS+=("$jt")
  PRED_OUTS+=("${out}_va.csv")
}

echo "[$(date -Iseconds)] submitting lung-crop training jobs..." >&2
submit_pair v1_3class_omnirad_lungcrop_s0 lc-om0 v1_3class_omnirad_lungcrop_s0 ${SUB16}/omnirad_lungcrop_s0
submit_pair v1_3class_omnirad_lungcrop_s1 lc-om1 v1_3class_omnirad_lungcrop_s1 ${SUB16}/omnirad_lungcrop_s1
submit_pair v1_3class_omnirad_lungcrop_s2 lc-om2 v1_3class_omnirad_lungcrop_s2 ${SUB16}/omnirad_lungcrop_s2
submit_pair v1_3class_eva02_lungcrop_s0  lc-ev0 v1_3class_eva02_lungcrop_s0  ${SUB16}/eva02_lungcrop_s0

# --- Wait for all prediction CSVs ---
echo "[$(date -Iseconds)] waiting for ${#PRED_OUTS[@]} prediction CSVs..." >&2
for f in "${PRED_OUTS[@]}"; do
  until [ -f "$f" ]; do sleep 120; done
  echo "  [$(date -Iseconds)] landed: $f" >&2
done
sleep 30

# --- Build ensemble candidates ---
echo "[$(date -Iseconds)] building lung-crop ensemble candidates..." >&2
uv run python build_lungcrop_ensemble.py 2>&1

# View-average the candidate ensembles
for v in LC_A50_pulmonary LC_B100_pulmonary LC_C70_pulmonary LC_D30_pulmonary; do
  f="${SUB16}/ladder/${v}.csv"
  if [ -f "$f" ]; then
    uv run python view_average.py --in "$f" --out "${SUB16}/ladder/${v}_va.csv" --force 2>&1 | tail -1
    mv "${SUB16}/ladder/${v}_va.csv" "$f"
  fi
done

echo "[$(date -Iseconds)] === LUNG-CROP PIPELINE DONE ===" >&2
echo "Candidates: ${SUB16}/ladder/LC_{A50,B100,C70,D30}_pulmonary.csv" >&2
