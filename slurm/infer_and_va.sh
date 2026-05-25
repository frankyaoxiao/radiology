#!/bin/bash
#SBATCH --job-name=infer-va
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/fxiao/misc/156/slurm_logs/%j.out

# Inference + view averaging in one job: produces both raw and _va CSVs.
# Launch: sbatch slurm/infer_and_va.sh <ckpt> <out_csv_basename> [--bf16] [--batch-size N]
# E.g.   sbatch slurm/infer_and_va.sh /data/.../ckpt_best.pt submissions/2026-05-11/foo

set -euo pipefail

CKPT="${1:?need ckpt path}"
OUT_BASE="${2:?need output basename without .csv}"
shift 2
RAW_CSV="${OUT_BASE}.csv"
VA_CSV="${OUT_BASE}_va.csv"

cd /home/fxiao/misc/156
mkdir -p "$(dirname "${OUT_BASE}")"

echo "=============================================="
echo "job id : ${SLURM_JOB_ID}"
echo "node   : $(hostname)"
echo "ckpt   : ${CKPT}"
echo "raw    : ${RAW_CSV}"
echo "va     : ${VA_CSV}"
date
echo "=============================================="

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

uv run python -u submit.py --ckpt "${CKPT}" --out "${RAW_CSV}" --force "$@"

uv run python view_average.py --in "${RAW_CSV}" --out "${VA_CSV}" --force

echo "=============================================="
echo "wrote ${RAW_CSV} and ${VA_CSV}"
ls -la "${RAW_CSV}" "${VA_CSV}"
