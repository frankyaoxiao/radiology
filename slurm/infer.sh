#!/bin/bash
#SBATCH --job-name=infer-csv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/fxiao/misc/156/slurm_logs/%j.out

# Inference launcher for Heimdall2.
# Launch: sbatch slurm/infer.sh <ckpt_path> <output_csv> [--ckpt extra1 extra2 ...]
#   sbatch slurm/infer.sh /data/.../ckpt_best.pt submissions/foo.csv
# For ensembling multiple ckpts in one CSV, pass them as a single --ckpt arg list.

set -euo pipefail

CKPT="${1:?need ckpt path}"
OUT="${2:?need output csv path}"
shift 2

cd /home/fxiao/misc/156
mkdir -p "$(dirname "${OUT}")"

echo "=============================================="
echo "job id : ${SLURM_JOB_ID}"
echo "node   : $(hostname)"
echo "ckpt   : ${CKPT}"
echo "out    : ${OUT}"
date
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "=============================================="

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

uv run python -u submit.py --ckpt "${CKPT}" --out "${OUT}" "$@"

echo "=============================================="
echo "wrote ${OUT}"
ls -la "${OUT}"
head -3 "${OUT}"
echo "..."
echo "rows: $(wc -l < "${OUT}")"
