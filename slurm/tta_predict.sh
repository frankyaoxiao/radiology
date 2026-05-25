#!/bin/bash
#SBATCH --job-name=tta_predict
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/fxiao/misc/156/slurm_logs/%j.out

set -uo pipefail
CKPT="${1:?need ckpt}"
OUT_BASE="${2:?need out path}"
shift 2

cd /home/fxiao/misc/156
echo "ckpt: ${CKPT}"
echo "out base: ${OUT_BASE}"
date

uv run python -u submit.py --ckpt "${CKPT}" --out "${OUT_BASE}.csv" --tta 3 --force "$@"
uv run python view_average.py --in "${OUT_BASE}.csv" --out "${OUT_BASE}_va.csv" --force
echo "done"
