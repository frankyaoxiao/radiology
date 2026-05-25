#!/bin/bash
#SBATCH --job-name=val_predict
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/fxiao/misc/156/slurm_logs/%j.out

# Usage: sbatch slurm/val_predict.sh <ckpt_path> <out_npz_path> [--bf16]
set -uo pipefail
CKPT="${1:?need ckpt}"
OUT="${2:?need out path}"
shift 2

cd /home/fxiao/misc/156
echo "ckpt: ${CKPT}"
echo "out:  ${OUT}"
date

uv run python -u val_predict.py --ckpt "${CKPT}" --out "${OUT}" "$@"
