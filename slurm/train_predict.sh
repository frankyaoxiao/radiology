#!/bin/bash
#SBATCH --job-name=train_predict
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/fxiao/misc/156/slurm_logs/%j.out

set -uo pipefail
CKPT="${1}"
OUT="${2}"
shift 2

cd /home/fxiao/misc/156
uv run python -u train_predict.py --ckpt "$CKPT" --out "$OUT" "$@"
