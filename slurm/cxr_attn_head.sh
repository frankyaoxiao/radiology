#!/bin/bash
#SBATCH --job-name=cxr_attn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/fxiao/misc/156/slurm_logs/%j.out

set -uo pipefail
SEED="${1:?need seed}"
OUT="${2:?need out csv}"

cd /home/fxiao/misc/156
date
uv run python -u cxr_attn_head_train.py --out-csv "${OUT}" --seed "${SEED}"
