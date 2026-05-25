#!/bin/bash
#SBATCH --job-name=mv-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/fxiao/misc/156/slurm_logs/%j.out

set -euo pipefail
CONFIG="${1:?need config}"

cd /home/fxiao/misc/156
echo "==============================================" 
echo "job id     : $SLURM_JOB_ID"
echo "config     : $CONFIG"
echo "cwd        : $PWD"
date
echo "=============================================="

uv run python -u train_multiview.py --config "$CONFIG"
