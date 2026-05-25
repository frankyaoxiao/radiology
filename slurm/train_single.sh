#!/bin/bash
#SBATCH --job-name=train_single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/home/fxiao/misc/156/slurm_logs/%j.out

# Single-GPU training launcher for Heimdall2.
# Launch: sbatch slurm/train_single.sh [config_path]
# Default config: configs/v1_3class_b16_s0.yaml
# train.py runs without torchrun in single-process mode (setup_ddp falls back).

set -euo pipefail

CONFIG="${1:-configs/v1_3class_b16_s0.yaml}"
cd /home/fxiao/misc/156

echo "=============================================="
echo "job id     : ${SLURM_JOB_ID}"
echo "node       : $(hostname)"
echo "config     : ${CONFIG}"
echo "cwd        : $(pwd)"
echo "cuda devs  : ${CUDA_VISIBLE_DEVICES:-unset}"
date
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo "=============================================="

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export TORCH_HOME="${HOME}/.cache/torch"
mkdir -p "${TORCH_HOME}"

uv run python -u train.py --config "${CONFIG}"

echo "=============================================="
echo "job ${SLURM_JOB_ID} finished at $(date)"
