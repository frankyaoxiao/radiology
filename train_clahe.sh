#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -A cs156b
#SBATCH -p gpu
#SBATCH -J clahe_train
#SBATCH --mail-user=akumarap@caltech.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o clahe_%j.out

CONFIG="${1:-configs/hpc_3class_448_clahe_s0.yaml}"

cd /resnick/groups/CS156b/from_central/2026/scalm_akumarap/code
uv run python -u train.py --config "$CONFIG"
