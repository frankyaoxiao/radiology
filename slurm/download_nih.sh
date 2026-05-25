#!/bin/bash
#SBATCH --job-name=dl-nih
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/fxiao/misc/156/slurm_logs/%j.out

set -euo pipefail
cd /home/fxiao/misc/156
date
uv run --with huggingface-hub python -u download_nih.py
date
