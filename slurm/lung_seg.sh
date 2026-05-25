#!/bin/bash
#SBATCH --job-name=lung_seg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/home/fxiao/misc/156/slurm_logs/%j.out

set -euo pipefail
cd /home/fxiao/misc/156
echo "==============================================" 
date
uv run python -u lung_seg_bboxes.py \
  --paths-csv /data/artifacts/frank/misc/lung_seg_paths.csv \
  --data-root /data/artifacts/frank/misc \
  --out /data/artifacts/frank/misc/labels/lung_bboxes.csv \
  --batch-size 8 \
  --num-workers 8
date
