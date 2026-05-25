#!/bin/bash
#SBATCH --job-name=val_mv_pi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/fxiao/misc/156/slurm_logs/%j.out

set -uo pipefail
CKPT="${1}"
OUT="${2}"
shift 2

cd /home/fxiao/misc/156
echo "ckpt: $CKPT  out: $OUT"
uv run python -u val_predict_multiview_perimage.py --ckpt "$CKPT" --out "$OUT" "$@"
