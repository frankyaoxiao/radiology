#!/bin/bash
#SBATCH --job-name=mv-infer-va
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/fxiao/misc/156/slurm_logs/%j.out

set -euo pipefail
CKPT="${1:?need ckpt}"
OUT_BASE="${2:?need out base (no .csv)}"
shift 2

cd /home/fxiao/misc/156
RAW="${OUT_BASE}.csv"
VA="${OUT_BASE}_va.csv"
mkdir -p "$(dirname "${OUT_BASE}")"

echo "=============================================="
echo "MV infer: ckpt=$CKPT out=$OUT_BASE"
date
echo "=============================================="

uv run python -u submit_multiview.py --ckpt "$CKPT" --out "$RAW" --force "$@"
uv run python view_average.py --in "$RAW" --out "$VA" --force
echo "done"
