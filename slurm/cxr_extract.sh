#!/bin/bash
#SBATCH --job-name=cxr_extract
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/fxiao/misc/156/slurm_logs/%j.out

# Usage: sbatch slurm/cxr_extract.sh <rank> <world>
# Each rank is a single-GPU process; world is the total shard count (e.g. 8).

set -uo pipefail
RANK="${1:?need rank}"
WORLD="${2:?need world}"

cd /home/fxiao/misc/156

echo "=============================================="
echo "job id : ${SLURM_JOB_ID}"
echo "node   : $(hostname)"
echo "rank   : ${RANK}/${WORLD}"
echo "cuda   : ${CUDA_VISIBLE_DEVICES:-unset}"
date
echo "=============================================="

/home/fxiao/misc/156/cxr_env/bin/python -u cxr_extract.py \
  --rank "${RANK}" \
  --world "${WORLD}" \
  --paths-csv /data/artifacts/frank/misc/cxr_extract_paths.csv \
  --out-dir   /data/artifacts/frank/misc/cxr_embeds \
  --gpu 0
