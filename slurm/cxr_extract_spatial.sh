#!/bin/bash
#SBATCH --job-name=cxr_ext_sp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/fxiao/misc/156/slurm_logs/%j.out

set -uo pipefail
RANK="${1:?need rank}"
WORLD="${2:?need world}"

cd /home/fxiao/misc/156
echo "rank ${RANK}/${WORLD} on $(hostname) gpu:${CUDA_VISIBLE_DEVICES:-unset}"
date

/home/fxiao/misc/156/cxr_env/bin/python -u cxr_extract_spatial.py \
  --rank "${RANK}" --world "${WORLD}" \
  --paths-csv /data/artifacts/frank/misc/cxr_extract_paths.csv \
  --out-dir   /data/artifacts/frank/misc/cxr_embeds_spatial \
  --gpu 0 --batch-size 4
