#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -A cs156b
#SBATCH -p any
#SBATCH -J download_rad_dino
#SBATCH --mail-user=akumarap@caltech.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o download_rad_dino_%j.out

module load anaconda3
# conda activate cs156b  # uncomment once env is set up

python -u -c "
from transformers import Dinov2Model
print('Downloading microsoft/rad-dino ...')
m = Dinov2Model.from_pretrained('microsoft/rad-dino')
save_path = '/resnick/groups/CS156b/from_central/2026/scalm_akumarap/runs/rad_dino_weights'
m.save_pretrained(save_path)
print(f'Saved to {save_path}')
"
