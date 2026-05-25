#!/bin/bash
# Wait for the 3 xrv_densenet_mimic CSVs, then build pool + ladder ensembles.
set -uo pipefail
cd /home/fxiao/misc/156
SUB="submissions/2026-05-11"

EXPECTED=(
  "${SUB}/xrv_densenet_mimic_s0_aug_trivial_va.csv"
  "${SUB}/xrv_densenet_mimic_s1_aug_trivial_va.csv"
  "${SUB}/xrv_densenet_mimic_s2_aug_trivial_va.csv"
)

echo "Waiting for 3 DenseNet CSVs..." >&2
for f in "${EXPECTED[@]}"; do
  until [ -f "$f" ]; do sleep 60; done
  echo "  $(date -Iseconds): $(basename $f) ready" >&2
done

echo "" >&2
echo "All 3 DenseNet CSVs landed. Assembling..." >&2

uv run python -c "
import pandas as pd, numpy as np
from config import LABEL_NAMES
LABELS = list(LABEL_NAMES)
SUB10 = 'submissions/2026-05-10'
SUB11 = 'submissions/2026-05-11'

# DenseNet 3-seed pool
csvs = [f'{SUB11}/xrv_densenet_mimic_s{s}_aug_trivial_va.csv' for s in (0,1,2)]
dns = [pd.read_csv(p).sort_values('Id').reset_index(drop=True) for p in csvs]
ids = dns[0]['Id'].to_numpy()
dn_pool = pd.DataFrame({'Id': ids})
for lab in LABELS: dn_pool[lab] = np.mean([d[lab].to_numpy() for d in dns], axis=0)
dn_pool.to_csv(f'{SUB11}/xrv_densenet_mimic_aug_trivial_3seed_mean.csv', index=False, float_format='%.6f')
print('wrote xrv_densenet_mimic_aug_trivial_3seed_mean.csv')

# Load other pools
om_pool   = pd.read_csv(f'{SUB11}/omnirad_aug_trivial_5split_mean.csv').sort_values('Id').reset_index(drop=True)
cnxl_pool = pd.read_csv(f'{SUB11}/cnxl_aug_trivial_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
hp_aug_pool = pd.read_csv(f'{SUB11}/hplus_aug_trivial_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
hp_old = pd.read_csv(f'{SUB10}/dinov3_hplus_3seed_mean.csv').sort_values('Id').reset_index(drop=True)

# Build 4 ladder variants — call them R, S, T, U (continue letters)
# R: standalone DenseNet pool — diagnostic
dn_pool.to_csv(f'{SUB11}/ladder/R_densenet_pool_alone.csv', index=False, float_format='%.6f')

# S: I (best so far) recipe with DenseNet added as a 4th leg
# .25 om + .25 hp + .35 cnxl + .15 densenet
ens_s = pd.DataFrame({'Id': ids})
for lab in LABELS:
    ens_s[lab] = 0.25*om_pool[lab].to_numpy() + 0.25*hp_old[lab].to_numpy() + 0.35*cnxl_pool[lab].to_numpy() + 0.15*dn_pool[lab].to_numpy()
ens_s.to_csv(f'{SUB11}/ladder/S_4leg_w25_25_35_15.csv', index=False, float_format='%.6f')

# T: same but with H+ aug pool
ens_t = pd.DataFrame({'Id': ids})
for lab in LABELS:
    ens_t[lab] = 0.25*om_pool[lab].to_numpy() + 0.25*hp_aug_pool[lab].to_numpy() + 0.35*cnxl_pool[lab].to_numpy() + 0.15*dn_pool[lab].to_numpy()
ens_t.to_csv(f'{SUB11}/ladder/T_4leg_w25_25_35_15_hpaug.csv', index=False, float_format='%.6f')

# U: J recipe with DenseNet added — drop weight from each existing leg proportionally
# J = .25/.25/.50, add .1 DenseNet: .225/.225/.45/.10
ens_u = pd.DataFrame({'Id': ids})
for lab in LABELS:
    ens_u[lab] = 0.225*om_pool[lab].to_numpy() + 0.225*hp_aug_pool[lab].to_numpy() + 0.45*cnxl_pool[lab].to_numpy() + 0.10*dn_pool[lab].to_numpy()
ens_u.to_csv(f'{SUB11}/ladder/U_J_plus_densenet_10pct.csv', index=False, float_format='%.6f')

print('wrote R, S, T, U')
"

# View-average all 4
for v in R_densenet_pool_alone S_4leg_w25_25_35_15 T_4leg_w25_25_35_15_hpaug U_J_plus_densenet_10pct; do
  uv run python view_average.py \
    --in submissions/2026-05-11/ladder/${v}.csv \
    --out submissions/2026-05-11/ladder/${v}_va.csv --force 2>&1 | tail -1
  mv submissions/2026-05-11/ladder/${v}_va.csv submissions/2026-05-11/ladder/${v}.csv
done

echo "" >&2
echo "=== DenseNet assembly DONE ===" >&2
ls -la "${SUB}/ladder/R_"* "${SUB}/ladder/S_"* "${SUB}/ladder/T_"* "${SUB}/ladder/U_"* >&2
