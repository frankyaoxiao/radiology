"""Assemble 5-OmniRad multi-split + H+ s0 + ConvNeXt-L into a 7-model ensemble."""
from pathlib import Path
import pandas as pd
import numpy as np
from config import LABEL_NAMES

LABELS = list(LABEL_NAMES)
SUB_DIR = Path("submissions/2026-05-10")

omnirad_csvs = [
    SUB_DIR / "omnirad_448.csv",          # split 42 (s0)
    SUB_DIR / "omnirad_448_s7.csv",
    SUB_DIR / "omnirad_448_s13.csv",
    SUB_DIR / "omnirad_448_s29.csv",
    SUB_DIR / "omnirad_448_s101.csv",
]
hplus_csv = SUB_DIR / "dinov3_hplus.csv"
cnxl_csv  = SUB_DIR / "dinov3_cnxl.csv"

def load_aligned(path):
    df = pd.read_csv(path).sort_values("Id").reset_index(drop=True)
    return df

# 1. OmniRad 5-split mean
oms = [load_aligned(p) for p in omnirad_csvs]
ids = oms[0]["Id"].to_numpy()
for d in oms[1:]:
    assert (d["Id"].to_numpy() == ids).all(), f"id mismatch in {d}"
om_mean = pd.DataFrame({"Id": ids})
for lab in LABELS:
    om_mean[lab] = np.mean([d[lab].to_numpy() for d in oms], axis=0)
om_mean.to_csv(SUB_DIR / "omnirad_5split_mean.csv", index=False, float_format="%.6f")
print(f"wrote omnirad_5split_mean.csv ({len(om_mean)} rows)")

# 2. 7-model ensemble: equal-weight average of (5-OmniRad mean, H+, ConvNeXt-L)
hp = load_aligned(hplus_csv)
cn = load_aligned(cnxl_csv)
ens = pd.DataFrame({"Id": ids})
for lab in LABELS:
    ens[lab] = (om_mean[lab].to_numpy() + hp[lab].to_numpy() + cn[lab].to_numpy()) / 3.0
ens.to_csv(SUB_DIR / "ensemble_7model.csv", index=False, float_format="%.6f")
print(f"wrote ensemble_7model.csv ({len(ens)} rows)")
