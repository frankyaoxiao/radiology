"""Try alternative Cardiomegaly cherry-picks based on val NMSE.

Top-3 Cardiomegaly backbones (split=42 val): siglip2 0.4121, eva02_ema 0.4122,
eva02_base 0.4128. Current cherry-pick I_D_with_cnxl_boost40 uses a CNXL-heavy
recipe — but CNXL was ranked 5th on Cardiomegaly val (0.4192). May be suboptimal.
"""
import pandas as pd, numpy as np
from config import LABEL_NAMES

LABELS = list(LABEL_NAMES)
SUB11 = "submissions/2026-05-11"
SUB10 = "submissions/2026-05-10"

# Load the 6 backbone pools
om   = pd.read_csv(f"{SUB11}/omnirad_aug_trivial_5split_mean.csv").sort_values("Id").reset_index(drop=True)
hp   = pd.read_csv(f"{SUB10}/dinov3_hplus_3seed_mean.csv").sort_values("Id").reset_index(drop=True)
cn   = pd.read_csv(f"{SUB11}/cnxl_aug_trivial_3seed_mean.csv").sort_values("Id").reset_index(drop=True)
cxr5 = pd.read_csv(f"{SUB11}/cxr_foundation_5seed_pool.csv").sort_values("Id").reset_index(drop=True)
sg14 = pd.read_csv(f"{SUB11}/siglip2_p14_384_3seed_mean.csv").sort_values("Id").reset_index(drop=True)
eva  = pd.read_csv(f"{SUB11}/eva02_3seed_mean.csv").sort_values("Id").reset_index(drop=True)
eva_ema = pd.read_csv(f"{SUB11}/eva02_ema_3seed_mean.csv").sort_values("Id").reset_index(drop=True)
ii   = pd.read_csv(f"{SUB11}/ladder/I_D_with_cnxl_boost40.csv").sort_values("Id").reset_index(drop=True)
ids = om["Id"].to_numpy()

# Cardiomegaly cherry-pick alternatives (val NMSE for sense):
# - siglip2: 0.4121
# - eva02_ema: 0.4122
# - eva02_base: 0.4128
# - om: 0.4143
# - cnxl: 0.4192
# - hplus: 0.4308
def build_pp(name, cardio_pred, fracture_pred=None):
    ens = pd.DataFrame({"Id": ids})
    for l in LABELS:
        if l == "Cardiomegaly":
            ens[l] = cardio_pred
        elif l == "Fracture":
            ens[l] = (0.30*cn[l].to_numpy() + 0.70*cxr5[l].to_numpy()) if fracture_pred is None else fracture_pred
        else:
            ens[l] = (0.10*om[l].to_numpy() + 0.10*hp[l].to_numpy() +
                      0.20*cn[l].to_numpy() + 0.14*cxr5[l].to_numpy() +
                      0.16*sg14[l].to_numpy() + 0.30*eva[l].to_numpy())
    out = f"{SUB11}/ladder/{name}.csv"
    ens.to_csv(out, index=False, float_format="%.6f")
    print(f"wrote {out}")

c = "Cardiomegaly"
# Build Cardiomegaly variants
build_pp("PP_Card_sgEva",     0.5*sg14[c].to_numpy() + 0.5*eva[c].to_numpy())
build_pp("PP_Card_sgEvaEMA",  0.5*sg14[c].to_numpy() + 0.5*eva_ema[c].to_numpy())
build_pp("PP_Card_top3",      0.34*sg14[c].to_numpy() + 0.33*eva[c].to_numpy() + 0.33*om[c].to_numpy())
build_pp("PP_Card_sgEvaCn",   0.4*sg14[c].to_numpy() + 0.4*eva[c].to_numpy() + 0.2*cn[c].to_numpy())
# Combine new Card cherry-pick with the existing other-label logic but lighter cnxl
