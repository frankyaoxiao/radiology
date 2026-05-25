"""Build ensemble candidates using lung-crop model predictions for pulmonary labels.

Lung-crop models are specialists for diseases inside the lung fields:
  Pneumonia, Lung Opacity, Pleural Effusion, Pleural Other.
For other labels (Cardiomegaly, Enlarged Card, Support Devices, Fracture, No Finding)
keep the existing full-image PP_6leg ensemble.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from config import LABEL_NAMES

LABELS = list(LABEL_NAMES)
RUNS = "/data/artifacts/frank/misc/runs"
SUB10, SUB11, SUB16 = "submissions/2026-05-10", "submissions/2026-05-11", "submissions/2026-05-16"

PULMONARY = ["Lung Opacity", "Pneumonia", "Pleural Effusion", "Pleural Other"]


def pool_mean(csvs):
    dfs = [pd.read_csv(p).sort_values("Id").reset_index(drop=True) for p in csvs if Path(p).exists()]
    if not dfs:
        return None
    ids = dfs[0]["Id"].to_numpy()
    out = pd.DataFrame({"Id": ids})
    for lab in LABELS:
        out[lab] = np.mean([d[lab].to_numpy() for d in dfs], axis=0)
    return out


def best_val_per_label(run):
    p = Path(f"{RUNS}/{run}/metrics.jsonl")
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p)]
    if not rows:
        return None
    best = min(rows, key=lambda r: r["val"]["nmse"].get("mean", 1e9))
    return best["val"]["nmse"]


def main():
    # --- Diagnostic: lung-crop val NMSE vs baseline on pulmonary labels ---
    print("=== val NMSE comparison (pulmonary labels) ===")
    baseline_runs = {
        "om_base": "v1_3class_omnirad_b14_s0_aug_trivial",
        "eva_base": "v1_3class_eva02_s0",
    }
    lungcrop_runs = {
        "om_lung_s0": "v1_3class_omnirad_lungcrop_s0",
        "om_lung_s1": "v1_3class_omnirad_lungcrop_s1",
        "om_lung_s2": "v1_3class_omnirad_lungcrop_s2",
        "eva_lung_s0": "v1_3class_eva02_lungcrop_s0",
    }
    hdr = f"{'run':<16}" + "".join(f"{l[:10]:>11}" for l in PULMONARY) + f"{'mean':>9}"
    print(hdr)
    for name, run in {**baseline_runs, **lungcrop_runs}.items():
        n = best_val_per_label(run)
        if n is None:
            print(f"{name:<16}  (no metrics)")
            continue
        vals = [n.get(l) for l in PULMONARY]
        row = f"{name:<16}" + "".join(f"{v:>11.4f}" if v else f"{'-':>11}" for v in vals)
        row += f"{n.get('mean', float('nan')):>9.4f}"
        print(row)

    # --- Pool lung-crop predictions ---
    om_lung = pool_mean([f"{SUB16}/omnirad_lungcrop_s{s}_va.csv" for s in (0, 1, 2)])
    eva_lung = pool_mean([f"{SUB16}/eva02_lungcrop_s0_va.csv"])
    lung_csvs = [c for c in [
        f"{SUB16}/omnirad_lungcrop_s0_va.csv", f"{SUB16}/omnirad_lungcrop_s1_va.csv",
        f"{SUB16}/omnirad_lungcrop_s2_va.csv", f"{SUB16}/eva02_lungcrop_s0_va.csv",
    ] if Path(c).exists()]
    lung_pool = pool_mean(lung_csvs)
    if lung_pool is None:
        print("ERROR: no lung-crop prediction CSVs found")
        return
    lung_pool.to_csv(f"{SUB16}/lungcrop_pool_mean.csv", index=False, float_format="%.6f")
    print(f"\nlung-crop pool: {len(lung_csvs)} models -> lungcrop_pool_mean.csv")

    # --- Base PP_6leg ensemble (current best) ---
    pp = pd.read_csv(f"{SUB11}/ladder/PP_6leg_eva_30.csv").sort_values("Id").reset_index(drop=True)
    lp = lung_pool.sort_values("Id").reset_index(drop=True)
    ids = pp["Id"].to_numpy()

    # Variant A: pulmonary labels = 0.5 PP + 0.5 lungcrop
    # Variant B: pulmonary labels = lungcrop only
    # Variant C: pulmonary labels = 0.3 PP + 0.7 lungcrop
    for tag, w_lung in [("A50", 0.5), ("B100", 1.0), ("C70", 0.7), ("D30", 0.3)]:
        ens = pp.copy()
        for lab in PULMONARY:
            ens[lab] = (1 - w_lung) * pp[lab].to_numpy() + w_lung * lp[lab].to_numpy()
        out = f"{SUB16}/ladder/LC_{tag}_pulmonary.csv"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        ens.to_csv(out, index=False, float_format="%.6f")
        print(f"  wrote LC_{tag}_pulmonary.csv (pulmonary = {1-w_lung:.1f} PP + {w_lung:.1f} lungcrop)")

    print("\ndone. View-average the LC_*.csv candidates before submitting.")


if __name__ == "__main__":
    main()
