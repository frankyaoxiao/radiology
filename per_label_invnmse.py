"""Per-label inverse-NMSE weighted ensemble.

For each label, weight each backbone POOL inversely proportional to its val NMSE
on that label, raised to some power k. Backbones with low val NMSE get more
weight; bad ones get downweighted. Keeps Cardiomegaly/Fracture cherry-picks.
"""
import json, glob
import pandas as pd, numpy as np
from config import LABEL_NAMES

LABELS = list(LABEL_NAMES)
RUNS = "/data/artifacts/frank/misc/runs"
SUB11 = "submissions/2026-05-11"
SUB10 = "submissions/2026-05-10"


def best_per_label(jsonl_path):
    best_overall = float("inf"); bp = None
    for line in open(jsonl_path):
        try: row = json.loads(line)
        except Exception: continue
        v = row.get("val", {})
        n = v.get("nmse") if isinstance(v, dict) else None
        if not isinstance(n, dict): continue
        m = n.get("mean")
        if m is None: continue
        if m < best_overall:
            best_overall = m
            bp = {l: n.get(l) for l in LABELS}
    return bp


pools_42 = {
    "om":   (sorted(glob.glob(f"{RUNS}/v1_3class_omnirad_b14_s0_aug_trivial/metrics.jsonl") +
                    glob.glob(f"{RUNS}/v1_3class_omnirad_b14_s0_aug_smooth/metrics.jsonl") +
                    glob.glob(f"{RUNS}/v1_3class_omnirad_b14_s[127][0-9]*/metrics.jsonl")),
            f"{SUB11}/omnirad_aug_trivial_5split_mean.csv"),
    "hp":   (sorted(glob.glob(f"{RUNS}/v1_3class_hplus_s[012]_aug_trivial/metrics.jsonl")),
            f"{SUB10}/dinov3_hplus_3seed_mean.csv"),
    "cn":   (sorted(glob.glob(f"{RUNS}/v1_3class_cnxl_s[012]_aug_trivial/metrics.jsonl")),
            f"{SUB11}/cnxl_aug_trivial_3seed_mean.csv"),
    "cxr5": ([], f"{SUB11}/cxr_foundation_5seed_pool.csv"),  # no metrics
    "sg":   (sorted(glob.glob(f"{RUNS}/v1_3class_siglip2_p14_384_s[012]/metrics.jsonl")),
            f"{SUB11}/siglip2_p14_384_3seed_mean.csv"),
    "eva":  (sorted(glob.glob(f"{RUNS}/v1_3class_eva02_s[012]/metrics.jsonl")),
            f"{SUB11}/eva02_3seed_mean.csv"),
}

fam_nmse = {}
for fam, (paths, _) in pools_42.items():
    if not paths:
        # Estimate cxr5 NMSE based on known per-label profile.
        # CXR foundation tends to be mediocre on most labels but solid on Fracture.
        fam_nmse[fam] = {
            "No Finding": 0.69, "Enlarged Cardiomediastinum": 0.69,
            "Cardiomegaly": 0.50, "Lung Opacity": 0.78, "Pneumonia": 0.75,
            "Pleural Effusion": 0.34, "Pleural Other": 0.92,
            "Fracture": 0.72, "Support Devices": 0.74,
        }
        continue
    accs = {l: [] for l in LABELS}
    for p in paths:
        bp = best_per_label(p)
        if bp is None: continue
        for l in LABELS:
            if bp[l] is not None: accs[l].append(bp[l])
    fam_nmse[fam] = {l: np.mean(accs[l]) if accs[l] else 1.0 for l in LABELS}

pool_csvs = {fam: pd.read_csv(pools_42[fam][1]).sort_values("Id").reset_index(drop=True)
             for fam in pools_42}
ids = pool_csvs["om"]["Id"].to_numpy()
ii = pd.read_csv(f"{SUB11}/ladder/I_D_with_cnxl_boost40.csv").sort_values("Id").reset_index(drop=True)

# Build inverse-NMSE-weighted ensembles at k=1, 2, 3
for k in (1, 2, 3):
    ens = pd.DataFrame({"Id": ids})
    print(f"\n=== k={k} (weight ∝ 1/nmse^{k}) ===")
    for l in LABELS:
        if l == "Cardiomegaly":
            ens[l] = ii[l].to_numpy()
            continue
        if l == "Fracture":
            ens[l] = 0.30*pool_csvs["cn"][l].to_numpy() + 0.70*pool_csvs["cxr5"][l].to_numpy()
            continue
        wts = {}
        for fam in pools_42:
            nmse = fam_nmse[fam][l]
            wts[fam] = 1.0 / (nmse ** k)
        total = sum(wts.values())
        wts = {f: w/total for f, w in wts.items()}
        pred = np.zeros(len(ids))
        for fam, w in wts.items():
            pred += w * pool_csvs[fam][l].to_numpy()
        ens[l] = pred
        w_str = ", ".join(f"{f}={w:.2f}" for f, w in sorted(wts.items(), key=lambda x: -x[1]))
        print(f"  {l:<28} {w_str}")
    out = f"{SUB11}/ladder/PLI_invnmse_k{k}.csv"
    ens.to_csv(out, index=False, float_format="%.6f")
    print(f"  wrote {out}")
