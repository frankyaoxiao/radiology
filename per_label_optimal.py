"""Per-label optimal backbone selection.

For each label, find the backbone family whose pool val NMSE is lowest.
Then build a test prediction where each label is from its optimal backbone pool.
"""
import json, glob, os
import pandas as pd, numpy as np
from config import LABEL_NAMES

LABELS = list(LABEL_NAMES)
RUNS = "/data/artifacts/frank/misc/runs"
SUB11 = "submissions/2026-05-11"
SUB10 = "submissions/2026-05-10"


def last_per_label(jsonl_path, max_steps=999999999):
    """Get per-label NMSE at the BEST-overall checkpoint (the one we'd use for inference)."""
    best_overall = float("inf")
    best_per_lab = None
    with open(jsonl_path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            v = row.get("val", {})
            n = v.get("nmse") if isinstance(v, dict) else None
            if not isinstance(n, dict):
                continue
            m = n.get("mean")
            if m is None or row.get("step", 0) > max_steps:
                continue
            if m < best_overall:
                best_overall = m
                best_per_lab = {l: n.get(l) for l in LABELS}
    return best_per_lab


# Map run names to backbone-family + test CSV path
# (backbone_family, test_pool_csv) for each set of runs we ensemble at test
backbone_pools = {
    "om_5split":      (sorted(glob.glob(f"{RUNS}/v1_3class_omnirad_b14_s0_aug_trivial*/metrics.jsonl")),
                       f"{SUB11}/omnirad_aug_trivial_5split_mean.csv"),
    "hplus":          (sorted(glob.glob(f"{RUNS}/v1_3class_hplus_s[012]_aug_trivial/metrics.jsonl")),
                       f"{SUB10}/dinov3_hplus_3seed_mean.csv"),
    "cnxl_trivial":   (sorted(glob.glob(f"{RUNS}/v1_3class_cnxl_s[012]_aug_trivial/metrics.jsonl")),
                       f"{SUB11}/cnxl_aug_trivial_3seed_mean.csv"),
    "cnxl_split":     (sorted(glob.glob(f"{RUNS}/v1_3class_cnxl_s[012]_aug_trivial_split*/metrics.jsonl")),
                       f"{SUB11}/cnxl_split_3seed_mean.csv"),
    "siglip2":        (sorted(glob.glob(f"{RUNS}/v1_3class_siglip2_p14_384_s[012]/metrics.jsonl")),
                       f"{SUB11}/siglip2_p14_384_3seed_mean.csv"),
    "siglip2_split":  (sorted(glob.glob(f"{RUNS}/v1_3class_siglip2_p14_384_s[012]_split*/metrics.jsonl")),
                       f"{SUB11}/siglip2_split_3seed_mean.csv"),
    "eva02_base":     (sorted(glob.glob(f"{RUNS}/v1_3class_eva02_s[012]/metrics.jsonl")),
                       f"{SUB11}/eva02_3seed_mean.csv"),
    "eva02_split":    (sorted(glob.glob(f"{RUNS}/v1_3class_eva02_s[012]_split*/metrics.jsonl")),
                       f"{SUB11}/eva02_split_3seed_mean.csv"),
    "eva02_ema":      (sorted(glob.glob(f"{RUNS}/v1_3class_eva02_s[012]_ema/metrics.jsonl")),
                       f"{SUB11}/eva02_ema_3seed_mean.csv"),
    "openclip":       (sorted(glob.glob(f"{RUNS}/v1_3class_openclip_s[012]/metrics.jsonl")),
                       f"{SUB11}/openclip_3seed_mean.csv"),
    "openclip_split": (sorted(glob.glob(f"{RUNS}/v1_3class_openclip_s[012]_split*/metrics.jsonl")),
                       f"{SUB11}/openclip_split_3seed_mean.csv"),
    "cxr5":           ([],  # CXR Foundation is sklearn-derived, no metrics.jsonl
                       f"{SUB11}/cxr_foundation_5seed_pool.csv"),
}

# Compute mean per-label NMSE across each family's seeds (proxy for pool NMSE)
def family_per_label(jsonl_paths):
    if not jsonl_paths:
        return None
    per_lab = {l: [] for l in LABELS}
    for p in jsonl_paths:
        bp = last_per_label(p)
        if bp is None:
            continue
        for l in LABELS:
            if bp[l] is not None:
                per_lab[l].append(bp[l])
    return {l: (np.mean(per_lab[l]) if per_lab[l] else None) for l in LABELS}


family_nmse = {}
for fam, (paths, _csv) in backbone_pools.items():
    family_nmse[fam] = family_per_label(paths)

# Print per-(family, label) NMSE
print(f"{'family':<18}", *[f"{l[:10]:>10}" for l in LABELS], "mean".rjust(10), sep=" ")
for fam in backbone_pools:
    pl = family_nmse[fam]
    if pl is None:
        print(f"{fam:<18}", *[f"{'--':>10}" for _ in LABELS], "--".rjust(10), sep=" ")
        continue
    means = [pl[l] for l in LABELS if pl[l] is not None]
    mean = np.mean(means) if means else float("nan")
    print(f"{fam:<18}",
          *[f"{(pl[l] if pl[l] is not None else float('nan')):>10.4f}" for l in LABELS],
          f"{mean:>10.4f}", sep=" ")

# Per-label winner
print()
print("=== Per-label winning pool ===")
winners = {}
for l in LABELS:
    best, best_fam = float("inf"), None
    for fam, pl in family_nmse.items():
        if pl is None or pl[l] is None:
            continue
        if pl[l] < best:
            best = pl[l]
            best_fam = fam
    winners[l] = (best_fam, best)
    print(f"  {l:<32} winner={best_fam:<18} nmse={best:.4f}")

# Build per-label optimal test prediction
print()
print("=== Building per-label optimal ensemble ===")
import os
pool_csvs = {}
for fam, (_paths, csv) in backbone_pools.items():
    if os.path.exists(csv):
        pool_csvs[fam] = pd.read_csv(csv).sort_values("Id").reset_index(drop=True)
    else:
        print(f"  MISSING: {csv}")

ids = pool_csvs["om_5split"]["Id"].to_numpy()
ens = pd.DataFrame({"Id": ids})
for l in LABELS:
    fam, _ = winners[l]
    if fam not in pool_csvs:
        print(f"  FALLBACK for {l}: using om_5split (winner {fam} CSV missing)")
        fam = "om_5split"
    ens[l] = pool_csvs[fam][l].to_numpy()
    print(f"  {l:<32} <- {fam}")

ens.to_csv(f"{SUB11}/ladder/PLO_per_label_optimal.csv", index=False, float_format="%.6f")
print(f"wrote PLO_per_label_optimal.csv")
