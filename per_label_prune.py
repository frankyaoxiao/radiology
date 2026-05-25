"""Per-label leg pruning: for each label, drop the worst backbones from the average.

All backbones used here trained on split_seed=42 (same val set), so per-label val
NMSE is comparable. We score each backbone for each label, drop the bottom-K, and
renormalize the remaining weights.
"""
import json, glob
import pandas as pd, numpy as np
from config import LABEL_NAMES

LABELS = list(LABEL_NAMES)
RUNS = "/data/artifacts/frank/misc/runs"
SUB11 = "submissions/2026-05-11"
SUB10 = "submissions/2026-05-10"


def best_per_label(jsonl_path):
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
            if m is None:
                continue
            if m < best_overall:
                best_overall = m
                best_per_lab = {l: n.get(l) for l in LABELS}
    return best_per_lab


# Same-split (=42) families
pools_42 = {
    "om":           (sorted(glob.glob(f"{RUNS}/v1_3class_omnirad_b14_s0_aug_trivial/metrics.jsonl") +
                            glob.glob(f"{RUNS}/v1_3class_omnirad_b14_s0_aug_smooth/metrics.jsonl") +
                            glob.glob(f"{RUNS}/v1_3class_omnirad_b14_s[127][0-9]*/metrics.jsonl")),
                     f"{SUB11}/omnirad_aug_trivial_5split_mean.csv",
                     0.10),
    "hp":           (sorted(glob.glob(f"{RUNS}/v1_3class_hplus_s[012]_aug_trivial/metrics.jsonl")),
                     f"{SUB10}/dinov3_hplus_3seed_mean.csv",
                     0.10),
    "cn":           (sorted(glob.glob(f"{RUNS}/v1_3class_cnxl_s[012]_aug_trivial/metrics.jsonl")),
                     f"{SUB11}/cnxl_aug_trivial_3seed_mean.csv",
                     0.20),
    "cxr5":         ([], f"{SUB11}/cxr_foundation_5seed_pool.csv", 0.14),  # no metrics
    "sg":           (sorted(glob.glob(f"{RUNS}/v1_3class_siglip2_p14_384_s[012]/metrics.jsonl")),
                     f"{SUB11}/siglip2_p14_384_3seed_mean.csv",
                     0.16),
    "eva":          (sorted(glob.glob(f"{RUNS}/v1_3class_eva02_s[012]/metrics.jsonl")),
                     f"{SUB11}/eva02_3seed_mean.csv",
                     0.30),
}


def family_per_label(jsonl_paths):
    if not jsonl_paths:
        return None
    accs = {l: [] for l in LABELS}
    for p in jsonl_paths:
        bp = best_per_label(p)
        if bp is None:
            continue
        for l in LABELS:
            if bp[l] is not None:
                accs[l].append(bp[l])
    return {l: np.mean(accs[l]) if accs[l] else None for l in LABELS}


fams = list(pools_42.keys())
fam_nmse = {fam: family_per_label(pools_42[fam][0]) for fam in fams}

# Per-label per-family NMSE table
print(f"{'family':<6} {'wt':>5}", *[f"{l[:9]:>9}" for l in LABELS], sep=" ")
for fam in fams:
    pl = fam_nmse[fam]
    wt = pools_42[fam][2]
    if pl is None:
        print(f"{fam:<6} {wt:>5.2f}", *[f"{'--':>9}" for _ in LABELS], sep=" ")
    else:
        print(f"{fam:<6} {wt:>5.2f}",
              *[f"{(pl[l] if pl[l] is not None else float('nan')):>9.4f}" for l in LABELS],
              sep=" ")

# Build a "pruned" per-label ensemble.
# For each label: drop backbones whose val NMSE on that label is in the WORST 1/3.
# Renormalize the remaining weights.
print()
print("=== per-label leg pruning (drop worst 1/3 by val NMSE on that label) ===")
pool_csvs = {fam: pd.read_csv(pools_42[fam][1]).sort_values("Id").reset_index(drop=True)
             for fam in fams}
ids = pool_csvs["om"]["Id"].to_numpy()
ii = pd.read_csv(f"{SUB11}/ladder/I_D_with_cnxl_boost40.csv").sort_values("Id").reset_index(drop=True)

ens = pd.DataFrame({"Id": ids})
for l in LABELS:
    if l == "Cardiomegaly":
        ens[l] = ii[l].to_numpy()
        print(f"  {l:<30} <- I_D_with_cnxl_boost40 (cherry-pick, kept)")
        continue
    if l == "Fracture":
        # Keep current Fracture cherry-pick (0.30*cn + 0.70*cxr)
        ens[l] = 0.30*pool_csvs["cn"][l].to_numpy() + 0.70*pool_csvs["cxr5"][l].to_numpy()
        print(f"  {l:<30} <- 0.30*cn + 0.70*cxr5 (kept)")
        continue

    # Rank families by their val NMSE on this label (lower is better)
    nmse_for_l = []
    for fam in fams:
        if fam == "cxr5":
            nmse = 0.75  # rough mid-tier estimate; no metrics.jsonl
        else:
            pl = fam_nmse[fam]
            nmse = (pl and pl.get(l)) or 1.0
        nmse_for_l.append((fam, nmse, pools_42[fam][2]))

    nmse_for_l.sort(key=lambda x: x[1])
    # Drop bottom 1/3: 6 families → drop 2 worst
    keep = nmse_for_l[:4]
    drop = nmse_for_l[4:]

    # Renormalize weights
    total_w = sum(w for _, _, w in keep)
    pred = np.zeros(len(ids))
    for fam, _, w in keep:
        pred += (w / total_w) * pool_csvs[fam][l].to_numpy()
    ens[l] = pred

    kept_str = ", ".join(f"{fam}({nm:.3f})" for fam, nm, _ in keep)
    drop_str = ", ".join(f"{fam}({nm:.3f})" for fam, nm, _ in drop)
    print(f"  {l:<30} keep: {kept_str}")
    print(f"  {'':<30} drop: {drop_str}")

out = f"{SUB11}/ladder/PLP_per_label_pruned.csv"
ens.to_csv(out, index=False, float_format="%.6f")
print(f"wrote {out}")
