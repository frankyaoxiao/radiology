"""Per-label optimal stacking using actual val predictions.

For each label, fit non-negative weights (summing to 1) over the val predictions
of each backbone POOL to minimize MSE against ground-truth labels. Apply those
weights to test predictions to build an optimally-stacked submission.

Requires val_preds.npz files for each (run, seed). We pool by averaging within
each backbone family.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import LABEL_NAMES

LABELS = list(LABEL_NAMES)
RUNS = "/data/artifacts/frank/misc/runs"
SUB10, SUB11 = "submissions/2026-05-10", "submissions/2026-05-11"


# Families with same split_seed=42 (val partition matches → can pool-stack)
FAMILIES_42 = {
    "om": {
        "runs": [
            "v1_3class_omnirad_b14_s0",  # split_seed=42 baseline
        ],
        "test_csv": f"{SUB11}/omnirad_aug_trivial_5split_mean.csv",
    },
    "hp": {
        "runs": [f"v1_3class_hplus_s{s}_aug_trivial" for s in (0, 1, 2)],
        "test_csv": f"{SUB10}/dinov3_hplus_3seed_mean.csv",
    },
    "cn": {
        "runs": [f"v1_3class_cnxl_s{s}_aug_trivial" for s in (0, 1, 2)],
        "test_csv": f"{SUB11}/cnxl_aug_trivial_3seed_mean.csv",
    },
    "sg": {
        "runs": [f"v1_3class_siglip2_p14_384_s{s}" for s in (0, 1, 2)],
        "test_csv": f"{SUB11}/siglip2_p14_384_3seed_mean.csv",
    },
    "oc": {
        "runs": [f"v1_3class_openclip_s{s}" for s in (0, 1, 2)],
        "test_csv": f"{SUB11}/openclip_3seed_mean.csv",
    },
    "eva": {
        "runs": [f"v1_3class_eva02_s{s}" for s in (0, 1, 2)],
        "test_csv": f"{SUB11}/eva02_3seed_mean.csv",
    },
    # CXR foundation: special — no val_preds because no DL training. Skip for stacking.
}


def load_pool_val(family_runs):
    """Load val_preds for all seeds of a family, pool by averaging.
    Returns (preds (N, L), labels (N, L), paths) — labels in {-1, 0, 1, nan}."""
    files = []
    for run in family_runs:
        path = Path(f"{RUNS}/{run}/val_preds.npz")
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        d = np.load(path, allow_pickle=True)
        files.append(d)
    if not files:
        return None, None, None
    # Stack predictions, average
    # raw_labels are in {-1, 0, 1, nan} format already
    preds = np.mean([f["preds"] for f in files], axis=0)
    labels = files[0]["raw_labels"]  # same val partition
    paths = files[0]["paths"]
    return preds, labels, paths


def main():
    print("=== loading val pools ===")
    fam_val = {}  # family -> (preds, labels) — labels common
    common_labels = None
    common_paths = None
    for fam, info in FAMILIES_42.items():
        preds, labels, paths = load_pool_val(info["runs"])
        if preds is None:
            print(f"  {fam}: SKIP (no val_preds)")
            continue
        if common_labels is None:
            common_labels = labels
            common_paths = paths
        else:
            # Verify same val partition
            if not np.array_equal(paths, common_paths):
                print(f"  {fam}: paths mismatch — skipping (different val split?)")
                continue
        fam_val[fam] = preds
        print(f"  {fam}: preds {preds.shape}")

    if not fam_val:
        print("no val_preds available — exit")
        return

    fam_names = list(fam_val.keys())
    K = len(fam_names)
    print(f"\nstacking across {K} backbone pools: {fam_names}")

    # Optimize per-label weights to minimize MSE on val
    print("\n=== per-label optimal weights ===")
    weights_per_label = {}
    for li, lab in enumerate(LABELS):
        # Get this label's preds and labels
        ys = common_labels[:, li]  # (N,) with NaN where masked
        ps = np.stack([fam_val[fam][:, li] for fam in fam_names], axis=1)  # (N, K)
        mask = ~np.isnan(ys)
        y = ys[mask]
        X = ps[mask]
        n = len(y)
        if n < 100:
            print(f"  {lab}: too few samples ({n}), skipping")
            continue

        # Non-negative weights summing to 1: optimize on simplex
        # Use scipy.minimize with constraints
        def loss(w):
            p = X @ w
            return float(np.mean((y - p) ** 2))

        # Initial: equal weights
        w0 = np.ones(K) / K
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]
        bounds = [(0.0, 1.0)] * K
        res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"ftol": 1e-9, "maxiter": 500})
        w = res.x
        # Sanity: clip and renormalize
        w = np.clip(w, 0, None)
        w = w / w.sum()
        baseline_loss = float(np.mean((y - X.mean(axis=1)) ** 2))
        opt_loss = float(np.mean((y - X @ w) ** 2))
        # NMSE
        var_y = float(np.var(y))
        baseline_nmse = baseline_loss / var_y if var_y > 0 else np.nan
        opt_nmse = opt_loss / var_y if var_y > 0 else np.nan
        weights_per_label[lab] = w
        w_str = ", ".join(f"{fam}={ww:.3f}" for fam, ww in zip(fam_names, w))
        print(f"  {lab[:30]:<30}  baseline_NMSE={baseline_nmse:.4f}  optimal_NMSE={opt_nmse:.4f}  weights: {w_str}")

    # Apply to test predictions
    print("\n=== applying weights to test predictions ===")
    test_csvs = {fam: pd.read_csv(FAMILIES_42[fam]["test_csv"]).sort_values("Id").reset_index(drop=True)
                 for fam in fam_names}
    ids = test_csvs[fam_names[0]]["Id"].to_numpy()
    ens = pd.DataFrame({"Id": ids})

    # Load Cardiomegaly cherry-pick + Fracture
    ii = pd.read_csv(f"{SUB11}/ladder/I_D_with_cnxl_boost40.csv").sort_values("Id").reset_index(drop=True)
    cxr5 = pd.read_csv(f"{SUB11}/cxr_foundation_5seed_pool.csv").sort_values("Id").reset_index(drop=True)
    cn_pool = test_csvs["cn"]

    for lab in LABELS:
        if lab == "Cardiomegaly":
            ens[lab] = ii[lab].to_numpy()
            continue
        if lab == "Fracture":
            ens[lab] = 0.30 * cn_pool[lab].to_numpy() + 0.70 * cxr5[lab].to_numpy()
            continue
        if lab not in weights_per_label:
            # Fallback: equal average
            ens[lab] = np.mean([test_csvs[fam][lab].to_numpy() for fam in fam_names], axis=0)
            continue
        w = weights_per_label[lab]
        pred = np.zeros(len(ids))
        for fam, ww in zip(fam_names, w):
            pred += ww * test_csvs[fam][lab].to_numpy()
        ens[lab] = pred

    out = f"{SUB11}/ladder/STK_optimal_stacked.csv"
    ens.to_csv(out, index=False, float_format="%.6f")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
