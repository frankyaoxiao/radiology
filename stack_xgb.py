"""Per-label XGBoost stacking using val predictions.

For each label, fit a small XGBoost regressor:
    X = (N, K) val pool predictions for K backbone families
    y = (N,) val labels in {-1, 0, +1} (or NaN to mask)

Apply trained model to test predictions (same K backbones) to get per-label test.

Compared to linear stacking, XGBoost can model non-linear interactions
(e.g., "trust EVA when CNXL is uncertain") but risks overfitting val.
Use early stopping with a holdout slice.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from config import LABEL_NAMES

LABELS = list(LABEL_NAMES)
RUNS = "/data/artifacts/frank/misc/runs"
SUB10, SUB11 = "submissions/2026-05-10", "submissions/2026-05-11"

FAMILIES_42 = {
    "om": {
        "runs": [f"v1_3class_omnirad_b14_s0"],  # original baseline (split_seed=42)
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
}


def load_pool_val(family_runs):
    files = []
    for run in family_runs:
        path = Path(f"{RUNS}/{run}/val_preds.npz")
        if path.exists():
            files.append(np.load(path, allow_pickle=True))
    if not files:
        return None, None, None
    preds = np.mean([f["preds"] for f in files], axis=0)
    labels = files[0]["raw_labels"]
    paths = files[0]["paths"]
    return preds, labels, paths


def main():
    from xgboost import XGBRegressor

    print("loading val pools…")
    fam_val = {}
    common_labels = None
    common_paths = None
    for fam, info in FAMILIES_42.items():
        preds, labels, paths = load_pool_val(info["runs"])
        if preds is None:
            print(f"  {fam}: SKIP")
            continue
        if common_labels is None:
            common_labels = labels
            common_paths = paths
        fam_val[fam] = preds
        print(f"  {fam}: {preds.shape}")

    fam_names = list(fam_val.keys())
    K = len(fam_names)
    print(f"\n{K} backbone pools: {fam_names}")

    # Split val into TRAIN (for XGBoost) and HELD-OUT (for eval)
    # Use stable split (first 80% by path order = train, last 20% = held-out)
    N = len(common_labels)
    np.random.seed(42)
    perm = np.random.permutation(N)
    cut = int(0.8 * N)
    train_idx = perm[:cut]
    held_idx = perm[cut:]
    print(f"val split: {len(train_idx)} train, {len(held_idx)} held-out")

    test_csvs = {fam: pd.read_csv(FAMILIES_42[fam]["test_csv"]).sort_values("Id").reset_index(drop=True)
                 for fam in fam_names}
    ids = test_csvs[fam_names[0]]["Id"].to_numpy()
    ens = pd.DataFrame({"Id": ids})
    ii = pd.read_csv(f"{SUB11}/ladder/I_D_with_cnxl_boost40.csv").sort_values("Id").reset_index(drop=True)
    cxr5 = pd.read_csv(f"{SUB11}/cxr_foundation_5seed_pool.csv").sort_values("Id").reset_index(drop=True)
    cn_pool = test_csvs["cn"]

    print("\n=== per-label XGBoost fit ===")
    for li, lab in enumerate(LABELS):
        if lab == "Cardiomegaly":
            ens[lab] = ii[lab].to_numpy()
            continue
        if lab == "Fracture":
            ens[lab] = 0.30 * cn_pool[lab].to_numpy() + 0.70 * cxr5[lab].to_numpy()
            continue

        # Stack features (N, K)
        ys = common_labels[:, li]
        Xs = np.stack([fam_val[fam][:, li] for fam in fam_names], axis=1)
        # Also include squares / products as features
        X_aug = np.hstack([Xs, Xs ** 2, Xs.mean(axis=1, keepdims=True)])  # (N, 2K+1)

        m_train = ~np.isnan(ys[train_idx])
        m_held = ~np.isnan(ys[held_idx])
        X_tr = X_aug[train_idx][m_train]
        y_tr = ys[train_idx][m_train]
        X_hd = X_aug[held_idx][m_held]
        y_hd = ys[held_idx][m_held]

        if len(y_tr) < 100 or len(y_hd) < 50:
            print(f"  {lab[:25]:<25}  too few samples, fallback to mean")
            ens[lab] = np.mean([test_csvs[fam][lab].to_numpy() for fam in fam_names], axis=0)
            continue

        # Baseline: simple mean
        mean_pred_hd = X_hd[:, :K].mean(axis=1)
        baseline_mse = float(np.mean((y_hd - mean_pred_hd) ** 2))

        # Fit XGBoost
        xgb = XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            reg_lambda=1.0, reg_alpha=0.1,
            objective="reg:squarederror", n_jobs=4, verbosity=0,
            early_stopping_rounds=20,
        )
        xgb.fit(X_tr, y_tr, eval_set=[(X_hd, y_hd)], verbose=False)
        pred_hd = xgb.predict(X_hd)
        xgb_mse = float(np.mean((y_hd - pred_hd) ** 2))
        var_y_hd = float(np.var(y_hd))
        bnmse = baseline_mse / var_y_hd
        xnmse = xgb_mse / var_y_hd
        print(f"  {lab[:25]:<25}  baseline NMSE={bnmse:.4f}  xgb NMSE={xnmse:.4f}  gain={(bnmse-xnmse):+.4f}")

        # Apply to test
        X_test_base = np.stack([test_csvs[fam][lab].to_numpy() for fam in fam_names], axis=1)
        X_test_aug = np.hstack([X_test_base, X_test_base ** 2,
                                X_test_base.mean(axis=1, keepdims=True)])
        pred_test = xgb.predict(X_test_aug)
        ens[lab] = pred_test

    out = f"{SUB11}/ladder/STKX_xgb_stacked.csv"
    ens.to_csv(out, index=False, float_format="%.6f")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
