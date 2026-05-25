"""XGBoost stacker: fit on TRAIN backbone-predictions, validate on VAL.

Early-stops on val NMSE per label. Applies to test predictions.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from config import LABEL_NAMES

LABELS = list(LABEL_NAMES)
RUNS = "/data/artifacts/frank/misc/runs"
SUB10, SUB11 = "submissions/2026-05-10", "submissions/2026-05-11"

POOLS = {
    "om":  (["v1_3class_omnirad_b14_s0"], f"{SUB11}/omnirad_aug_trivial_5split_mean.csv"),
    "hp":  ([f"v1_3class_hplus_s{s}_aug_trivial" for s in (0,1,2)], f"{SUB10}/dinov3_hplus_3seed_mean.csv"),
    "cn":  ([f"v1_3class_cnxl_s{s}_aug_trivial" for s in (0,1,2)], f"{SUB11}/cnxl_aug_trivial_3seed_mean.csv"),
    "sg":  ([f"v1_3class_siglip2_p14_384_s{s}" for s in (0,1,2)], f"{SUB11}/siglip2_p14_384_3seed_mean.csv"),
    "oc":  ([f"v1_3class_openclip_s{s}" for s in (0,1,2)], f"{SUB11}/openclip_3seed_mean.csv"),
    "eva": ([f"v1_3class_eva02_s{s}" for s in (0,1,2)], f"{SUB11}/eva02_3seed_mean.csv"),
}


def load_pool(runs, split):
    files = [np.load(Path(f"{RUNS}/{r}/{split}.npz"), allow_pickle=True)
             for r in runs if Path(f"{RUNS}/{r}/{split}.npz").exists()]
    if not files: return None, None
    return np.mean([f["preds"] for f in files], axis=0), files[0]["raw_labels"]


def main():
    from xgboost import XGBRegressor

    pool_train, pool_val = {}, {}
    train_labels, val_labels = None, None
    for name, (runs, _) in POOLS.items():
        tr_p, tr_l = load_pool(runs, "train_preds")
        v_p, v_l = load_pool(runs, "val_preds")
        if tr_p is None or v_p is None:
            print(f"  {name}: skip"); continue
        if train_labels is None:
            train_labels, val_labels = tr_l, v_l
        pool_train[name] = tr_p; pool_val[name] = v_p
        print(f"  {name}: train={tr_p.shape}, val={v_p.shape}")

    pools = list(pool_train.keys())
    K = len(pools)
    print(f"\nstacking {K} pools: {pools}")

    test_csvs = {p: pd.read_csv(POOLS[p][1]).sort_values("Id").reset_index(drop=True) for p in pools}
    ids = test_csvs[pools[0]]["Id"].to_numpy()
    ens = pd.DataFrame({"Id": ids})

    total_b, total_v, n_lab = 0, 0, 0
    for li, lab in enumerate(LABELS):
        X_tr = np.stack([pool_train[p][:, li] for p in pools], axis=1)
        y_tr = train_labels[:, li]
        X_v = np.stack([pool_val[p][:, li] for p in pools], axis=1)
        y_v = val_labels[:, li]
        m_tr = ~np.isnan(y_tr); m_v = ~np.isnan(y_v)
        X_tr, y_tr = X_tr[m_tr], y_tr[m_tr]
        X_v, y_v = X_v[m_v], y_v[m_v]
        if len(y_v) < 100:
            ens[lab] = np.mean([test_csvs[p][lab].to_numpy() for p in pools], axis=0)
            continue

        # Add squares and pairwise products as engineered features
        X_tr_aug = np.hstack([X_tr, X_tr ** 2])
        X_v_aug = np.hstack([X_v, X_v ** 2])

        # Baseline
        baseline_v = float(np.mean((y_v - X_v.mean(axis=1)) ** 2)) / float(np.var(y_v))

        xgb = XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            reg_lambda=2.0, reg_alpha=0.5, subsample=0.8, colsample_bytree=0.8,
            n_jobs=4, verbosity=0, objective="reg:squarederror",
            early_stopping_rounds=30,
        )
        xgb.fit(X_tr_aug, y_tr, eval_set=[(X_v_aug, y_v)], verbose=False)
        pred_v = xgb.predict(X_v_aug)
        xgb_v_nmse = float(np.mean((y_v - pred_v) ** 2)) / float(np.var(y_v))
        total_b += baseline_v; total_v += xgb_v_nmse; n_lab += 1
        print(f"  {lab[:25]:<25}  baseline={baseline_v:.4f}  xgb_val={xgb_v_nmse:.4f}  Δ={baseline_v-xgb_v_nmse:+.4f}")

        X_test = np.stack([test_csvs[p][lab].to_numpy() for p in pools], axis=1)
        X_test_aug = np.hstack([X_test, X_test ** 2])
        ens[lab] = xgb.predict(X_test_aug)

    print(f"\nmean baseline NMSE: {total_b/n_lab:.4f}")
    print(f"mean XGB val NMSE: {total_v/n_lab:.4f}")
    print(f"mean gain: {(total_b-total_v)/n_lab:+.4f}")

    ii = pd.read_csv(f"{SUB11}/ladder/I_D_with_cnxl_boost40.csv").sort_values("Id").reset_index(drop=True)
    cn = test_csvs.get("cn")
    cxr5 = pd.read_csv(f"{SUB11}/cxr_foundation_5seed_pool.csv").sort_values("Id").reset_index(drop=True)
    ens["Cardiomegaly"] = ii["Cardiomegaly"].to_numpy()
    if cn is not None:
        ens["Fracture"] = 0.30 * cn["Fracture"].to_numpy() + 0.70 * cxr5["Fracture"].to_numpy()

    out = "submissions/2026-05-16/ladder/STK_xgb_trainfit.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ens.to_csv(out, index=False, float_format="%.6f")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
