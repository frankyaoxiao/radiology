"""Expanded stacking with more backbone pools (EMA, SWA, MV variants).

Uses val_preds.npz (and val_preds_swa.npz) for each pool.
"""
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from config import LABEL_NAMES

LABELS = list(LABEL_NAMES)
RUNS = "/data/artifacts/frank/misc/runs"
SUB10, SUB11 = "submissions/2026-05-10", "submissions/2026-05-11"


# Each entry: (val_preds_paths, test_csv_path)
POOLS = {
    "om":      ([f"{RUNS}/v1_3class_omnirad_b14_s0/val_preds.npz"],
                f"{SUB11}/omnirad_aug_trivial_5split_mean.csv"),
    "hp":      ([f"{RUNS}/v1_3class_hplus_s{s}_aug_trivial/val_preds.npz" for s in (0,1,2)],
                f"{SUB10}/dinov3_hplus_3seed_mean.csv"),
    "cn":      ([f"{RUNS}/v1_3class_cnxl_s{s}_aug_trivial/val_preds.npz" for s in (0,1,2)],
                f"{SUB11}/cnxl_aug_trivial_3seed_mean.csv"),
    "sg":      ([f"{RUNS}/v1_3class_siglip2_p14_384_s{s}/val_preds.npz" for s in (0,1,2)],
                f"{SUB11}/siglip2_p14_384_3seed_mean.csv"),
    "oc":      ([f"{RUNS}/v1_3class_openclip_s{s}/val_preds.npz" for s in (0,1,2)],
                f"{SUB11}/openclip_3seed_mean.csv"),
    "eva":     ([f"{RUNS}/v1_3class_eva02_s{s}/val_preds.npz" for s in (0,1,2)],
                f"{SUB11}/eva02_3seed_mean.csv"),
    "eva_ema": ([f"{RUNS}/v1_3class_eva02_s{s}_ema/val_preds.npz" for s in (0,1,2)],
                f"{SUB11}/eva02_ema_3seed_mean.csv"),
    "eva_swa": ([f"{RUNS}/v1_3class_eva02_s{s}/val_preds_swa.npz" for s in (0,1,2)],
                f"{SUB11}/eva02_swa_3seed_mean.csv"),
    "cn_swa":  ([f"{RUNS}/v1_3class_cnxl_s{s}_aug_trivial/val_preds_swa.npz" for s in (0,1,2)],
                f"{SUB11}/cnxl_swa_3seed_mean.csv"),
    "hp_swa":  ([f"{RUNS}/v1_3class_hplus_s{s}_aug_trivial/val_preds_swa.npz" for s in (0,1,2)],
                f"{SUB11}/hplus_swa_3seed_mean.csv"),
    "sg_swa":  ([f"{RUNS}/v1_3class_siglip2_p14_384_s{s}/val_preds_swa.npz" for s in (0,1,2)],
                f"{SUB11}/siglip2_swa_3seed_mean.csv"),
    "mv":      ([f"{RUNS}/v1_3class_multiview_omnirad_pooled/val_preds.npz"],
                f"{SUB11}/multiview_omnirad_3variant_mean.csv"),
}


def load_pool(paths):
    files = []
    for p in paths:
        if Path(p).exists():
            files.append(np.load(p, allow_pickle=True))
    if not files:
        return None, None
    preds = np.mean([f["preds"] for f in files], axis=0)
    labels = files[0]["raw_labels"]
    return preds, labels


def main():
    pool_val = {}
    common_labels = None
    common_shape = None
    for name, (val_paths, _test) in POOLS.items():
        preds, labels = load_pool(val_paths)
        if preds is None:
            print(f"  {name}: NOT READY (no val_preds)")
            continue
        if common_labels is None:
            common_labels = labels
            common_shape = preds.shape
        elif preds.shape != common_shape:
            print(f"  {name}: SHAPE MISMATCH {preds.shape} vs {common_shape} — skipping")
            continue
        pool_val[name] = preds
        print(f"  {name}: shape {preds.shape}")

    if not pool_val:
        print("no pools ready")
        return

    pools = list(pool_val.keys())
    K = len(pools)
    print(f"\nstacking {K} pools: {pools}")

    test_csvs = {}
    for name in pools:
        try:
            test_csvs[name] = pd.read_csv(POOLS[name][1]).sort_values("Id").reset_index(drop=True)
        except FileNotFoundError:
            print(f"  WARN: no test csv for {name}, dropping from stacking")
            del pool_val[name]
    pools = [n for n in pools if n in test_csvs]
    K = len(pools)

    # K-fold CV stacking
    ids = test_csvs[pools[0]]["Id"].to_numpy()
    ens = pd.DataFrame({"Id": ids})

    print("\n=== K-fold (K=5) stacking ===")
    total_b, total_o = 0, 0
    n_labels = 0
    for li, lab in enumerate(LABELS):
        ys = common_labels[:, li]
        Xs = np.stack([pool_val[p][:, li] for p in pools], axis=1)
        mask = ~np.isnan(ys)
        y_all, X_all = ys[mask], Xs[mask]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_weights = []
        fold_oof_preds = np.full(len(y_all), np.nan)
        for tr, te in kf.split(X_all):
            X_tr, y_tr = X_all[tr], y_all[tr]
            def loss(w): return float(np.mean((y_tr - X_tr @ w) ** 2))
            w0 = np.ones(K) / K
            cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            bounds = [(0.0, 1.0)] * K
            res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"ftol": 1e-9, "maxiter": 500})
            w = np.clip(res.x, 0, None); w = w / w.sum()
            fold_weights.append(w)
            fold_oof_preds[te] = X_all[te] @ w
        avg_w = np.mean(fold_weights, axis=0); avg_w = avg_w / avg_w.sum()

        oof_nmse = float(np.mean((y_all - fold_oof_preds) ** 2)) / float(np.var(y_all))
        base_nmse = float(np.mean((y_all - X_all.mean(axis=1)) ** 2)) / float(np.var(y_all))
        w_str = ", ".join(f"{p}={ww:.2f}" for p, ww in sorted(zip(pools, avg_w), key=lambda x: -x[1]) if ww > 0.02)
        print(f"  {lab[:24]:<24}  base={base_nmse:.4f}  oof={oof_nmse:.4f}  Δ={base_nmse-oof_nmse:+.4f}  top: {w_str}")
        total_b += base_nmse; total_o += oof_nmse; n_labels += 1

        # Apply to test
        pred = np.zeros(len(ids))
        for p, w in zip(pools, avg_w):
            pred += w * test_csvs[p][lab].to_numpy()
        ens[lab] = pred

    print(f"\nmean baseline NMSE: {total_b/n_labels:.4f}")
    print(f"mean OOF NMSE: {total_o/n_labels:.4f}")
    print(f"mean gain: {(total_b-total_o)/n_labels:+.4f}")

    # Apply Cardiomegaly and Fracture cherry-picks (LB-validated)
    ii = pd.read_csv(f"{SUB11}/ladder/I_D_with_cnxl_boost40.csv").sort_values("Id").reset_index(drop=True)
    cn = test_csvs.get("cn")
    cxr5 = pd.read_csv(f"{SUB11}/cxr_foundation_5seed_pool.csv").sort_values("Id").reset_index(drop=True)
    ens["Cardiomegaly"] = ii["Cardiomegaly"].to_numpy()
    if cn is not None:
        ens["Fracture"] = 0.30 * cn["Fracture"].to_numpy() + 0.70 * cxr5["Fracture"].to_numpy()

    out = f"{SUB11}/ladder/STK_expanded.csv"
    ens.to_csv(out, index=False, float_format="%.6f")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
