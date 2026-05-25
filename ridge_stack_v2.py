"""Ridge stacking over 6 backbone POOLS (each seed-averaged before ridge).

Components: om_s0, hp_pool, cnxl_pool, cxr_pool, sg14_pool, eva_pool.
Fits ridge with GroupKFold by patient to pick alpha, then applies weights
to TEST CSV pools and writes a submission.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from config import LABEL_NAMES
from metrics import per_label_nmse


def _load(key, val_dir):
    d = np.load(val_dir / f"val_{key}.npz", allow_pickle=True)
    return d["preds"], d["paths"], d["raw_labels"]


def _pool_mean(keys, val_dir):
    preds_list = []
    paths = None; labels = None
    for k in keys:
        p, pa, la = _load(k, val_dir)
        if paths is None:
            paths = pa; labels = la
        preds_list.append(p)
    return np.mean(preds_list, axis=0), paths, labels


def _pid_from_paths(paths):
    import re
    pat = re.compile(r"pid(\d+)")
    return np.array([int(pat.search(p).group(1)) if pat.search(p) else -1 for p in paths])


def _ridge_fit_predict(P_train, y_train, m_train, P_val, alpha, per_label=False):
    N, K, L = P_train.shape
    Nv = P_val.shape[0]
    if per_label:
        coefs = np.zeros((L, K), dtype=np.float32)
        pred = np.zeros((Nv, L), dtype=np.float32)
        for li in range(L):
            X = P_train[:, :, li]
            y = y_train[:, li]
            m = m_train[:, li]
            Xm, ym = X[m], y[m]
            if len(Xm) == 0: continue
            mdl = Ridge(alpha=alpha, fit_intercept=False, positive=False)
            mdl.fit(Xm, ym)
            coefs[li] = mdl.coef_
            pred[:, li] = mdl.predict(P_val[:, :, li])
        return pred, coefs
    else:
        X = P_train.transpose(0, 2, 1).reshape(N * L, K)
        y = y_train.reshape(N * L)
        m = m_train.reshape(N * L)
        Xm, ym = X[m], y[m]
        model = Ridge(alpha=alpha, fit_intercept=False, positive=False)
        model.fit(Xm, ym)
        Xv = P_val.transpose(0, 2, 1).reshape(Nv * L, K)
        pred = model.predict(Xv).reshape(Nv, L)
        return pred, model.coef_


def _nmse_mean(pred, label, mask):
    yt = label.copy(); yt[~mask] = np.nan
    out = per_label_nmse(yt, pred, list(LABEL_NAMES))
    return out["mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", type=Path, default=Path("/data/artifacts/frank/misc/val_preds"))
    ap.add_argument("--per-label", action="store_true")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    print("Loading val pools...")
    om_pred, paths, raw_labels = _load("om_s0", args.val_dir)
    hp_pool_pred, _, _ = _pool_mean(["hp_s0", "hp_s1", "hp_s2"], args.val_dir)
    cnxl_pool_pred, _, _ = _pool_mean(["cnxl_s0", "cnxl_s1", "cnxl_s2"], args.val_dir)
    cxr_pred, _, _ = _load("cxr_pool", args.val_dir)
    sg14_pool_pred, _, _ = _pool_mean(["sg14_s0", "sg14_s1", "sg14_s2"], args.val_dir)
    eva_pool_pred, _, _ = _pool_mean(["eva_s0", "eva_s1", "eva_s2"], args.val_dir)

    P = np.stack([om_pred, hp_pool_pred, cnxl_pool_pred, cxr_pred, sg14_pool_pred, eva_pool_pred], axis=1)
    comp_names = ["om", "hp_pool", "cnxl_pool", "cxr_pool", "sg14_pool", "eva_pool"]
    mask = ~np.isnan(raw_labels)
    labels = np.nan_to_num(raw_labels, nan=0.0)
    print(f"val rows: {P.shape[0]}  K={P.shape[1]}")

    eq_pred = P.mean(axis=1)
    print(f"equal-weight val nmse: {_nmse_mean(eq_pred, labels, mask):.4f}")

    pids = _pid_from_paths(paths)
    gkf = GroupKFold(n_splits=args.n_folds)

    alphas = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    cv_nmse = {}
    for a in alphas:
        fold_nmses = []
        for tr_idx, va_idx in gkf.split(P, labels[:, 0], groups=pids):
            pred, _ = _ridge_fit_predict(P[tr_idx], labels[tr_idx], mask[tr_idx], P[va_idx], a, per_label=args.per_label)
            fold_nmses.append(_nmse_mean(pred, labels[va_idx], mask[va_idx]))
        cv_nmse[a] = float(np.mean(fold_nmses))
        print(f"  alpha={a:>8.2f}  cv_nmse={cv_nmse[a]:.4f}")
    best_alpha = min(cv_nmse, key=cv_nmse.get)
    print(f"\nbest alpha: {best_alpha}  cv_nmse: {cv_nmse[best_alpha]:.4f}")

    final_pred, final_coef = _ridge_fit_predict(P, labels, mask, P, best_alpha, per_label=args.per_label)
    print(f"final fit-on-train val nmse: {_nmse_mean(final_pred, labels, mask):.4f}")

    if args.per_label:
        print("\nfinal coefficients per label:")
        print(f"{'label':30s}  " + "  ".join(f"{n:>10s}" for n in comp_names))
        for li, lab in enumerate(LABEL_NAMES):
            print(f"{lab:30s}  " + "  ".join(f"{final_coef[li, k]:+10.4f}" for k in range(len(comp_names))))
    else:
        print("\nfinal coefficients:")
        for n, c in zip(comp_names, final_coef):
            print(f"  {n:12s}: {c:+.4f}")

    # Apply to TEST pools
    SUB10 = "submissions/2026-05-10"
    SUB11 = "submissions/2026-05-11"
    test_csvs = [
        f"{SUB11}/omnirad_aug_trivial_5split_mean.csv",
        f"{SUB10}/dinov3_hplus_3seed_mean.csv",
        f"{SUB11}/cnxl_aug_trivial_3seed_mean.csv",
        f"{SUB11}/cxr_foundation_5seed_pool.csv",
        f"{SUB11}/siglip2_p14_384_3seed_mean.csv",
        f"{SUB11}/eva02_3seed_mean.csv",
    ]
    test_dfs = [pd.read_csv(p).sort_values("Id").reset_index(drop=True) for p in test_csvs]
    ids = test_dfs[0]["Id"].to_numpy()
    out = pd.DataFrame({"Id": ids})
    for li, lab in enumerate(LABEL_NAMES):
        vals = np.zeros(len(ids), dtype=np.float32)
        weights = final_coef[li] if args.per_label else final_coef
        for d, w in zip(test_dfs, weights):
            vals += w * d[lab].to_numpy()
        out[lab] = vals
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, float_format="%.6f")
    print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
