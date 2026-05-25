"""Ridge stacking on val with K-fold CV.

Loads per-component val predictions from val_preds/*.npz, fits ridge weights
using K-fold CV across val patients to pick alpha, then applies the resulting
weights to test-set CSVs and writes a combined submission.

Two modes:
  --mode pool      : 3 components (om_s0, hp_pool, cnxl_pool) — pool == mean of seeds
  --mode individual: 7 components (om_s0, hp_s0/s1/s2, cnxl_s0/s1/s2)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from config import LABEL_NAMES


def _load_val_preds(comp_keys: list[str], val_dir: Path):
    """Returns (P, labels, mask, paths) where P is (N, K, 9)."""
    preds_list = []
    labels = paths = None
    for key in comp_keys:
        d = np.load(val_dir / f"val_{key}.npz", allow_pickle=True)
        if labels is None:
            labels = d["raw_labels"]
            paths = d["paths"]
        else:
            assert (d["paths"] == paths).all(), f"paths mismatch in {key}"
        preds_list.append(d["preds"])
    P = np.stack(preds_list, axis=1)  # (N, K, 9)
    mask = ~np.isnan(labels)
    labels = np.nan_to_num(labels, nan=0.0)  # zeros where masked; we'll never use them
    return P.astype(np.float32), labels.astype(np.float32), mask, paths


def _pid_from_path(paths: np.ndarray) -> np.ndarray:
    import re
    pat = re.compile(r"pid(\d+)")
    return np.array([int(pat.search(p).group(1)) if pat.search(p) else -1 for p in paths])


def _ridge_fit_predict(P_train, y_train, m_train, P_val, alpha, per_label: bool = False):
    """Fit ridge. If per_label, fit a separate K-d weight vector per label.
    Returns (y_pred, coef) where coef is (K,) for global or (L, K) for per-label.
    """
    N, K, L = P_train.shape
    Nv = P_val.shape[0]
    if per_label:
        coefs = np.zeros((L, K), dtype=np.float32)
        pred = np.zeros((Nv, L), dtype=np.float32)
        for li in range(L):
            X = P_train[:, :, li]  # (N, K)
            y = y_train[:, li]
            m = m_train[:, li]
            Xm, ym = X[m], y[m]
            if len(Xm) == 0:
                continue
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


def _nmse(pred, label, mask):
    """Per-label NMSE: (sum(err^2)) / (sum(label^2)), averaged across labels."""
    nmse_per = []
    for li in range(label.shape[1]):
        m = mask[:, li]
        if m.sum() == 0:
            nmse_per.append(np.nan)
            continue
        e = pred[m, li] - label[m, li]
        sse = float((e ** 2).sum())
        sst = float((label[m, li] ** 2).sum())
        nmse_per.append(sse / max(sst, 1e-8))
    return float(np.nanmean(nmse_per))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", type=Path, default=Path("/data/artifacts/frank/misc/val_preds"))
    ap.add_argument("--mode", choices=["pool", "individual"], default="pool")
    ap.add_argument("--per-label", action="store_true",
                    help="Fit a separate ridge per label (9 weight vectors)")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    if args.mode == "pool":
        # We average hp/cnxl seeds inside the val tensor, then ridge is K=3.
        # Build pool by averaging seed columns.
        comp_keys = ["om_s0", "hp_s0", "hp_s1", "hp_s2", "cnxl_s0", "cnxl_s1", "cnxl_s2"]
        P_all, labels, mask, paths = _load_val_preds(comp_keys, args.val_dir)
        # Pool: om_s0, mean(hp), mean(cnxl)
        om = P_all[:, 0:1, :]
        hp_pool = P_all[:, 1:4, :].mean(axis=1, keepdims=True)
        cnxl_pool = P_all[:, 4:7, :].mean(axis=1, keepdims=True)
        P = np.concatenate([om, hp_pool, cnxl_pool], axis=1)  # (N, 3, 9)
        comp_names = ["om", "hp_pool", "cnxl_pool"]
    else:
        comp_keys = ["om_s0", "hp_s0", "hp_s1", "hp_s2", "cnxl_s0", "cnxl_s1", "cnxl_s2"]
        P, labels, mask, paths = _load_val_preds(comp_keys, args.val_dir)
        comp_names = comp_keys

    K = P.shape[1]
    N = P.shape[0]
    print(f"val rows: {N}  components: {K}  ({comp_names})")
    # Baseline: pre-CV val NMSE with equal weights
    equal_pred = P.mean(axis=1)
    print(f"equal-weight val nmse: {_nmse(equal_pred, labels, mask):.4f}")

    # GroupKFold by patient
    pids = _pid_from_path(paths)
    gkf = GroupKFold(n_splits=args.n_folds)

    # Sweep alphas
    alphas = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    cv_nmse = {}
    for alpha in alphas:
        fold_nmses = []
        for tr_idx, va_idx in gkf.split(P, labels[:, 0], groups=pids):
            pred, _ = _ridge_fit_predict(P[tr_idx], labels[tr_idx], mask[tr_idx], P[va_idx], alpha, per_label=args.per_label)
            n = _nmse(pred, labels[va_idx], mask[va_idx])
            fold_nmses.append(n)
        cv_nmse[alpha] = float(np.mean(fold_nmses))
        print(f"  alpha={alpha:>8.2f}  cv_nmse={cv_nmse[alpha]:.4f}")

    best_alpha = min(cv_nmse, key=cv_nmse.get)
    print(f"\nbest alpha: {best_alpha}  cv_nmse: {cv_nmse[best_alpha]:.4f}")

    # Final fit on all val with best alpha
    final_pred, final_coef = _ridge_fit_predict(P, labels, mask, P, best_alpha, per_label=args.per_label)
    print(f"final fit-on-train val nmse: {_nmse(final_pred, labels, mask):.4f}")
    if args.per_label:
        print("\nfinal coefficients per label:")
        print(f"{'label':30s}  " + "  ".join(f"{n:>10s}" for n in comp_names))
        for li, lab in enumerate(LABEL_NAMES):
            print(f"{lab:30s}  " + "  ".join(f"{final_coef[li, k]:+10.4f}" for k in range(len(comp_names))))
    else:
        print("\nfinal coefficients:")
        for n, c in zip(comp_names, final_coef):
            print(f"  {n:12s}: {c:+.4f}")

    # Apply weights to TEST predictions
    print(f"\napplying weights to test predictions...")
    SUB10 = "submissions/2026-05-10"
    SUB11 = "submissions/2026-05-11"

    if args.mode == "pool":
        # OmniRad: use 5-split aug_trivial pool at test
        om_test = pd.read_csv(f"{SUB11}/omnirad_aug_trivial_5split_mean.csv").sort_values("Id").reset_index(drop=True)
        hp_test = pd.read_csv(f"{SUB11}/hplus_aug_trivial_3seed_mean.csv").sort_values("Id").reset_index(drop=True)
        cnxl_test = pd.read_csv(f"{SUB11}/cnxl_aug_trivial_3seed_mean.csv").sort_values("Id").reset_index(drop=True)
        test_preds = [om_test, hp_test, cnxl_test]
    else:
        # Individual seeds
        om_test = pd.read_csv(f"{SUB11}/omnirad_aug_trivial_5split_mean.csv").sort_values("Id").reset_index(drop=True)
        hp0 = pd.read_csv(f"{SUB11}/hplus_aug_trivial_va.csv").sort_values("Id").reset_index(drop=True)
        hp1 = pd.read_csv(f"{SUB11}/hplus_s1_aug_trivial_va.csv").sort_values("Id").reset_index(drop=True)
        hp2 = pd.read_csv(f"{SUB11}/hplus_s2_aug_trivial_va.csv").sort_values("Id").reset_index(drop=True)
        cn0 = pd.read_csv(f"{SUB11}/cnxl_aug_trivial_va.csv").sort_values("Id").reset_index(drop=True)
        cn1 = pd.read_csv(f"{SUB11}/cnxl_s1_aug_trivial_va.csv").sort_values("Id").reset_index(drop=True)
        cn2 = pd.read_csv(f"{SUB11}/cnxl_s2_aug_trivial_va.csv").sort_values("Id").reset_index(drop=True)
        test_preds = [om_test, hp0, hp1, hp2, cn0, cn1, cn2]

    ids = test_preds[0]["Id"].to_numpy()
    out = pd.DataFrame({"Id": ids})
    for li, lab in enumerate(LABEL_NAMES):
        vals = np.zeros(len(ids), dtype=np.float32)
        weights = final_coef[li] if args.per_label else final_coef
        for c, w in zip(test_preds, weights):
            vals += w * c[lab].to_numpy()
        out[lab] = vals
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, float_format="%.6f")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
