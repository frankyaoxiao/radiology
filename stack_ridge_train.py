"""Ridge-shrinkage stacker: linear weights with L2 toward manual prior.

For each label, minimize:
  ‖y_train - X_train @ w‖² + λ‖w - w_manual‖²
subject to w ≥ 0, sum(w) = 1.

Pick λ via val NMSE on a held-out val_preds.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
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
# Manual prior weights (matches PP_6leg without cxr, renormalized)
W_MANUAL = {"om": 0.10/0.86, "hp": 0.10/0.86, "cn": 0.20/0.86,
            "sg": 0.16/0.86, "oc": 0.0, "eva": 0.30/0.86}


def load_pool(runs, split):
    files = [np.load(Path(f"{RUNS}/{r}/{split}.npz"), allow_pickle=True)
             for r in runs if Path(f"{RUNS}/{r}/{split}.npz").exists()]
    if not files: return None, None
    return np.mean([f["preds"] for f in files], axis=0), files[0]["raw_labels"]


def main():
    pool_train, pool_val = {}, {}
    train_labels = val_labels = None
    for name, (runs, _) in POOLS.items():
        tr_p, tr_l = load_pool(runs, "train_preds")
        v_p, v_l = load_pool(runs, "val_preds")
        if tr_p is None or v_p is None:
            print(f"  {name}: skip"); continue
        if train_labels is None: train_labels, val_labels = tr_l, v_l
        pool_train[name] = tr_p; pool_val[name] = v_p
        print(f"  {name}: train={tr_p.shape}, val={v_p.shape}")

    pools = list(pool_train.keys()); K = len(pools)
    w_prior = np.array([W_MANUAL.get(p, 1.0/K) for p in pools])
    w_prior = w_prior / w_prior.sum()
    print(f"\npools: {pools}\nw_prior: {dict(zip(pools, w_prior.round(3)))}")

    test_csvs = {p: pd.read_csv(POOLS[p][1]).sort_values("Id").reset_index(drop=True) for p in pools}
    ids = test_csvs[pools[0]]["Id"].to_numpy()
    ens = pd.DataFrame({"Id": ids})

    lambdas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
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
            ens[lab] = np.mean([test_csvs[p][lab].to_numpy() for p in pools], axis=0); continue

        baseline_v = float(np.mean((y_v - X_v.mean(axis=1)) ** 2)) / float(np.var(y_v))
        # Manual baseline
        manual_v = float(np.mean((y_v - X_v @ w_prior) ** 2)) / float(np.var(y_v))

        best_v = float("inf"); best_lam = None; best_w = None
        for lam in lambdas:
            # Loss: train MSE + lam * (w - w_prior)'(w - w_prior)
            def loss(w):
                resid = y_tr - X_tr @ w
                return float(np.mean(resid ** 2) + lam * np.sum((w - w_prior) ** 2))
            w0 = w_prior.copy()
            cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            res = minimize(loss, w0, method="SLSQP", bounds=[(0.0, 1.0)] * K,
                           constraints=cons, options={"ftol": 1e-10, "maxiter": 500})
            w = np.clip(res.x, 0, None); w = w / w.sum()
            v_nmse = float(np.mean((y_v - X_v @ w) ** 2)) / float(np.var(y_v))
            if v_nmse < best_v:
                best_v, best_lam, best_w = v_nmse, lam, w

        wstr = ", ".join(f"{p}={ww:.2f}" for p, ww in sorted(zip(pools, best_w), key=lambda x: -x[1]) if ww > 0.05)
        print(f"  {lab[:24]:<24}  base={baseline_v:.4f}  manual={manual_v:.4f}  ridge_val={best_v:.4f}  λ={best_lam}  {wstr}")
        total_b += manual_v; total_v += best_v; n_lab += 1

        pred = np.zeros(len(ids))
        for p, w in zip(pools, best_w):
            pred += w * test_csvs[p][lab].to_numpy()
        ens[lab] = pred

    print(f"\nmean manual NMSE: {total_b/n_lab:.4f}")
    print(f"mean ridge NMSE: {total_v/n_lab:.4f}")
    print(f"mean gain over manual: {(total_b-total_v)/n_lab:+.4f}")

    ii = pd.read_csv(f"{SUB11}/ladder/I_D_with_cnxl_boost40.csv").sort_values("Id").reset_index(drop=True)
    cn = test_csvs.get("cn")
    cxr5 = pd.read_csv(f"{SUB11}/cxr_foundation_5seed_pool.csv").sort_values("Id").reset_index(drop=True)
    ens["Cardiomegaly"] = ii["Cardiomegaly"].to_numpy()
    if cn is not None:
        ens["Fracture"] = 0.30 * cn["Fracture"].to_numpy() + 0.70 * cxr5["Fracture"].to_numpy()

    out = "submissions/2026-05-16/ladder/STK_ridge_trainfit.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ens.to_csv(out, index=False, float_format="%.6f")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
