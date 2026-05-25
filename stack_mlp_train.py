"""MLP stacker trained on TRAIN predictions, validated on VAL predictions.

For each label:
  X_train = (N_tr, K) backbone train predictions
  y_train = (N_tr,) raw labels {-1, 0, +1, nan}
  X_val   = (N_v, K)  backbone val predictions
  y_val   = (N_v,)    val labels

Train small MLP with early stopping on val NMSE. Apply to test predictions.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import LABEL_NAMES

LABELS = list(LABEL_NAMES)
RUNS = "/data/artifacts/frank/misc/runs"
SUB10, SUB11 = "submissions/2026-05-10", "submissions/2026-05-11"

# Pool definitions: (list of {train_preds, val_preds} pairs, test_csv)
POOLS = {
    "om":  (["v1_3class_omnirad_b14_s0"], f"{SUB11}/omnirad_aug_trivial_5split_mean.csv"),
    "hp":  ([f"v1_3class_hplus_s{s}_aug_trivial" for s in (0,1,2)], f"{SUB10}/dinov3_hplus_3seed_mean.csv"),
    "cn":  ([f"v1_3class_cnxl_s{s}_aug_trivial" for s in (0,1,2)], f"{SUB11}/cnxl_aug_trivial_3seed_mean.csv"),
    "sg":  ([f"v1_3class_siglip2_p14_384_s{s}" for s in (0,1,2)], f"{SUB11}/siglip2_p14_384_3seed_mean.csv"),
    "oc":  ([f"v1_3class_openclip_s{s}" for s in (0,1,2)], f"{SUB11}/openclip_3seed_mean.csv"),
    "eva": ([f"v1_3class_eva02_s{s}" for s in (0,1,2)], f"{SUB11}/eva02_3seed_mean.csv"),
}


def load_pool(runs, split):
    """split is 'train_preds' or 'val_preds'"""
    files = []
    for r in runs:
        p = Path(f"{RUNS}/{r}/{split}.npz")
        if p.exists():
            files.append(np.load(p, allow_pickle=True))
    if not files:
        return None, None
    preds = np.mean([f["preds"] for f in files], axis=0)
    labels = files[0]["raw_labels"]
    return preds, labels


def main():
    # Load all pools for both splits
    pool_train = {}; pool_val = {}
    train_labels = None; val_labels = None
    for name, (runs, _test) in POOLS.items():
        tr_p, tr_l = load_pool(runs, "train_preds")
        v_p, v_l = load_pool(runs, "val_preds")
        if tr_p is None or v_p is None:
            print(f"  {name}: skip (missing train or val preds)"); continue
        if train_labels is None:
            train_labels, val_labels = tr_l, v_l
        pool_train[name] = tr_p
        pool_val[name] = v_p
        print(f"  {name}: train={tr_p.shape}, val={v_p.shape}")

    pools = list(pool_train.keys())
    K = len(pools)
    print(f"\nstacking {K} pools: {pools}")

    test_csvs = {p: pd.read_csv(POOLS[p][1]).sort_values("Id").reset_index(drop=True) for p in pools}
    ids = test_csvs[pools[0]]["Id"].to_numpy()
    ens = pd.DataFrame({"Id": ids})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_b, total_v, n_lab = 0, 0, 0
    for li, lab in enumerate(LABELS):
        # Build per-label features
        X_tr = np.stack([pool_train[p][:, li] for p in pools], axis=1)
        y_tr = train_labels[:, li]
        X_v = np.stack([pool_val[p][:, li] for p in pools], axis=1)
        y_v = val_labels[:, li]

        # Mask out NaN labels (uncertain in val; in train, U-Ones means uncertain→1)
        m_tr = ~np.isnan(y_tr)
        m_v = ~np.isnan(y_v)
        X_tr_m, y_tr_m = X_tr[m_tr], y_tr[m_tr]
        X_v_m, y_v_m = X_v[m_v], y_v[m_v]

        if len(y_v_m) < 100:
            print(f"  {lab[:30]:<30}: too few val samples")
            ens[lab] = np.mean([test_csvs[p][lab].to_numpy() for p in pools], axis=0)
            continue

        # Baseline: equal-weight average (val NMSE)
        baseline_v = float(np.mean((y_v_m - X_v_m.mean(axis=1)) ** 2)) / float(np.var(y_v_m))

        # Train MLP
        X_tr_t = torch.from_numpy(X_tr_m).float().to(device)
        y_tr_t = torch.from_numpy(y_tr_m).float().to(device)
        X_v_t = torch.from_numpy(X_v_m).float().to(device)
        y_v_t = torch.from_numpy(y_v_m).float().to(device)

        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(K, 16), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(16, 1),
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        var_yv = torch.var(y_v_t).item()
        best_v_nmse = float("inf")
        best_state = None
        patience = 50
        no_improve = 0
        for step in range(500):
            model.train()
            # full-batch update (cheap, ~150k samples)
            pred = model(X_tr_t).squeeze(-1)
            loss = ((y_tr_t - pred) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                model.eval()
                pv = model(X_v_t).squeeze(-1)
                v_mse = ((y_v_t - pv) ** 2).mean().item()
                v_nmse = v_mse / var_yv
                if v_nmse < best_v_nmse:
                    best_v_nmse = v_nmse
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

        # Apply best MLP to test predictions
        if best_state is not None:
            model.load_state_dict(best_state)
        X_test = np.stack([test_csvs[p][lab].to_numpy() for p in pools], axis=1)
        X_test_t = torch.from_numpy(X_test).float().to(device)
        with torch.no_grad():
            model.eval()
            pred_test = model(X_test_t).squeeze(-1).cpu().numpy()
        ens[lab] = pred_test
        print(f"  {lab[:25]:<25}  baseline={baseline_v:.4f}  mlp_val={best_v_nmse:.4f}  Δ={baseline_v-best_v_nmse:+.4f}")
        total_b += baseline_v; total_v += best_v_nmse; n_lab += 1

    print(f"\nmean baseline NMSE: {total_b/n_lab:.4f}")
    print(f"mean MLP val NMSE: {total_v/n_lab:.4f}")
    print(f"mean gain: {(total_b-total_v)/n_lab:+.4f}")

    # Cherry-picks for Card/Fracture
    ii = pd.read_csv(f"{SUB11}/ladder/I_D_with_cnxl_boost40.csv").sort_values("Id").reset_index(drop=True)
    cn = test_csvs.get("cn")
    cxr5 = pd.read_csv(f"{SUB11}/cxr_foundation_5seed_pool.csv").sort_values("Id").reset_index(drop=True)
    ens["Cardiomegaly"] = ii["Cardiomegaly"].to_numpy()
    if cn is not None:
        ens["Fracture"] = 0.30 * cn["Fracture"].to_numpy() + 0.70 * cxr5["Fracture"].to_numpy()

    out = "submissions/2026-05-16/ladder/STK_mlp_trainfit.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ens.to_csv(out, index=False, float_format="%.6f")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
