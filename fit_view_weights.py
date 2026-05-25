"""Fit per-label (w_F, w_L) view averaging weights on a val_preds npz.

Input:  npz from val_predict.py with arrays preds, paths, raw_labels.
Output: JSON with per-label [w_F, w_L] mapping.

Method:
  - Group val by (pid, study). Restrict to multi-view groups with both frontal & lateral.
  - For each of 9 labels, grid-search (w_F, w_L) ∈ search_space minimizing per-label NMSE
    on the multi-view subset (NaN labels masked, just like the leaderboard).

Usage:
    uv run python fit_view_weights.py \\
        --val-npz /data/.../runs/<run>/val_preds.npz \\
        --out view_weights_omnirad448.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import LABEL_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-npz", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--regularize-toward-default", type=float, default=0.0,
                    help="L2 toward (3, 1) per label. 0 = unregularized; try 0.1 for shrinkage.")
    args = ap.parse_args()

    z = np.load(str(args.val_npz), allow_pickle=True)
    preds   = z["preds"]                   # (N, 9), in [-1, 1]
    paths   = z["paths"].astype(str)       # (N,)
    labels  = z["raw_labels"]              # (N, 9), NaN for blanks

    # Parse pid/study/view from path
    df = pd.DataFrame({"path": paths})
    df["pid"]   = df["path"].str.extract(r"(pid\d+)", expand=False)
    df["study"] = df["path"].str.extract(r"(study\d+)", expand=False)
    df["is_frontal"] = df["path"].str.contains("frontal", regex=False)

    # For each (pid, study) group, find frontal index and lateral index (if both exist)
    groups = df.groupby(["pid", "study"])
    pairs = []
    for (pid, study), grp in groups:
        f_idx = grp.index[grp["is_frontal"]].tolist()
        l_idx = grp.index[~grp["is_frontal"]].tolist()
        if len(f_idx) >= 1 and len(l_idx) >= 1:
            # Pick first of each (rare to have multiple of same view)
            pairs.append((f_idx[0], l_idx[0]))
    print(f"val (pid, study) groups: {len(groups)}; multi-view (both views) groups: {len(pairs)}")

    if len(pairs) < 50:
        raise RuntimeError("Too few multi-view val groups to fit reliably.")

    f_idx = np.array([p[0] for p in pairs])
    l_idx = np.array([p[1] for p in pairs])

    pred_F = preds[f_idx]   # (M, 9)
    pred_L = preds[l_idx]   # (M, 9)
    # Use the frontal row's label as the study label (label is per-study, same across views)
    y      = labels[f_idx]  # (M, 9), NaN where blank

    # Grid search per label
    grid = [0, 1, 2, 3, 5]
    weights = {}
    for j, lab in enumerate(LABEL_NAMES):
        y_j   = y[:, j]
        mask  = ~np.isnan(y_j)
        if mask.sum() < 10:
            print(f"  {lab:30s}  insufficient val samples ({mask.sum()}) — defaulting to (3, 1)")
            weights[lab] = [3.0, 1.0]
            continue
        var = float(np.var(y_j[mask]))
        if var <= 0:
            weights[lab] = [3.0, 1.0]
            continue

        best = None
        best_loss = float("inf")
        # baseline: global (3, 1) for reference
        baseline_blend = (3.0 * pred_F[:, j] + 1.0 * pred_L[:, j]) / 4.0
        baseline_blend = np.clip(baseline_blend, -1, 1)
        baseline_nmse = np.mean((y_j[mask] - baseline_blend[mask]) ** 2) / var

        for w_F in grid:
            for w_L in grid:
                if w_F == 0 and w_L == 0:
                    continue
                blend = (w_F * pred_F[:, j] + w_L * pred_L[:, j]) / (w_F + w_L)
                blend = np.clip(blend, -1, 1)
                err = np.mean((y_j[mask] - blend[mask]) ** 2) / var
                if args.regularize_toward_default > 0:
                    reg = args.regularize_toward_default * ((w_F - 3.0) ** 2 + (w_L - 1.0) ** 2)
                    err = err + reg * 0.0001  # tiny regularizer scaled to NMSE units
                if err < best_loss:
                    best_loss = err
                    best = (w_F, w_L)

        weights[lab] = [float(best[0]), float(best[1])]
        improvement = baseline_nmse - best_loss
        marker = "  ✓" if improvement > 0.001 else ("  ~" if improvement > 0 else "  -")
        print(f"  {lab:30s}  best=({best[0]}, {best[1]})  val_nmse={best_loss:.4f}  vs (3,1)={baseline_nmse:.4f}  Δ={improvement:+.4f}{marker}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
