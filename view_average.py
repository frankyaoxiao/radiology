"""Apply (pid, study)-level view averaging to a submission CSV.

For each test image, group by (patient_id, study_number) and replace its label
predictions with the weighted mean across views in the same study. Frontal
views are weighted higher than lateral by default (3:1) — the validated
recipe that gave RAD-DINO ensemble ~+0.004 on the leaderboard.

Usage:
    # global 3:1 frontal:lateral
    uv run python view_average.py --in foo.csv --out foo_va.csv

    # custom global weights
    uv run python view_average.py --in foo.csv --out foo_va.csv \\
        --frontal-weight 3 --lateral-weight 1

    # per-label weights from JSON (overrides --frontal-weight / --lateral-weight)
    uv run python view_average.py --in foo.csv --out foo_va.csv \\
        --weights-json view_weights_omnirad448.json

JSON format: {label_name: [w_F, w_L], ...}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import LABEL_NAMES


def _resolve_weights(args, labels):
    """Return per-label arrays (w_F, w_L) of length len(labels)."""
    if args.weights_json is not None:
        with open(args.weights_json) as f:
            data = json.load(f)
        w_F = np.array([float(data[lab][0]) for lab in labels])
        w_L = np.array([float(data[lab][1]) for lab in labels])
    else:
        w_F = np.full(len(labels), float(args.frontal_weight))
        w_L = np.full(len(labels), float(args.lateral_weight))
    return w_F, w_L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--test-ids-csv",
        type=str,
        default="/data/artifacts/frank/misc/labels/test_ids.csv",
    )
    ap.add_argument("--frontal-weight", type=float, default=3.0)
    ap.add_argument("--lateral-weight", type=float, default=1.0)
    ap.add_argument("--weights-json", type=Path, default=None,
                    help="Optional path to per-label weights JSON {label_name: [w_F, w_L]}.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        raise FileExistsError(f"{args.out} exists; pass --force to overwrite.")

    labels = list(LABEL_NAMES)
    w_F_arr, w_L_arr = _resolve_weights(args, labels)

    df_pred = pd.read_csv(args.in_path)
    df_test = pd.read_csv(args.test_ids_csv)

    # Extract grouping fields
    df_test["pid"] = df_test["Path"].str.extract(r"(pid\d+)", expand=False)
    df_test["study"] = df_test["Path"].str.extract(r"(study\d+)", expand=False)
    df_test["is_frontal"] = df_test["Path"].str.contains("frontal", regex=False)
    merged = df_pred.merge(
        df_test[["Id", "pid", "study", "is_frontal"]], on="Id", how="left"
    )
    if merged["pid"].isna().any():
        raise RuntimeError("some Ids in the prediction CSV did not match any row in test_ids.csv")

    # Per-row weight matrix: (N, L) — w_F if frontal, w_L if lateral, per label
    is_frontal_mat = merged["is_frontal"].to_numpy(dtype=bool)[:, None]            # (N, 1)
    per_row_weights = np.where(is_frontal_mat, w_F_arr[None, :], w_L_arr[None, :])  # (N, L)
    pred_mat = merged[labels].to_numpy()                                            # (N, L)
    weighted_vals = pred_mat * per_row_weights                                      # (N, L)

    # Build a frame for per-label groupby
    merged_w = merged[["pid", "study"]].copy()
    for j, lab in enumerate(labels):
        merged_w[lab] = weighted_vals[:, j]
        merged_w[f"_w_{lab}"] = per_row_weights[:, j]

    grp = merged_w.groupby(["pid", "study"], sort=False)
    sums = grp[labels].sum()                                                       # (G, L)
    denoms = pd.DataFrame({lab: grp[f"_w_{lab}"].sum() for lab in labels})         # (G, L)
    # Avoid div-by-zero when both weights are 0 for some label (shouldn't happen unless JSON has [0,0])
    denoms_arr = denoms.to_numpy().copy()  # to_numpy() can return a read-only view
    denoms_arr[denoms_arr == 0] = 1.0
    group_means = pd.DataFrame(
        sums.to_numpy() / denoms_arr,
        index=sums.index, columns=sums.columns,
    )

    # Map each Id back to its group's per-label mean
    keys = list(zip(merged["pid"], merged["study"]))
    out = pd.DataFrame({"Id": merged["Id"].to_numpy()})
    for lab in labels:
        out[lab] = group_means.loc[keys, lab].to_numpy()
    out = out.sort_values("Id").reset_index(drop=True)
    out.to_csv(args.out, index=False, float_format="%.6f")

    print(f"wrote {args.out} ({len(out)} rows)", flush=True)
    print(f"\nPer-label weights and mean shifts (orig -> va):")
    for j, lab in enumerate(labels):
        a = float(df_pred[lab].mean())
        b = float(out[lab].mean())
        print(f"  {lab:30s}  weights=({w_F_arr[j]:.1f}, {w_L_arr[j]:.1f})  "
              f"orig={a:+.4f}  va={b:+.4f}  diff={b-a:+.6f}")

    # Multi-view stats
    sizes = grp.size().reset_index(name="n_views")
    multi_groups = (sizes["n_views"] > 1).sum()
    n_in_multi = int(sizes.loc[sizes["n_views"] > 1, "n_views"].sum())
    print(f"\nGroups: {len(sizes)}  multi-view: {multi_groups}  "
          f"images in multi-view groups: {n_in_multi}/{len(out)} "
          f"({100*n_in_multi/len(out):.1f}%)")


if __name__ == "__main__":
    main()
