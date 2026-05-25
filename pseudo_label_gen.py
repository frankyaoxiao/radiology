"""Generate pseudo-labels for test images from an existing ensemble CSV.

Uses high-confidence thresholding: |pred| > tau → sign(pred), else NaN (masked from loss).

Output schema matches train2023.csv (Path + 9 label columns), letting the regular
dataset pipeline absorb it via cfg.pseudo_label_csv.

Usage:
    uv run python pseudo_label_gen.py \\
        --ensemble-csv submissions/2026-05-10/ensemble_3family_w442_va.csv \\
        --test-ids-csv /data/artifacts/frank/misc/labels/test_ids.csv \\
        --threshold 0.5 \\
        --out /data/artifacts/frank/misc/labels/pseudo_labels_t05.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import LABEL_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-csv", required=True, type=Path,
                    help="CSV with Id + 9 label columns of [-1, 1] predictions")
    ap.add_argument("--test-ids-csv", required=True, type=Path,
                    help="test_ids.csv with Id, Path columns")
    ap.add_argument("--threshold", required=True, type=float,
                    help="confidence threshold |pred| > τ → use sign as label, else mask (NaN)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    LABELS = list(LABEL_NAMES)
    pred_df = pd.read_csv(args.ensemble_csv).sort_values("Id").reset_index(drop=True)
    test_df = pd.read_csv(args.test_ids_csv).sort_values("Id").reset_index(drop=True)
    if not (pred_df["Id"].to_numpy() == test_df["Id"].to_numpy()).all():
        raise RuntimeError("ensemble CSV Ids don't match test_ids.csv")

    out = pd.DataFrame({"Path": test_df["Path"].to_numpy()})

    n_kept = np.zeros(len(LABELS), dtype=int)
    n_pos = np.zeros(len(LABELS), dtype=int)
    n_neg = np.zeros(len(LABELS), dtype=int)
    n_unc = np.zeros(len(LABELS), dtype=int)
    for j, lab in enumerate(LABELS):
        pred = pred_df[lab].to_numpy()
        out_col = np.full(len(pred), np.nan, dtype=np.float32)
        # |pred| > τ → ±1, but mid-range (−0.2..0.2 for τ=0.5? no — we want |pred| ≤ τ → uncertain class 0)
        # Simpler scheme: hard 3-class assignment with two thresholds {±τ}
        out_col[pred >  args.threshold] =  1.0
        out_col[pred < -args.threshold] = -1.0
        # Optional middle band → "uncertain" (raw 0). Use a tighter band so we only
        # tag clear-uncertain cases; the rest stay NaN (blank, masked from loss).
        unc_band = max(0.05, args.threshold * 0.2)
        mid = (pred >= -unc_band) & (pred <= unc_band)
        out_col[mid] = 0.0
        out[lab] = out_col
        n_kept[j] = (~np.isnan(out_col)).sum()
        n_pos[j] = (out_col == 1.0).sum()
        n_neg[j] = (out_col == -1.0).sum()
        n_unc[j] = (out_col == 0.0).sum()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} rows)\n")
    print(f"{'Label':30s}  {'kept':>7s}  {'+1':>7s}  {'-1':>7s}  {'unc(0)':>8s}  {'kept%':>7s}")
    for j, lab in enumerate(LABELS):
        kept_pct = 100.0 * n_kept[j] / len(out)
        print(f"  {lab:28s}  {n_kept[j]:>7d}  {n_pos[j]:>7d}  {n_neg[j]:>7d}  {n_unc[j]:>8d}  {kept_pct:>6.2f}%")


if __name__ == "__main__":
    main()
