"""Average predictions from 5 multi-split 3-class DenseNet models.

Produces two submission CSVs:
  A) Pure identity average: just average the 5 caches' predictions.
  B) Bias-only shift: fit a single bias per label on pooled val data.

Usage:
    python -u multisplit_avg.py \
        --runs-root /resnick/groups/CS156b/from_central/2026/scalm_akumarap/runs \
        --labels-csv /resnick/groups/CS156b/from_central/data/student_labels/train2023.csv \
        --test-ids-csv /resnick/groups/CS156b/from_central/data/student_labels/test_ids.csv \
        --data-root /resnick/groups/CS156b/from_central/data \
        --out-identity submission_multisplit_identity.csv \
        --out-bias submission_multisplit_bias.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from config import LABEL_NAMES


# Cache directory names and their corresponding split seeds.
CACHE_CONFIGS = [
    ("3class_448_calib_cache", 42),
    ("3class_448_split7_cache", 7),
    ("3class_448_split13_cache", 13),
    ("3class_448_split29_cache", 29),
    ("3class_448_split101_cache", 101),
]


def _drop_junk_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnamed/empty index columns from train2023.csv."""
    drop = [c for c in df.columns if c == "" or str(c).startswith("Unnamed")]
    return df.drop(columns=drop, errors="ignore")


def _extract_pid(path_series: pd.Series) -> pd.Series:
    return path_series.str.extract(r"(pid\d+)", expand=False)


def load_labels_df(labels_csv: str) -> pd.DataFrame:
    """Load train2023.csv, clean it, and add patient IDs."""
    df = pd.read_csv(labels_csv)
    df = _drop_junk_cols(df)
    df["pid"] = _extract_pid(df["Path"])
    df = df[df["pid"].notna()].reset_index(drop=True)
    return df


def get_val_pids(all_pids: np.ndarray, seed: int, val_frac: float = 0.1) -> set:
    """Return set of val patient IDs for a given split seed."""
    rng = np.random.default_rng(seed)
    pids = all_pids.copy()
    rng.shuffle(pids)
    n_val = int(round(len(pids) * val_frac))
    return set(pids[:n_val].tolist())


def load_cache(cache_dir: Path, kind: str) -> dict:
    """Load an .npz cache file and return a dict with 'preds' and optionally 'ids'.

    Handles both 'logits' and 'probs' keys. For 3-class models these are
    already P(+1)-P(-1) expected values in [-1, 1].
    Val caches don't have 'ids' — only test caches do.
    """
    npz_path = cache_dir / f"{kind}_ensemble_cache.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Cache not found: {npz_path}")
    data = np.load(str(npz_path))

    if "logits" in data:
        preds = data["logits"].astype(np.float64)
    elif "probs" in data:
        preds = data["probs"].astype(np.float64)
    else:
        raise KeyError(f"Neither 'logits' nor 'probs' found in {npz_path}")

    ids = data["ids"] if "ids" in data else None
    return {"preds": preds, "ids": ids}


def fit_bias(val_pred: np.ndarray, val_true: np.ndarray) -> float:
    """Fit a single bias b minimizing MSE of clip(pred + b, -1, 1) vs true.

    val_pred: (N,) predictions
    val_true: (N,) true labels (may contain NaN)
    Returns optimal bias b.
    """
    mask = ~np.isnan(val_true)
    if mask.sum() < 5:
        return 0.0
    xm = val_pred[mask]
    ym = val_true[mask]

    def objective(b):
        return np.mean((ym - np.clip(xm + b, -1, 1)) ** 2)

    result = minimize_scalar(objective, bounds=(-0.5, 0.5), method="bounded",
                             options={"xatol": 1e-7, "maxiter": 1000})
    return float(result.x)


def write_submission(path: Path, ids: np.ndarray, preds: np.ndarray,
                     label_names: List[str]) -> None:
    """Write a submission CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id"] + label_names)
        for sample_id, row in zip(ids, preds):
            writer.writerow([int(sample_id)] + [f"{v:.6f}" for v in row])
    print(f"Wrote {path} ({len(ids)} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Average multi-split 3-class predictions into submission CSVs."
    )
    ap.add_argument("--runs-root", required=True, type=Path,
                    help="Root directory containing cache subdirectories")
    ap.add_argument("--labels-csv", required=True, type=str,
                    help="Path to train2023.csv")
    ap.add_argument("--test-ids-csv", required=True, type=str,
                    help="Path to test_ids.csv")
    ap.add_argument("--data-root", required=True, type=str,
                    help="Data root (unused but kept for CLI consistency)")
    ap.add_argument("--out-identity", required=True, type=Path,
                    help="Output CSV for pure identity average")
    ap.add_argument("--out-bias", required=True, type=Path,
                    help="Output CSV for bias-corrected average")
    args = ap.parse_args()

    label_names = list(LABEL_NAMES)
    n_labels = len(label_names)

    # ------------------------------------------------------------------
    # 1. Load test predictions from all 5 caches
    # ------------------------------------------------------------------
    print("=" * 80, flush=True)
    print("LOADING TEST PREDICTIONS", flush=True)
    print("=" * 80, flush=True)

    test_preds_list = []
    test_ids = None

    for cache_name, seed in CACHE_CONFIGS:
        cache_dir = args.runs_root / cache_name
        print(f"  Loading test from {cache_dir} (seed={seed})...", flush=True)
        cache = load_cache(cache_dir, "test")
        test_preds_list.append(cache["preds"])

        if test_ids is None:
            test_ids = cache["ids"]
        else:
            # Verify IDs match across all caches
            if not np.array_equal(test_ids, cache["ids"]):
                raise ValueError(
                    f"Test IDs mismatch between first cache and {cache_name}. "
                    f"First has {len(test_ids)} ids, this has {len(cache['ids'])} ids."
                )
        print(f"    shape={cache['preds'].shape}, "
              f"mean={np.mean(cache['preds']):.4f}, "
              f"range=[{np.min(cache['preds']):.4f}, {np.max(cache['preds']):.4f}]",
              flush=True)

    # Stack and average
    test_stack = np.stack(test_preds_list, axis=0)  # (5, N, 9)
    test_avg = np.mean(test_stack, axis=0)           # (N, 9)
    test_avg_clipped = np.clip(test_avg, -1, 1)

    print(f"\nAveraged {len(test_preds_list)} caches -> shape {test_avg.shape}", flush=True)

    # ------------------------------------------------------------------
    # CSV A: Pure identity average
    # ------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("CSV A: PURE IDENTITY AVERAGE (no calibration)", flush=True)
    print("=" * 80, flush=True)

    write_submission(args.out_identity, test_ids, test_avg_clipped, label_names)

    # ------------------------------------------------------------------
    # 2. Load val predictions and true labels for bias fitting
    # ------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("LOADING VAL PREDICTIONS FOR BIAS FITTING", flush=True)
    print("=" * 80, flush=True)

    # Load the full labels dataframe
    df_labels = load_labels_df(args.labels_csv)
    all_pids = df_labels["pid"].drop_duplicates().to_numpy()
    print(f"Total patients: {len(all_pids)}", flush=True)

    # For each split, load val predictions and match with true labels
    all_val_preds = []   # list of (n_val_samples, 9) arrays
    all_val_trues = []   # list of (n_val_samples, 9) arrays

    for cache_name, seed in CACHE_CONFIGS:
        cache_dir = args.runs_root / cache_name
        print(f"\n  Loading val from {cache_dir} (seed={seed})...", flush=True)

        cache = load_cache(cache_dir, "val")
        val_preds = cache["preds"]   # (n_val, 9)

        # Determine which patients are in this split's val set
        val_pids = get_val_pids(all_pids, seed)
        print(f"    Val patients for seed {seed}: {len(val_pids)}", flush=True)

        # Get val rows from labels df
        df_val = df_labels[df_labels["pid"].isin(val_pids)].reset_index(drop=True)

        # The val predictions should match the val split size
        if len(val_preds) != len(df_val):
            print(f"    WARNING: val cache has {len(val_preds)} samples but "
                  f"split has {len(df_val)} samples. Using min.", flush=True)
            n_use = min(len(val_preds), len(df_val))
            val_preds = val_preds[:n_use]
            df_val = df_val.iloc[:n_use]

        # Extract raw val labels (1.0=pos, 0.0=uncertain, -1.0=neg, NaN=blank)
        val_true = df_val[label_names].to_numpy(dtype=np.float64)

        # For raw labels in the -1/0/1 scale, keep as-is (including 0.0 uncertain)
        # NaN stays NaN and will be masked during bias fitting.

        print(f"    Val preds shape: {val_preds.shape}, "
              f"Val true shape: {val_true.shape}", flush=True)

        n_valid = np.sum(~np.isnan(val_true), axis=0)
        print(f"    Valid (non-NaN) counts per label: {n_valid}", flush=True)

        all_val_preds.append(val_preds)
        all_val_trues.append(val_true)

    # Pool all val predictions and true labels
    pooled_preds = np.concatenate(all_val_preds, axis=0)  # (total_val, 9)
    pooled_trues = np.concatenate(all_val_trues, axis=0)  # (total_val, 9)
    print(f"\nPooled val samples: {len(pooled_preds)}", flush=True)

    # ------------------------------------------------------------------
    # 3. Fit per-label bias
    # ------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("FITTING PER-LABEL BIAS (y = x + b)", flush=True)
    print("=" * 80, flush=True)

    biases = np.zeros(n_labels)
    for i, name in enumerate(label_names):
        b = fit_bias(pooled_preds[:, i], pooled_trues[:, i])
        biases[i] = b

        mask = ~np.isnan(pooled_trues[:, i])
        n_valid = mask.sum()
        pred_mean = np.mean(pooled_preds[:, i][mask]) if n_valid > 0 else float("nan")
        true_mean = np.mean(pooled_trues[:, i][mask]) if n_valid > 0 else float("nan")
        mse_before = np.mean((pooled_trues[:, i][mask] -
                              np.clip(pooled_preds[:, i][mask], -1, 1)) ** 2) if n_valid > 0 else float("nan")
        mse_after = np.mean((pooled_trues[:, i][mask] -
                             np.clip(pooled_preds[:, i][mask] + b, -1, 1)) ** 2) if n_valid > 0 else float("nan")

        print(f"  {name:30s}  bias={b:+.6f}  n_valid={n_valid:5d}  "
              f"pred_mean={pred_mean:+.4f}  true_mean={true_mean:+.4f}  "
              f"MSE: {mse_before:.6f} -> {mse_after:.6f}", flush=True)

    # ------------------------------------------------------------------
    # CSV B: Bias-corrected average
    # ------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("CSV B: BIAS-CORRECTED AVERAGE", flush=True)
    print("=" * 80, flush=True)

    test_bias = np.clip(test_avg + biases[np.newaxis, :], -1, 1)
    write_submission(args.out_bias, test_ids, test_bias, label_names)

    # ------------------------------------------------------------------
    # 4. Sanity checks
    # ------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("SANITY CHECK: PER-LABEL STATISTICS", flush=True)
    print("=" * 80, flush=True)

    header = (f"{'Label':30s}  {'Identity Mean':>14s}  {'Bias Mean':>12s}  "
              f"{'Bias':>8s}  {'Id Min':>8s}  {'Id Max':>8s}  "
              f"{'Bi Min':>8s}  {'Bi Max':>8s}")
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for i, name in enumerate(label_names):
        id_col = test_avg_clipped[:, i]
        bi_col = test_bias[:, i]
        print(f"  {name:28s}  {np.mean(id_col):+14.6f}  {np.mean(bi_col):+12.6f}  "
              f"{biases[i]:+8.5f}  {np.min(id_col):+8.4f}  {np.max(id_col):+8.4f}  "
              f"{np.min(bi_col):+8.4f}  {np.max(bi_col):+8.4f}", flush=True)

    # Per-cache agreement
    print("\n" + "=" * 80, flush=True)
    print("PER-CACHE MEAN PREDICTIONS (test set)", flush=True)
    print("=" * 80, flush=True)

    header2 = f"{'Cache':35s}  " + "  ".join(f"{n[:8]:>10s}" for n in label_names)
    print(header2, flush=True)
    print("-" * len(header2), flush=True)

    for j, (cache_name, seed) in enumerate(CACHE_CONFIGS):
        means = [np.mean(test_preds_list[j][:, i]) for i in range(n_labels)]
        line = f"  {cache_name:33s}  " + "  ".join(f"{m:+10.4f}" for m in means)
        print(line, flush=True)

    means_avg = [np.mean(test_avg_clipped[:, i]) for i in range(n_labels)]
    line = f"  {'AVERAGE':33s}  " + "  ".join(f"{m:+10.4f}" for m in means_avg)
    print(line, flush=True)

    # Spread across caches
    print("\n" + "=" * 80, flush=True)
    print("INTER-CACHE SPREAD (std of per-sample predictions across 5 caches)", flush=True)
    print("=" * 80, flush=True)

    per_sample_std = np.std(test_stack, axis=0)  # (N, 9)
    for i, name in enumerate(label_names):
        std_col = per_sample_std[:, i]
        print(f"  {name:28s}  mean_std={np.mean(std_col):.4f}  "
              f"max_std={np.max(std_col):.4f}  "
              f"median_std={np.median(std_col):.4f}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
