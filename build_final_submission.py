"""Build final submission from RAD-DINO + MAIRA-2 ensembles.

Averages:
1. RAD-DINO 3-seed ensemble (split 42)
2. RAD-DINO multi-split (splits 7, 13, 29, 101)
3. MAIRA-2 3-seed ensemble

Optionally applies patient-level view averaging (group by patient, average predictions).

Usage:
    python -u build_final_submission.py \
        --runs-root /resnick/groups/CS156b/from_central/2026/scalm_akumarap/runs \
        --test-ids-csv /resnick/groups/CS156b/from_central/data/student_labels/test_ids.csv \
        --out submission_final.csv \
        --out-view-avg submission_final_viewavg.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import LABEL_NAMES


def load_test_cache(cache_dir: Path) -> dict:
    npz = np.load(str(cache_dir / "test_ensemble_cache.npz"))
    preds = npz["logits"] if "logits" in npz else npz["probs"]
    return {"preds": preds.astype(np.float64), "ids": npz["ids"]}


def patient_view_average(df_submission: pd.DataFrame, test_ids_csv: str) -> pd.DataFrame:
    """Average predictions across views of the same patient."""
    df_test = pd.read_csv(test_ids_csv)
    df_test["pid"] = df_test["Path"].str.extract(r"(pid\d+)")

    merged = df_submission.merge(df_test[["Id", "pid"]], on="Id")
    labels = list(LABEL_NAMES)

    patient_means = merged.groupby("pid")[labels].transform("mean")
    result = df_submission.copy()
    result[labels] = patient_means
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True, type=Path)
    ap.add_argument("--test-ids-csv", required=True, type=str)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--out-view-avg", type=Path, default=None)
    args = ap.parse_args()

    labels = list(LABEL_NAMES)

    # Load all caches
    caches = {}
    cache_dirs = {
        "rd_3seed": args.runs_root / "rad_dino_cache",
        "rd_split7": args.runs_root / "rd_split7_cache",
        "rd_split13": args.runs_root / "rd_split13_cache",
        "rd_split29": args.runs_root / "rd_split29_cache",
        "rd_split101": args.runs_root / "rd_split101_cache",
        "maira2": args.runs_root / "maira2_cache",
    }

    test_ids = None
    for name, cache_dir in cache_dirs.items():
        if not (cache_dir / "test_ensemble_cache.npz").exists():
            print(f"  SKIP {name}: {cache_dir} not ready", flush=True)
            continue
        data = load_test_cache(cache_dir)
        caches[name] = data["preds"]
        if test_ids is None:
            test_ids = data["ids"]
        print(f"  Loaded {name}: shape={data['preds'].shape} mean={np.mean(data['preds']):.4f}", flush=True)

    if not caches:
        raise RuntimeError("No caches loaded")

    # Average all available caches
    all_preds = np.stack(list(caches.values()), axis=0)
    avg_preds = np.clip(np.mean(all_preds, axis=0), -1, 1)

    print(f"\nAveraged {len(caches)} caches: {list(caches.keys())}", flush=True)
    print(f"Shape: {avg_preds.shape}", flush=True)

    # Write main submission
    df_out = pd.DataFrame({"Id": test_ids.astype(int)})
    for i, l in enumerate(labels):
        df_out[l] = avg_preds[:, i]
    df_out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(df_out)} rows)", flush=True)

    # Per-label means
    print(f"\n{'Label':30s}  {'Mean':>10s}  {'Std':>10s}")
    for i, l in enumerate(labels):
        print(f"  {l:28s}  {np.mean(avg_preds[:, i]):+10.4f}  {np.std(avg_preds[:, i]):10.4f}", flush=True)

    # View averaging
    if args.out_view_avg:
        print(f"\nApplying patient-level view averaging...", flush=True)
        df_va = patient_view_average(df_out, args.test_ids_csv)
        df_va.to_csv(args.out_view_avg, index=False)
        print(f"Wrote {args.out_view_avg} ({len(df_va)} rows)", flush=True)

        print(f"\nView-avg per-label means:")
        for l in labels:
            orig = df_out[l].mean()
            va = df_va[l].mean()
            print(f"  {l:28s}  orig={orig:+.4f}  viewavg={va:+.4f}  diff={va-orig:+.6f}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
