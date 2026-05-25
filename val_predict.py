"""Run val inference for a single ckpt and save per-image predictions + paths.

Output: npz with arrays
    preds   (N_val, 9)  P(+1) - P(-1) per label
    paths   (N_val,)    image paths (for grouping by pid/study)
    labels  (N_val, 9)  raw -1/0/1/NaN labels (NaN = blank, masked from NMSE)

Usage:
    uv run python val_predict.py \\
        --ckpt /data/.../ckpts/ckpt_best.pt \\
        --out  /data/.../runs/<run_name>/val_preds.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config, LABEL_NAMES
from dataset import CheXpertDataset, build_val_transform, load_and_split, _drop_junk_cols, _extract_pid
from submit import load_model, _set_deterministic


def main():
    _set_deterministic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, meta = load_model(args.ckpt, device)
    print(f"loaded ckpt step={meta['step']} best_nmse={meta['best_metric']:.4f}", flush=True)

    # Reproduce the same val split used at training time
    df_train, df_val, y_train, y_val = load_and_split(cfg)
    print(f"val rows: {len(df_val):,}", flush=True)

    # Build a clean val dataset (no augmentation, just the val transform)
    val_ds = CheXpertDataset(
        df_val, y_val, cfg.data_root, build_val_transform(cfg),
        clahe=cfg.clahe, clahe_clip_limit=cfg.clahe_clip_limit, clahe_tile_size=cfg.clahe_tile_size,
    )
    loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Run inference
    all_logits = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            if args.bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x)
            else:
                logits = model(x)
            all_logits.append(logits.float().cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)  # (N, 9, 3) for 3-class

    # Collapse 3-class to per-label expected value: P(+1) - P(-1)
    if cfg.target_type == "3class":
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = e / e.sum(axis=-1, keepdims=True)
        preds = probs[:, :, 2] - probs[:, :, 0]  # (N, 9)
        preds = np.clip(preds, -1, 1)
    elif cfg.target_type == "raw":
        preds = np.clip(logits, -1, 1)
    else:
        preds = 1.0 / (1.0 + np.exp(-logits))

    # Pull raw labels from the original CSV so we have NaN for blanks (the leaderboard scores against these)
    df_full = pd.read_csv(cfg.labels_csv)
    df_full = _drop_junk_cols(df_full)
    df_full["pid"] = _extract_pid(df_full["Path"])
    df_full = df_full[df_full["pid"].notna()].reset_index(drop=True)
    # Match rows by Path
    val_paths = df_val["Path"].to_numpy()
    val_path_to_row = {p: i for i, p in enumerate(df_full["Path"].to_numpy())}
    raw_labels = np.full((len(val_paths), len(LABEL_NAMES)), np.nan, dtype=np.float32)
    for i, p in enumerate(val_paths):
        if p in val_path_to_row:
            row = df_full.iloc[val_path_to_row[p]]
            for j, lab in enumerate(LABEL_NAMES):
                v = row[lab]
                raw_labels[i, j] = np.nan if pd.isna(v) else float(v)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, preds=preds, paths=val_paths, raw_labels=raw_labels)
    print(f"wrote {args.out}: preds {preds.shape}, raw_labels {raw_labels.shape}", flush=True)


if __name__ == "__main__":
    main()
