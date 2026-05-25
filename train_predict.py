"""Run train-set inference for a single ckpt and save predictions + labels."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config, LABEL_NAMES
from dataset import CheXpertDataset, build_val_transform, load_and_split
from submit import load_model, _set_deterministic


def main():
    _set_deterministic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max-samples", type=int, default=0, help="if >0, subsample train")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, meta = load_model(args.ckpt, device)
    print(f"loaded ckpt step={meta['step']} best_nmse={meta['best_metric']:.4f}", flush=True)

    df_train, df_val, y_train, y_val = load_and_split(cfg)
    if args.max_samples > 0:
        df_train = df_train.head(args.max_samples).reset_index(drop=True)
        y_train = y_train[:args.max_samples]
    print(f"train rows: {len(df_train):,}", flush=True)

    train_ds = CheXpertDataset(df_train, y_train, cfg.data_root, build_val_transform(cfg))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    all_preds = []
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(device, non_blocking=True)
            if args.bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x)
            else:
                logits = model(x)
            logits = logits.float()
            if cfg.target_type == "3class":
                p = logits.softmax(dim=-1)
                pred = (p[..., 2] - p[..., 0]).cpu().numpy()
            else:
                pred = torch.sigmoid(logits).cpu().numpy() * 2 - 1
            all_preds.append(pred)

    preds = np.concatenate(all_preds, axis=0).astype(np.float32)
    paths = df_train["Path"].to_numpy()

    if cfg.target_type == "3class":
        raw_labels = np.full_like(preds, np.nan, dtype=np.float32)
        raw_labels[y_train == 0] = -1.0
        raw_labels[y_train == 1] = 0.0
        raw_labels[y_train == 2] = 1.0
    else:
        raw_labels = y_train.astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, preds=preds, paths=paths, raw_labels=raw_labels)
    print(f"wrote {args.out}  shape={preds.shape}", flush=True)


if __name__ == "__main__":
    main()
