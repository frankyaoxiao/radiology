"""Train a multi-view fusion model: shared backbone over (frontal, lateral),
concat features → MLP head.

Adapts train.py:main for multi-view I/O. Single GPU.
"""
from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config, LABEL_NAMES
from dataset import load_and_split, build_train_transform, build_val_transform
from metrics import per_label_nmse, per_label_auroc
from model import CheXpertModel
from train import (
    masked_3class_ce_loss, masked_3class_aux_mse, masked_bce_with_logits,
    masked_mse_loss, masked_smooth_l1_loss, build_scheduler, _json_safe,
)

from multiview import MultiViewCheXpertDataset, MultiViewModel


def make_inner_backbone(cfg: Config) -> Tuple[nn.Module, int]:
    """Build an inner model only for its `.backbone`. Return (inner, hidden_dim)."""
    inner = CheXpertModel(cfg, pretrained=True)
    if hasattr(inner, "hidden_dim"):
        D = int(inner.hidden_dim)
    elif hasattr(inner.backbone, "num_features"):
        D = int(inner.backbone.num_features)
    elif hasattr(inner.backbone, "embed_dim"):
        D = int(inner.backbone.embed_dim)
    else:
        # Probe one forward pass
        with torch.no_grad():
            x = torch.zeros(1, 3, cfg.image_size, cfg.image_size)
            tokens = inner.backbone.forward_features(x)
            D = tokens.shape[-1]
    return inner, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    run_dir = cfg.run_dir
    ckpt_dir = run_dir / "ckpts"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_yaml(run_dir / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==============================================")
    print(f"multi-view training")
    print(f"config: {args.config}")
    print(f"run_dir: {run_dir}")
    print(f"device: {device}")
    print(f"model_type (backbone): {cfg.model_type}")
    print(f"image_size: {cfg.image_size}, batch_size_per_gpu: {cfg.batch_size_per_gpu}")
    print(f"==============================================", flush=True)

    # ---- data ----
    print(f"[multiview] loading dataset …", flush=True)
    df_train, df_val, y_train, y_val = load_and_split(cfg)
    print(f"[multiview] train={len(df_train):,}  val={len(df_val):,}", flush=True)

    paired_only = getattr(cfg, "mv_paired_only", False)
    train_ds = MultiViewCheXpertDataset(
        df_train, y_train, cfg.data_root, build_train_transform(cfg),
        paired_only=paired_only,
    )
    # Val always uses all frontals (so val NMSE is comparable across runs)
    val_ds = MultiViewCheXpertDataset(
        df_val, y_val, cfg.data_root, build_val_transform(cfg),
        paired_only=False,
    )
    print(f"[multiview] train frontals={len(train_ds):,}  val frontals={len(val_ds):,}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size_per_gpu, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size_per_gpu, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    # ---- model ----
    print(f"[multiview] building model …", flush=True)
    inner, hidden_dim = make_inner_backbone(cfg)
    print(f"[multiview] backbone hidden_dim={hidden_dim}", flush=True)
    head_kind = getattr(cfg, "mv_head_kind", "mlp")
    print(f"[multiview] head_kind: {head_kind}", flush=True)
    model = MultiViewModel(
        inner_model=inner,
        hidden_dim=hidden_dim,
        num_labels=cfg.num_labels,
        num_classes_per_label=(3 if cfg.target_type == "3class" else 1),
        dropout=cfg.dropout,
        head_kind=head_kind,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[multiview] params: {n_params/1e6:.1f}M", flush=True)

    # ---- optimizer ----
    param_groups = model.param_groups(cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = build_scheduler(optimizer, total_steps, cfg.warmup_ratio)
    print(f"[multiview] steps_per_epoch={steps_per_epoch} total_steps={total_steps} "
          f"warmup={int(cfg.warmup_ratio*total_steps)}", flush=True)

    # ---- loss ----
    if cfg.target_type == "3class":
        loss_fn = masked_3class_ce_loss
    elif cfg.loss_fn == "bce":
        loss_fn = masked_bce_with_logits
    elif cfg.loss_fn == "mse":
        loss_fn = masked_mse_loss
    elif cfg.loss_fn == "smooth_l1":
        loss_fn = masked_smooth_l1_loss
    else:
        raise ValueError(f"unsupported loss_fn for multi-view: {cfg.loss_fn}")

    # ---- training loop ----
    scored_names = list(cfg.label_names)
    scored_idx = list(range(len(scored_names)))
    metrics_file = run_dir / "metrics.jsonl"
    best_metric = float("inf")
    step = 0
    t0 = time.time()
    log_acc = []

    def evaluate_mv():
        model.eval()
        preds, ytrue = [], []
        with torch.no_grad():
            for x_f, x_l, lp, y in val_loader:
                x_f, x_l, lp = x_f.to(device, non_blocking=True), x_l.to(device, non_blocking=True), lp.to(device, non_blocking=True)
                logits = model(x_f, x_l, lp)
                # For 3-class, convert logits to scalar prediction (P(+1) - P(-1))
                if cfg.target_type == "3class":
                    p = logits.softmax(dim=-1)  # (B, L, 3) order: [-1, 0, +1]
                    pred = p[..., 2] - p[..., 0]  # (B, L)
                else:
                    pred = torch.sigmoid(logits) * 2 - 1  # roughly map to [-1, +1]
                preds.append(pred.cpu().numpy())
                ytrue.append(y.numpy())
        preds = np.concatenate(preds, 0)
        ytrue = np.concatenate(ytrue, 0)
        # ytrue for 3-class is (B, L) integers 0/1/2 (or -100 for blank).
        # Map 0->-1, 1->0, 2->+1, anything else (-100) -> NaN to mask.
        if cfg.target_type == "3class":
            yt = np.full_like(preds, np.nan)
            yt[ytrue == 0] = -1.0
            yt[ytrue == 1] = 0.0
            yt[ytrue == 2] = 1.0
        else:
            yt = ytrue.astype(np.float32)
        nmse_per = per_label_nmse(yt, preds, scored_names)
        nmse_mean = float(np.mean([v for v in nmse_per.values() if v is not None]))
        return nmse_per, nmse_mean

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        for x_f, x_l, lp, y in train_loader:
            x_f = x_f.to(device, non_blocking=True)
            x_l = x_l.to(device, non_blocking=True)
            lp = lp.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x_f, x_l, lp)
            loss = loss_fn(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            log_acc.append((float(loss.detach()), float(gn.detach())))

            if step % cfg.log_every_steps == 0:
                avg_loss = np.mean([x[0] for x in log_acc])
                avg_gn = np.mean([x[1] for x in log_acc])
                log_acc = []
                lr_bb = optimizer.param_groups[0]["lr"]
                lr_hd = optimizer.param_groups[2]["lr"]
                el = int(time.time() - t0)
                print(f"step {step:>6,}/{total_steps:,}  epoch {epoch}/{cfg.epochs}  "
                      f"loss {avg_loss:.4f}  gn {avg_gn:.2f}  "
                      f"lr_bb {lr_bb:.2e}  lr_hd {lr_hd:.2e}  elapsed {el:>6d}s", flush=True)

            if step % cfg.eval_every_steps == 0 or step == total_steps:
                nmse_per, nmse_mean = evaluate_mv()
                model.train()
                row = {"step": step, "epoch": epoch, "val": {"nmse": dict(nmse_per, mean=nmse_mean)}}
                with open(metrics_file, "a") as f:
                    f.write(json.dumps(_json_safe(row)) + "\n")
                # ckpt
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step, "epoch": epoch,
                    "best_metric": min(best_metric, nmse_mean),
                    "primary_metric": "nmse",
                    "config": cfg.to_dict(),
                }
                torch.save(state, ckpt_dir / "ckpt_last.pt")
                if nmse_mean < best_metric:
                    best_metric = nmse_mean
                    torch.save(state, ckpt_dir / "ckpt_best.pt")
                    print(f"[val @ step {step}] new best nmse={nmse_mean:.4f} — saved ckpt_best.pt", flush=True)
                else:
                    print(f"[val @ step {step}] nmse={nmse_mean:.4f}  (best {best_metric:.4f})", flush=True)


if __name__ == "__main__":
    main()
