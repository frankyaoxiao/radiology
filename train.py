"""DDP training loop for CheXpert 2023 × DINOv3 ViT-H+/16.

Launch via::

    torchrun --standalone --nproc_per_node=4 train.py --config configs/default.yaml

Parameters, paths, and hyperparameters are all read from the YAML.
The resolved config is copied into the run directory as provenance.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import Config
from dataset import (
    CheXpertDataset,
    build_train_transform,
    build_val_transform,
    load_and_split,
)
from metrics import per_label_auroc
from model import CheXpertModel


def _json_safe(obj):
    """Convert nan/inf to None so the output is RFC-8259 JSON, not Python JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# DDP helpers
# --------------------------------------------------------------------------- #
def setup_ddp() -> Tuple[int, int, int, torch.device]:
    """Initialise torch.distributed from torchrun environment variables.

    Returns (rank, world_size, local_rank, device). On single-process runs
    (torchrun not used) returns (0, 1, 0, cuda:0 or cpu).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def all_gather_tensor(t: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-gather a CUDA tensor and concatenate along dim 0."""
    if world_size == 1:
        return t
    gathered = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t.contiguous())
    return torch.cat(gathered, dim=0)


# --------------------------------------------------------------------------- #
# mixup
# --------------------------------------------------------------------------- #
def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multi-label mixup: blend image pairs and label pairs by lam ~ Beta(a, a).

    Works natively with BCE (fractional targets are valid).
    """
    if alpha <= 0:
        return x, y
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x = lam * x + (1.0 - lam) * x[perm]
    y = lam * y + (1.0 - lam) * y[perm]
    return x, y


# --------------------------------------------------------------------------- #
# schedule
# --------------------------------------------------------------------------- #
def build_scheduler(optimizer, total_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    world_size: int,
    rank: int,
) -> Dict[str, float]:
    model.eval()
    logit_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        logit_chunks.append(logits.float())
        label_chunks.append(y)

    local_logits = torch.cat(logit_chunks, dim=0) if logit_chunks else torch.zeros(0, cfg.num_labels, device=device)
    local_labels = torch.cat(label_chunks, dim=0) if label_chunks else torch.zeros(0, cfg.num_labels, device=device)

    # Gather across ranks. DistributedSampler pads to equal length per rank
    # so all_gather is safe (may cause a tiny bit of sample duplication in
    # the final concatenation; negligible for AUROC).
    all_logits = all_gather_tensor(local_logits, world_size)
    all_labels = all_gather_tensor(local_labels, world_size)

    metrics: Dict[str, float] = {}
    if is_main(rank):
        yp = torch.sigmoid(all_logits).cpu().numpy()
        yt = all_labels.cpu().numpy()  # contains nan for uncertains
        metrics = per_label_auroc(yt, yp, cfg.label_names)
    model.train()
    return metrics


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="path to YAML config")
    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_ddp()
    cfg = Config.from_yaml(args.config)

    # Seed
    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    torch.backends.cudnn.benchmark = True

    run_dir = cfg.run_dir
    ckpt_dir = run_dir / "ckpts"
    if is_main(rank):
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        cfg.save_yaml(run_dir / "config.yaml")
        # also copy the *original* yaml in case it has comments the user wants
        try:
            shutil.copy(args.config, run_dir / "config.source.yaml")
        except Exception:
            pass

    if world_size > 1:
        dist.barrier()

    # -------------------- data --------------------
    if is_main(rank):
        print(f"[rank 0] loading dataset …", flush=True)
    df_train, df_val, y_train, y_val = load_and_split(cfg)
    # Smoke / debug subsetting (both 0 in normal runs)
    if cfg.max_train_samples > 0:
        df_train = df_train.head(cfg.max_train_samples).reset_index(drop=True)
        y_train = y_train[: cfg.max_train_samples]
    if cfg.max_val_samples > 0:
        df_val = df_val.head(cfg.max_val_samples).reset_index(drop=True)
        y_val = y_val[: cfg.max_val_samples]
    train_ds = CheXpertDataset(df_train, y_train, cfg.data_root, build_train_transform(cfg))
    val_ds = CheXpertDataset(df_val, y_val, cfg.data_root, build_val_transform(cfg))
    if is_main(rank):
        print(f"[rank 0] train={len(train_ds):,}  val={len(val_ds):,}", flush=True)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size_per_gpu,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size_per_gpu,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    # -------------------- model --------------------
    if is_main(rank):
        print(f"[rank 0] building model …", flush=True)
    model = CheXpertModel(cfg).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    inner = model.module if isinstance(model, DDP) else model
    optimizer = torch.optim.AdamW(
        inner.param_groups(cfg.lr_backbone, cfg.lr_head, cfg.weight_decay),
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_steps > 0:
        total_steps = min(total_steps, cfg.max_steps)
    scheduler = build_scheduler(optimizer, total_steps, cfg.warmup_ratio)

    if is_main(rank):
        print(
            f"[rank 0] steps_per_epoch={steps_per_epoch} total_steps={total_steps} "
            f"warmup={int(total_steps * cfg.warmup_ratio)}",
            flush=True,
        )

    # -------------------- training loop --------------------
    global_step = 0
    best_mean_auc = float("-inf")
    metrics_path = run_dir / "metrics.jsonl"
    if is_main(rank):
        metrics_path.touch(exist_ok=True)

    # Running stats for the current eval window. loss_sum / grad_sum
    # stay on the GPU as 0-d tensors so accumulating them does not
    # trigger a host-device sync on every step; we only .item() them
    # at log / eval boundaries. (DDP synchronizes grads on backward(),
    # so grad_norm is identical across ranks.)
    def fresh_accum() -> dict:
        return {
            "loss_sum": torch.zeros((), device=device),
            "grad_sum": torch.zeros((), device=device),
            "n": 0,
            "samples": 0,
            "t_start": time.time(),
        }

    accum = fresh_accum()
    t_loop_start = time.time()
    model.train()
    for epoch in range(cfg.epochs):
        train_sampler.set_epoch(epoch)
        for x, y in train_loader:
            global_step += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            local_bs = x.size(0)

            x, y = mixup_batch(x, y, cfg.mixup_alpha)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(inner.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            # Accumulate running stats on-device. No .item() here —
            # that would force a host/device sync every step and
            # serialise the pipeline.
            accum["loss_sum"] += loss.detach()
            accum["grad_sum"] += grad_norm.detach()
            accum["n"] += 1
            accum["samples"] += local_bs * world_size

            if is_main(rank) and global_step % cfg.log_every_steps == 0:
                lr_bb = optimizer.param_groups[0]["lr"]
                lr_hd = optimizer.param_groups[-1]["lr"]
                elapsed = time.time() - t_loop_start
                # These .item() calls are the ONLY per-log-interval sync points.
                step_loss = loss.detach().item()
                step_gn = grad_norm.detach().item()
                print(
                    f"step {global_step:>6,}/{total_steps:<6,}  "
                    f"epoch {epoch+1}/{cfg.epochs}  "
                    f"loss {step_loss:.4f}  "
                    f"gn {step_gn:.2f}  "
                    f"lr_bb {lr_bb:.2e}  lr_hd {lr_hd:.2e}  "
                    f"elapsed {elapsed:>6.0f}s",
                    flush=True,
                )

            if global_step % cfg.eval_every_steps == 0 or global_step == total_steps:
                metrics = evaluate(model, val_loader, device, cfg, world_size, rank)
                if is_main(rank):
                    mean_auc = metrics.get("mean", float("nan"))
                    window_elapsed = time.time() - accum["t_start"]
                    # Single sync point for all accumulators.
                    loss_avg = (accum["loss_sum"] / max(1, accum["n"])).item()
                    gn_avg = (accum["grad_sum"] / max(1, accum["n"])).item()
                    # Update best_mean_auc BEFORE writing ckpt_last so the
                    # saved metadata in ckpt_last.pt reflects the most recent
                    # best (not the one-eval-stale best).
                    if not math.isnan(mean_auc) and mean_auc > best_mean_auc:
                        best_mean_auc = mean_auc
                        is_new_best = True
                    else:
                        is_new_best = False

                    line = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "train_loss_step": loss.detach().item(),
                        "train_loss_avg": loss_avg,
                        "grad_norm_avg": gn_avg,
                        "lr_backbone": optimizer.param_groups[0]["lr"],
                        "lr_head": optimizer.param_groups[-1]["lr"],
                        "samples_per_sec": accum["samples"] / window_elapsed if window_elapsed > 0 else 0.0,
                        "elapsed_window_sec": window_elapsed,
                        "elapsed_total_sec": time.time() - t_loop_start,
                        "val": metrics,
                    }
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps(_json_safe(line)) + "\n")
                    per_lab = "  ".join(
                        f"{n.split()[0][:4]}:{metrics.get(n, float('nan')):.3f}"
                        for n in cfg.label_names
                    )
                    print(
                        f"[val @ step {global_step}] mean_auc={mean_auc:.4f}  "
                        f"loss_avg={loss_avg:.4f}  "
                        f"gn_avg={gn_avg:.2f}  "
                        f"sps={line['samples_per_sec']:.1f}  |  {per_lab}",
                        flush=True,
                    )

                    save_ckpt(ckpt_dir / "ckpt_last.pt", inner, optimizer, scheduler, global_step, epoch, best_mean_auc, cfg)
                    if is_new_best:
                        save_ckpt(ckpt_dir / "ckpt_best.pt", inner, optimizer, scheduler, global_step, epoch, best_mean_auc, cfg)
                        print(f"[val @ step {global_step}] new best mean_auc={best_mean_auc:.4f} — saved ckpt_best.pt", flush=True)
                # reset eval-window accumulators on every rank
                accum = fresh_accum()
                if world_size > 1:
                    dist.barrier()

            if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                break
        if cfg.max_steps > 0 and global_step >= cfg.max_steps:
            break

    if is_main(rank):
        print(f"done. best mean_auc={best_mean_auc:.4f}", flush=True)

    cleanup_ddp()


def save_ckpt(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    step: int,
    epoch: int,
    best_mean_auc: float,
    cfg: Config,
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "best_mean_auc": best_mean_auc,
        "config": cfg.to_dict(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


if __name__ == "__main__":
    main()
