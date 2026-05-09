"""Average model weights across seeds (model soups).

Loads multiple checkpoints, averages their state_dicts, saves merged model.
No GPU needed — runs on CPU.

Usage:
    python -u merge_checkpoints.py \
        --ckpts ckpt0.pt ckpt1.pt ckpt2.pt \
        --out merged_ckpt.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from collections import OrderedDict

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    print(f"Merging {len(args.ckpts)} checkpoints...", flush=True)

    merged_sd = None
    n = len(args.ckpts)

    for i, path in enumerate(args.ckpts):
        print(f"  Loading {path.name}...", flush=True)
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        sd = ckpt["model"]

        if merged_sd is None:
            merged_sd = OrderedDict()
            for k, v in sd.items():
                merged_sd[k] = v.float() / n
        else:
            for k, v in sd.items():
                merged_sd[k] += v.float() / n

    # Save as a checkpoint with the merged state_dict and config from first ckpt
    first_ckpt = torch.load(str(args.ckpts[0]), map_location="cpu", weights_only=False)
    out_ckpt = {
        "model": merged_sd,
        "config": first_ckpt["config"],
        "step": first_ckpt.get("step"),
        "epoch": first_ckpt.get("epoch"),
        "best_metric": first_ckpt.get("best_metric"),
        "primary_metric": first_ckpt.get("primary_metric"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, str(args.out))
    print(f"Saved merged checkpoint to {args.out}", flush=True)

    # Sanity: compare a few param norms
    print(f"\nParam norm comparison (first 5 keys):", flush=True)
    keys = list(merged_sd.keys())[:5]
    for k in keys:
        merged_norm = merged_sd[k].norm().item()
        orig_norm = first_ckpt["model"][k].float().norm().item()
        print(f"  {k}: merged={merged_norm:.4f}  orig_s0={orig_norm:.4f}", flush=True)


if __name__ == "__main__":
    main()
