"""Generate val predictions for a multi-view ckpt — same val partition as
single-view runs (so the output can feed into stack_per_label.py)."""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config, LABEL_NAMES
from dataset import load_and_split, build_val_transform
from model import CheXpertModel
from multiview import MultiViewCheXpertDataset, MultiViewModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    cfg_dict = ck["config"]
    known = {f.name for f in Config.__dataclass_fields__.values()}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in known}
    cfg = Config(**cfg_dict)

    inner = CheXpertModel(cfg, pretrained=False)
    if hasattr(inner, "hidden_dim"):
        D = int(inner.hidden_dim)
    elif hasattr(inner.backbone, "num_features"):
        D = int(inner.backbone.num_features)
    else:
        D = 768
    head_kind = getattr(cfg, "mv_head_kind", "mlp")
    model = MultiViewModel(inner_model=inner, hidden_dim=D, num_labels=cfg.num_labels,
                            num_classes_per_label=(3 if cfg.target_type == "3class" else 1),
                            dropout=cfg.dropout, head_kind=head_kind)
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).eval()

    df_train, df_val, y_train, y_val = load_and_split(cfg)
    print(f"val rows: {len(df_val)}")
    val_ds = MultiViewCheXpertDataset(df_val, y_val, cfg.data_root, build_val_transform(cfg),
                                       paired_only=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Iterate val_ds in same order as built-in CheXpertDataset (per-frontal-row)
    # The val_preds.npz format is per-row paths matching df_val for the FRONTAL rows of val_ds.
    # We need to mirror that ordering.
    all_preds, all_paths, all_labels = [], [], []
    with torch.no_grad():
        for x_f, x_l, lp, y in val_loader:
            x_f = x_f.to(device, non_blocking=True)
            x_l = x_l.to(device, non_blocking=True)
            lp = lp.to(device, non_blocking=True)
            if args.bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x_f, x_l, lp)
            else:
                logits = model(x_f, x_l, lp)
            logits = logits.float()
            if cfg.target_type == "3class":
                p = logits.softmax(dim=-1)
                pred = (p[..., 2] - p[..., 0]).cpu().numpy()
            else:
                pred = torch.sigmoid(logits).cpu().numpy() * 2 - 1
            all_preds.append(pred)
            all_labels.append(y.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    raw_labels_int = np.concatenate(all_labels, axis=0)

    # Convert 3-class integer labels back to {-1, 0, 1, NaN}
    if cfg.target_type == "3class":
        raw_labels = np.full_like(preds, np.nan, dtype=np.float32)
        raw_labels[raw_labels_int == 0] = -1.0
        raw_labels[raw_labels_int == 1] = 0.0
        raw_labels[raw_labels_int == 2] = 1.0
    else:
        raw_labels = raw_labels_int.astype(np.float32)

    # Need paths. Use the frontal_indices of val_ds.
    paths = np.array([val_ds.df.loc[int(i), "Path"] for i in val_ds.frontal_indices])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, preds=preds.astype(np.float32), paths=paths, raw_labels=raw_labels)
    print(f"wrote {args.out}  preds={preds.shape}")


if __name__ == "__main__":
    main()
