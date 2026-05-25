"""Train a head on spatial CXR features (4 pooling stats × 1376 = 5504-d).

Same flow as cxr_head_train.py but expects npz shards with 5504-d features.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import LABEL_NAMES


def _u_ones_3class(values):
    y = np.full(values.shape, -1, dtype=np.int64)
    mask = np.zeros(values.shape, dtype=bool)
    y[values == -1] = 0
    y[values == 0]  = 1
    y[values == 1]  = 2
    mask[~np.isnan(values)] = True
    return y, mask


def _build_pid_split(df, val_frac, split_seed):
    pids = df["pid"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(split_seed)
    rng.shuffle(pids)
    n_val = int(round(len(pids) * val_frac))
    return set(pids[:n_val].tolist())


def _extract_pid(path_col):
    return path_col.str.extract(r"pid(\d+)", expand=False).astype("Int64")


class MLPHead(nn.Module):
    def __init__(self, in_dim, num_labels, hidden_dim=1024, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels * 3),
        )
        self.num_labels = num_labels

    def forward(self, x):
        return self.net(x).view(-1, self.num_labels, 3)


def masked_3class_ce(logits, targets, mask):
    B, L, _ = logits.shape
    lp = F.cross_entropy(logits.reshape(B * L, 3), targets.reshape(B * L).clamp_min(0), reduction="none").reshape(B, L)
    return (lp * mask).sum() / mask.sum().clamp_min(1)


def predict_score_3class(logits):
    p = F.softmax(logits, dim=-1)
    return p[..., 2] - p[..., 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards-dir", default="/data/artifacts/frank/misc/cxr_embeds_spatial")
    p.add_argument("--labels-csv", default="/data/artifacts/frank/misc/labels/train2023.csv")
    p.add_argument("--test-ids-csv", default="/data/artifacts/frank/misc/labels/test_ids.csv")
    p.add_argument("--paths-csv", default="/data/artifacts/frank/misc/cxr_extract_paths.csv")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--feat-dim", type=int, default=5504)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    LABELS = list(LABEL_NAMES)
    NL = len(LABELS)

    paths_df = pd.read_csv(args.paths_csv)
    n_total = len(paths_df)
    feats = np.zeros((n_total, args.feat_dim), dtype=np.float32)
    seen = np.zeros(n_total, dtype=bool)
    shards = sorted(Path(args.shards_dir).glob("shard_*_of_*.npz"))
    print(f"loading {len(shards)} shards from {args.shards_dir}...")
    for shp in shards:
        d = np.load(shp)
        ids = d["idxs"]
        feats[ids] = d["feats"]
        seen[ids] = True
    print(f"loaded {seen.sum()}/{n_total} embeddings")
    if not seen.all():
        print(f"WARNING: {(~seen).sum()} missing")

    df_full = pd.read_csv(args.labels_csv)
    df_full["pid"] = _extract_pid(df_full["Path"])
    parseable = df_full["pid"].notna().to_numpy()
    df_train = df_full[parseable].reset_index(drop=True)
    df_test = pd.read_csv(args.test_ids_csv)

    n_train_full = len(df_full)
    train_feats = feats[:n_train_full][parseable]
    n_train = len(df_train)

    label_mat = df_train[LABELS].to_numpy(dtype=np.float32)
    targets, masks = _u_ones_3class(label_mat)
    val_pids = _build_pid_split(df_train, args.val_frac, args.split_seed)
    is_val = df_train["pid"].isin(val_pids).to_numpy()
    train_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]
    print(f"train: {len(train_idx)}  val: {len(val_idx)}  feat_dim={args.feat_dim}")

    Xtr = torch.from_numpy(train_feats[train_idx])
    ytr = torch.from_numpy(targets[train_idx])
    mtr = torch.from_numpy(masks[train_idx])
    Xv  = torch.from_numpy(train_feats[val_idx])
    Xte = torch.from_numpy(feats[n_train_full:])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPHead(args.feat_dim, NL, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    dl = DataLoader(TensorDataset(Xtr, ytr, mtr), batch_size=args.batch_size, shuffle=True, drop_last=True)

    best = math.inf; best_state = None
    for ep in range(args.epochs):
        model.train()
        avg = 0
        for x, y, m in dl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits = model(x)
            loss = masked_3class_ce(logits, y, m)
            opt.zero_grad(); loss.backward(); opt.step()
            avg += loss.item()
        avg /= len(dl); sched.step()

        model.eval()
        with torch.no_grad():
            sse = sst = 0.0
            pred = predict_score_3class(model(Xv.to(device))).cpu().numpy()
            yv = label_mat[val_idx]; mv = masks[val_idx]
            for li in range(NL):
                m = mv[:, li]
                yt = yv[m, li]; yp = pred[m, li]
                if len(yt):
                    sse += float(((yp - yt) ** 2).sum())
                    sst += float((yt ** 2).sum())
            val_nmse = sse / max(sst, 1e-8)
        if val_nmse < best:
            best = val_nmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep+1:3d}/{args.epochs}  loss={avg:.4f}  val_nmse={val_nmse:.4f}  best={best:.4f}")

    print(f"\n=== best val_nmse: {best:.4f} ===")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = predict_score_3class(model(Xte.to(device))).cpu().numpy()

    out = pd.DataFrame({"Id": df_test["Id"]})
    for li, lab in enumerate(LABELS):
        out[lab] = pred[:, li]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, float_format="%.6f")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
