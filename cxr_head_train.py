"""Train a small MLP head on Google CXR Foundation (ELIXR-C) embeddings.

After cxr_extract.py has populated /data/.../cxr_embeds/shard_*.npz, this
loads + aggregates them, splits by the same patient-level split_seed=42
val_frac=0.1 used elsewhere, trains a 3-class CE head, and writes a CSV in
the same format as our other submissions.
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

from config import Config, LABEL_NAMES


def _u_ones_3class(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map raw -1/0/+1 (and NaN = uncertain) to 3-class targets {0,1,2} + mask.

    -1 -> class 0, 0 -> class 1, +1 -> class 2.
    NaN/missing -> ignored via mask.
    """
    y = np.full(values.shape, -1, dtype=np.int64)
    mask = np.zeros(values.shape, dtype=bool)
    y[values == -1] = 0
    y[values == 0]  = 1
    y[values == 1]  = 2
    mask[~np.isnan(values)] = True
    return y, mask


def _build_pid_split(df: pd.DataFrame, val_frac: float, split_seed: int):
    pids = df["pid"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(split_seed)
    rng.shuffle(pids)
    n_val = int(round(len(pids) * val_frac))
    val_pids = set(pids[:n_val].tolist())
    return val_pids


def _extract_pid(path_col: pd.Series) -> pd.Series:
    import re
    return path_col.str.extract(r"pid(\d+)", expand=False).astype("Int64")


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_labels: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels * 3),
        )
        self.num_labels = num_labels

    def forward(self, x):
        return self.net(x).view(-1, self.num_labels, 3)


def masked_3class_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # logits: (B, L, 3); targets: (B, L) int; mask: (B, L) bool
    B, L, _ = logits.shape
    loss_per = F.cross_entropy(
        logits.reshape(B * L, 3),
        targets.reshape(B * L).clamp_min(0),
        reduction="none",
    ).reshape(B, L)
    masked = loss_per * mask
    return masked.sum() / mask.sum().clamp_min(1)


def predict_score_3class(logits: torch.Tensor) -> torch.Tensor:
    # logits: (B, L, 3). Return P(+1) - P(-1)
    probs = F.softmax(logits, dim=-1)
    return probs[..., 2] - probs[..., 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards-dir", type=str, default="/data/artifacts/frank/misc/cxr_embeds")
    p.add_argument("--labels-csv", type=str, default="/data/artifacts/frank/misc/labels/train2023.csv")
    p.add_argument("--test-ids-csv", type=str, default="/data/artifacts/frank/misc/labels/test_ids.csv")
    p.add_argument("--paths-csv", type=str, default="/data/artifacts/frank/misc/cxr_extract_paths.csv")
    p.add_argument("--out-csv", type=str, required=True,
                   help="Where to write the submission CSV")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    LABELS = list(LABEL_NAMES)
    NL = len(LABELS)

    # Aggregate shards
    paths_df = pd.read_csv(args.paths_csv)
    n_total = len(paths_df)
    feats = np.zeros((n_total, 1376), dtype=np.float32)
    seen = np.zeros(n_total, dtype=bool)
    shards = sorted(Path(args.shards_dir).glob("shard_*_of_*.npz"))
    print(f"loading {len(shards)} shards...")
    for shp in shards:
        d = np.load(shp)
        ids = d["idxs"]
        feats[ids] = d["feats"]
        seen[ids] = True
    print(f"loaded {seen.sum()}/{n_total} embeddings")
    if not seen.all():
        missing = np.where(~seen)[0]
        print(f"WARNING: {len(missing)} missing embeddings  (first 5 idxs: {missing[:5]})")

    # Build train + test dataframes aligned with the global idx
    df_train_full = pd.read_csv(args.labels_csv)
    df_train_full["pid"] = _extract_pid(df_train_full["Path"])
    # Find rows with unparseable paths — these are dropped by dataset.load_and_split
    parseable_mask = df_train_full["pid"].notna().to_numpy()
    df_train = df_train_full[parseable_mask].reset_index(drop=True)
    keep = ["Path", "pid"] + LABELS
    df_train = df_train[keep]

    df_test = pd.read_csv(args.test_ids_csv)

    n_train_full = len(df_train_full)  # 178158 — what's in paths_df
    # Global idx -> split: paths_df has 178158 train + 22596 test
    paths_df_train = paths_df.iloc[:n_train_full]
    paths_df_test  = paths_df.iloc[n_train_full:]
    assert (paths_df_train["split"] == "train").all()
    assert (paths_df_test["split"]  == "test").all()

    # Filter train features+labels to parseable rows only
    train_feats = feats[:n_train_full][parseable_mask]
    n_train = len(df_train)

    # Build label matrix for train rows
    label_mat = np.full((n_train, NL), np.nan, dtype=np.float32)
    for li, lab in enumerate(LABELS):
        col = df_train[lab].to_numpy()
        # Map U=-1 to 0 (U-Ones rule for training); -1/0/+1 already correct
        # Some labels use 'U' (-1) as a class — keep as -1 for class 0
        label_mat[:, li] = col

    targets, masks = _u_ones_3class(label_mat)
    print(f"train rows: {n_train}  mask fill: {masks.mean():.3f}")

    # Patient-level split (matching dataset.py)
    val_pids = _build_pid_split(df_train, args.val_frac, args.split_seed)
    is_val = df_train["pid"].isin(val_pids).to_numpy()
    train_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]
    print(f"train: {len(train_idx)}  val: {len(val_idx)}")

    X_train = torch.from_numpy(train_feats[train_idx])
    y_train = torch.from_numpy(targets[train_idx])
    m_train = torch.from_numpy(masks[train_idx])

    X_val = torch.from_numpy(train_feats[val_idx])
    y_val = torch.from_numpy(targets[val_idx])
    m_val = torch.from_numpy(masks[val_idx])

    X_test = torch.from_numpy(feats[n_train_full:])
    # Test rows: paths_df.iloc[n_train_full:] in original order — these match df_test order

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPHead(1376, NL, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    train_ds = TensorDataset(X_train, y_train, m_train)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    best_val = math.inf
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        loss_avg = 0.0
        for x, y, m in train_dl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits = model(x)
            loss = masked_3class_ce(logits, y, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_avg += loss.item()
        loss_avg /= len(train_dl)
        sched.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device))
            val_scores = predict_score_3class(val_logits).cpu().numpy()
            # NMSE on val: target is the raw -1/0/+1 column where present
            sse = 0.0; sst = 0.0
            for li in range(NL):
                vmask = masks[val_idx][:, li]
                yt = label_mat[val_idx][vmask, li]
                yp = val_scores[vmask, li]
                if len(yt) == 0:
                    continue
                sse += float(((yp - yt) ** 2).sum())
                sst += float((yt ** 2).sum())
            val_nmse = sse / max(sst, 1e-8)
        if val_nmse < best_val:
            best_val = val_nmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch+1:3d}/{args.epochs}  loss={loss_avg:.4f}  val_nmse={val_nmse:.4f}  best={best_val:.4f}")

    print(f"\n=== best val_nmse: {best_val:.4f} ===")
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_logits = model(X_test.to(device))
        test_scores = predict_score_3class(test_logits).cpu().numpy()

    # Write submission CSV
    out = pd.DataFrame({"Id": df_test["Id"]})
    for li, lab in enumerate(LABELS):
        out[lab] = test_scores[:, li]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, float_format="%.6f")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
