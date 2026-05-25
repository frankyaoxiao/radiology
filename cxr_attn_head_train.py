"""Train an attention-pool head on CXR Foundation 8x8x1376 spatial features.

Instead of GAP-ing to 1376 (averaging all 64 spatial locations equally), use
multi-head attention over the 64 tokens to learn label-specific spatial
attention. This should help labels with localized findings (Fracture,
Cardiomegaly heart-region, etc.).
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
    y[values == -1] = 0; y[values == 0] = 1; y[values == 1] = 2
    mask[~np.isnan(values)] = True
    return y, mask


def _build_pid_split(df, val_frac, split_seed):
    pids = df["pid"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(split_seed); rng.shuffle(pids)
    return set(pids[:int(round(len(pids) * val_frac))].tolist())


def _extract_pid(s):
    return s.str.extract(r"pid(\d+)", expand=False).astype("Int64")


class AttentionPoolHead(nn.Module):
    """Per-label learned-query attention pooling over (B, 64, 1376) features.

    Each of the 9 labels has its own learnable query vector; the query attends
    over the 64 spatial tokens (with learned positional bias) to produce a
    per-label pooled feature, which then goes through a small linear to 3
    classes (negative/uncertain/positive).
    """

    def __init__(self, in_dim=1376, num_labels=9, num_heads=8, dropout=0.3):
        super().__init__()
        self.num_labels = num_labels
        self.in_dim = in_dim
        self.queries = nn.Parameter(torch.randn(num_labels, in_dim) * 0.02)
        # Learned 2D positional bias for 8x8 grid
        self.pos_bias = nn.Parameter(torch.zeros(num_labels, 64))
        self.proj = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_dim, num_labels * 3)
        # Per-label scale + bias for the pooled feature
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        # x: (B, 64, 1376)
        B, T, D = x.shape
        # Project keys
        k = self.proj(x)  # (B, 64, 1376)
        # Attention scores per label: (B, num_labels, 64)
        scores = torch.einsum("ld,btd->blt", self.queries, k) / math.sqrt(D)
        scores = scores + self.pos_bias.unsqueeze(0)  # (B, L, 64)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # Pool: (B, L, 64) x (B, 64, D) -> (B, L, D)
        pooled = torch.einsum("blt,btd->bld", attn, x)
        # Per-label features through shared classifier (use diagonal slicing)
        # Reshape pooled to (B*L, D), apply classifier, reshape (B*L, L*3) — overkill
        # Simpler: produce per-label 3-logits by linear from each label's pooled feat
        # Approach: one linear that maps each pooled feat to 3 logits using per-label params
        # We use a single shared linear but with per-label rows separated.
        # Actually simplest: project each pooled feature down to 3 logits using a single shared linear.
        # That means classifier output is (B, L, num_labels*3) — but we only want label-specific 3.
        # Use diagonal: each label's pooled feature -> classify against that label's 3 classes only.
        # Implementation: classifier weight shape (num_labels*3, D). Reshape pooled (B*L, D), apply classifier -> (B*L, L*3). Then for each row li, pick rows li*3:li*3+3 of L*3.
        logits_all = self.classifier(pooled)  # (B, L, L*3)
        # Select the diagonal: for each label li, take logits_all[:, li, li*3:li*3+3]
        out = torch.stack(
            [logits_all[:, li, li * 3:(li + 1) * 3] for li in range(self.num_labels)],
            dim=1,
        )  # (B, L, 3)
        return out


def masked_3class_ce(logits, targets, mask):
    B, L, _ = logits.shape
    lp = F.cross_entropy(logits.reshape(B * L, 3), targets.reshape(B * L).clamp_min(0), reduction="none").reshape(B, L)
    return (lp * mask).sum() / mask.sum().clamp_min(1)


def predict_score_3class(logits):
    p = F.softmax(logits, dim=-1)
    return p[..., 2] - p[..., 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", default="/data/artifacts/frank/misc/cxr_embeds_v2")
    ap.add_argument("--labels-csv", default="/data/artifacts/frank/misc/labels/train2023.csv")
    ap.add_argument("--test-ids-csv", default="/data/artifacts/frank/misc/labels/test_ids.csv")
    ap.add_argument("--paths-csv", default="/data/artifacts/frank/misc/cxr_extract_paths.csv")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    LABELS = list(LABEL_NAMES); NL = len(LABELS)

    # Load shards (8x8x1376 per image)
    paths_df = pd.read_csv(args.paths_csv)
    n_total = len(paths_df)
    feats = np.zeros((n_total, 64, 1376), dtype=np.float32)
    seen = np.zeros(n_total, dtype=bool)
    shards = sorted(Path(args.shards_dir).glob("shard_*_of_*.npz"))
    print(f"loading {len(shards)} shards from {args.shards_dir}...")
    for shp in shards:
        d = np.load(shp)
        ids = d["idxs"]
        # d["feats"] is (N_shard, 8, 8, 1376); reshape to (N_shard, 64, 1376)
        shard_feats = d["feats"].reshape(-1, 64, 1376)
        feats[ids] = shard_feats
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

    label_mat = df_train[LABELS].to_numpy(dtype=np.float32)
    targets, masks = _u_ones_3class(label_mat)
    val_pids = _build_pid_split(df_train, args.val_frac, args.split_seed)
    is_val = df_train["pid"].isin(val_pids).to_numpy()
    train_idx = np.where(~is_val)[0]; val_idx = np.where(is_val)[0]
    print(f"train: {len(train_idx)}  val: {len(val_idx)}")

    Xtr = torch.from_numpy(train_feats[train_idx])
    ytr = torch.from_numpy(targets[train_idx])
    mtr = torch.from_numpy(masks[train_idx])
    Xv  = torch.from_numpy(train_feats[val_idx])
    Xte = torch.from_numpy(feats[n_train_full:])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionPoolHead(in_dim=1376, num_labels=NL, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    dl = DataLoader(TensorDataset(Xtr, ytr, mtr), batch_size=args.batch_size, shuffle=True, drop_last=True)

    best = math.inf; best_state = None
    for ep in range(args.epochs):
        model.train()
        avg = 0
        for x, y, m in dl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            loss = masked_3class_ce(model(x), y, m)
            opt.zero_grad(); loss.backward(); opt.step()
            avg += loss.item()
        avg /= len(dl); sched.step()

        model.eval()
        with torch.no_grad():
            # Eval in chunks (val has 18k items, 64x1376 each)
            preds = []
            for i in range(0, len(Xv), 256):
                chunk = Xv[i:i+256].to(device)
                preds.append(predict_score_3class(model(chunk)).cpu().numpy())
            pred = np.concatenate(preds, axis=0)
            yv = label_mat[val_idx]; mv = masks[val_idx]
            sse = sst = 0.0
            for li in range(NL):
                m_li = mv[:, li]; yt = yv[m_li, li]; yp = pred[m_li, li]
                if len(yt):
                    sse += float(((yp - yt) ** 2).sum()); sst += float((yt ** 2).sum())
            val_nmse = sse / max(sst, 1e-8)
        if val_nmse < best:
            best = val_nmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"ep {ep+1:3d}/{args.epochs}  loss={avg:.4f}  val_nmse={val_nmse:.4f}  best={best:.4f}")

    print(f"\n=== best val_nmse: {best:.4f} ===")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(Xte), 256):
            chunk = Xte[i:i+256].to(device)
            preds.append(predict_score_3class(model(chunk)).cpu().numpy())
        test_pred = np.concatenate(preds, axis=0)

    out = pd.DataFrame({"Id": df_test["Id"]})
    for li, lab in enumerate(LABELS):
        out[lab] = test_pred[:, li]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, float_format="%.6f")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
