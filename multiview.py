"""Multi-view (frontal+lateral) fusion model and dataset.

Trains a siamese backbone that processes frontal and lateral images and fuses
the features through an MLP head. For studies without a lateral view, a
learned "no lateral" embedding stands in.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Callable
import re
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from config import Config, LABEL_NAMES
from dataset import build_train_transform, build_val_transform, IMAGENET_MEAN, IMAGENET_STD


_PID_STUDY_RE = re.compile(r"(pid\d+)/(study\d+)")


def _pid_study(path: str) -> Tuple[str, str] | None:
    m = _PID_STUDY_RE.search(path)
    if not m:
        return None
    return m.group(1), m.group(2)


class MultiViewCheXpertDataset(Dataset):
    """Each item is a (frontal_image, lateral_image, lat_present_flag, label) tuple.

    Iteration is over FRONTAL rows. The lateral image is sampled at random from
    the same study; if no lateral exists, a zero tensor is returned and
    lat_present=0.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        data_root: str | Path,
        transform: Callable,
        paired_only: bool = False,
    ) -> None:
        # All rows from df. Build a map study -> [lateral row indices].
        df = df.reset_index(drop=True).copy()
        assert "Frontal/Lateral" in df.columns
        self.df = df
        self.y = y
        self.data_root = Path(data_root)
        self.transform = transform

        is_frontal = (df["Frontal/Lateral"] == "Frontal").to_numpy()
        is_lateral = (df["Frontal/Lateral"] == "Lateral").to_numpy()

        # Map study -> [lateral global row indices]
        self._study_to_lats: dict[Tuple[str, str], List[int]] = {}
        for i in np.where(is_lateral)[0]:
            ps = _pid_study(df.loc[i, "Path"])
            if ps:
                self._study_to_lats.setdefault(ps, []).append(int(i))

        all_frontal = np.where(is_frontal)[0]
        if paired_only:
            keep = []
            for i in all_frontal:
                ps = _pid_study(df.loc[i, "Path"])
                if ps and ps in self._study_to_lats:
                    keep.append(int(i))
            self.frontal_indices = np.asarray(keep, dtype=np.int64)
            print(f"[multiview/paired_only] filtered to {len(self.frontal_indices)} paired frontals "
                  f"(from {len(all_frontal)} total)", flush=True)
        else:
            self.frontal_indices = all_frontal
            paired = sum(1 for i in self.frontal_indices
                         if _pid_study(df.loc[i, "Path"]) in self._study_to_lats)
            print(f"[multiview] {len(self.frontal_indices)} frontals; "
                  f"{paired} ({paired/len(self.frontal_indices)*100:.1f}%) have a lateral pair",
                  flush=True)

    def __len__(self) -> int:
        return len(self.frontal_indices)

    def _load(self, row_idx: int) -> torch.Tensor:
        full = self.data_root / self.df.loc[row_idx, "Path"]
        with Image.open(full) as img:
            img = img.convert("RGB")
            return self.transform(img)

    def __getitem__(self, idx: int):
        row_idx = int(self.frontal_indices[idx])
        x_front = self._load(row_idx)
        y = torch.from_numpy(np.array(self.y[row_idx]))

        ps = _pid_study(self.df.loc[row_idx, "Path"])
        lats = self._study_to_lats.get(ps, []) if ps else []
        if lats:
            lat_idx = random.choice(lats)
            x_lat = self._load(lat_idx)
            lat_present = torch.tensor(1.0)
        else:
            x_lat = torch.zeros_like(x_front)
            lat_present = torch.tensor(0.0)

        return x_front, x_lat, lat_present, y


class MultiViewModel(nn.Module):
    """Siamese: shared backbone applied to frontal AND lateral; fuse features
    via a configurable head. For absent lateral, swap in a learned
    `no_lateral` embedding.

    `head_kind`:
      - "mlp": 2D -> D -> out (LayerNorm + Linear + GELU + Dropout + Linear)
      - "linear": 2D -> out (single Linear, like single-view but doubled input)
      - "sum_linear": features summed (D), Linear(D -> out) (forces shared rep)
      - "cross_attn": frontal query attends to {frontal, lateral} keys (1-layer
                      multi-head attention), then Linear(D -> out).
    """

    def __init__(self, inner_model: nn.Module, hidden_dim: int, num_labels: int = 9,
                 num_classes_per_label: int = 3, dropout: float = 0.3,
                 head_kind: str = "mlp"):
        super().__init__()
        self.inner = inner_model
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_classes_per_label = num_classes_per_label
        self.head_kind = head_kind

        self.no_lateral_embedding = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.trunc_normal_(self.no_lateral_embedding, std=0.02)

        out_dim = num_labels * num_classes_per_label
        if head_kind == "mlp":
            self.head = nn.Sequential(
                nn.LayerNorm(2 * hidden_dim),
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, out_dim),
            )
            nn.init.trunc_normal_(self.head[-1].weight, std=0.02)
            nn.init.zeros_(self.head[-1].bias)
        elif head_kind == "linear":
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(2 * hidden_dim, out_dim),
            )
            nn.init.trunc_normal_(self.head[-1].weight, std=0.02)
            nn.init.zeros_(self.head[-1].bias)
        elif head_kind == "sum_linear":
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, out_dim),
            )
            nn.init.trunc_normal_(self.head[-1].weight, std=0.02)
            nn.init.zeros_(self.head[-1].bias)
        elif head_kind == "cross_attn":
            # Frontal token queries {frontal, lateral} keys/values
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True, dropout=dropout)
            self.attn_norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, out_dim),
            )
            nn.init.trunc_normal_(self.head[-1].weight, std=0.02)
            nn.init.zeros_(self.head[-1].bias)
        else:
            raise ValueError(f"unknown head_kind: {head_kind!r}")

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract per-image features using the inner backbone's forward_features.

        Falls back to assuming the inner model has a `.backbone` attribute that
        produces tokens or pooled features.
        """
        bb = self.inner.backbone
        if hasattr(bb, "forward_features"):
            tokens = bb.forward_features(x)
            if tokens.ndim == 3:  # (B, N, D) - take CLS
                return tokens[:, 0, :]
            return tokens  # already pooled (B, D)
        # Fallback: try a generic forward and assume output is (B, D)
        return bb(x)

    def forward(self, x_front: torch.Tensor, x_lat: torch.Tensor,
                lat_present: torch.Tensor) -> torch.Tensor:
        B = x_front.size(0)
        f_front = self._features(x_front)
        f_lat_raw = self._features(x_lat)
        # lat_present: (B,) 0 or 1
        mask = lat_present.unsqueeze(-1).bool()
        f_lat = torch.where(mask, f_lat_raw, self.no_lateral_embedding.expand(B, -1))
        if self.head_kind == "sum_linear":
            feats = f_front + f_lat
        elif self.head_kind == "cross_attn":
            # query: frontal (B, 1, D); keys/values: {frontal, lateral} (B, 2, D)
            q = f_front.unsqueeze(1)
            kv = torch.stack([f_front, f_lat], dim=1)
            attended, _ = self.attn(q, kv, kv)
            feats = self.attn_norm(attended.squeeze(1) + f_front)  # residual
        else:
            feats = torch.cat([f_front, f_lat], dim=-1)
        out = self.head(feats)
        if self.num_classes_per_label == 1:
            return out
        return out.view(B, self.num_labels, self.num_classes_per_label)

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float):
        """Two param groups: backbone (lr_backbone) and head + no_lateral_embedding (lr_head)."""
        backbone_params = list(self.inner.backbone.parameters())
        head_params = list(self.head.parameters()) + [self.no_lateral_embedding]

        # Split each group into decay/no_decay
        def split(params):
            decay, no_decay = [], []
            for p in params:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1:
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        bb_d, bb_nd = split(backbone_params)
        hd_d, hd_nd = split(head_params)
        return [
            {"params": bb_d,  "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": bb_nd, "lr": lr_backbone, "weight_decay": 0.0},
            {"params": hd_d,  "lr": lr_head,     "weight_decay": weight_decay},
            {"params": hd_nd, "lr": lr_head,     "weight_decay": 0.0},
        ]
