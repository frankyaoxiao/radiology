"""DINOv3 ViT-H+/16 backbone + classification head.

Two head types:
  - "attention" (default): single-query multi-head attention pool over
    all backbone tokens (CLS + storage registers + all patches), then
    Linear → num_labels. Lets the head learn to weight spatial tokens
    per task — useful for CheXpert where findings are often localized.
  - "cls": plain Linear on the CLS token only.

Uses the native Meta DINOv3 loader via a local repo clone + .pth file
(the HF mirror is gated). Backbone uses RoPE so any image size works.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn

from config import Config


# --------------------------------------------------------------------------- #
# backbone loader
# --------------------------------------------------------------------------- #
def _load_dinov3_backbone(cfg: Config) -> nn.Module:
    """Build the DINOv3 architecture from the local repo clone and load
    weights from a local .pth. Bypasses both ``torch.hub.load`` (which
    triggers hubconf.py → torchmetrics) and the factory's built-in
    weights loader (which routes every path through
    ``load_state_dict_from_url`` and hits a read-only cache dir).
    """
    import sys, importlib
    repo_dir = str(Path(cfg.dinov3_repo).expanduser().resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    backbones_mod = importlib.import_module("dinov3.hub.backbones")
    factory = getattr(backbones_mod, cfg.dinov3_arch)

    model = factory(pretrained=False)

    weights_path = Path(cfg.dinov3_weights).expanduser().resolve()
    sd = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"DINOv3 state_dict mismatch: missing={missing} unexpected={unexpected}"
        )
    return model


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #
class ClsHead(nn.Module):
    """Linear classifier on the CLS token only."""

    def __init__(self, dim: int, num_labels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, num_labels)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feats: dict) -> torch.Tensor:
        cls = feats["x_norm_clstoken"]  # (B, D)
        return self.proj(cls)


class AttentionPoolHead(nn.Module):
    """Single-query multi-head attention pool over all backbone tokens.

    Tokens supplied to the pool = concat(CLS, storage_tokens, patch_tokens).
    A single learned query attends across them; the (1, D) output is
    LayerNormed and fed to a Linear classifier.
    """

    def __init__(self, dim: int, num_labels: int, num_heads: int = 8) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm_out = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, num_labels)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feats: dict) -> torch.Tensor:
        cls = feats["x_norm_clstoken"].unsqueeze(1)      # (B, 1, D)
        storage = feats["x_storage_tokens"]               # (B, n_storage, D)
        patches = feats["x_norm_patchtokens"]             # (B, N_patch, D)
        tokens = torch.cat([cls, storage, patches], dim=1)  # (B, 1+n_storage+N_patch, D)

        kv = self.norm_kv(tokens)
        q = self.query.expand(tokens.size(0), -1, -1)    # (B, 1, D)
        pooled, _ = self.attn(query=q, key=kv, value=kv, need_weights=False)
        pooled = self.norm_out(pooled.squeeze(1))        # (B, D)
        return self.proj(pooled)


def _build_head(cfg: Config, dim: int) -> nn.Module:
    if cfg.head_type == "cls":
        return ClsHead(dim, cfg.num_labels)
    elif cfg.head_type == "attention":
        return AttentionPoolHead(dim, cfg.num_labels, num_heads=cfg.attn_pool_heads)
    else:
        raise ValueError(f"unknown head_type: {cfg.head_type!r}")


# --------------------------------------------------------------------------- #
# top-level wrapper
# --------------------------------------------------------------------------- #
class CheXpertModel(nn.Module):
    """DINOv3 backbone + classification head."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = _load_dinov3_backbone(cfg)
        hidden_dim: int = int(self.backbone.embed_dim)
        self.hidden_dim = hidden_dim
        self.head = _build_head(cfg, hidden_dim)

    # -------------------------------------------------------------- #
    # forward
    # -------------------------------------------------------------- #
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values : (B, 3, H, W) float tensor (ImageNet-normalized).

        Returns (B, num_labels) logits.
        """
        feats = self.backbone.forward_features(pixel_values)
        return self.head(feats)

    # -------------------------------------------------------------- #
    # optimizer helpers
    # -------------------------------------------------------------- #
    def param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float,
    ) -> List[dict]:
        """Two-group AdamW layout with no-decay on biases/1-D params.

        The learned attention query is 3-D ((1,1,D)) so the ndim<=1 rule
        puts it in the decay bucket; we special-case it into no-decay
        since it's functionally a learned "position / search" vector.
        """
        def split_decay(named: Iterable[tuple[str, nn.Parameter]]) -> tuple[list, list]:
            decay, no_decay = [], []
            for name, p in named:
                if not p.requires_grad:
                    continue
                # biases, LayerNorm weights, cls_token, storage_tokens,
                # mask_token, attention pool query — all no-decay.
                if p.ndim <= 1 or name.endswith("query") or "tokens" in name:
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        bb_decay, bb_no_decay = split_decay(self.backbone.named_parameters())
        hd_decay, hd_no_decay = split_decay(self.head.named_parameters())

        groups = [
            {"params": bb_decay,    "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"},
            {"params": bb_no_decay, "lr": lr_backbone, "weight_decay": 0.0,          "name": "backbone_nodecay"},
            {"params": hd_decay,    "lr": lr_head,     "weight_decay": weight_decay, "name": "head_decay"},
            {"params": hd_no_decay, "lr": lr_head,     "weight_decay": 0.0,          "name": "head_nodecay"},
        ]
        return [g for g in groups if len(g["params"]) > 0]
