"""Model backbones for CheXpert classification.

Supports model types (selected via cfg.model_type):

  - "dinov3": DINOv3 ViT-H+/16 (840M params) with attention pool or CLS head.
  - "densenet121": torchvision DenseNet-121 (7M params) with ImageNet pretraining,
    GAP + dropout + linear head.
  - "rad_dino": microsoft/rad-dino ViT-B/14 (86M params) pretrained on chest X-rays,
    CLS token + dropout + linear head.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Linear classifier on the CLS token only, with optional pre-classifier dropout."""

    def __init__(self, dim: int, num_labels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Linear(dim, num_labels)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feats: dict) -> torch.Tensor:
        cls = feats["x_norm_clstoken"]  # (B, D)
        return self.proj(self.drop(cls))


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


class MlpHead(nn.Module):
    """2-layer MLP classifier on the CLS token: Linear -> GELU -> Dropout -> Linear.

    Adds capacity for the head to learn label inter-correlations beyond what a
    single Linear can express. The activation lives between the two Linears so
    the total mapping is genuinely nonlinear.
    """

    def __init__(self, dim: int, num_labels: int, hidden_dim: int = 512, dropout: float = 0.3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, num_labels)
        for m in (self.fc1, self.fc2):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, feats: dict) -> torch.Tensor:
        cls = feats["x_norm_clstoken"]  # (B, D)
        return self.fc2(self.drop(self.act(self.fc1(cls))))


def _build_head(cfg: Config, dim: int) -> nn.Module:
    num_classes_per_label = 3 if cfg.target_type == "3class" else 1
    out_dim = cfg.num_labels * num_classes_per_label
    if cfg.head_type == "cls":
        return ClsHead(dim, out_dim, dropout=cfg.head_dropout)
    elif cfg.head_type == "attention":
        return AttentionPoolHead(dim, out_dim, num_heads=cfg.attn_pool_heads)
    elif cfg.head_type == "mlp":
        return MlpHead(dim, out_dim, hidden_dim=cfg.head_hidden_dim, dropout=cfg.dropout)
    else:
        raise ValueError(f"unknown head_type: {cfg.head_type!r}")


# --------------------------------------------------------------------------- #
# DenseNet-121 backbone
# --------------------------------------------------------------------------- #
class DenseNet121Model(nn.Module):
    """DenseNet-121 with ImageNet pretrained weights, GAP + dropout + linear head."""

    def __init__(self, cfg: Config, pretrained: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes_per_label = 3 if cfg.target_type == "3class" else 1
        import torchvision.models as models
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.densenet121(weights=weights)
        self.features = base.features
        self.hidden_dim = base.classifier.in_features  # 1024
        self.drop = nn.Dropout(p=cfg.dropout)
        out_dim = cfg.num_labels * self.num_classes_per_label
        self.classifier = nn.Linear(self.hidden_dim, out_dim)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.features(pixel_values)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1))
        feats = torch.flatten(feats, 1)
        feats = self.drop(feats)
        out = self.classifier(feats)
        if self.num_classes_per_label == 3:
            return out.view(-1, self.cfg.num_labels, 3)
        return out

    def param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float,
    ) -> List[dict]:
        def split_decay(named: Iterable[tuple[str, nn.Parameter]]) -> tuple[list, list]:
            decay, no_decay = [], []
            for name, p in named:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1:
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        bb_decay, bb_no_decay = split_decay(self.features.named_parameters())
        hd_decay, hd_no_decay = split_decay(
            list(self.classifier.named_parameters())
        )

        groups = [
            {"params": bb_decay,    "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"},
            {"params": bb_no_decay, "lr": lr_backbone, "weight_decay": 0.0,          "name": "backbone_nodecay"},
            {"params": hd_decay,    "lr": lr_head,     "weight_decay": weight_decay, "name": "head_decay"},
            {"params": hd_no_decay, "lr": lr_head,     "weight_decay": 0.0,          "name": "head_nodecay"},
        ]
        return [g for g in groups if len(g["params"]) > 0]


# --------------------------------------------------------------------------- #
# top-level wrapper (DINOv3)
# --------------------------------------------------------------------------- #
class CheXpertDINOv3Model(nn.Module):
    """DINOv3 backbone + classification head."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes_per_label = 3 if cfg.target_type == "3class" else 1
        self.backbone = _load_dinov3_backbone(cfg)
        hidden_dim: int = int(self.backbone.embed_dim)
        self.hidden_dim = hidden_dim
        self.head = _build_head(cfg, hidden_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(pixel_values)
        out = self.head(feats)
        if self.num_classes_per_label == 3:
            return out.view(-1, self.cfg.num_labels, 3)
        return out

    def param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float,
    ) -> List[dict]:
        _NO_DECAY_SUFFIXES = (
            "cls_token",
            "storage_tokens",
            "mask_token",
            "query",
        )

        def split_decay(named: Iterable[tuple[str, nn.Parameter]]) -> tuple[list, list]:
            decay, no_decay = [], []
            for name, p in named:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1 or name.endswith(_NO_DECAY_SUFFIXES):
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


# --------------------------------------------------------------------------- #
# ConvNeXt backbone (via timm)
# --------------------------------------------------------------------------- #
class ConvNeXtModel(nn.Module):
    """ConvNeXt-Base/Small with timm pretrained weights, GAP + dropout + linear head."""

    def __init__(self, cfg: Config, pretrained: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        import timm
        self.num_classes_per_label = 3 if cfg.target_type == "3class" else 1
        out_dim = cfg.num_labels * self.num_classes_per_label
        model_name = {
            "convnext_base": "convnext_base.fb_in22k_ft_in1k",
            "convnext_small": "convnext_small.fb_in22k_ft_in1k",
        }.get(cfg.model_type, cfg.model_type)
        self.backbone = timm.create_model(
            model_name, pretrained=False, num_classes=0, global_pool="avg",
            drop_path_rate=getattr(cfg, "drop_path_rate", 0.0),
        )
        if pretrained and hasattr(cfg, "convnext_weights") and cfg.convnext_weights:
            weights_path = Path(cfg.convnext_weights).expanduser().resolve()
            if weights_path.suffix == ".safetensors":
                from safetensors.torch import load_file
                sd = load_file(str(weights_path))
            else:
                sd = torch.load(str(weights_path), map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(sd, strict=False)
        self.hidden_dim = self.backbone.num_features
        self.drop = nn.Dropout(p=cfg.dropout)
        self.classifier = nn.Linear(self.hidden_dim, out_dim)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(pixel_values)
        feats = self.drop(feats)
        out = self.classifier(feats)
        if self.num_classes_per_label == 3:
            return out.view(-1, self.cfg.num_labels, 3)
        return out

    def param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float,
    ) -> List[dict]:
        def split_decay(named: Iterable[tuple[str, nn.Parameter]]) -> tuple[list, list]:
            decay, no_decay = [], []
            for name, p in named:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1:
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        bb_decay, bb_no_decay = split_decay(self.backbone.named_parameters())
        hd_decay, hd_no_decay = split_decay(
            list(self.classifier.named_parameters())
        )
        groups = [
            {"params": bb_decay,    "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"},
            {"params": bb_no_decay, "lr": lr_backbone, "weight_decay": 0.0,          "name": "backbone_nodecay"},
            {"params": hd_decay,    "lr": lr_head,     "weight_decay": weight_decay, "name": "head_decay"},
            {"params": hd_no_decay, "lr": lr_head,     "weight_decay": 0.0,          "name": "head_nodecay"},
        ]
        return [g for g in groups if len(g["params"]) > 0]


# --------------------------------------------------------------------------- #
# RAD-DINO backbone (microsoft/rad-dino via HuggingFace transformers)
# --------------------------------------------------------------------------- #
class RadDinoModel(nn.Module):
    """RAD-DINO ViT-B/14 pretrained on ~800K chest X-rays.

    Supports head_type="cls" (dropout + linear on CLS token) or
    "attention" (attention pooling over CLS + patch tokens).
    """

    def __init__(self, cfg: Config, pretrained: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes_per_label = 3 if cfg.target_type == "3class" else 1
        from transformers import AutoModel, AutoConfig
        model_path = cfg.rad_dino_path
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_path)
        else:
            config = AutoConfig.from_pretrained(model_path)
            self.backbone = AutoModel.from_config(config)
        self.hidden_dim = int(self.backbone.config.hidden_size)
        out_dim = cfg.num_labels * self.num_classes_per_label
        self.head_type = cfg.head_type
        if self.head_type == "attention":
            self.head = AttentionPoolHead(self.hidden_dim, out_dim, num_heads=cfg.attn_pool_heads)
        else:
            self.drop = nn.Dropout(p=cfg.dropout)
            self.classifier = nn.Linear(self.hidden_dim, out_dim)
            nn.init.trunc_normal_(self.classifier.weight, std=0.02)
            nn.init.zeros_(self.classifier.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        use_hidden = self.head_type == "attention"
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=use_hidden)
        if self.head_type == "attention":
            hidden = outputs.hidden_states[self.cfg.rad_dino_layer]
            feats = {
                "x_norm_clstoken": outputs.last_hidden_state[:, 0, :],
                "x_storage_tokens": hidden[:, :0, :],
                "x_norm_patchtokens": hidden[:, 1:, :],
            }
            out = self.head(feats)
        else:
            cls_token = outputs.last_hidden_state[:, 0, :]
            cls_token = self.drop(cls_token)
            out = self.classifier(cls_token)
        if self.num_classes_per_label == 3:
            return out.view(-1, self.cfg.num_labels, 3)
        return out

    def param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float,
    ) -> List[dict]:
        def split_decay(named: Iterable[tuple[str, nn.Parameter]]) -> tuple[list, list]:
            decay, no_decay = [], []
            for name, p in named:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1:
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        bb_decay, bb_no_decay = split_decay(self.backbone.named_parameters())
        if self.head_type == "attention":
            head_params = self.head.named_parameters()
        else:
            head_params = list(self.classifier.named_parameters())
        hd_decay, hd_no_decay = split_decay(head_params)

        groups = [
            {"params": bb_decay,    "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"},
            {"params": bb_no_decay, "lr": lr_backbone, "weight_decay": 0.0,          "name": "backbone_nodecay"},
            {"params": hd_decay,    "lr": lr_head,     "weight_decay": weight_decay, "name": "head_decay"},
            {"params": hd_no_decay, "lr": lr_head,     "weight_decay": 0.0,          "name": "head_nodecay"},
        ]
        return [g for g in groups if len(g["params"]) > 0]


# --------------------------------------------------------------------------- #
# Generic timm-loaded medical foundation model (Radio-DINO, OmniRad)
# --------------------------------------------------------------------------- #
class RadioDinoModel(nn.Module):
    """timm-loaded ViT for medical foundation models published with timm-style
    state dict keys (e.g. patch_embed.proj.weight, pos_embed, cls_token).

    Works for: hf_hub:Snarcy/OmniRad-base, hf_hub:Snarcy/RadioDino-b16, etc.
    Reuses the cfg.rad_dino_path config field; expected to start with `hf_hub:`.
    """

    def __init__(self, cfg: Config, pretrained: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes_per_label = 3 if cfg.target_type == "3class" else 1
        import timm
        self.backbone = timm.create_model(
            cfg.rad_dino_path,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            img_size=cfg.image_size,
            drop_path_rate=getattr(cfg, "drop_path_rate", 0.0),
        )
        self.hidden_dim = int(self.backbone.num_features)
        out_dim = cfg.num_labels * self.num_classes_per_label
        self.head_type = cfg.head_type
        if self.head_type == "attention":
            self.head = AttentionPoolHead(self.hidden_dim, out_dim, num_heads=cfg.attn_pool_heads)
        elif self.head_type == "mlp":
            self.head = MlpHead(self.hidden_dim, out_dim, hidden_dim=cfg.head_hidden_dim, dropout=cfg.dropout)
        else:
            self.drop = nn.Dropout(p=cfg.dropout)
            self.classifier = nn.Linear(self.hidden_dim, out_dim)
            nn.init.trunc_normal_(self.classifier.weight, std=0.02)
            nn.init.zeros_(self.classifier.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(pixel_values)  # (B, 1+N, D)
        if self.head_type == "attention":
            feats = {
                "x_norm_clstoken": tokens[:, 0, :],
                "x_storage_tokens": tokens[:, :0, :],
                "x_norm_patchtokens": tokens[:, 1:, :],
            }
            out = self.head(feats)
        elif self.head_type == "mlp":
            feats = {"x_norm_clstoken": tokens[:, 0, :]}
            out = self.head(feats)
        else:
            cls_token = tokens[:, 0, :]
            cls_token = self.drop(cls_token)
            out = self.classifier(cls_token)
        if self.num_classes_per_label == 3:
            return out.view(-1, self.cfg.num_labels, 3)
        return out

    def param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float,
    ) -> List[dict]:
        def split_decay(named: Iterable[tuple[str, nn.Parameter]]) -> tuple[list, list]:
            decay, no_decay = [], []
            for name, p in named:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1:
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        bb_decay, bb_no_decay = split_decay(self.backbone.named_parameters())
        if self.head_type in ("attention", "mlp"):
            head_params = self.head.named_parameters()
        else:
            head_params = list(self.classifier.named_parameters())
        hd_decay, hd_no_decay = split_decay(head_params)

        groups = [
            {"params": bb_decay,    "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"},
            {"params": bb_no_decay, "lr": lr_backbone, "weight_decay": 0.0,          "name": "backbone_nodecay"},
            {"params": hd_decay,    "lr": lr_head,     "weight_decay": weight_decay, "name": "head_decay"},
            {"params": hd_no_decay, "lr": lr_head,     "weight_decay": 0.0,          "name": "head_nodecay"},
        ]
        return [g for g in groups if len(g["params"]) > 0]


class OpenCLIPModel(nn.Module):
    """OpenCLIP ViT-L/14 vision encoder + 3-class CE head.

    Default checkpoint: `vit_large_patch14_clip_336.laion2b_ft_in12k_in1k`.
    Pretraining: contrastive image-text on LAION-2B (~2B image-text pairs,
    no medical data) → fine-tuned on ImageNet-12K → fine-tuned on ImageNet-1K.

    Different SSL paradigm from DINOv3 (self-distillation), SigLIP 2 (sigmoid
    contrastive), CXR Foundation (contrastive on chest X-rays), and EVA-02
    (masked image modeling). 5th distinct pretraining style for our ensemble.
    """

    def __init__(self, cfg: Config, pretrained: bool = True) -> None:
        super().__init__()
        import timm
        self.cfg = cfg
        self.num_classes_per_label = 3 if cfg.target_type == "3class" else 1
        model_id = getattr(
            cfg, "openclip_model_id",
            "vit_large_patch14_clip_336.laion2b_ft_in12k_in1k",
        )
        self.backbone = timm.create_model(
            model_id, pretrained=pretrained, num_classes=0,
            drop_path_rate=getattr(cfg, "drop_path_rate", 0.0),
        )
        self.hidden_dim = int(self.backbone.num_features)
        self.drop = nn.Dropout(p=cfg.dropout)
        out_dim = cfg.num_labels * self.num_classes_per_label
        self.classifier = nn.Linear(self.hidden_dim, out_dim)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(pixel_values)
        feats = self.drop(feats)
        logits = self.classifier(feats)
        if self.num_classes_per_label == 3:
            return logits.view(-1, self.cfg.num_labels, 3)
        return logits

    def param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float,
    ) -> List[dict]:
        def split_decay(named: Iterable[tuple[str, nn.Parameter]]) -> tuple[list, list]:
            decay, no_decay = [], []
            for name, p in named:
                if not p.requires_grad: continue
                if p.ndim <= 1: no_decay.append(p)
                else: decay.append(p)
            return decay, no_decay

        bb_decay, bb_no_decay = split_decay(self.backbone.named_parameters())
        hd_decay, hd_no_decay = split_decay(list(self.classifier.named_parameters()))
        groups = [
            {"params": bb_decay,    "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"},
            {"params": bb_no_decay, "lr": lr_backbone, "weight_decay": 0.0,          "name": "backbone_nodecay"},
            {"params": hd_decay,    "lr": lr_head,     "weight_decay": weight_decay, "name": "head_decay"},
            {"params": hd_no_decay, "lr": lr_head,     "weight_decay": 0.0,          "name": "head_nodecay"},
        ]
        return [g for g in groups if len(g["params"]) > 0]


class BiomedCLIPModel(nn.Module):
    """BiomedCLIP ViT-B/16 vision encoder + 3-class (or coral) CE head.

    Default checkpoint: `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`.
    Pretraining: CLIP-style contrastive on PMC-15M (15M biomedical image-text
    pairs from PubMed Central, biomedical domain, distinct paradigm from
    DINOv3/EVA-02/SigLIP/OpenCLIP). 224² input, 768-d trunk features.

    Loaded via open_clip; we use the trunk only (pre-CLIP-projection features).
    """

    def __init__(self, cfg: Config, pretrained: bool = True) -> None:
        super().__init__()
        import open_clip
        self.cfg = cfg
        if cfg.target_type == "3class":
            self.num_classes_per_label = 3
        elif cfg.target_type == "coral":
            self.num_classes_per_label = 2
        else:
            self.num_classes_per_label = 1
        model_id = getattr(
            cfg, "biomedclip_model_id",
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )
        oc_model, _, _ = open_clip.create_model_and_transforms(model_id)
        self.backbone = oc_model.visual.trunk  # timm ViT-B/16 trunk; outputs (B, 768)
        self.hidden_dim = int(self.backbone.num_features)
        self.drop = nn.Dropout(p=cfg.dropout)
        out_dim = cfg.num_labels * self.num_classes_per_label
        self.classifier = nn.Linear(self.hidden_dim, out_dim)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(pixel_values)
        feats = self.drop(feats)
        logits = self.classifier(feats)
        if self.num_classes_per_label in (3, 2):
            return logits.view(-1, self.cfg.num_labels, self.num_classes_per_label)
        return logits

    def param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float,
    ) -> List[dict]:
        def split_decay(named):
            decay, no_decay = [], []
            for name, p in named:
                if not p.requires_grad: continue
                if p.ndim <= 1: no_decay.append(p)
                else: decay.append(p)
            return decay, no_decay
        bb_d, bb_nd = split_decay(self.backbone.named_parameters())
        hd_d, hd_nd = split_decay(self.classifier.named_parameters())
        groups = [
            {"params": bb_d,  "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"},
            {"params": bb_nd, "lr": lr_backbone, "weight_decay": 0.0,          "name": "backbone_nodecay"},
            {"params": hd_d,  "lr": lr_head,     "weight_decay": weight_decay, "name": "head_decay"},
            {"params": hd_nd, "lr": lr_head,     "weight_decay": 0.0,          "name": "head_nodecay"},
        ]
        return [g for g in groups if len(g["params"]) > 0]


class EVA02Model(nn.Module):
    """EVA-02 ViT-Large vision encoder + 3-class CE head.

    EVA-02 was pretrained with masked image modeling (MIM) on Merged-38M
    (IN-22K + CC12M + CC3M + COCO + ADE20K + Object365 + OpenImages — no
    medical imaging, no CheXpert), then fine-tuned on IN-22K and IN-1K.
    Different SSL objective from DINOv3 (self-distillation), so adds
    architectural-and-pretraining diversity to the ensemble.

    Uses the `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` checkpoint
    (304M params, 448² native, 1024-d output).
    """

    def __init__(self, cfg: Config, pretrained: bool = True) -> None:
        super().__init__()
        import timm
        self.cfg = cfg
        if cfg.target_type == "3class":
            self.num_classes_per_label = 3
        elif cfg.target_type == "coral":
            self.num_classes_per_label = 2  # 2 cumulative threshold logits per label
        else:
            self.num_classes_per_label = 1
        model_id = getattr(
            cfg, "eva02_model_id",
            "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        )
        self.backbone = timm.create_model(
            model_id, pretrained=pretrained, num_classes=0,
            drop_path_rate=getattr(cfg, "drop_path_rate", 0.0),
        )
        self.hidden_dim = int(self.backbone.num_features)
        self.drop = nn.Dropout(p=cfg.dropout)
        out_dim = cfg.num_labels * self.num_classes_per_label
        self.classifier = nn.Linear(self.hidden_dim, out_dim)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(pixel_values)  # (B, hidden_dim)
        feats = self.drop(feats)
        logits = self.classifier(feats)
        if self.num_classes_per_label in (3, 2):
            return logits.view(-1, self.cfg.num_labels, self.num_classes_per_label)
        return logits

    def param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float,
    ) -> List[dict]:
        def split_decay(named: Iterable[tuple[str, nn.Parameter]]) -> tuple[list, list]:
            decay, no_decay = [], []
            for name, p in named:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1:
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        llrd_decay = float(getattr(self.cfg, "llrd_decay", 0.0))
        if llrd_decay > 0.0 and llrd_decay < 1.0:
            return self._llrd_param_groups(lr_backbone, lr_head, weight_decay, llrd_decay)

        bb_decay, bb_no_decay = split_decay(self.backbone.named_parameters())
        hd_decay, hd_no_decay = split_decay(list(self.classifier.named_parameters()))

        groups = [
            {"params": bb_decay,    "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"},
            {"params": bb_no_decay, "lr": lr_backbone, "weight_decay": 0.0,          "name": "backbone_nodecay"},
            {"params": hd_decay,    "lr": lr_head,     "weight_decay": weight_decay, "name": "head_decay"},
            {"params": hd_no_decay, "lr": lr_head,     "weight_decay": 0.0,          "name": "head_nodecay"},
        ]
        return [g for g in groups if len(g["params"]) > 0]

    def _llrd_param_groups(self, lr_backbone, lr_head, weight_decay, decay):
        """Layer-wise LR decay for ViT backbones.

        For timm's eva02_large model, params are named like 'patch_embed.*',
        'cls_token', 'pos_embed', 'blocks.{i}.*', 'norm.*'. We assign:
          - patch_embed/cls/pos: layer 0
          - blocks.i: layer i+1
          - norm/head: deepest layer (full LR)
        Then lr_layer = lr_backbone * decay^(N - layer).
        Head gets lr_head as before.
        """
        import re
        blocks = getattr(self.backbone, "blocks", None)
        num_blocks = len(blocks) if blocks is not None else 24
        # Total backbone "depth" = patch_embed (0) + N blocks + final norm
        N = num_blocks + 1  # last layer index

        def layer_id(name: str) -> int:
            if name.startswith(("patch_embed", "cls_token", "pos_embed")):
                return 0
            m = re.match(r"blocks\.(\d+)\.", name)
            if m:
                return int(m.group(1)) + 1
            # norm, etc.
            return N

        groups = []
        # Build per-layer groups (decay vs no-decay × N+1 layers)
        per_layer: dict[int, dict[str, list]] = {}
        for name, p in self.backbone.named_parameters():
            if not p.requires_grad:
                continue
            lid = layer_id(name)
            if lid not in per_layer:
                per_layer[lid] = {"decay": [], "no_decay": []}
            (per_layer[lid]["decay"] if p.ndim > 1 else per_layer[lid]["no_decay"]).append(p)

        for lid in sorted(per_layer.keys()):
            lr_l = lr_backbone * (decay ** (N - lid))
            if per_layer[lid]["decay"]:
                groups.append({"params": per_layer[lid]["decay"], "lr": lr_l,
                               "weight_decay": weight_decay, "name": f"bb_L{lid}_decay"})
            if per_layer[lid]["no_decay"]:
                groups.append({"params": per_layer[lid]["no_decay"], "lr": lr_l,
                               "weight_decay": 0.0, "name": f"bb_L{lid}_nodecay"})

        # Head at full lr_head
        hd_decay, hd_no_decay = [], []
        for name, p in self.classifier.named_parameters():
            if not p.requires_grad: continue
            (hd_decay if p.ndim > 1 else hd_no_decay).append(p)
        if hd_decay:
            groups.append({"params": hd_decay,    "lr": lr_head, "weight_decay": weight_decay, "name": "head_decay"})
        if hd_no_decay:
            groups.append({"params": hd_no_decay, "lr": lr_head, "weight_decay": 0.0,          "name": "head_nodecay"})
        return groups


class SigLIP2Model(nn.Module):
    """SigLIP 2 SO400M image encoder + 3-class CE head.

    Supports two variants:
    - "patch14_384": fixed 384x384 input
    - "naflex_512":  NaFlex with max_num_patches=1024 (=> 512x512 max for patch16)

    For NaFlex, the pixel values are pre-patched flat tensors of shape
    (B, 1024, 768) plus attention_mask (B, 1024) and spatial_shapes (B, 2).
    Our dataset normally produces (B, 3, H, W). For NaFlex we do the patching
    in-model on the fly so the rest of the pipeline is unaffected.
    """

    def __init__(self, cfg: Config, pretrained: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes_per_label = 3 if cfg.target_type == "3class" else 1
        variant = getattr(cfg, "siglip2_variant", "patch14_384")
        self.variant = variant

        from transformers import AutoModel
        if variant == "patch14_384":
            model_id = "google/siglip2-so400m-patch14-384"
            self.expected_size = 384
            self.is_naflex = False
            self.patch_size = 14
        elif variant == "naflex_512":
            model_id = "google/siglip2-so400m-patch16-naflex"
            self.expected_size = 512
            self.is_naflex = True
            self.patch_size = 16
            self.num_patches_side = 32  # 512 / 16
        else:
            raise ValueError(f"unknown siglip2_variant: {variant}")

        full = AutoModel.from_pretrained(model_id)
        self.backbone = full.vision_model
        self.hidden_dim = int(self.backbone.config.hidden_size)
        self.drop = nn.Dropout(p=cfg.dropout)
        out_dim = cfg.num_labels * self.num_classes_per_label
        self.classifier = nn.Linear(self.hidden_dim, out_dim)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        if self.is_naflex:
            self.register_buffer(
                "spatial_shapes_const",
                torch.tensor([[self.num_patches_side, self.num_patches_side]], dtype=torch.long),
                persistent=False,
            )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (B, 3, H, W) -> (B, num_patches, patch*patch*3) for NaFlex."""
        B, C, H, W = x.shape
        ps = self.patch_size
        # B, C, H, W -> B, num_patches_h, num_patches_w, C, ps, ps
        x = x.unfold(2, ps, ps).unfold(3, ps, ps)  # (B, C, nh, nw, ps, ps)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, nh, nw, C, ps, ps)
        x = x.view(B, -1, C * ps * ps)  # (B, num_patches, C*ps*ps)
        return x

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if not self.is_naflex:
            # Standard 384x384 input: pass directly
            out = self.backbone(pixel_values=pixel_values)
            pooled = out.pooler_output  # (B, hidden_dim)
        else:
            # NaFlex 512: patchify on the fly + provide spatial_shapes + attention_mask
            B = pixel_values.size(0)
            patches = self._patchify(pixel_values)  # (B, 1024, 768)
            num_patches = patches.shape[1]
            spatial_shapes = self.spatial_shapes_const.expand(B, -1)
            attention_mask = torch.ones(
                B, num_patches, dtype=torch.int32, device=pixel_values.device
            )
            out = self.backbone(
                pixel_values=patches,
                attention_mask=attention_mask,
                spatial_shapes=spatial_shapes,
            )
            pooled = out.pooler_output

        pooled = self.drop(pooled)
        logits = self.classifier(pooled)
        if self.num_classes_per_label == 3:
            return logits.view(-1, self.cfg.num_labels, 3)
        return logits

    def param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float,
    ) -> List[dict]:
        def split_decay(named: Iterable[tuple[str, nn.Parameter]]) -> tuple[list, list]:
            decay, no_decay = [], []
            for name, p in named:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1:
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        bb_decay, bb_no_decay = split_decay(self.backbone.named_parameters())
        hd_decay, hd_no_decay = split_decay(list(self.classifier.named_parameters()))

        groups = [
            {"params": bb_decay,    "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"},
            {"params": bb_no_decay, "lr": lr_backbone, "weight_decay": 0.0,          "name": "backbone_nodecay"},
            {"params": hd_decay,    "lr": lr_head,     "weight_decay": weight_decay, "name": "head_decay"},
            {"params": hd_no_decay, "lr": lr_head,     "weight_decay": 0.0,          "name": "head_nodecay"},
        ]
        return [g for g in groups if len(g["params"]) > 0]


# --------------------------------------------------------------------------- #
# factory
class XRVDenseNet121Model(nn.Module):
    """DenseNet-121 with TorchXRayVision MIMIC-CXR pretrained weights.

    The backbone was pretrained on chest X-rays only (no CheXpert in training).
    Expects 1-channel grayscale input normalized to [-1024, 1024] (xrv convention).
    We accept our standard 3-channel ImageNet-normalized input and convert
    in-model so the rest of the pipeline doesn't need to change.
    """

    def __init__(self, cfg: Config, pretrained: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes_per_label = 3 if cfg.target_type == "3class" else 1
        # Build a torchvision DenseNet121 features module with 1-channel conv0.
        # This avoids importing torchxrayvision at train time (its submodule
        # baseline_models/jfhealthcare collides with our top-level model.py).
        import torchvision.models as tvm
        backbone = tvm.densenet121(weights=None)
        features = backbone.features
        old_conv0 = features.conv0
        features.conv0 = nn.Conv2d(
            1, old_conv0.out_channels,
            kernel_size=old_conv0.kernel_size,
            stride=old_conv0.stride,
            padding=old_conv0.padding,
            bias=False,
        )
        self.features = features
        if pretrained:
            xrv_weight_path = getattr(
                cfg, "xrv_weights_path",
                "/data/artifacts/frank/misc/xrv_densenet121_mimic_ch_features.pt",
            )
            sd = torch.load(xrv_weight_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.features.load_state_dict(sd, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"xrv weight load mismatch: missing={missing[:3]}... "
                    f"unexpected={unexpected[:3]}..."
                )
        self.hidden_dim = 1024
        self.drop = nn.Dropout(p=cfg.dropout)
        out_dim = cfg.num_labels * self.num_classes_per_label
        self.classifier = nn.Linear(self.hidden_dim, out_dim)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        # ImageNet stats used by our dataloader; we undo them before applying xrv norm.
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _imagenet_to_xrv(self, x: torch.Tensor) -> torch.Tensor:
        # 3-channel ImageNet-normalized -> 1-channel xrv-normalized
        # Step 1: undo ImageNet norm -> [0, 1] range
        x = x * self.imagenet_std + self.imagenet_mean
        # Step 2: average to 1 channel (chest X-rays are grayscale)
        x = x.mean(dim=1, keepdim=True)
        # Step 3: map [0, 1] -> [-1024, 1024] (xrv.datasets.normalize, maxval=1)
        x = (2.0 * x - 1.0) * 1024.0
        return x

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self._imagenet_to_xrv(pixel_values)
        feats = self.features(x)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1))
        feats = torch.flatten(feats, 1)
        feats = self.drop(feats)
        out = self.classifier(feats)
        if self.num_classes_per_label == 3:
            return out.view(-1, self.cfg.num_labels, 3)
        return out

    def param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float,
    ) -> List[dict]:
        def split_decay(named: Iterable[tuple[str, nn.Parameter]]) -> tuple[list, list]:
            decay, no_decay = [], []
            for name, p in named:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1:
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        bb_decay, bb_no_decay = split_decay(self.features.named_parameters())
        hd_decay, hd_no_decay = split_decay(list(self.classifier.named_parameters()))

        groups = [
            {"params": bb_decay,    "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"},
            {"params": bb_no_decay, "lr": lr_backbone, "weight_decay": 0.0,          "name": "backbone_nodecay"},
            {"params": hd_decay,    "lr": lr_head,     "weight_decay": weight_decay, "name": "head_decay"},
            {"params": hd_no_decay, "lr": lr_head,     "weight_decay": 0.0,          "name": "head_nodecay"},
        ]
        return [g for g in groups if len(g["params"]) > 0]


# --------------------------------------------------------------------------- #
# Backward-compatible alias used by train.py and submit.py
def CheXpertModel(cfg: Config, pretrained: bool = True) -> nn.Module:
    if cfg.model_type == "densenet121":
        return DenseNet121Model(cfg, pretrained=pretrained)
    elif cfg.model_type in ("convnext_base", "convnext_small"):
        return ConvNeXtModel(cfg, pretrained=pretrained)
    elif cfg.model_type == "dinov3":
        return CheXpertDINOv3Model(cfg)
    elif cfg.model_type == "rad_dino":
        return RadDinoModel(cfg, pretrained=pretrained)
    elif cfg.model_type == "radio_dino":
        return RadioDinoModel(cfg, pretrained=pretrained)
    elif cfg.model_type == "xrv_densenet121":
        return XRVDenseNet121Model(cfg, pretrained=pretrained)
    elif cfg.model_type == "siglip2":
        return SigLIP2Model(cfg, pretrained=pretrained)
    elif cfg.model_type == "eva02":
        return EVA02Model(cfg, pretrained=pretrained)
    elif cfg.model_type == "openclip":
        return OpenCLIPModel(cfg, pretrained=pretrained)
    elif cfg.model_type == "biomedclip":
        return BiomedCLIPModel(cfg, pretrained=pretrained)
    else:
        raise ValueError(f"unknown model_type: {cfg.model_type!r}")
