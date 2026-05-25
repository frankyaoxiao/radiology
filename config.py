"""Training configuration for CheXpert 2023 × DINOv3 ViT-H+/16.

Source of truth is a YAML file passed to ``train.py --config <path>``.
This module defines the schema and defaults; any field not set in YAML
falls back to the default below. The resolved config is copied into the
run directory at training start for provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


# The 9 labels in train2023.csv, in the exact order expected by the submission CSV.
LABEL_NAMES: List[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Pneumonia",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


@dataclass
class Config:
    # --- run identity (REQUIRED in YAML; drives output paths) ---
    run_name: str = "default"
    runs_root: str = "/data/artifacts/frank/misc/runs"

    # --- data ---
    data_root: str = "/data/artifacts/frank/misc"
    labels_csv: str = "/data/artifacts/frank/misc/labels/train2023.csv"
    test_ids_csv: str = "/data/artifacts/frank/misc/labels/test_ids.csv"
    val_frac: float = 0.1
    split_seed: int = 42

    # --- image ---
    image_size: int = 768
    crop_scale_min: float = 0.85
    crop_scale_max: float = 1.00
    rotation_deg: float = 10.0
    brightness: float = 0.2
    contrast: float = 0.2
    hflip: bool = False

    # --- preprocessing ---
    clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8

    # --- multi-view blending ---
    multiview_blend: bool = False
    multiview_blend_prob: float = 0.3
    multiview_blend_alpha_min: float = 0.3
    multiview_blend_alpha_max: float = 0.7

    # --- labels ---
    # label_names: every label the MODEL is trained on. Default is the 9 scored
    # labels. For 14-label aux training, set this to the full 14-label list
    # (9 scored + 5 aux). num_labels must match len(label_names).
    label_names: List[str] = field(default_factory=lambda: list(LABEL_NAMES))
    num_labels: int = 9
    # scored_label_names: the subset of label_names that the LEADERBOARD cares
    # about. Used for val metric computation and submission CSV output. If
    # None, falls back to label_names (existing 9-label behavior). For 14-label
    # aux training, set this to the 9 scored labels.
    scored_label_names: Optional[List[str]] = None
    # Train defaults to the original U-Ones setup: uncertain -> 1,
    # blank/unmentioned -> 0. For score-aligned experiments, set
    # default_uncertain_strategy="ignore" and blank_strategy="ignore".
    #
    # Val always masks uncertain labels. blank_strategy controls whether
    # blank/unmentioned val labels are counted as negatives ("zeros", legacy)
    # or excluded from metrics ("ignore", score-aligned).
    #
    # Per-label uncertain handling: a dict mapping label name to strategy.
    # Strategies: "ones" (uncertain→1), "zeros" (uncertain→0),
    #             "ignore" (uncertain→nan, masked from loss).
    # Labels not listed default to default_uncertain_strategy.
    # Example: {"Pneumonia": "ignore", "Pleural Other": "ignore"}
    default_uncertain_strategy: str = "ones"
    blank_strategy: str = "zeros"
    uncertain_strategy: Optional[dict] = None

    # --- model ---
    # model_type: "dinov3" (original ViT-H+) or "densenet121" (torchvision).
    model_type: str = "dinov3"
    # DenseNet-121 / ConvNeXt / RAD-DINO / Radio-DINO head dropout
    dropout: float = 0.5
    # Optional dropout in the DINOv3 ClsHead path (default 0 preserves existing
    # behavior; set to 0.3 to match RAD-DINO's recipe in the DINOv3 path).
    head_dropout: float = 0.0
    convnext_weights: str = ""
    # RAD-DINO (microsoft/rad-dino): HuggingFace model ID or local path.
    rad_dino_path: str = "microsoft/rad-dino"
    # XRV DenseNet121 pre-extracted features-only state_dict path (only used by
    # model_type="xrv_densenet121"). We avoid importing torchxrayvision at train
    # time because its submodule collides with our top-level model.py.
    # Pre-saved options (all legal, no CheXpert in pretraining):
    #   xrv_densenet121_mimic_ch_features.pt  (MIMIC-CXR — recommended)
    #   xrv_densenet121_mimic_nb_features.pt  (MIMIC-CXR, negative-binding variant)
    #   xrv_densenet121_nih_features.pt        (NIH only)
    xrv_weights_path: str = "/data/artifacts/frank/misc/xrv_densenet121_mimic_ch_features.pt"
    # SigLIP 2 variant: "patch14_384" or "naflex_512"
    siglip2_variant: str = "patch14_384"
    # Which hidden layer to use for patch tokens in attention pooling.
    # -1 = last layer, -2 = second-to-last, etc. Only used with head_type="attention".
    rad_dino_layer: int = -2
    # Native DINOv3 loader: we call torch.hub.load on a local clone of
    # facebookresearch/dinov3 and point at a downloaded .pth. The HF repo
    # is gated; bypassing HF avoids the access barrier.
    dinov3_repo: str = "/data/artifacts/frank/misc/dinov3_repo"
    dinov3_arch: str = "dinov3_vith16plus"
    dinov3_weights: str = "/data/artifacts/frank/misc/labels/dino/dinov3_hplus.pth"
    # Head type: "attention" (attention pooling over CLS + storage + patches),
    # "cls" (linear on CLS token only), or "mlp" (2-layer MLP on CLS token).
    head_type: str = "cls"
    attn_pool_heads: int = 8
    # Hidden dimension for the MLP head (only used when head_type == "mlp").
    head_hidden_dim: int = 512

    # --- optimization ---
    epochs: int = 5
    batch_size_per_gpu: int = 8  # global 32 on 4× H100 (bs=16 OOMs at 768²)
    num_workers: int = 8
    lr_head: float = 2e-5
    lr_backbone: float = 1e-6
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    mixup_alpha: float = 0.1  # Beta(a, a); 0 disables
    # "micro" matches the original code: average BCE across every valid
    # (sample, label) element. "macro" averages valid elements per label,
    # then averages labels equally; useful when blanks are masked and label
    # density differs by orders of magnitude.
    loss_reduction: str = "micro"
    # Label smoothing for 3-class CE (and the soft-target mixup path).
    # 0.0 = no smoothing (default); 0.1 = standard; 0.2 = aggressive.
    label_smoothing: float = 0.0
    # Focal loss exponent for 3-class CE. 0 = standard CE; 1-3 = focal modulation.
    # focal_gamma > 0: per-sample loss is multiplied by (1 - p_true_class)^gamma.
    focal_gamma: float = 0.0
    # Auxiliary MSE loss weight on (P(+1) - P(-1)) vs raw targets, alongside 3-class CE.
    # 0.0 = pure CE (default). Total = (1 - aux_mse_weight) * CE + aux_mse_weight * MSE.
    aux_mse_weight: float = 0.0
    # Augmentation pipeline: "basic" (default crop+rotate+colorjitter), "randaug_light",
    # "randaug_std", "randaug_strong", or "trivial" (TrivialAugmentWide).
    augmentation: str = "basic"
    # Optional pseudo-label CSV merged into the train set during training.
    # Path schema is the same as labels_csv (with Path + 9 label columns).
    # data_root for these images comes from data_root unless pseudo_label_data_root is set.
    pseudo_label_csv: str = ""
    pseudo_label_data_root: str = ""
    # "binary": train targets are 0/1, loss is BCE, submit sigmoid probs.
    # "raw": train targets are -1/0/1, loss is MSE, submit clipped linear outputs.
    # "3class": train targets are class indices {0=-1, 1=0, 2=+1}, loss is per-label CE,
    #           submit P(+1) - P(-1).
    target_type: str = "binary"
    # "bce" (default for binary), "mse" (default for raw), "smooth_l1", "ce" (for 3class)
    loss_fn: str = "bce"
    # Per-label loss weights. If set, multiply each label's loss by its weight.
    # Map from label name to float. Labels not listed get weight 1.0.
    label_weights: Optional[dict] = None
    # In raw target mode: list of label names whose uncertain (0) values
    # should be masked from loss. Useful for labels like Pneumonia where
    # uncertain dominates and raw MSE learns "predict 0" instead of
    # useful pos/neg separation.
    raw_uncertain_mask: Optional[List[str]] = None
    # Optimizer: "adamw" (default), "rmsprop", or "muon"
    # Muon uses orthogonalized momentum for 2D backbone weights; non-2D
    # params (biases, layernorms, embeddings, head) go to AdamW.
    optimizer: str = "adamw"
    rmsprop_momentum: float = 0.9
    rmsprop_alpha: float = 0.99
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    # EMA (exponential moving average of weights)
    ema: bool = False
    ema_decay: float = 0.999
    # Layer-wise LR decay (LLRD). If > 0 and < 1, applies per-block decay so
    # early layers get a smaller LR than later layers. Standard fine-tuning
    # trick for large pretrained models (ViTs especially). lr_backbone is the
    # LR for the LAST backbone block; earlier blocks get lr_backbone * decay^k.
    # 0.0 (default) disables.
    llrd_decay: float = 0.0

    # Stochastic depth (drop_path) for ViT-style backbones. timm models
    # (EVA-02, SigLIP-2, OpenCLIP, ConvNeXt) accept drop_path_rate at construction.
    # 0.0 (default) disables. Typical values: 0.1 for ViT-L, 0.2 for ViT-H.
    drop_path_rate: float = 0.0

    # Optional warm-start: load model weights ONLY from this ckpt at init
    # (NOT optimizer/scheduler/step). Used for two-stage training:
    # stage 1 trains on external data (e.g., NIH), stage 2 inits from stage 1
    # and trains on CheXpert from scratch (fresh LR schedule).
    init_from_ckpt: str = ""

    # Lung-crop: path to a CSV (Path, x0, y0, x1, y1) of per-image lung bounding
    # boxes. If set, images are cropped to the lung region before augmentation.
    # Used to train a pulmonary-label specialist on lung-focused crops.
    lung_crop_csv: str = ""

    # Multi-view fusion: head architecture for combining frontal + lateral features.
    # "mlp" (default): LayerNorm -> Linear(2D, D) -> GELU -> Dropout -> Linear(D, out)
    # "linear":        Dropout -> Linear(2D, out)
    # "sum_linear":    sum features -> Dropout -> Linear(D, out)
    mv_head_kind: str = "mlp"

    # If true, multi-view training filters to only paired (frontal+lateral) studies
    # (~17% of training data). Forces the model to learn cross-view fusion.
    mv_paired_only: bool = False

    # --- eval / ckpt ---
    eval_every_steps: int = 500
    # "nmse" (lower is better, matches judge's scoring) or "auroc"
    # (higher is better, standard CheXpert convention).
    primary_metric: str = "nmse"

    # --- misc ---
    seed: int = 0
    log_every_steps: int = 20
    # Smoke / debug knobs. Leave 0 for full runs.
    max_train_samples: int = 0  # if >0, slice train set to first N rows
    max_val_samples: int = 0    # if >0, slice val set to first N rows
    max_steps: int = 0          # if >0, stop training after this many steps

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @property
    def run_dir(self) -> Path:
        return Path(self.runs_root) / self.run_name

    @property
    def effective_scored_labels(self) -> List[str]:
        """Labels used for val metrics and submission output.

        Falls back to label_names when scored_label_names is not set, so
        existing 9-label configs behave identically to before.
        """
        return list(self.scored_label_names) if self.scored_label_names else list(self.label_names)

    @property
    def scored_indices(self) -> List[int]:
        """Indices of effective_scored_labels within label_names.

        Used to slice model outputs (num_labels wide) down to the scored
        subset for metrics and submission.
        """
        scored = self.effective_scored_labels
        name_to_idx = {n: i for i, n in enumerate(self.label_names)}
        missing = [n for n in scored if n not in name_to_idx]
        if missing:
            raise ValueError(
                f"scored_label_names contains labels not in label_names: {missing}"
            )
        return [name_to_idx[n] for n in scored]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        unknown = set(data) - {f.name for f in cls.__dataclass_fields__.values()}
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def save_yaml(self, path: str | Path) -> None:
        import yaml
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
