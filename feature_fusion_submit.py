"""Multi-view feature fusion submission.

Instead of averaging predictions (post-head), average CLS token features
(pre-head) across views of the same study. The classification head sees
combined frontal+lateral information simultaneously.

Usage:
    uv run python -u feature_fusion_submit.py \
        --ckpt ckpt0.pt ckpt1.pt ckpt2.pt \
        --test-ids-csv /path/to/test_ids.csv \
        --data-root /path/to/data \
        --out submission.csv \
        --frontal-weight 3.0 --lateral-weight 1.0
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import Config, LABEL_NAMES
from dataset import build_val_transform
from model import CheXpertModel
from submit import _set_deterministic


class FeatureDataset(Dataset):
    def __init__(self, df, data_root, transform):
        self.ids = df["Id"].tolist()
        self.paths = df["Path"].tolist()
        self.root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        with Image.open(self.root / self.paths[idx]) as img:
            img = img.convert("RGB")
            x = self.transform(img)
        return self.ids[idx], x


def extract_features_and_predict(model, loader, device, study_map, view_types,
                                  frontal_weight, lateral_weight):
    """Extract CLS features, average per study, predict through head."""
    model.eval()

    # Step 1: extract CLS features for all images
    all_ids = []
    all_features = []
    with torch.no_grad():
        for ids, x in loader:
            x = x.to(device, non_blocking=True)
            outputs = model.backbone(pixel_values=x)
            cls_tokens = outputs.last_hidden_state[:, 0, :]  # (B, 768)
            all_features.append(cls_tokens.cpu())
            all_ids.extend([int(i) for i in ids])

    all_features = torch.cat(all_features, dim=0)  # (N, 768)

    # Step 2: weighted average features per study
    fused_features = torch.zeros_like(all_features)
    for study, img_indices in study_map.items():
        if len(img_indices) == 1:
            fused_features[img_indices[0]] = all_features[img_indices[0]]
        else:
            weights = []
            feats = []
            for idx in img_indices:
                vt = view_types[idx]
                w = frontal_weight if vt == "frontal" else lateral_weight
                weights.append(w)
                feats.append(all_features[idx])
            weights = torch.tensor(weights, dtype=torch.float32)
            weights = weights / weights.sum()
            fused = sum(w * f for w, f in zip(weights, feats))
            for idx in img_indices:
                fused_features[idx] = fused

    # Step 3: pass fused features through head
    fused_features = fused_features.to(device)
    with torch.no_grad():
        if hasattr(model, 'drop'):
            out = model.classifier(model.drop(fused_features))
        elif hasattr(model, 'head'):
            # For attention pool head, just use linear proj
            out = model.head.proj(model.head.norm_out(fused_features))
        else:
            out = model.classifier(fused_features)

    if model.num_classes_per_label == 3:
        logits = out.view(-1, 9, 3).cpu().numpy()
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        preds = np.clip(probs[:, :, 2] - probs[:, :, 0], -1, 1)
    else:
        preds = out.cpu().numpy()

    return all_ids, preds


def main():
    _set_deterministic()

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, nargs="+", type=Path)
    ap.add_argument("--test-ids-csv", required=True, type=str)
    ap.add_argument("--data-root", required=True, type=str)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--frontal-weight", type=float, default=3.0)
    ap.add_argument("--lateral-weight", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.test_ids_csv)
    df["study"] = df["Path"].str.extract(r"(pid\d+/study\d+)")
    df["view_type"] = df["Path"].str.extract(r"(frontal|lateral)")

    # Build study -> image index mapping
    study_map = defaultdict(list)
    view_types = {}
    for i, row in df.iterrows():
        study_map[row["study"]].append(i)
        view_types[i] = row["view_type"]

    print(f"Test: {len(df)} images, {len(study_map)} studies", flush=True)
    print(f"Weights: frontal={args.frontal_weight}, lateral={args.lateral_weight}", flush=True)

    # Ensemble across checkpoints
    ensemble_preds = None
    for i, ckpt_path in enumerate(args.ckpt):
        print(f"\nCheckpoint {i+1}/{len(args.ckpt)}: {ckpt_path.name}", flush=True)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg_dict = ckpt["config"]
        known = {f.name for f in Config.__dataclass_fields__.values()}
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in known}
        sd_keys = set(ckpt["model"].keys())
        if "classifier.weight" in sd_keys and "head.query" not in sd_keys:
            cfg_dict["head_type"] = "cls"
        cfg = Config(**cfg_dict)

        model = CheXpertModel(cfg, pretrained=False)
        model.load_state_dict(ckpt["model"], strict=True)
        model.to(device)

        transform = build_val_transform(cfg)
        ds = FeatureDataset(df, cfg.data_root, transform)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

        ids, preds = extract_features_and_predict(
            model, loader, device, study_map, view_types,
            args.frontal_weight, args.lateral_weight
        )

        if ensemble_preds is None:
            ensemble_preds = preds
        else:
            ensemble_preds += preds
        del model, ckpt
        torch.cuda.empty_cache()

    ensemble_preds /= len(args.ckpt)
    ensemble_preds = np.clip(ensemble_preds, -1, 1)

    # Write submission
    labels = list(LABEL_NAMES)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id"] + labels)
        for sample_id, row in zip(ids, ensemble_preds):
            writer.writerow([int(sample_id)] + [f"{v:.6f}" for v in row])

    print(f"\nWrote {args.out} ({len(ids)} rows)", flush=True)
    for i, l in enumerate(labels):
        print(f"  {l:28s}  mean={np.mean(ensemble_preds[:, i]):+.4f}", flush=True)


if __name__ == "__main__":
    main()
