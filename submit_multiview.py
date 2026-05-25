"""Multi-view inference: load a multi-view model ckpt + run prediction on test.

For each test (Id, Path) row, find the matching frontal+lateral pair within
the same pid/study and run the multi-view model. Output a CSV with one row per
test Id, in the same format as submit.py.
"""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from config import Config, LABEL_NAMES
from dataset import build_val_transform
from model import CheXpertModel
from multiview import MultiViewModel


_PID_STUDY_RE = re.compile(r"(pid\d+)/(study\d+)")


def _pid_study(path: str):
    m = _PID_STUDY_RE.search(path)
    return (m.group(1), m.group(2)) if m else None


class MultiViewSubmitDataset(Dataset):
    """Iterate over test rows (Id, Path) but on __getitem__ return BOTH the
    row image and the matching lateral_or_null from the same study.
    """
    def __init__(self, df_test: pd.DataFrame, data_root: str | Path, transform):
        df_test = df_test.reset_index(drop=True).copy()
        df_test["pid_study"] = df_test["Path"].apply(_pid_study)
        df_test["is_frontal"] = df_test["Path"].str.contains("frontal", case=False)
        df_test["is_lateral"] = df_test["Path"].str.contains("lateral", case=False)
        self.df = df_test
        self.data_root = Path(data_root)
        self.transform = transform

        # Build study -> [(idx, "frontal"|"lateral", path)]
        self.study_to_rows = {}
        for i, row in df_test.iterrows():
            ps = row["pid_study"]
            if ps is None:
                continue
            view = "frontal" if row["is_frontal"] else ("lateral" if row["is_lateral"] else "unknown")
            self.study_to_rows.setdefault(ps, []).append((int(i), view, row["Path"]))

    def __len__(self):
        return len(self.df)

    def _load(self, path: str):
        full = self.data_root / path
        with Image.open(full) as img:
            return self.transform(img.convert("RGB"))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ps = row["pid_study"]
        path = row["Path"]

        # x_front: prefer the test row's image if it's frontal; otherwise use study's first frontal; else this row.
        x_row = self._load(path)
        # Look up frontal + lateral for this study
        study_rows = self.study_to_rows.get(ps, [])
        frontals = [p for _, v, p in study_rows if v == "frontal"]
        laterals = [p for _, v, p in study_rows if v == "lateral"]

        if row["is_frontal"]:
            x_front = x_row
        else:
            if frontals:
                x_front = self._load(frontals[0])
            else:
                x_front = x_row  # this is a lateral, no frontal; reuse it

        if laterals:
            # If the row is itself a lateral, prefer this row (for consistency)
            x_lat = x_row if row["is_lateral"] else self._load(laterals[0])
            lat_present = torch.tensor(1.0)
        else:
            x_lat = torch.zeros_like(x_front)
            lat_present = torch.tensor(0.0)

        return int(row["Id"]), x_front, x_lat, lat_present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(f"out exists: {args.out}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load checkpoint
    ck = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    cfg_dict = ck["config"]
    known = {f.name for f in Config.__dataclass_fields__.values()}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in known}
    cfg = Config(**cfg_dict)
    print(f"loaded ckpt step={ck.get('step')}  best_metric={ck.get('best_metric')}", flush=True)

    # Build inner backbone
    inner = CheXpertModel(cfg, pretrained=False)
    if hasattr(inner, "hidden_dim"):
        D = int(inner.hidden_dim)
    elif hasattr(inner.backbone, "num_features"):
        D = int(inner.backbone.num_features)
    else:
        D = 768  # fallback
    head_kind = getattr(cfg, "mv_head_kind", "mlp")
    model = MultiViewModel(
        inner_model=inner,
        hidden_dim=D,
        num_labels=cfg.num_labels,
        num_classes_per_label=(3 if cfg.target_type == "3class" else 1),
        dropout=cfg.dropout,
        head_kind=head_kind,
    )
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).eval()

    # Load test set
    df_test = pd.read_csv(cfg.test_ids_csv)
    print(f"test rows: {len(df_test):,}", flush=True)

    ds = MultiViewSubmitDataset(df_test, cfg.data_root, build_val_transform(cfg))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    all_ids, all_pred = [], []
    t0 = time.time()
    with torch.no_grad():
        for ids, x_f, x_l, lp in loader:
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
            all_pred.append(pred)
            all_ids.extend([int(i) for i in ids])
    preds = np.concatenate(all_pred, axis=0)
    print(f"inference done in {time.time()-t0:.1f}s  shape={preds.shape}", flush=True)

    # Build output CSV
    out_df = pd.DataFrame({"Id": all_ids})
    for i, l in enumerate(LABEL_NAMES):
        out_df[l] = preds[:, i]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False, float_format="%.6f")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
