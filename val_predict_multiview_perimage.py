"""Per-image MV val predictions matching submit_multiview's behavior.

Iterates ALL val rows (frontal AND lateral), pairing each with same-study counterpart.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from config import Config, LABEL_NAMES
from dataset import load_and_split, build_val_transform
from model import CheXpertModel
from multiview import MultiViewModel

_PID_STUDY_RE = re.compile(r"(pid\d+)/(study\d+)")
def _pid_study(path):
    m = _PID_STUDY_RE.search(str(path))
    return (m.group(1), m.group(2)) if m else None


class ValPerImageMV(Dataset):
    """Iterate ALL val rows. For each, find frontal+lateral from same study."""
    def __init__(self, df_val, y_val, data_root, transform):
        df = df_val.reset_index(drop=True).copy()
        is_frontal = (df["Frontal/Lateral"] == "Frontal").to_numpy()
        is_lateral = (df["Frontal/Lateral"] == "Lateral").to_numpy()
        df["is_frontal"] = is_frontal
        df["is_lateral"] = is_lateral
        self.df = df
        self.y = y_val
        self.data_root = Path(data_root)
        self.transform = transform

        self._study_frontals = {}
        self._study_laterals = {}
        for i in np.where(is_frontal)[0]:
            ps = _pid_study(df.loc[i, "Path"])
            if ps:
                self._study_frontals.setdefault(ps, []).append(int(i))
        for i in np.where(is_lateral)[0]:
            ps = _pid_study(df.loc[i, "Path"])
            if ps:
                self._study_laterals.setdefault(ps, []).append(int(i))

    def __len__(self):
        return len(self.df)

    def _load(self, idx):
        with Image.open(self.data_root / self.df.loc[idx, "Path"]) as img:
            return self.transform(img.convert("RGB"))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_row = self._load(idx)
        ps = _pid_study(row["Path"])
        frontals = self._study_frontals.get(ps, []) if ps else []
        laterals = self._study_laterals.get(ps, []) if ps else []

        if row["is_frontal"]:
            x_front = x_row
        else:
            x_front = self._load(frontals[0]) if frontals else x_row

        if laterals:
            x_lat = x_row if row["is_lateral"] else self._load(laterals[0])
            lat_present = torch.tensor(1.0)
        else:
            x_lat = torch.zeros_like(x_front)
            lat_present = torch.tensor(0.0)

        y = torch.from_numpy(np.array(self.y[idx]))
        return x_front, x_lat, lat_present, y, idx


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
    val_ds = ValPerImageMV(df_val, y_val, cfg.data_root, build_val_transform(cfg))
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    N = len(val_ds)
    all_preds = np.zeros((N, len(LABEL_NAMES)), dtype=np.float32)
    all_paths = np.array([val_ds.df.loc[i, "Path"] for i in range(N)], dtype=object)
    all_labels = np.array([val_ds.y[i] for i in range(N)])

    with torch.no_grad():
        for x_f, x_l, lp, y, idxs in loader:
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
            for k, idx in enumerate(idxs):
                all_preds[int(idx)] = pred[k]

    # Convert 3class int labels back to {-1,0,1,nan}
    if cfg.target_type == "3class":
        raw_labels = np.full_like(all_preds, np.nan, dtype=np.float32)
        raw_labels[all_labels == 0] = -1.0
        raw_labels[all_labels == 1] = 0.0
        raw_labels[all_labels == 2] = 1.0
    else:
        raw_labels = all_labels.astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, preds=all_preds, paths=all_paths, raw_labels=raw_labels)
    print(f"wrote {args.out}  preds={all_preds.shape}")


if __name__ == "__main__":
    main()
