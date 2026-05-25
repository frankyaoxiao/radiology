"""Run lung segmentation on train + test images, save bbox coordinates.

Output CSV columns: Path, x0, y0, x1, y1, mask_area, used_fallback.
"""
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from PIL import Image

import torchxrayvision as xrv
from torchxrayvision.datasets import XRayResizer


SEG = None
RESIZE = XRayResizer(512)


def _load_seg(device):
    global SEG
    SEG = xrv.baseline_models.chestx_det.PSPNet().eval().to(device)


def _preprocess_one(path, data_root):
    """Open image, normalize, resize to 512. Returns (np array [1,512,512], W, H, path)
    or None if the file is missing/unreadable (caller skips it)."""
    full = Path(data_root) / path
    try:
        with Image.open(full) as img:
            img = img.convert("L")
            W, H = img.size
            arr = np.array(img)
    except (FileNotFoundError, OSError):
        return None
    arr = xrv.datasets.normalize(arr, 255)[None]  # (1, h, w)
    t = RESIZE(arr)  # (1, 512, 512)
    return t.astype(np.float32), W, H, path


def _bbox_from_logits(out_one, W, H, pad_frac=0.05, min_frac=0.25):
    """out_one: (14, 512, 512) tensor on cpu. Returns (x0,y0,x1,y1, mask_area, fallback)."""
    lung_logits = out_one[4] + out_one[5]
    mask = (lung_logits > 0).cpu().numpy()
    area_frac = mask.sum() / mask.size
    if area_frac < 0.02:
        return (0, 0, W, H, area_frac, True)
    ys, xs = np.where(mask)
    y0_5, y1_5 = ys.min(), ys.max()
    x0_5, x1_5 = xs.min(), xs.max()
    sx = W / 512.0
    sy = H / 512.0
    x0 = int(x0_5 * sx); x1 = int(x1_5 * sx)
    y0 = int(y0_5 * sy); y1 = int(y1_5 * sy)
    crop_w_frac = (x1 - x0) / W
    crop_h_frac = (y1 - y0) / H
    if crop_w_frac < min_frac or crop_h_frac < min_frac:
        return (0, 0, W, H, area_frac, True)
    pw = int((x1 - x0) * pad_frac)
    ph = int((y1 - y0) * pad_frac)
    x0 = max(0, x0 - pw); x1 = min(W, x1 + pw)
    y0 = max(0, y0 - ph); y1 = min(H, y1 + ph)
    return (x0, y0, x1, y1, float(area_frac), False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths-csv", required=True, type=Path,
                    help="CSV with a 'Path' column listing relative paths")
    ap.add_argument("--data-root", default="/data/artifacts/frank/misc")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _load_seg(device)

    paths_df = pd.read_csv(args.paths_csv)
    paths = paths_df["Path"].tolist()
    print(f"processing {len(paths)} paths", flush=True)

    out_rows = []
    pool = ThreadPoolExecutor(max_workers=args.num_workers)

    def submit_batch(start, end):
        return list(pool.map(lambda p: _preprocess_one(p, args.data_root), paths[start:end]))

    t0 = time.time()
    i = 0
    bs = args.batch_size
    # Prefetch the first batch
    next_future = pool.submit(submit_batch, 0, bs)
    while i < len(paths):
        raw_items = next_future.result()
        # Always advance the cursor by the batch size; prefetch the next bs paths.
        i += bs
        if i < len(paths):
            next_future = pool.submit(submit_batch, i, min(i + bs, len(paths)))
        # Drop any missing-file entries (None) before the GPU forward pass.
        items = [it for it in raw_items if it is not None]
        if not items:
            continue
        arrs = torch.from_numpy(np.stack([it[0] for it in items])).to(device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = SEG(arrs)  # (B, 14, 512, 512)
        out = out.float().cpu()
        for k, (_, W, H, p) in enumerate(items):
            x0, y0, x1, y1, area, fallback = _bbox_from_logits(out[k], W, H)
            out_rows.append({
                "Path": p, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "mask_area_frac": area, "used_fallback": fallback,
            })
        if i % 5000 < bs:
            el = time.time() - t0
            done = len(out_rows)
            print(f"  {i}/{len(paths)} ({done} ok)  {i/el:.1f} img/sec  ETA {(len(paths)-i)/(i/el)/60:.0f} min", flush=True)

    df = pd.DataFrame(out_rows)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  total {len(df)} rows", flush=True)
    print(f"fallback (full image): {df['used_fallback'].sum()} ({df['used_fallback'].mean()*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
