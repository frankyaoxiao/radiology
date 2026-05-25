"""Re-extract Google CXR Foundation features WITH spatial structure (40x40x1376
instead of GAP'd 1376). Uses m.prune() to bypass the batch=1 input pipeline.

Per-image still ~160ms on H100 (model has internal serialization), but we get
full spatial features for spatial-pool head training.

Output per shard: npz with (N, 40, 40, 1376) features.
"""
from __future__ import annotations

import argparse
import io
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image


def _preprocess_image(img: Image.Image, target_size: int = 1280) -> np.ndarray:
    """Replicate the SavedModel's per-image preprocessing in PyTorch/numpy:
    grayscale -> resize bilinear to target_size -> min-max norm to [0, 1]
    -> tile to 3 channels (grayscale_to_rgb).
    Returns (1, target_size, target_size, 3) float32.
    """
    img = img.convert("L")
    img = img.resize((target_size, target_size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32)
    # Min-max normalize per-image (matches the SavedModel's cond/truediv path)
    a_min, a_max = a.min(), a.max()
    if a_max > a_min:
        a = (a - a_min) / (a_max - a_min)
    else:
        a = a * 0  # all-zero image
    a = a[None, :, :, None]  # (1, H, W, 1)
    a = np.tile(a, (1, 1, 1, 3))  # grayscale_to_rgb
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--paths-csv", type=str, required=True)
    p.add_argument("--model-dir", type=str,
                   default="/data/artifacts/hf_cache/hub/models--google--cxr-foundation/snapshots/e5af8ea44a17bad5504f7e485388d6b05786860f")
    p.add_argument("--out-dir", type=str,
                   default="/data/artifacts/frank/misc/cxr_embeds_spatial")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4,
                   help="GPU batch size; larger doesn't help much (model has internal serial ops)")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    import tensorflow as tf
    import tensorflow_text  # noqa: F401
    import pandas as pd

    t0 = time.time()
    paths = pd.read_csv(args.paths_csv)
    my_paths = paths.iloc[args.rank::args.world].reset_index(drop=True)
    n_local = len(my_paths)
    print(f"[rank {args.rank}/{args.world} gpu {args.gpu}] {n_local} images, batch={args.batch_size}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{args.rank:02d}_of_{args.world:02d}.npz"
    if out_path.exists():
        print(f"[rank {args.rank}] shard exists, skipping", flush=True)
        return

    print(f"[rank {args.rank}] loading elixr_c ...", flush=True)
    m = tf.saved_model.load(f"{args.model_dir}/elixr-c-v2-pooled")
    pruned = m.prune(
        feeds=["EfficientNet/grayscale_to_rgb:0"],
        fetches=["EfficientNet/dropout/Identity:0"],
    )
    print(f"[rank {args.rank}] model loaded in {time.time()-t0:.1f}s", flush=True)

    # Warmup
    dummy = tf.zeros([args.batch_size, 1280, 1280, 3], dtype=tf.float32)
    _ = pruned(dummy)
    print(f"[rank {args.rank}] warmup done", flush=True)

    # Allocate output: (N_local, 40, 40, 1376) is huge — 200K images × 40 × 40 × 1376 × 4 bytes = 1.7TB total
    # Per shard: 25K × 40 × 40 × 1376 × 4 = 220GB. WAY too much.
    # Compromise: save GAP (1376), max-pool (1376), and std (1376) = 4128-d per image
    # Plus the central 5x5 region's mean = 1376 (focuses on lung area assuming centered patient)
    # Total: 5504-d per image = 25K × 5504 × 4 = 0.55GB per shard. Tractable.
    n_feat = 1376 * 4  # GAP + max + std + center5x5_mean
    feats = np.empty((n_local, n_feat), dtype=np.float32)
    idxs = np.empty(n_local, dtype=np.int64)

    last_log = time.time()
    start = time.time()
    bs = args.batch_size

    i = 0
    while i < n_local:
        end = min(i + bs, n_local)
        batch_imgs = []
        batch_idxs = []
        for j in range(i, end):
            row = my_paths.iloc[j]
            try:
                img = Image.open(row["path"])
                pre = _preprocess_image(img)  # (1, 1280, 1280, 3)
                batch_imgs.append(pre[0])
                batch_idxs.append(int(row["idx"]))
            except Exception as e:
                print(f"[rank {args.rank}] image {j} failed: {e}", flush=True)
                batch_imgs.append(np.zeros((1280, 1280, 3), dtype=np.float32))
                batch_idxs.append(int(row["idx"]))

        x = tf.constant(np.stack(batch_imgs, axis=0), dtype=tf.float32)
        try:
            out = pruned(x)
            spatial = out[0] if isinstance(out, (list, tuple)) else out  # (B, 40, 40, 1376)
            spatial_np = spatial.numpy()
        except Exception as e:
            print(f"[rank {args.rank}] batch {i}-{end} failed at inference: {e}", flush=True)
            spatial_np = np.zeros((end - i, 40, 40, 1376), dtype=np.float32)

        # Compute 4 pooling statistics per image
        for k in range(end - i):
            sm = spatial_np[k]  # (40, 40, 1376)
            gap = sm.reshape(-1, 1376).mean(axis=0)
            max_pool = sm.reshape(-1, 1376).max(axis=0)
            std_pool = sm.reshape(-1, 1376).std(axis=0)
            # Central 20x20 region (lungs typically centered)
            center = sm[10:30, 10:30, :].reshape(-1, 1376).mean(axis=0)
            feats[i + k] = np.concatenate([gap, max_pool, std_pool, center])
            idxs[i + k] = batch_idxs[k]

        if time.time() - last_log > 30:
            elapsed = time.time() - start
            rate = (end) / elapsed
            eta = (n_local - end) / rate
            print(f"[rank {args.rank}] {end}/{n_local} ({rate:.1f}/s, eta {eta/60:.1f}min)", flush=True)
            last_log = time.time()
        i = end

    np.savez_compressed(out_path, feats=feats, idxs=idxs)
    print(f"[rank {args.rank}] DONE -> {out_path}  ({time.time()-start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
