"""Re-extract Google CXR Foundation features using TF preprocessing in Python
(bit-identical to original) + batched inference via prune() at the
EfficientNet/dropout/Identity tensor (post-EfficientNet, pre-broken-Reshape).

Saves the model's 8x8x1376 features per image, computed by manually
average-pooling the 40x40x1376 EfficientNet output with kernel=5 stride=5
(matches the saved model's internal avg pool exactly).
"""
from __future__ import annotations

import argparse
import io
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--paths-csv", type=str, required=True)
    p.add_argument("--model-dir", type=str,
                   default="/data/artifacts/hf_cache/hub/models--google--cxr-foundation/snapshots/e5af8ea44a17bad5504f7e485388d6b05786860f")
    p.add_argument("--out-dir", type=str,
                   default="/data/artifacts/frank/misc/cxr_embeds_v2")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
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
    print(f"[rank {args.rank}/{args.world} gpu {args.gpu}] {n_local} images bs={args.batch_size}", flush=True)

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
        fetches=["EfficientNet/dropout/Identity:0"],  # (B, 40, 40, 1376)
    )
    print(f"[rank {args.rank}] model loaded in {time.time()-t0:.1f}s", flush=True)

    def tf_preprocess(img_bytes):
        # Bit-identical to SavedModel input pipeline
        decoded = tf.image.decode_png(img_bytes, channels=1)
        resized = tf.compat.v1.image.resize_bilinear(
            decoded[None], [1280, 1280], align_corners=False, half_pixel_centers=False
        )[0]
        resized = tf.cast(resized, tf.float32)
        mn = tf.reduce_min(resized)
        mx = tf.reduce_max(resized)
        norm = tf.cond(tf.greater(mx, mn),
                       lambda: (resized - mn) / (mx - mn),
                       lambda: resized * 0)
        return tf.tile(norm[None], [1, 1, 1, 3])

    # Warmup
    dummy = tf.zeros([args.batch_size, 1280, 1280, 3], dtype=tf.float32)
    _ = pruned(dummy)

    feats = np.empty((n_local, 8, 8, 1376), dtype=np.float32)
    idxs = np.empty(n_local, dtype=np.int64)

    last_log = time.time()
    start = time.time()

    i = 0
    while i < n_local:
        end = min(i + args.batch_size, n_local)
        batch_pre = []
        batch_idxs = []
        for j in range(i, end):
            row = my_paths.iloc[j]
            try:
                with open(row["path"], "rb") as f:
                    img_bytes = f.read()
                pre = tf_preprocess(img_bytes)  # (1, 1280, 1280, 3)
                batch_pre.append(pre)
                batch_idxs.append(int(row["idx"]))
            except Exception as e:
                print(f"[rank {args.rank}] image {j} failed preproc: {e}", flush=True)
                batch_pre.append(tf.zeros([1, 1280, 1280, 3], dtype=tf.float32))
                batch_idxs.append(int(row["idx"]))

        try:
            x = tf.concat(batch_pre, axis=0)
            out = pruned(x)
            sp = out[0] if isinstance(out, (list, tuple)) else out  # (B, 40, 40, 1376)
            # Match the SavedModel's internal AvgPool: kernel=5 stride=5 valid
            gap8 = tf.nn.avg_pool2d(sp, ksize=5, strides=5, padding="VALID").numpy()  # (B, 8, 8, 1376)
        except Exception as e:
            print(f"[rank {args.rank}] batch {i}-{end} failed at inference: {e}", flush=True)
            gap8 = np.zeros((end - i, 8, 8, 1376), dtype=np.float32)

        for k in range(end - i):
            feats[i + k] = gap8[k]
            idxs[i + k] = batch_idxs[k]

        if time.time() - last_log > 30:
            elapsed = time.time() - start
            rate = end / elapsed
            eta = (n_local - end) / rate
            print(f"[rank {args.rank}] {end}/{n_local} ({rate:.1f}/s, eta {eta/60:.1f}min)", flush=True)
            last_log = time.time()
        i = end

    np.savez_compressed(out_path, feats=feats, idxs=idxs)
    print(f"[rank {args.rank}] DONE -> {out_path}  ({time.time()-start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
