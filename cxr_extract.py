"""Extract Google CXR Foundation (ELIXR-C) image embeddings.

Per-process extraction of feature_maps_0 from elixr-c-v2-pooled, sharded over
images by --rank/--world. Saves a (N, 8, 8, 1376) -> GAP -> (N, 1376) npz to
disk per shard. SavedModel is batch=1 locked so each rank is a serial loop.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def _example_for(img_bytes: bytes):
    import tensorflow as tf
    return tf.train.Example(features=tf.train.Features(
        feature={'image/encoded': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[img_bytes])
        )}
    )).SerializeToString()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--paths-csv", type=str, required=True,
                   help="CSV with columns 'idx,path' giving 0-indexed image id + abs path")
    p.add_argument("--model-dir", type=str,
                   default="/data/artifacts/hf_cache/hub/models--google--cxr-foundation/snapshots/e5af8ea44a17bad5504f7e485388d6b05786860f")
    p.add_argument("--out-dir", type=str,
                   default="/data/artifacts/frank/misc/cxr_embeds")
    p.add_argument("--gpu", type=int, default=None,
                   help="GPU index to pin (defaults to rank)")
    args = p.parse_args()

    gpu = args.rank if args.gpu is None else args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    import tensorflow as tf
    import tensorflow_text  # noqa: F401  needed for SentencepieceOp
    import pandas as pd

    t0 = time.time()
    paths = pd.read_csv(args.paths_csv)
    # shard
    my_paths = paths.iloc[args.rank::args.world].reset_index(drop=True)
    n_local = len(my_paths)
    print(f"[rank {args.rank}/{args.world} gpu {gpu}] {n_local} images", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{args.rank:02d}_of_{args.world:02d}.npz"
    if out_path.exists():
        print(f"[rank {args.rank}] shard exists, skipping", flush=True)
        return

    print(f"[rank {args.rank}] loading elixr_c ...", flush=True)
    elixr_c = tf.saved_model.load(f"{args.model_dir}/elixr-c-v2-pooled")
    infer = elixr_c.signatures["serving_default"]
    print(f"[rank {args.rank}] model loaded in {time.time()-t0:.1f}s", flush=True)

    # warm up XLA
    img = Image.new("L", (1024, 1024), 128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    _ = infer(input_example=tf.constant([_example_for(buf.getvalue())]))

    feats = np.empty((n_local, 1376), dtype=np.float32)
    idxs = np.empty(n_local, dtype=np.int64)
    last_log = time.time()
    start = time.time()

    for i, row in my_paths.iterrows():
        try:
            img = Image.open(row["path"]).convert("L")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            ex = _example_for(buf.getvalue())
            out = infer(input_example=tf.constant([ex]))
            fm = out["feature_maps_0"].numpy()[0]  # (8, 8, 1376)
            feats[i] = fm.reshape(-1, 1376).mean(axis=0)  # GAP -> 1376
            idxs[i] = int(row["idx"])
        except Exception as e:
            print(f"[rank {args.rank}] image {i} failed: {e}", flush=True)
            feats[i] = 0.0
            idxs[i] = int(row["idx"])

        if time.time() - last_log > 30:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (n_local - i - 1) / rate
            print(f"[rank {args.rank}] {i+1}/{n_local} ({rate:.1f}/s, eta {eta/60:.1f}min)", flush=True)
            last_log = time.time()

    np.savez_compressed(out_path, feats=feats, idxs=idxs)
    print(f"[rank {args.rank}] DONE -> {out_path}  ({time.time()-start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
