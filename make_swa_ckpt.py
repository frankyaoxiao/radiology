"""SWA-2: average ckpt_best + ckpt_last weights and save as ckpt_swa."""
import argparse
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", required=True, type=Path)
    ap.add_argument("--last", required=True, type=Path)
    ap.add_argument("--out",  required=True, type=Path)
    args = ap.parse_args()

    print(f"loading {args.best}")
    cb = torch.load(args.best, map_location="cpu", weights_only=False)
    print(f"loading {args.last}")
    cl = torch.load(args.last, map_location="cpu", weights_only=False)

    sb = cb["model"]
    sl = cl["model"]
    if set(sb.keys()) != set(sl.keys()):
        raise ValueError("state dicts differ in keys")

    avg = {}
    for k in sb:
        if sb[k].dtype.is_floating_point:
            avg[k] = (sb[k].float() + sl[k].float()) / 2.0
            avg[k] = avg[k].to(sb[k].dtype)
        else:
            avg[k] = sl[k]

    out = dict(cb)  # copy metadata from best
    out["model"] = avg
    out["swa_note"] = f"average of {args.best.name} and {args.last.name}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
