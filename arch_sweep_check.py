"""For each historical OmniRad arch sweep, find best val NMSE mean."""
import json, glob

RUNS = "/data/artifacts/frank/misc/runs"


def best_mean_nmse(jsonl_path):
    best = float("inf")
    for line in open(jsonl_path):
        try:
            row = json.loads(line)
        except Exception:
            continue
        v = row.get("val", {})
        n = v.get("nmse") if isinstance(v, dict) else None
        if isinstance(n, dict):
            m = n.get("mean")
            if m is not None and m < best:
                best = m
    return best


variants = sorted(glob.glob(f"{RUNS}/v1_3class_omnirad_b14_s0_*/metrics.jsonl"))
rows = []
for p in variants:
    name = p.split("/")[-2]
    short = name.replace("v1_3class_omnirad_b14_s0_", "")
    rows.append((short, best_mean_nmse(p)))

rows.sort(key=lambda r: r[1])
print(f"{'arch':<30} {'val_nmse_best':>14}")
for short, v in rows:
    print(f"  {short:<30} {v:>14.4f}")

# Also check baseline omnirad
base = glob.glob(f"{RUNS}/v1_3class_omnirad_b14_s0/metrics.jsonl")
if base:
    print()
    print(f"baseline (no suffix): {best_mean_nmse(base[0]):.4f}")
