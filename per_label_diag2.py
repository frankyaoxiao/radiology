"""Verify Pneumonia gap. Show per-seed best for OmniRad trivial vs other om configs."""
import json, glob

LABELS = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
          "Lung Opacity", "Pneumonia", "Pleural Effusion",
          "Pleural Other", "Fracture", "Support Devices"]
RUNS = "/data/artifacts/frank/misc/runs"


def best_per_label(jsonl_path):
    best = {l: float("inf") for l in LABELS}
    last = {l: float("nan") for l in LABELS}
    with open(jsonl_path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            v = row.get("val", {})
            n = v.get("nmse") if isinstance(v, dict) else None
            if not isinstance(n, dict):
                continue
            for lab in LABELS:
                x = n.get(lab)
                if x is None:
                    continue
                if x < best[lab]:
                    best[lab] = x
                last[lab] = x
    return best, last


for run in sorted(glob.glob(f"{RUNS}/v1_3class_omnirad*/metrics.jsonl")):
    name = run.split("/")[-2]
    b, l = best_per_label(run)
    pn_best = b["Pneumonia"]
    pn_last = l["Pneumonia"]
    print(f"  {name:<45} pneumonia best={pn_best:.4f} last={pn_last:.4f}")

print()
print("=== Compare top 5 single-config Pneumonia bests across all backbones ===")
all_pn = []
for run in sorted(glob.glob(f"{RUNS}/v1_3class_*/metrics.jsonl")):
    name = run.split("/")[-2]
    b, l = best_per_label(run)
    all_pn.append((b["Pneumonia"], name))
all_pn.sort()
for v, n in all_pn[:15]:
    print(f"  pneumonia={v:.4f}  {n}")
