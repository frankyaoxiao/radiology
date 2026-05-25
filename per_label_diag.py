"""Per-label val NMSE diagnostic across all trained backbones.

For each backbone family, find the best per-label val NMSE (lowest) from
metrics.jsonl. Surface where switching legs may help.
"""
import json
import glob
from collections import defaultdict

LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Pneumonia", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

RUNS_DIR = "/data/artifacts/frank/misc/runs"


def best_per_label(jsonl_path):
    """Return {label: best_nmse} from a metrics.jsonl file."""
    best = {l: float("inf") for l in LABELS}
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
                if x is not None and x < best[lab]:
                    best[lab] = x
    return best


# Map run name -> family
def family_of(run):
    if "omnirad_b14" in run:
        if "aug_smooth" in run: return "om_smooth"
        if "aug_trivial" in run: return "om_trivial"
        return "om_other"
    if "cnxl" in run:
        if "aug_trivial" in run: return "cnxl_trivial"
        if "clahe" in run: return "cnxl_clahe"
        return "cnxl_other"
    if "hplus" in run:
        if "aug_trivial" in run: return "hplus_trivial"
        return "hplus_other"
    if "eva02" in run:
        if "ema" in run: return "eva02_ema"
        if "llrd" in run: return "eva02_llrd"
        return "eva02_base"
    if "siglip2" in run: return "siglip2"
    if "openclip" in run: return "openclip"
    if "b16" in run: return "b16"
    return "other"


# Walk all metrics files
runs = sorted([p for p in glob.glob(f"{RUNS_DIR}/v1_3class_*/metrics.jsonl")])
print(f"found {len(runs)} run jsonl files")

family_to_runs = defaultdict(list)
for path in runs:
    run = path.split("/")[-2]
    family_to_runs[family_of(run)].append((run, path))

# For each family, take MIN per-label NMSE across all its seeds/configs (best-case per label)
family_best = {}
for fam, items in family_to_runs.items():
    per_lab_best = {l: float("inf") for l in LABELS}
    for run, path in items:
        try:
            bpl = best_per_label(path)
        except Exception as e:
            print(f"  skip {run}: {e}")
            continue
        for l in LABELS:
            if bpl[l] < per_lab_best[l]:
                per_lab_best[l] = bpl[l]
    family_best[fam] = per_lab_best

# Pretty print: family rows, label columns
fams = sorted(family_best.keys())
print()
print("=" * 130)
print("Per-label MIN val NMSE per backbone family (across all seeds/configs)")
print("=" * 130)
header = f"{'family':<16}"
for l in LABELS:
    header += f" {l[:10]:>10}"
header += f" {'mean':>10}"
print(header)
for fam in fams:
    pb = family_best[fam]
    means = [pb[l] for l in LABELS if pb[l] < 1e9]
    avg = sum(means) / len(means) if means else float("nan")
    row = f"{fam:<16}"
    for l in LABELS:
        v = pb[l]
        row += f" {v:>10.4f}" if v < 1e9 else " " + "-".rjust(10)
    row += f" {avg:>10.4f}"
    print(row)

# For each label, show winner
print()
print("=" * 100)
print("Per-label winner")
print("=" * 100)
for l in LABELS:
    best_fam, best_v = None, float("inf")
    for fam, pb in family_best.items():
        if pb[l] < best_v:
            best_v = pb[l]
            best_fam = fam
    # Second-best for delta
    second_v = float("inf")
    for fam, pb in family_best.items():
        if fam != best_fam and pb[l] < second_v:
            second_v = pb[l]
    delta = second_v - best_v
    print(f"  {l:<35} winner={best_fam:<16} nmse={best_v:.4f}  Δ_2nd={delta:+.4f}")
