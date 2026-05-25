"""Summarize all OmniRad-448 experiment runs (round 1).

Reads each run's metrics.jsonl + slurm log + final CSV (if any), produces a ranked table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

VARIANTS = [
    "aug_randlight", "aug_randstd", "aug_randstrong", "aug_trivial",
    "focal_g1", "focal_g2", "focal_g3",
    "mse_w30", "mse_w50", "mse_w70",
    "pseudo_t03", "pseudo_t05", "pseudo_t07",
]

# Existing reference points
REFERENCE = {
    "[baseline] OmniRad-448 s0": (0.6777, 0.670),
    "[baseline] om_mlp (Linear→MLP)": (None, None),  # filled if found
    "[ablation] om_B (lr_hd=1e-4)": (0.6802, None),
    "[ablation] om_ls01 (LS=0.1)": (0.6821, None),
    "[ablation] om_mu01 (mu=0.1)": (0.6901, None),
}


def get_best_val(slurm_log: Path) -> float | None:
    if not slurm_log.exists():
        return None
    pat = re.compile(r"new best nmse=([\d.]+)")
    best = None
    for line in slurm_log.read_text().splitlines():
        m = pat.search(line)
        if m:
            v = float(m.group(1))
            if best is None or v < best:
                best = v
    return best


def get_state(jid: int) -> str:
    import subprocess
    try:
        r = subprocess.run(["sacct", "-j", str(jid), "-h", "-P", "-X", "-o", "State"],
                           capture_output=True, text=True, timeout=5)
        out = r.stdout.strip().splitlines()
        return out[0] if out else "?"
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=Path("/data/artifacts/frank/misc/runs"))
    ap.add_argument("--slurm-logs", type=Path, default=Path("/home/fxiao/misc/156/slurm_logs"))
    ap.add_argument("--csv-dir", type=Path, default=Path("submissions/2026-05-11"))
    args = ap.parse_args()

    rows = []
    # Existing reference (look up baseline + the running om_mlp)
    for name, (val, pub) in REFERENCE.items():
        rows.append({"name": name, "val": val, "public": pub, "csv": None, "state": "REF"})

    # Find each new variant's slurm log by its run_name
    for variant in VARIANTS:
        run_name = f"v1_3class_omnirad_b14_s0_{variant}"
        run_dir = args.runs_root / run_name
        # Look up most recent slurm log for this variant by checking ckpt_best.pt mtime + finding job id
        slurm_logs = sorted(args.slurm_logs.glob("*.out"), key=lambda p: p.stat().st_mtime, reverse=True)
        best_val = None
        state = "PEND"
        slurm_id = None
        for log in slurm_logs:
            text = log.read_text()
            if f"config     : configs/v1_3class_omnirad_b14_s0_{variant}.yaml" in text:
                slurm_id = log.stem
                state = get_state(int(slurm_id)) if slurm_id.isdigit() else "?"
                best_val = get_best_val(log)
                break
        # Find CSV
        short = variant.replace("aug_randlight", "aug_rlight").replace("aug_randstd", "aug_rstd").replace("aug_randstrong", "aug_rstrong").replace("aug_trivial", "aug_triv")
        csv = args.csv_dir / f"omnirad_{short}_va.csv"
        rows.append({
            "name": variant, "val": best_val, "public": None,
            "csv": str(csv) if csv.exists() else None,
            "state": state,
        })

    # Rank by val (None last)
    rows_sorted = sorted(rows, key=lambda r: r["val"] if r["val"] is not None else 9.0)
    print(f"{'rank':>4s}  {'name':35s}  {'state':<10s}  {'val':>7s}  {'public':>7s}  {'csv':<60s}")
    print("-" * 130)
    for i, r in enumerate(rows_sorted, 1):
        val = f"{r['val']:.4f}" if r['val'] is not None else "—"
        pub = f"{r['public']:.3f}" if r['public'] is not None else "—"
        csv = r['csv'] or "—"
        print(f"{i:>4d}  {r['name']:35s}  {r['state']:<10s}  {val:>7s}  {pub:>7s}  {csv:<60s}")


if __name__ == "__main__":
    main()
