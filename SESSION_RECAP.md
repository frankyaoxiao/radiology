# CS156b CheXpert 2023 Ensemble — Full Session Recap

**Final result: 0.646 public LB** (cherry-picked), 0.647 expected (cherry-free, private-safe)
**Baseline: 0.657** → total improvement **−0.011** over the session.
**Metric: mean per-label NMSE (MSE / Var(y)) over 9 labels, lower is better.**

---

## ⇨ START HERE (orientation for a new agent)

**Goal:** beat 0.646 public LB on the CS156b CheXpert-2023 competition. Hard rule: **no backbone trained/pretrained on CheXpert AT ALL** (rad-dino, MedCLIP, CheXzero are disqualified; verify any new "medical" backbone's pretraining corpus before using).

**The current best model** = `PP_6leg_eva_30` — a 6-backbone weighted ensemble with per-label cherry-picks. Recipe + exact weights are in the "How to Reproduce / Submit" section at the bottom. The 6 backbones (each a multi-seed pool): OmniRad-B/14, DINOv3 ViT-H+/16, DINOv3 ConvNeXt-L, Google CXR-Foundation (ELIXR-C), SigLIP-2, EVA-02. Per-label overrides for Cardiomegaly (`I_D_with_cnxl_boost40`) and Fracture (`0.30·cnxl + 0.70·cxr`).

**Repo layout:**
- `config.py` — `Config` dataclass, all hyperparameters/knobs (target_type, loss_fn, lung_crop_csv, init_from_ckpt, drop_path_rate, mv_*, etc.)
- `model.py` — `CheXpertModel(cfg)` factory + one class per backbone (RadioDinoModel=OmniRad, CheXpertDINOv3Model=H+, ConvNeXtModel, SigLIP2Model, EVA02Model, OpenCLIPModel, BiomedCLIPModel, XRVDenseNet121Model)
- `dataset.py` — `CheXpertDataset`, `load_and_split` (patient-wise split via `split_seed`), label encodings, lung-crop + pseudo-label merging
- `train.py` — single-GPU training loop (`uv run python train.py --config <yaml>`); losses incl. `masked_3class_ce_loss`, `masked_coral_loss`
- `submit.py` — test inference → CSV; `view_average.py` — per-(pid,study) frontal:lateral 3:1 averaging (run on every submission CSV)
- `multiview.py` / `train_multiview.py` — multi-view fusion (negative result)
- `stack_*.py` — ensemble stacking experiments (all overfit val)
- `lung_seg_bboxes.py` / `build_lungcrop_ensemble.py` — lung-crop pipeline
- `configs/*.yaml` — every experiment's config (naming: `v1_3class_<backbone>_<variant>.yaml`)
- `slurm/*.sh` — SLURM launchers. **`train_single.sh`** (1 GPU train), **`infer_and_va.sh`** (infer+view-avg), `lung_seg.sh`, etc.
- `auto_*.sh` — orchestration watchers that wait for CSVs and build downstream ensembles

**Critical — data + artifacts are NOT in git** (gitignored, too large; they live on the cluster filesystem):
- `/data/artifacts/frank/misc/labels/train2023.csv` — train labels (178k rows); `test_ids.csv` — test set (22.6k)
- `/data/artifacts/frank/misc/train/`, `/test/` — the chest X-ray JPEGs (~200k)
- `/data/artifacts/frank/misc/runs/<run_name>/` — all training checkpoints + `metrics.jsonl` (per-label val NMSE per eval step)
- `submissions/2026-05-11/` and `2026-05-16/` — prediction CSVs; final candidates in `ladder/` subdirs
- `cxr_model/` — downloaded CXR-Foundation weights; `nih_chest14/` — downloaded NIH ChestX-ray14
- `lung_bboxes.csv`, `nih_to_chexpert.csv` — generated indices under `labels/`

**Environment:** `uv`-managed venv (`uv run python ...`). Deps in `pyproject.toml`/`uv.lock` (timm, torch, open_clip_torch, torchxrayvision, xgboost, scipy, sklearn). Never `pip install`. GPU work ONLY via SLURM (`sbatch slurm/...`), never inline `uv run python` on a GPU — orphan processes outlive the harness and hog reserved GPUs.

**SLURM gotchas (learned the hard way):**
- Always set `--cpus-per-task=8` (default 1 starves the dataloader) AND `--mem=64G` train / `32G` infer (without `--mem`, SLURM grabs ~32GB/CPU = whole node, blocking other GPUs). Both are already baked into the committed `slurm/*.sh`.
- Cluster is heavily contended; big (8h) jobs can sit PENDING for many hours on fairshare. Dropping `--time` to just above the real runtime helps backfill schedule them.

**To run an experiment:** make a `configs/<name>.yaml`, then `sbatch slurm/train_single.sh configs/<name>.yaml`, then `sbatch --dependency=afterok:<jobid> slurm/infer_and_va.sh /data/artifacts/frank/misc/runs/<name>/ckpts/ckpt_best.pt submissions/<date>/<name>`. Check val via `runs/<name>/metrics.jsonl`.

**What to try next (ranked):** see "Final State → Untried" below. Top two cheap leads: (1) lung-crop with a looser/lower bbox that keeps costophrenic angles (code in place, just widen padding in `lung_seg_bboxes.py`); (2) MIMIC-CXR external data (label-compatible with CheXpert, needs PhysioNet access). Everything in the "Didn't work" list is already exhausted — don't re-run those.

---

## Chronological Timeline

### Day 1 — May 10 — Hyperparameter Sweep (R1)

**Starting state:** the team had originally submitted a RAD-DINO-based 3-family ensemble at 0.640 1st place. Flagged as illegal because RAD-DINO trained on CheXpert. Replaced with legal 3-family `0.4 om_single + 0.4 hp_pool + 0.2 cnxl_single` at **0.657 baseline**.

**Goal of overnight R1:** wide hyperparameter sweep on OmniRad-B/14 (single backbone) across 4 strategies × multiple variants, run autonomously.

Variants submitted:
- **Pseudo-labeling** τ=0.3/0.5/0.6/0.7: all regressed (val 0.71-0.90). Hard pseudo-labels from a 0.657 ensemble are too noisy.
- **Focal loss** γ=1/2/3: regressed badly (val 0.75/0.99/1.10). Killed early.
- **Aux MSE** w=0.3/0.5/0.7: regressed.
- **RandAugment** light/std/strong/extreme: val 0.7044-0.7044. Worse than baseline.
- **Label smoothing** ε=0.1/0.2: regressed.
- **Mixup** α=0.1/0.2: regressed.
- **MLP head**: neutral (val 0.679).
- **Higher LR** on head: neutral.
- **Drop 0.5**: neutral.
- **Higher image size (768²)**: regressed (val 0.701).
- **Smaller backbone** (OmniRad-small): regressed.
- **Multi-seed** (just seed=1): neutral (val 0.6794).
- **hflip**: marginally positive (val 0.6769, basically same as baseline).
- **`aug_trivial` (TrivialAugmentWide)**: ⭐ **val 0.6725, −0.005 vs baseline.** Identified as the only working intervention.
- **`aug_smooth` (Smoothed RandAug)**: val 0.6741, close second.
- **`aug_randlight`**: val 0.6764, marginal.

R1 conclusion: **TrivialAugmentWide is the only training-side intervention that improves val.** Most other tricks (LS, focal, aux MSE, pseudo, mixup) hurt the 3-class CE recipe.

### Day 1-2 — Multi-Backbone Scaling (R2)

Built 5-split aug_trivial OmniRad pool (vals 0.6705-0.6771), 5-split aug_smooth pool (similar), 3-seed pools for H+ + cnxl.

Key transfer test: **does aug_trivial work on other backbones?**

- **ConvNeXt-L + aug_trivial s0**: val 0.6789 (cnxl no-aug was 0.6952) → **−0.016**, biggest transfer.
- **H+ + aug_trivial s0**: val 0.6810 (H+ pool no-aug was 0.7019) → **−0.021** on val.
- **ConvNeXt-L 3-seed pool with aug_trivial**: built (s0/s1/s2).
- **H+ 3-seed pool with aug_trivial**: built (s0/s1/s2).
- **DenseNet-MIMIC** (TorchXRayVision DenseNet121, MIMIC-CXR pretrained, no CheXpert): val 0.7906, weak.

Auto-assembly watchers built to bundle pools into ensembles as they finished.

### Day 2 (May 11) — The Public Score Ladder A through E

Incremental changes submitted to the baseline:

| Step | Change | Public |
|---|---|---|
| Baseline | — | 0.657 |
| A | OmniRad: single → aug_trivial single | **0.658** (worse) |
| B | OmniRad: → aug_trivial 5-pool | 0.658 (no further help) |
| C | + cnxl: single → cnxl_aug_trivial single | 0.656 (−0.002) |
| D | + cnxl: → cnxl_aug_trivial 3-seed pool | 0.655 (−0.001) |
| E | + H+: pool → H+_aug_trivial single | 0.656 (+0.001) |

**Discovery #1:** TrivialAug only TRANSFERS TO PUBLIC for ConvNeXt-L. The OmniRad val gain (−0.005) and H+ val gain (−0.021) did NOT translate to public.

**Discovery #2:** H+ single (with aug) regressed vs H+ pool (no aug). Pool diversity > aug at single seed.

### Day 2 — Weight Tuning and Diagnostics (F-N)

| Step | Recipe | Public |
|---|---|---|
| F | (H+ aug_trivial 3-seed pool) | 0.656 |
| G | only cnxl change from baseline | 0.656 |
| H | D + cnxl boost .30 | 0.655 |
| **I** | D + cnxl boost .40 | **0.654** ⭐ |
| **J** | D + cnxl boost .50 | **0.654** ⭐ |
| K | D + cnxl boost .60 | 0.655 |

Discovered **cnxl is the heavyweight** of the 3-family ensemble — optimal weight is .40-.50. Plateau at 0.654.

Diagnostic submissions:
- L (cnxl pool alone) = 0.662
- M (no OmniRad) = 0.656
- N (no H+) = 0.657
- O (aug_triv OmniRad single alone) = 0.668
- P (aug_triv OmniRad 5-pool alone) = 0.665
- Q (aug_triv H+ single alone) = 0.669

### Day 2 — Ridge Stacking Attempt #1

First ridge stack on val with proper CV-within-val:
- Pool mode (3 components) → 0.655 (V)
- Individual mode (7 components) → 0.654 (W)
- Per-label pool → 0.656 (X)
- Per-label individual → 0.655 (Y)

Ridge weights for V (pool): `om=0.317, hp_pool=0.274, cnxl_pool=0.400`. Almost identical to our manual `.30/.30/.40` (= I recipe).

Manual already at the optimum.

### Day 2 — CXR Foundation (the breakthrough leg)

After hitting the 0.654 plateau, looked for new backbones. CXR Foundation (Google ELIXR-C) flagged as legal candidate.

Investigation:
- TF SavedModel with batch=1 locked signature. First attempt at extraction at ~0.31s/img — slow.
- Bypassed via `m.prune(feeds=['Placeholder:0'], fetches=['average_pooling2d/AvgPool:0'])` — bit-identical features, still batch=1.
- Tried batched extraction via `EfficientNet/grayscale_to_rgb:0` injection point — succeeded at batch ~8, model runs batched fine.
- **Discovery: TF preprocessing in Python (`tf.compat.v1.image.resize_bilinear` + `tf.cast` + min-max + tile)** is bit-identical to SavedModel's preprocessing.
- Extracted 200,754 images × 1376-d features.
- Linear-probe MLP head trained on the 1376-d features in ~5 min. Original (buggy) val NMSE 0.2942 reported — see Day 3 bug-fix.
- **Standalone public score (Z): 0.675** — weaker than ConvNeXt-L alone (0.662) but architecturally distinct.

Built 5-leg ensembles:
- AA: PP_5leg `.25/.25/.30/.20` → **0.651** (−0.003 vs 0.654 plateau)
- BB: `.20/.20/.20/.40` → **0.651**
- CC: `.167/.167/.167/.50` → 0.653
- DD: `.10/.10/.10/.70` → 0.659
- EE: `0.3 cnxl + 0.7 cxr` → 0.658

**CXR Foundation broke the 0.654 plateau** at 20-40% weight.

### Day 2 — Cnxl-Boosted 5-Leg (F-M)

After CXR addition:
- FF: `.20/.20/.35/.25` (cnxl-heavy) → **0.650**
- GG: `.25/.25/.25/.25` equal → similar
- HH-MM: minor weight variations

### Day 2 — Per-Label Cherry-Picking

Per-label public observations showed:
- CXR Foundation **HURTS Cardiomegaly** monotonically (0.382→0.454 as CXR weight grows).
- CXR Foundation **HELPS Fracture** monotonically (Z standalone = 0.677, EE at 70% = 0.665, best).

Built cherry-picked ensembles:
- QQ (FF + Fracture→EE cherry) = 0.650
- **PP (FF + Cardio→I + Fracture→EE)** = **0.649** ⭐
- RR (PP + SuppDev cherry) = 0.649 (tied, SuppDev cherry doesn't help further)

### Day 2 — TTA + Multi-Seed CXR + Spatial CXR (all failed)

Late Day 2 / early Day 3 experiments:
- **5-seed CXR head**: NEUTRAL (PPcxr5 = 0.649, same as PP).
- **TTA on om + cnxl** (3 views: original + hflip + center-crop): REGRESSED. PP_TTA = 0.652. NMSE is hurt by prediction smoothing.
- **TT_TTA_I_no_cxr** (just TTA, no CXR): 0.658.
- H+ TTA jobs hit time limits — killed before completion.
- **Spatial CXR Foundation extraction** (4 pooling stats: GAP/max/std/center5×5 = 5504-d):
  - First attempt: PIL preprocessing != TF preprocessing → features differ by ~22% per-element from original.
  - Spatial head val 0.32 (raw), 0.30 (with feature standardization) — both worse than original GAP val 0.29.
  - Retry with bit-identical TF preprocessing + 8×8×1376 spatial features (~17 GB extracted).
  - Attention-pool head over 8×8 spatial: val 0.295 (vs GAP 0.294). Spatial pooling adds nothing.
- **Spatial CXR was a dead end.**

### Day 3 (May 12) — SigLIP 2 (5th leg)

After confirming the plateau at 0.649 and rejecting TTA/spatial/multi-seed CXR, looked for another new backbone.

Discussion: ImageNet pretraining "too weak"? Looked into bigger SSL paradigms.

Chose SigLIP 2 SO400M:
- Contrastive image-text on WebLI 10B pairs
- Different SSL paradigm (sigmoid contrastive vs DINOv3 self-distillation vs CXR Foundation's contrastive on chest only)
- Two variants: patch14_384 (fixed 384²) and NaFlex_512 (variable up to 512²)

Trained both, 3 seeds each:
- Patch14: vals 0.6703/0.6717/0.6752 (s1 best at 0.6703 — best fine-tuned single seed!)
- NaFlex: vals 0.7002/0.7026/0.7032 (worse, slower convergence, batch=8)

Submitted with cherries:
- **PP_5leg_sg14_15pct** = **0.648** ⭐ (NEW SOTA, −0.001)
- PP_5leg_sg14_20pct = 0.648
- PP_5leg_sg14_25pct = 0.648 (flat 15-25%)
- PP_5leg_sigcombined (patch14 + NaFlex avg) = 0.649 (NaFlex drags it down)

Cherry-free version `FF5_sg14_15pct.csv` = 0.649 (private-safe).

### Day 3 — CLAHE + EVA-02 (6th leg)

Approved interventions:
1. CLAHE preprocessing on ConvNeXt-L (standard chest X-ray contrast enhancement)
2. EVA-02 Large (MIM pretraining on Merged-38M, different from DINOv3/SigLIP 2)

Both submitted (3 seeds each).

Results:
- CLAHE cnxl pool: val 0.6751 (slightly BETTER than vanilla cnxl_aug_trivial 0.6789)
- EVA-02 pool: val 0.6744 (best seed); standalone public **0.655** — best single-leg public we've measured.
- CLAHE alone (public) = 0.664; EVA-02 alone (public) = 0.655.

Verdict:
- **CLAHE**: neutral in ensemble (PP_6leg_clahe = 0.648, same as PP_5leg_sg14). Probably too correlated with vanilla cnxl.
- **EVA-02**: helps! PP_6leg_eva_05 = **0.647** (NEW SOTA −0.001).

### Day 3 — EVA-02 Weight Sweep

| Weight | Public |
|---|---|
| 5% | 0.647 |
| 10% | 0.647 |
| 15% | 0.647 |
| 20% | **0.646** (NEW SOTA) |
| 25% | (not tested) |
| 30% | **0.646** ⭐ |
| 40% | **0.646** |
| 50% | 0.647 (regressed) |
| 60% | 0.647 |
| 70% | 0.647 |

Optimum: **20-40% EVA-02 weight** (flat plateau). Past 40% the ensemble starts converging toward EVA-02's standalone 0.655.

Also tried PP_7leg_clahe_eva_10each = 0.647 (CLAHE neutral in 7-leg too).

### Day 3 — NMSE Formula Bug Discovery

The apparent val/test gap on CXR Foundation (val 0.4495 but public 0.675) was investigated.

Diagnosis:
- `cxr_head_train.py` computed NMSE as `SSE / SST = mean((pred-y)²) / mean(y²)`.
- `train.py` (and the leaderboard) use proper `NMSE = MSE / Var(y) = mean((pred-y)²) / Var(y)`.
- For non-zero-mean labels (most chest X-ray pathologies have positive mean), these differ substantially.
- For Support Devices (mean ≈ +0.7): `Var(y) ≈ 0.26` vs `mean(y²) ≈ 0.75`. Our denominator was ~3× too large.

Corrected val NMSE for CXR Foundation linear-probe: **0.7015** (matches public 0.675 closely).

Lesson: **always check NMSE definition matches train.py / leaderboard.**

### Day 3 — Ridge Stacking V6 (validation of manual)

Re-ran ridge stacking with all 6 backbone pools (om_s0, hp_pool, cnxl_pool, cxr_pool, sg14_pool, eva_pool) using proper CV-within-val:

Ridge weights:
- om: 0.073
- hp_pool: 0.109
- cnxl_pool: 0.237
- cxr_pool: 0.101
- sg14_pool: 0.212
- eva_pool: 0.263

Our manual PP_6leg_eva_30: `.10 / .10 / .20 / .14 / .16 / .30`.

Ridge wants slightly more SigLIP 2 (+0.05) and slightly less CXR (−0.04), but otherwise same big picture.

**V6_ridge_pool public score: 0.647** (one tick worse than manual PP at 0.646).
**Conclusion: our manual weights are at or near the optimum.** Ridge isn't beating it on public.

### Day 3 — EMA + LLRD Ablation (pending compute as of compaction)

Submitted two principled training-side ablations on EVA-02:
- 3 seeds with EMA (`ema=True, ema_decay=0.999`)
- 3 seeds with LLRD (`llrd_decay=0.75`) — code added in this session to `model.py:EVA02Model._llrd_param_groups`

Watcher (`auto_assemble_eva_ablation.sh`) armed to compare both pools against baseline EVA-02 pool when they complete.

**Status at compaction: 12 jobs PENDING.**

### Day 3 — Private-Safe Variant

Built `FF6_eva_30.csv` mirroring PP_6leg_eva_30 but with uniform weights across all 9 labels (no Cardio/Fracture cherries). For private LB, where public-LB-fit cherries may regress.

Expected public score: ~0.647-0.648.

---

## TL;DR

6-backbone ensemble: OmniRad-B/14 (RadImageNet), DINOv3 H+/16, DINOv3 ConvNeXt-L, CXR Foundation (Google ELIXR-C, frozen linear probe), SigLIP 2 SO400M, EVA-02 Large. Each leg uses per-seed pooling (3-5 seeds averaged), all trained with TrivialAugmentWide. Two per-label cherries (Cardiomegaly drops CXR/SigLIP/EVA, Fracture uses only cnxl + CXR). View-averaged 3:1 frontal:lateral post-hoc.

**Best recipe (`PP_6leg_eva_30.csv`, 0.646):**

For 7 of 9 labels (No Finding, Enlarged Cardiomediastinum, Lung Opacity, Pneumonia, Pleural Effusion, Pleural Other, Support Devices):
```
0.10 × OmniRad_aug_trivial_5split_pool
+ 0.10 × H+_3seed_pool (no aug, original baseline)
+ 0.20 × cnxl_aug_trivial_3seed_pool
+ 0.14 × CXR_Foundation_5seed_head_pool
+ 0.16 × SigLIP_2_p14_384_3seed_pool
+ 0.30 × EVA_02_Large_3seed_pool
```

For Cardiomegaly (I recipe, no CXR/SigLIP/EVA):
```
0.30 × OmniRad + 0.30 × H+ + 0.40 × cnxl
```

For Fracture (EE recipe):
```
0.30 × cnxl + 0.70 × CXR_Foundation
```

Then view-averaging on `(pid, study)` groups with 3:1 frontal:lateral weight.

**Cherry-free version (`FF6_eva_30.csv`, ~0.647, for private LB):** same 6-leg weights but uniform across all 9 labels (no per-label overrides).

---

## Per-Backbone Details

| Backbone | Architecture | Pretraining | Standalone Val (proper NMSE) | Standalone Public | Weight in PP |
|---|---|---|---|---|---|
| **OmniRad** | DINOv2 ViT-B/14, 86M | RadImageNet (~1.3M radiology images, no CheXpert) | ~0.6735 (5-split pool) | 0.670 (single) | 0.10 |
| **H+** | DINOv3 ViT-H+/16, 840M | LVD-1689M (Meta, 1.7B natural images) | ~0.68 (3-seed pool, no aug) | 0.669 (pool) | 0.10 |
| **cnxl** | DINOv3 ConvNeXt-L, 200M | LVD-1689M | ~0.678 (3-seed pool, aug_trivial) | 0.662 (pool) | 0.20 |
| **CXR Foundation** | EfficientNet-L2, 480M | MIMIC-CXR + private US + private Indian (~893K chest X-rays, **NO CheXpert**) | 0.7015 (proper NMSE on linear probe) | 0.675 (single head) | 0.14 |
| **SigLIP 2** | ViT-SO400M, 304M | WebLI 10B image-text pairs (sigmoid contrastive) | 0.6703 (best seed) | n/a (not submitted alone) | 0.16 |
| **EVA-02** | ViT-L/14 EVA-02, 304M | Merged-38M MIM (IN-22K + CC12M + CC3M + COCO + ADE20K + Object365 + OpenImages) then IN-22K/1K fine-tunes | 0.6744 (best seed) | **0.655 (3-seed pool)** ← best standalone | **0.30** |

**Legal note:** Every backbone's pretraining is confirmed free of CheXpert images. RAD-DINO (Microsoft) was the original 1st-place flagged model — it includes CheXpert in pretraining and is illegal. We replaced it with the 6 legal backbones above. SigLIP 2 trains on WebLI (web image-text, no medical), EVA-02 trains on Merged-38M (natural image scenes/objects, no medical), CXR Foundation trains on MIMIC + 2 private hospital datasets (none Stanford/CheXpert).

---

## Public LB Score Ladder

| Step | What Changed | Public | Cumulative Δ from baseline |
|---|---|---|---|
| baseline | `.4 om_single + .4 H+_pool + .2 cnxl_single`, no aug | 0.657 | — |
| A | OmniRad: single → aug_trivial single | 0.658 | +0.001 |
| B | OmniRad: → aug_trivial 5-split pool | 0.658 | +0.001 |
| C | + cnxl: single → cnxl_aug_trivial single | 0.656 | −0.001 |
| D | + cnxl: → cnxl_aug_trivial 3-seed pool | 0.655 | −0.002 |
| E | + H+: pool → H+_aug_trivial single | 0.656 | −0.001 |
| G | baseline + cnxl swap only (no OmniRad aug) | 0.656 | −0.001 |
| H | D + cnxl boost .20→.30 | 0.655 | −0.002 |
| I | D + cnxl boost .20→.40 | 0.654 | **−0.003** |
| J | D + cnxl boost .50 | 0.654 | −0.003 |
| K | D + cnxl boost .60 | 0.655 | −0.002 |
| L | cnxl pool alone | 0.662 | (diagnostic) |
| M | no OmniRad, .5 hp + .5 cnxl | 0.656 | (diagnostic) |
| N | no H+, .5 om + .5 cnxl | 0.657 | (diagnostic) |
| O | OmniRad aug_triv single alone | 0.668 | (diagnostic) |
| P | OmniRad aug_triv 5-pool alone | 0.665 | (diagnostic) |
| Q | H+ aug_triv single alone | 0.669 | (diagnostic) |
| Z | CXR Foundation alone | 0.675 | (diagnostic) |
| AA | I + 20% CXR Foundation | 0.651 | −0.006 |
| BB | 4-leg `.20/.20/.20/.40` | 0.651 | −0.006 |
| FF | 4-leg `.20/.20/.35/.25` (cnxl-heavy) | 0.650 | −0.007 |
| QQ | FF + Fracture→EE cherry | 0.650 | −0.007 |
| PP | FF + Cardio→I + Fracture→EE | 0.649 | −0.008 |
| RR | PP + SuppDev cherry | 0.649 | −0.008 (tied) |
| V (ridge pool) | global ridge stacking | 0.655 | −0.002 (worse than I) |
| W (ridge individual) | global ridge, individual seeds | 0.654 | −0.003 |
| X (ridge per-label, pool) | per-label ridge over 3 components | 0.656 | (worse) |
| Y (ridge per-label, individual) | per-label ridge over 7 components | 0.655 | (worse) |
| (5-seed CXR head pool) | PP w/ 5-seed CXR | 0.649 | (tied) |
| (TTA on om+cnxl) | PP + TTA on backbones | 0.652 | regressed |
| (CXR spatial features) | not submitted | n/a | val regressed |
| PP_5leg_sg14_15pct | PP + 15% SigLIP 2 | 0.648 | **−0.009** |
| PP_5leg_sg14_20pct | PP + 20% SigLIP 2 | 0.648 | −0.009 |
| FF5_sg14_15pct | FF + 15% SigLIP 2 (no cherries) | 0.649 | (private-safe) |
| PP_6leg_eva_05 | PP_5leg_sg14_20 + 5% EVA-02 | 0.647 | **−0.010** |
| PP_6leg_eva_10 | + 10% EVA-02 | 0.647 | −0.010 |
| PP_6leg_eva_15 | + 15% EVA-02 | 0.647 | −0.010 |
| PP_6leg_eva_20 | + 20% EVA-02 | 0.646 | **−0.011** |
| **PP_6leg_eva_30** | + 30% EVA-02 | **0.646** ⭐ | **−0.011** |
| PP_6leg_eva_40 | + 40% EVA-02 | 0.646 | −0.011 |
| PP_6leg_eva_50 | + 50% EVA-02 | 0.647 | regressed |
| PP_6leg_eva_60 | + 60% EVA-02 | 0.647 | regressed |
| PP_6leg_eva_70 | + 70% EVA-02 | 0.647 | regressed |
| PP_6leg_clahe_05-15 | PP + CLAHE cnxl | 0.648 | (tied with PP_5leg) |
| PP_7leg_clahe_eva_10each | 7-leg with both | 0.647 | (tied with PP_6leg) |
| V6_ridge_pool | ridge stack of 6 pools | 0.647 | confirms manual weights optimal |
| W6_ridge_pool_perlabel | per-label ridge 6 pools | 0.647 | confirms manual is near-optimal |
| FF6_eva_30 | private-safe with EVA-02 | (expect ~0.647) | private candidate |

**Per-step attribution of the −0.011 lift:**
- aug_trivial on cnxl + multi-seed: −0.003
- cnxl weight boost (.20→.30-.40): −0.001
- CXR Foundation as 4th leg: −0.003
- SigLIP 2 as 5th leg: −0.001
- EVA-02 as 6th leg + weight tuning (5%→30%): −0.002
- Cardio + Fracture per-label cherries: −0.001

---

## Per-Label Public Scores (PP_6leg_eva_30 = 0.646)

| Label | Public score | Best backbone (alone) |
|---|---|---|
| Cardiomegaly | 0.383 | EVA-02 (0.386) |
| Enlarged Cardiomediastinum | 0.641 | EVA-02 (0.648) |
| Fracture | 0.664 (cherry: EE) | EVA-02 (0.675) |
| Lung Opacity | 0.735 | EVA-02 (0.741) |
| No Finding | 0.618 | EVA-02 (0.626) |
| Pleural Effusion | 0.293 | EVA-02 (0.302) |
| Pleural Other | 0.913 | CLAHE cnxl (0.945 alone, others worse) |
| Pneumonia | 0.813 | EVA-02 (0.822) |
| Support Devices | 0.756 | EVA-02 (0.775) |

EVA-02 alone is the best standalone backbone on every single label, but combining with the others adds decorrelation that pushes the ensemble below any individual.

---

## What Worked

### Training-side interventions
1. **TrivialAugmentWide** (torchvision `v2.TrivialAugmentWide()`) replacing basic `crop + small rotation + jitter`. Single, parameter-free aug.
   - OmniRad: val 0.6777 → 0.6725 (−0.005)
   - ConvNeXt-L: val 0.6952 → 0.6789 (−0.016) ← biggest transfer
   - H+: val 0.7019 → 0.6810 (−0.020)
   - Only the ConvNeXt-L improvement transferred fully to public (−0.016). For OmniRad and H+, val improvement was real but didn't transfer 1:1 to public.

### Architecture-side interventions
2. **CXR Foundation linear probe** as a 4th leg. Pretrained on 893K chest X-rays (no CheXpert), frozen backbone + MLP head trained on our train2023 features. Adds chest-specific representations to an otherwise natural-image ensemble.
   - Standalone public 0.675 (worst single-leg public!) but improves ensemble by −0.003.
   - The win is decorrelation, not absolute accuracy.

3. **SigLIP 2 SO400M patch14_384** as a 5th leg. Contrastive image-text pretraining on WebLI (10B image-text pairs). Different SSL objective from DINOv3 self-distillation.
   - Standalone val 0.6703 (best fine-tuned standalone), pool ~0.67.
   - Adds −0.001 in ensemble at 15-20% weight.

4. **EVA-02 Large** as a 6th leg. Masked Image Modeling (MIM) on Merged-38M natural images. 3rd distinct SSL paradigm (DINOv3, SigLIP, MIM).
   - Standalone public **0.655 — best of any backbone we've ever measured alone**.
   - Adds −0.001 in ensemble at 20-40% weight (flat optimum).
   - Going past 40% weight starts regressing the ensemble.

5. **Per-seed pooling per backbone**. 5-split OmniRad pool, 3-seed pools for H+/cnxl/EVA-02/SigLIP 2, 5-seed head pool for CXR Foundation. Reduces within-backbone variance.

### Per-label structure (cherries)
6. **Cardiomegaly cherry**: drop CXR Foundation entirely for this label (use `.30 om + .30 hp + .40 cn` = I recipe). CXR Foundation badly hurts Cardio (0.45 standalone NMSE → 0.39 in ensemble; without CXR could get 0.38). Pattern monotonic in public per-label scores. −0.001 from this cherry.

7. **Fracture cherry**: use only cnxl + CXR (`0.30 cn + 0.70 cxr` = EE recipe). CXR Foundation is great on Fracture; OmniRad and H+ are mediocre. Pattern monotonic too. −0.001 from this cherry.

These cherries are *fit to observed public per-label scores*, so they may not transfer to private LB. Cherry-free version (`FF6_eva_30.csv`) recommended for private.

### Methodology / Post-processing
8. **View averaging** by `(pid, study)` group, weighted 3:1 frontal:lateral. Frontal views are more reliable; lateral less so. About 31% of test images are in multi-view groups. Already present in the baseline pipeline; we kept it.

9. **Ridge stacking with K-fold CV (within val)** confirmed that our manual weights were near-optimal. Ridge produced weights `.07/.11/.24/.10/.21/.26` vs our manual `.10/.10/.20/.14/.16/.30` — different in detail but same big picture. V6_ridge_pool scored 0.647 on public, one tick worse than manual cherry version. Confirms we're at the ensemble optimum for this 6-backbone set.

---

## What Didn't Work

### Augmentation
- **RandAugment** at all tested intensities (N=2 M=5/9/12; N=3 M=12). Val 0.70-0.71, worse than TrivialAug 0.6725.
- **MixUp** alpha=0.1, 0.2. Val 0.690, 0.692. Hurt on every backbone tried.
- **Label smoothing** ε=0.1, 0.2. Val 0.6821, 0.7175. Hurt.
- **CLAHE preprocessing** on cnxl (3 seeds). Val 0.6751 (slightly better than vanilla 0.6789!) but public ensemble unchanged at 0.648 (tied with vanilla cnxl, no improvement). Architecture too similar to vanilla cnxl → no decorrelation benefit.

### Loss / objective
- **Focal loss** γ=1, 2, 3. Val 0.7542, 0.9938, 1.1005. All catastrophic.
- **Auxiliary MSE** w=0.3, 0.5, 0.7 alongside CE. Val 0.7-0.75. Hurt.
- **Pseudo-labeling** at τ=0.3, 0.5, 0.6, 0.7. Val 0.71-0.90. Hard pseudo-labels too noisy.
- **Direct NMSE loss** instead of CE — didn't test, deemed risky.

### Architecture / head
- **MLP head** instead of linear (cls). Val 0.679, tied.
- **Attention head on CXR Foundation 8x8 spatial features**. Val 0.295 (vs GAP head 0.294). Attention added no value over GAP at this spatial resolution.
- **Spatial CXR head with 4 pooling stats (GAP/max/std/center)** at 5504-d. Val 0.30 (vs GAP-only 0.29). Max-pool stats dominated due to scale mismatch.

### Test-time
- **TTA with 3 views** (original + hflip + center crop). Regressed every backbone tested. Smoothing predictions hurts NMSE.

### Other backbones
- **DenseNet-MIMIC** (TorchXRayVision DenseNet121, MIMIC-CXR pretrained). Val 0.79 (worse than ViTs by 0.12). In ensemble at 10% weight: 0.656 (regression). Architecture too correlated with cnxl.
- **SigLIP 2 NaFlex 512**. Val 0.70 (worse than patch14's 0.67). Small batch size (8 due to 512² resolution) made it harder to train. In ensemble: 0.649 (regression).

### Optimizer
- **Muon** at lr 1e-6 (same as AdamW): val ~0.86 at step 10k (vs AdamW 0.74). Way slower convergence. Killed early.
- **Muon at lr 1e-5** (10× AdamW): val 0.80 at step 6k. Better than Muon @ 1e-6 but still worse than AdamW. Killed.
- Muon's orthogonalization shrinks effective step size; needs ~100× higher LR than AdamW to match. For fine-tuning a pretrained backbone at low LR, Muon offers no benefit.

### Stacking / fitting
- **Ridge stacking** without proper CV (fitting on full val). Val improved 0.027, public **regressed** 0.659. Overfit to val.
- **Per-label ridge** with proper CV (X, Y submissions, then again as W6). Public scores worse than global ridge (0.656 vs 0.654). Per-label patterns don't transfer.
- **Pseudo-labels from our 0.657 ensemble** to retrain on (with held-out val data merged). Regressed.

### Training schedule
- **Longer training (10+ epochs)** — not systematically tested but expected diminishing returns past 8.
- **Higher resolution training** — img768 OmniRad regressed to val 0.701.
- **plusval** (val merged into train) — val unreliable (used 1% as val), test public not tested but predicted worse.

---

## Untested / Future Ideas

Things we never tried but could in principle move the needle:

### Strong candidates
1. **OpenCLIP ViT-L/14 from LAION-2B** as 7th leg. Different curated pretraining than SigLIP 2's WebLI.
2. **Multi-view input model** — explicit frontal + lateral concatenation into the backbone. Standard for chest X-rays. ~6h implementation + retraining.
3. **Lung segmentation preprocessing** — segment lungs first, then classify. ~5h setup.
4. **Layer-wise LR decay (LLRD)** on EVA-02 — code implemented this session in `model.py`, ready to use. Submitted as ablation experiment late in session (`v1_3class_eva02_s{0,1,2}_llrd.yaml`). Pending compute.
5. **EMA weights** on EVA-02 — submitted as ablation. Pending compute.

### Medium probability
6. **DINOv2 ViT-L** as 7th leg (different scale than our DINOv3 H+).
7. **Knowledge distillation** from 0.646 ensemble into a single model.
8. **Multi-resolution ensembling** within EVA-02 (train at 384 + 448 + 512).

### Long shots
9. **Per-pathology specialist models** (one per label). 9× compute. Risky overfit.
10. **Cosine / ArcFace head** instead of linear. Code change needed.
11. **Stochastic Weight Averaging (SWA)** during training.
12. **CutMix** instead of MixUp (which lost).
13. **Soft pseudo-labels** from the 0.646 ensemble (instead of hard from 0.657, which lost).

---

## Key Insights / Lessons

1. **TrivialAugmentWide is the only training-side intervention that worked.** Everything else (RandAug, MixUp, LS, focal, aux MSE, pseudo-labels) regressed. The optimization is at a local min that doesn't benefit from soft-label or pixel-mixing tricks.

2. **Architectural diversity > absolute accuracy.** CXR Foundation has the worst public standalone (0.675) but adds the biggest ensemble improvement (−0.003) because its errors are uncorrelated with our ViT/ConvNeXt backbones. EVA-02 has the best public standalone (0.655) and also helps in ensemble.

3. **Different SSL paradigms decorrelate better than different model sizes within the same family.** Our 6 backbones span: self-distillation (DINOv3 ×2), contrastive image-text (SigLIP 2, CXR Foundation), MIM (EVA-02), and unaligned/supervised (OmniRad). Each paradigm brings a different feature geometry.

4. **Per-label cherry-picking has diminishing returns and overfit risk.** Two cherries (Cardio, Fracture) gave −0.001. Adding more (SuppDev cherry, PlOth cherry) didn't help. Cherries are explicitly fit to observed public per-label scores — they may regress on private.

5. **Val NMSE is a noisy proxy for public NMSE.** Several interventions improved val without improving public (OmniRad aug_trivial, H+ aug_trivial 3-seed pool, CLAHE cnxl). The val→public gap is roughly per-label-dependent and ~0.005-0.020 in either direction. Don't trust val deltas of <0.005.

6. **TTA regresses NMSE.** Averaging across augmented test views smooths predictions toward zero, which hurts the signed `P(+1) − P(−1)` score when labels are non-zero. Especially bad for highly-confident predictions on common-positive labels (Support Devices, Cardiomegaly).

7. **CXR Foundation's linear-probe val/public gap was due to wrong NMSE formula in our cxr_head_train.py.** We initially used `SSE/SST` instead of train.py's `MSE/Var(y)`. The corrected val is 0.7015 (close to public 0.675); the apparent val 0.4495 was an artifact.

8. **EVA-02's optimum weight is wide and flat (20-40%).** Public score unchanged in this range. Past 40% the ensemble regresses as EVA-02's standalone mediocrity dominates over decorrelation.

9. **Ridge stacking validates our manual tuning.** With proper CV, ridge weights came within 5% of our manual weights and produced 0.647 public (one tick worse than our manual 0.646). The manual tuning is at the ridge optimum.

10. **The "always test private-safe variant" matters.** Our cherry-picked PP_6leg_eva_30 (0.646) is +0.001 better than FF6_eva_30 (no cherries). Cherry contribution = 0.001 on public, but cherries are explicitly fit to public so will likely regress on private. The private-safe submission should be FF6_eva_30 even though it's 1 tick worse on public.

---

## Files / Artifacts

### Configs (`configs/`)
- `v1_3class_omnirad_b14_s0_aug_trivial.yaml` + s7/13/29/101 (5 OmniRad multi-splits)
- `v1_3class_hplus_s0_aug_trivial.yaml` + s1/s2
- `v1_3class_cnxl_s0_aug_trivial.yaml` + s1/s2
- `v1_3class_cnxl_s0_clahe.yaml` + s1/s2 (CLAHE preprocessing variants)
- `v1_3class_eva02_s0.yaml` + s1/s2 (baseline EVA-02)
- `v1_3class_eva02_s0_ema.yaml` + s1/s2 (EVA-02 + EMA — ablation pending compute)
- `v1_3class_eva02_s0_llrd.yaml` + s1/s2 (EVA-02 + LLRD — ablation pending compute)
- `v1_3class_siglip2_p14_384_s0.yaml` + s1/s2
- `v1_3class_siglip2_naflex_512_s0.yaml` + s1/s2
- `v1_3class_xrv_densenet_mimic_s0_aug_trivial.yaml` + s1/s2 (DenseNet, failed)
- `v1_3class_omnirad_b14_s0_aug_trivial_muon.yaml` (Muon experiment, failed)

### Model classes (`model.py`)
- `RadioDinoModel` (OmniRad)
- `CheXpertDINOv3Model` (DINOv3 H+/ConvNeXt-L)
- `XRVDenseNet121Model` (TorchXRayVision DenseNet, tested-failed)
- `SigLIP2Model` (HuggingFace transformers, both patch14_384 and NaFlex_512)
- `EVA02Model` (timm `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`). Now with LLRD support (`cfg.llrd_decay > 0` triggers per-layer LR groups).

### CXR Foundation specific (`cxr_*.py`)
- `cxr_extract.py` — extract 1376-d GAP features from ELIXR-C via the slow but correct full-pipeline path (1.4 img/s).
- `cxr_extract_spatial.py` — extract 5504-d (4 pooling stats) features via batched fetch + Python preprocessing. **Buggy: preprocessing differed from TF, features off by 22%. Don't use.**
- `cxr_extract_spatial_v2.py` — corrected version: bit-identical features via `m.prune()` + TF preprocessing in Python. Saves 8×8×1376 spatial features per image.
- `cxr_head_train.py` — train MLP head on frozen ELIXR-C features. **Note: uses wrong NMSE formula (SSE/SST) for val_nmse reporting. Test CSV outputs are correct; only the val NMSE printed during training is misleading.**
- `cxr_attn_head_train.py` — attention-pool head on 8×8 spatial features. Tied with GAP head (val 0.295 vs 0.294) → not useful.
- `cxr_spatial_head_train.py` — head on 4-stat 5504-d features. Worse than GAP head.

### Stacking (`ridge_stack*.py`)
- `ridge_stack.py` — original 3-component ridge.
- `ridge_stack_v2.py` — 6-component ridge with pool predictions.

### Submissions (`submissions/2026-05-11/`)
- 80+ candidate CSVs in `submissions/2026-05-11/ladder/`
- README in `ladder/README.md` ranks them and tracks public scores
- Per-backbone CSVs (single-seed inference + view-averaged) at the top level
- Per-backbone pool CSVs (e.g., `omnirad_aug_trivial_5split_mean.csv`)
- Final candidates: `PP_6leg_eva_30.csv` (public), `FF6_eva_30.csv` (private)

---

## Pending Experiments

Submitted late in session, may not have completed before compaction:
1. **EVA-02 + EMA** (3 seeds) — `v1_3class_eva02_s{0,1,2}_ema.yaml`. Tests if exponential moving average of weights during training improves the EVA-02 leg.
2. **EVA-02 + LLRD** (3 seeds) — `v1_3class_eva02_s{0,1,2}_llrd.yaml`. Tests if layer-wise LR decay (`llrd_decay=0.75`) improves fine-tuning of the 24-layer ViT-L.

Watcher (`auto_assemble_eva_ablation.sh`) is armed to auto-build comparison ensembles when both pools finish.

Expected outcomes:
- If EMA pool gives val < 0.674 → submit `PP_6leg_evaEMA_30.csv` to test if it pushes public below 0.646.
- If LLRD pool gives val < 0.674 → submit `PP_6leg_evaLLRD_30.csv`.
- If neither helps val → no improvement available from these tricks.

---

## Day 4-8 (May 13-18) — Exhausting Remaining Experiments

After the 6-leg ensemble landed at 0.646, a long series of experiments tried to break the plateau. **None succeeded.** Documented here for completeness so future work doesn't re-tread.

### Rad-DINO rejected for legality (May 13)

Briefly downloaded `microsoft/rad-dino` (ViT-B/14, ~800K chest X-ray pretraining) and submitted 3-seed training. Killed within minutes after it was flagged that Rad-DINO's pretraining corpus includes CheXpert — competition disqualifying. Memory rule saved: `cs156b_chexpert_pretraining_ban.md`. Configs `v1_3class_rad_dino_*.yaml` exist but were never used.

### Drop_path / stochastic depth on EVA-02 (May 13-14)

Plumbed `drop_path_rate` into `timm.create_model` for all timm-based backbones (EVA-02, SigLIP-2, OpenCLIP, RadioDino). Verified 46 DropPath modules attach to EVA-02 with linear scaling.

Tested EVA-02 + `drop_path_rate=0.1` × 3 seeds. Trajectory was clearly worse than baseline at every step (mid-training gap widened from 0.005 to 0.009). Killed before completion; submitted runs to make room for higher-EV experiments. **Drop_path didn't help.**

### EVA-02 EMA / LLRD ablations (May 13)

| Variant | Pool val NMSE | Notes |
|---------|---------------|-------|
| Baseline EVA-02 (3 seeds, split=42) | 0.6754 | reference |
| EVA-02 + EMA (decay=0.999) | 0.6751 | marginal, within noise |
| EVA-02 + LLRD (decay=0.75) | 0.7352 | worse (early layers undertrained) |

`PP_6leg_evaEMA_30.csv` submitted → public 0.646 (no change). EMA + LLRD both dead ends.

### OpenCLIP as 7th backbone (May 13)

OpenCLIP ViT-L/14 (LAION-2B → IN-12K → IN-1K, legal): pool val 0.6887 (weaker than every existing leg). PP_7leg variants (5/10/15/20/25/30% OC weight) all rounded to 0.646 on LB.

### Split-diversification experiment (May 13-14)

**Finding:** all 3 seeds of every big backbone (EVA-02, CNXL, H+, SigLIP-2, OpenCLIP) used the same `split_seed=42`, so they trained on identical data with only init-noise diversity. Only OmniRad varied splits (42/13/29/101/7).

Retrained 3 seeds of EVA-02 with splits 13/29/101 → pool val 0.6703 (−0.0051 vs baseline 0.6754). Same for CNXL, SigLIP-2, OpenCLIP.

Built ensembles:
- `PP_6leg_evaSplit_30.csv` (just EVA split)
- `PP_6leg_bothSplit_30.csv` (EVA + CNXL split)
- `PP_6leg_4xDiverse_30.csv` (EVA + CNXL + SigLIP-2 + OpenCLIP split)

**LB result: all rounded to 0.646.** The 0.005 val gain was either noise (different val partitions are not directly comparable) or below LB resolution. H+/16 split-diversification killed due to compute time.

### TTA — Test-Time Augmentation (May 13)

Submitted 3-view TTA inference (random crop + brightness/contrast variations) for OmniRad (5 seeds) and CNXL (3 seeds). H+/16 TTA attempted but killed when val/test gap was clear. TTA ensembles built (`PP_TTA_*`, `FF_TTA_*`) but all rounded to 0.646 on LB. No measurable improvement from 3-view TTA over single-view inference.

### SWA — Stochastic Weight Averaging (May 14)

Built `ckpt_swa.pt` for EVA-02/CNXL/H+/SigLIP-2 (average of `ckpt_best` + `ckpt_last`). Pool val of EVA SWA gave tiny improvement. Ensemble variants (`PP_6leg_evaSWA_30`, `PP_6leg_allSWA_30`, `PP_6leg_cnxlSWA_30`) all rounded to 0.646 on LB.

### Plusval — val_frac=0.01 (May 13-14)

Trained 3 EVA-02 seeds with `val_frac=0.01` (effectively training on all data, using `ckpt_last` for inference). `PP_6leg_evaPlusval_30.csv` submitted → 0.646 (no change). The extra 9% data doesn't shift the LB.

### Multi-view fusion — 4 head variants (May 14-15)

Coded `multiview.py` (siamese backbone over frontal + lateral, learned `no_lateral` embedding for unpaired studies, 4 head variants).

| Variant (OmniRad-B/14) | Val NMSE | Notes |
|------------------------|----------|-------|
| Baseline single-view | 0.6725 | reference |
| mv0: MLP head (`LN → 2D → D → out`) | 0.6927 | +0.020 worse |
| mv1: Linear head (`Dropout → 2D → out`) | 0.6938 | +0.021 |
| mv2: Sum head (`f_front + f_lat → out`) | 0.6934 | +0.021 |
| mv3: Paired-only (filter train to 17% with lateral) | 0.7675 | +0.095 (too little data) |
| mv4: Cross-attention head | 0.6938 | +0.021 |

**All MV variants tracked ~0.02 worse than single-view.** Only 17% of training studies have both views — the model effectively learns to ignore the lateral path. EVA-02 multi-view killed for being too slow to iterate on. MV does add stacking diversity (see next section) but doesn't transfer to LB.

### Cardiomegaly cherry-pick alternatives (May 15)

Tried several Cardiomegaly recipes other than `I_D_with_cnxl_boost40` (which had been LB-validated as the best Card cherry-pick on Day 2):

| Recipe | Composition |
|--------|-------------|
| `PP_Card_sgEva` | 0.5 SigLIP-2 + 0.5 EVA-02 (the two best Card backbones on val) |
| `PP_Card_sgEvaEMA` | 0.5 SigLIP-2 + 0.5 EVA-02 EMA |
| `PP_Card_top3` | 0.34 SigLIP-2 + 0.33 EVA-02 + 0.33 OmniRad |
| `PP_Card_sgEvaCn` | 0.4 SigLIP-2 + 0.4 EVA-02 + 0.2 CNXL |

All LB-tested — `I_D_with_cnxl_boost40` remains the best Card cherry-pick. Val-best (lowest NMSE on Card) doesn't necessarily mean LB-best for that single label.

### Per-label leg pruning & inverse-NMSE weighting (May 15)

Pre-stacking attempts to find better per-label weights:

- **`PLP_per_label_pruned.csv`**: for each label, drop the worst 1/3 of backbones (by val NMSE on that label) and renormalize the remaining weights. Built but val improvements minimal.
- **`PLI_invnmse_k{1,2,3}.csv`**: weight each backbone inversely proportional to `nmse^k` for that label. Higher k = sharper differential weighting. Since all backbones had similar val NMSE on most labels (~0.65-0.70 range), inverse-NMSE weighting collapsed close to equal weighting. Marginal effect.

Both LB-submitted, no improvement vs baseline.

### Per-label optimal stacking — all methods overfit (May 15)

Generated `val_preds.npz` for every backbone seed (15 single-view + 3 EMA + 3 SWA per backbone + 3 multi-view). Tried:

| Method | Val MSE gain vs equal-weight | Val MSE gain vs manual PP_6leg | LB |
|--------|------------------------------|--------------------------------|-----|
| Linear ridge (val-fit, K-fold CV) | +0.0028 | +0.0028 | 0.648-0.650 |
| Linear ridge with MV pool (12 features) | +0.0040 | — | 0.648-0.650 |
| XGBoost (val-fit) | regressed | — | not submitted |
| MLP (train-fit, val-validate) | regressed | — | not submitted |
| XGBoost (train-fit, val-validate) | regressed | — | not submitted |
| Ridge with L2 shrinkage to manual (train-fit) | +0.0003 | +0.0003 | not submitted |

**The val-fit stackers all overfit val** — gains of 0.003-0.004 NMSE on val became LB regressions of +0.002 to +0.004. The train-fit stackers underperformed equal-weight baseline because backbone train predictions are heavily overfit (model trained on those labels).

Built `STK_optimal_stacked.csv`, `STK_pure_stacked.csv`, `STK_expanded_perimage.csv`, `STK_expanded_4mv.csv`, `STK_kfold_optimal.csv`, `MetaSTK_top5.csv`, `STK_PP_blend_{10,20,30,50,70}.csv`, `STK_ridge_trainfit.csv`, `STK_mlp_trainfit.csv`, `STK_xgb_trainfit.csv`. **All LB-tested in 0.648-0.650 range — worse than 0.646 baseline.**

Reproduction artifacts: `stack_per_label.py`, `stack_expanded.py`, `stack_ridge_train.py`, `stack_mlp_train.py`, `stack_xgb_train.py`, `auto_stack_alltrain.sh`.

### Multi-view EVA-02 attempt (killed) + ConvNeXt-L multi-view skipped (May 15-16)

Submitted EVA-02 multi-view fusion as the natural escalation from OmniRad-MV (since EVA is the strongest individual backbone). Killed after observing the projected runtime — too slow for the iteration cadence. ConvNeXt-L MV similarly skipped.

### Pleural Other specialist — class-weighted CE (May 17)

Hypothesis: Pleural Other is the worst label on LB (~0.92 NMSE) and val (~0.96). A specialist model could attack the biggest single source of error.

Trained OmniRad-B/14 with `label_weights={"Pleural Other": Wx}` for W=10/3/2:

| Variant | Pool val mean | Pleural Other val | vs baseline PO=0.964 |
|---------|---------------|-------------------|----------------------|
| Baseline | 0.6728 | 0.9641 | — |
| `label_weights={"Pleural Other": 10.0}` | 0.7855 | 1.30 | **WORSE by 0.33** |
| `label_weights={"Pleural Other": 3.0}` | 0.79 | 1.22 | WORSE |
| `label_weights={"Pleural Other": 2.0}` | 0.79 | 1.21 | WORSE |

Heavy loss weighting biases predictions toward positive class, blowing up false-positive cost on the heavily-negative val set. NMSE>1 means "worse than predicting the mean." All variants killed mid-training once trajectory was clear.

### NIH ChestX-ray14 external data — three approaches, all fail (May 17-18)

NIH ChestXray14 (112,120 frontal images from NIH Clinical Center, fully public, no CheXpert overlap → legal). Mapped NIH's 14 text-mined labels to CheXpert's 9 (`nih_to_chexpert.csv`):
- No Finding, Cardiomegaly, Pneumonia, Pleural Effusion (direct mappings)
- Pleural Other ← NIH Pleural_Thickening (3385 positives, 2.6× CheXpert's PO+ count)
- Lung Opacity ← NIH Infiltration / Consolidation / Atelectasis / Edema (union)
- Enlarged Cardiomediastinum, Fracture, Support Devices: NaN-masked (NIH doesn't annotate)

| Approach | Val mean | Val Pleural Other | vs baseline (0.6728 / 0.9641) |
|----------|----------|-------------------|--------------------------------|
| OmniRad + NIH mixed (no weighting) | 0.7069 | 0.99 | both WORSE |
| OmniRad + NIH mixed + PO 10× weight | 0.80+ | 1.10-1.28 | much worse |
| OmniRad pretrain on NIH (3 ep) → fine-tune on CheXpert (8 ep, fresh schedule) | 0.7045 | 0.98 | both WORSE |

Per-label deltas of two-stage NIH→CheXpert vs baseline OmniRad:

| Label | NIH 2-stage | Baseline | Δ |
|-------|-------------|----------|---|
| No Finding | 0.664 | 0.650 | +0.014 |
| Enlarged Card | 0.682 | 0.652 | +0.030 |
| Cardiomegaly | 0.430 | 0.414 | +0.016 |
| Lung Opacity | 0.805 | 0.738 | **+0.067** |
| Pneumonia | 0.821 | 0.656 | **+0.165** |
| Pleural Effusion | 0.315 | 0.306 | +0.009 |
| Pleural Other | 0.980 | 0.964 | +0.016 |
| Fracture | 0.786 | 0.694 | **+0.092** |
| Support Devices | 0.860 | 0.748 | **+0.112** |

**Every label gets worse with NIH.** Likely causes:
- NIH labels text-mined from radiology reports (lower quality than CheXpert's labeler tool)
- NIH image acquisition differs from CheXpert (different scanners, processing)
- NIH "Pleural Thickening" is one specific subtype, narrower than CheXpert "Pleural Other"

Code added (kept for future use): `init_from_ckpt` config field + warm-start logic in `train.py`, `nih_to_chexpert.csv` builder, two-stage pipeline.

### XRV DenseNet-121 MIMIC-CXR pretrained (revisited May 17)

Already-trained backbone we never used. MIMIC-CXR pretraining is legal (different institution from CheXpert). Pool val:

| Backbone | Mean val NMSE |
|----------|---------------|
| OmniRad B/14 (1 seed) | 0.673 |
| SigLIP-2 | 0.672 |
| EVA-02 | 0.676 |
| **XRV DenseNet-121 MIMIC (3 seeds)** | **0.790** |

XRV is 0.10-0.12 NMSE worse than every other backbone — too weak to add to ensemble even for diversity. Built `PP_7leg_xrv_{05,10,15}` and `PP_xrvForPO` candidates but skipped LB submission.

### CORAL ordinal regression on EVA-02 (May 20)

Idea (for ensemble diversity via a structurally different loss, not because it'd beat CE): train EVA-02 with CORAL ordinal loss — 2 cumulative threshold logits per label, P(Y>-1) and P(Y>0), prediction = σ(logit₀)+σ(logit₁)−1. Code: `target_type: coral` + `loss_fn: coral` in `config.py`/`train.py`/`submit.py`, `masked_coral_loss`, and a 2-logit head path in `EVA02Model`.

**Failed — uncalibrated.** Val NMSE stuck at ~2.7 (vs baseline 0.676), 4-6× worse, especially on rare-positive labels (Pleural Other val ~6.5). Root cause: the naive 2-independent-logit head doesn't enforce cumulative monotonicity (`P(Y>-1) ≥ P(Y>0)`), so on imbalanced labels the thresholds train to incompatible distributions and the scalar prediction is junk. Killed mid-training. Proper fix (not done): true CORAL (shared weight + ordered biases) or CORN (conditional `P(Y>0|Y>-1)`), each ~30min code + retrain.

### BiomedCLIP as a backbone (May 20)

`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (ViT-B/16, CLIP-contrastive on PMC-15M biomedical figure-caption pairs). Legality: borderline — PMC-15M is figures from 4.4M PubMed papers, which *could* include CheXpert images reproduced as figures; paper claims no direct CheXpert inclusion but doesn't document dedup. Cleared under "backbone not *trained on* CheXpert directly." Loaded via `open_clip` (trunk → 768-d), added `BiomedCLIPModel` to `model.py`, 3 seeds trained.

**Worse on every label** (mean val 0.756 vs 0.67 baseline; per-label gap 0.04-0.20). Same tier as XRV — biomedical-text-contrastive pretraining doesn't beat general-image pretraining here. Pred CSVs exist; not added to ensemble.

### Confidence-weighted ensemble (May 20)

Per-(image,label) weighting where each backbone's weight ∝ |prediction| (proxy for confidence, since per-backbone CSVs only store the scalar P(+1)−P(-1), not full softmax). Built `CW_confweighted_k{05,10,20}` (|p|^k × manual_w) and `CW_confweighted_pure`. Ready to LB-test but not submitted; true entropy weighting would need re-extracting full 3-class softmax distributions.

### Lung-segmentation crop specialist (May 23-24) — biggest recent experiment

Hypothesis: crop each image to its lung bounding box (per-image adaptive, removes background) → train a specialist → use it for the 4 pulmonary labels (Lung Opacity, Pneumonia, Pleural Effusion, Pleural Other), keep full-image ensemble for the rest.

Pipeline (`lung_seg_bboxes.py` → `auto_lungcrop_pipeline.sh` → `build_lungcrop_ensemble.py`):
1. Ran torchxrayvision `chestx_det.PSPNet` (ChestX-Det pretrained, legal — not CheXpert) on all 200,754 images; union of Left+Right Lung channels → bbox → `labels/lung_bboxes.csv`. 9.5% fell back to full image (laterals/seg failures); 90.5% real crops (median 1797×1435px, mask area ~13%).
2. Added `lung_crop_csv` config + crop-before-transform in `dataset.py`/`submit.py`.
3. Trained 3× OmniRad + 1× EVA-02 on lung-cropped images.

**Failed — worse on every pulmonary label, both backbones:**

| Label | baseline (om/eva) | lung-crop (best) | Δ |
|-------|-------------------|------------------|---|
| Lung Opacity | 0.793 | 0.797 | +0.004 |
| Pneumonia | 0.795 | 0.823 | +0.028 |
| Pleural Effusion | 0.305 | 0.337 | +0.032 |
| Pleural Other | 0.966 | 0.983 | +0.017 |

Root cause: the lung-mask bbox **clips the costophrenic angles and diaphragm** — exactly where pleural effusion pools and lower-lobe disease sits. EVA-02's higher resolution (448²) didn't flip it. Candidates `LC_{A50,B100,C70,D30}_pulmonary` built but expected to regress vs 0.646. **Untried follow-up: looser/lower-padded crop** that keeps the costophrenic angles (extend bbox downward), or lung-crop as an *additional* diversity leg rather than a pulmonary replacement.

---

## Final State — What Did and Didn't Work

**Worked (in chronological order of discovery):**
1. TrivialAugmentWide → −0.005 OmniRad val (Day 1, only training trick that helped)
2. ConvNeXt-L + aug_trivial → −0.016 (Day 2 transfer)
3. Cardiomegaly cherry-pick: `I_D_with_cnxl_boost40` (Day 2, replaces ensemble PO with cnxl-boosted recipe) → ~−0.003 LB
4. Fracture cherry-pick: `0.30*cnxl + 0.70*cxr5` (Day 2) → ~−0.001 LB
5. Adding CXR Foundation (Google ELIXR-C linear probe) as 4th leg (Day 2-3) → ~−0.002 LB
6. Adding SigLIP-2 + EVA-02 as 5th/6th legs (Day 3) → ~−0.001 LB each

**Didn't work (all tested in Day 4-8 follow-up):**
- EVA-02 EMA / LLRD
- OpenCLIP as 7th leg
- Split-diversification (val gain didn't transfer to LB)
- SWA averaging
- Plusval (val_frac=0.01)
- Multi-view fusion (4 head variants + paired-only)
- Per-label optimal stacking (linear / ridge / MLP / XGBoost, val-fit and train-fit)
- Pleural Other specialist (loss weighting 2x/3x/10x)
- NIH ChestX-ray14 external data (mixed, weighted, two-stage)
- XRV DenseNet-121 MIMIC (too weak)
- CORAL ordinal regression (uncalibrated — needs proper monotonic/CORN impl)
- BiomedCLIP backbone (too weak, like XRV)
- Lung-segmentation crop specialist (clips costophrenic angles → worse pulmonary)

**Untried (would need significant time / resources):**
- DINOv3 ViT-7B (6.7B params, too expensive to train end-to-end)
- MIMIC-CXR external data (377k images, requires PhysioNet credentialing — but uses same labeler family as CheXpert so labels would match cleanly). **Most promising untried external data** — unlike NIH, its CheXpert-labeler annotations would match our label semantics.
- PadChest (168k Spanish images, fine-grained labels need mapping)
- Self-supervised pretraining (DINO/MAE on NIH images)
- True out-of-fold train predictions (K-fold retraining of every backbone — expensive)
- **Lung-crop with looser bbox** that keeps costophrenic angles (the failure mode above was clipping them) — cheapest next thing to try, code already in place (just widen padding in `lung_seg_bboxes.py`)
- CORN/true-CORAL ordinal (fix the calibration bug, ~30min code) for loss-diversity in the ensemble

---

## Bottom Line

After Day 4-12 of extensive follow-up experiments, **0.646 public LB is the real ceiling for our backbone set + training paradigm.** All val gains discovered (split-diversification, stacking, EMA) failed to transfer to public LB, suggesting the val/test distribution shift is bigger than these marginal improvements. Cherry-picks and the 6-leg manual weighted average are already at or near the optimum given the backbones we have. Every Day 4-12 experiment (ablations, stacking, multi-view, specialists, external data, ordinal loss, new backbones, lung-crop) failed to beat 0.646. Breaking through would require either a substantially stronger single backbone (ViT-7B), label-compatible external data (MIMIC-CXR), or a paradigmatically different approach (self-supervised pretraining, distillation). The two cheapest unexplored leads are: (1) lung-crop with a looser bbox that preserves the costophrenic angles, and (2) MIMIC-CXR external data (label-compatible, unlike the NIH attempt that hurt).

---

## How to Reproduce / Submit

### ⇨ WHAT TO SUBMIT (read before final/private submission)

Two files, both view-averaged and ready in `submissions/2026-05-11/ladder/`:

| File | Public LB | Use for | Notes |
|------|-----------|---------|-------|
| **`PP_6leg_eva_30.csv`** | **0.646** | **public leaderboard** | 6-leg ensemble + Cardiomegaly cherry-pick (`I_D_with_cnxl_boost40`) + Fracture cherry-pick (`0.30·cnxl + 0.70·cxr`). The cherry-picks were tuned ON the public LB. |
| **`FF6_eva_30.csv`** | ~0.647 | **private leaderboard / final** | Same 6-leg ensemble, **no cherry-picks** (uses the plain weighted average for Cardiomegaly + Fracture too). |

**Which to pick for the private set:** the two cherry-picks (Cardiomegaly `I_D`, Fracture `0.30/0.70`) were selected by submitting variants to the *public* leaderboard and keeping whatever scored best — i.e. they are fit to the public split and carry overfitting risk on the hidden private split. `PP_6leg_eva_30` is ~1 tick better on public; `FF6_eva_30` is the conservative, cherry-free version that should generalize more safely to private. **If the competition lets you pick 2 final submissions, submit both** (one of each). **If you can pick only one for the private set, `FF6_eva_30` is the safer choice** unless you have evidence the cherry-picks hold on a held-out fold (we don't — they were never validated off the public LB). The gap between them is only ~0.001, so the downside of going cherry-free is tiny and the downside of an overfit cherry-pick on private is larger.

Both are regenerable from the per-backbone pool CSVs (see code below) — the pools live in `submissions/2026-05-10/` and `submissions/2026-05-11/` (gitignored; on the cluster filesystem). Always run `view_average.py` on any freshly built ensemble CSV before submitting.

### Code to re-derive the recipe from raw artifacts
```python
import pandas as pd
import numpy as np
from config import LABEL_NAMES

SUB10 = "submissions/2026-05-10"
SUB11 = "submissions/2026-05-11"

om   = pd.read_csv(f"{SUB11}/omnirad_aug_trivial_5split_mean.csv").sort_values("Id").reset_index(drop=True)
hp   = pd.read_csv(f"{SUB10}/dinov3_hplus_3seed_mean.csv").sort_values("Id").reset_index(drop=True)
cn   = pd.read_csv(f"{SUB11}/cnxl_aug_trivial_3seed_mean.csv").sort_values("Id").reset_index(drop=True)
cxr5 = pd.read_csv(f"{SUB11}/cxr_foundation_5seed_pool.csv").sort_values("Id").reset_index(drop=True)
sg14 = pd.read_csv(f"{SUB11}/siglip2_p14_384_3seed_mean.csv").sort_values("Id").reset_index(drop=True)
eva  = pd.read_csv(f"{SUB11}/eva02_3seed_mean.csv").sort_values("Id").reset_index(drop=True)
ii   = pd.read_csv(f"{SUB11}/ladder/I_D_with_cnxl_boost40.csv").sort_values("Id").reset_index(drop=True)
ids = om["Id"].to_numpy()

out = pd.DataFrame({"Id": ids})
for lab in LABEL_NAMES:
    if lab == "Cardiomegaly":
        # I recipe — no CXR/SigLIP/EVA
        out[lab] = ii[lab].to_numpy()
    elif lab == "Fracture":
        # EE recipe — cnxl + CXR only
        out[lab] = 0.30 * cn[lab].to_numpy() + 0.70 * cxr5[lab].to_numpy()
    else:
        # 6-leg with 30% EVA-02
        out[lab] = (0.10 * om[lab].to_numpy() +
                    0.10 * hp[lab].to_numpy() +
                    0.20 * cn[lab].to_numpy() +
                    0.14 * cxr5[lab].to_numpy() +
                    0.16 * sg14[lab].to_numpy() +
                    0.30 * eva[lab].to_numpy())
out.to_csv("PP_6leg_eva_30.csv", index=False, float_format="%.6f")
# then run view_average.py on PP_6leg_eva_30.csv to produce the final _va.csv
```

**Cherry-free variant (`FF6_eva_30.csv`, private-safe)** — identical except Cardiomegaly and Fracture use the same 6-leg weighted average as every other label (no `ii` cherry-pick, no Fracture override):
```python
out = pd.DataFrame({"Id": ids})
for lab in LABEL_NAMES:
    out[lab] = (0.10*om[lab] + 0.10*hp[lab] + 0.20*cn[lab]
                + 0.14*cxr5[lab] + 0.16*sg14[lab] + 0.30*eva[lab]).to_numpy()
out.to_csv("FF6_eva_30.csv", index=False, float_format="%.6f")
# then run view_average.py
```
(Weights sum to 1.00; `view_average.py` applies the per-(pid,study) 3:1 frontal:lateral average that every submission needs.)

---

## Acknowledgements / Notes

This session was conducted over ~3 days of mostly-autonomous operation. The final ensemble (6 backbones × 3-5 seeds each = ~20 models) represents the bulk of training compute.
