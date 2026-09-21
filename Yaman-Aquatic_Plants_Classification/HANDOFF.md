# Session handoff — aquatic plant classification (YNLT paper + training repo)

Use this file to start a **new conversation** and continue the work. It summarizes what was done, what is where, and what still needs to be done.

---

## 0. Update, 21 Sep 2026 (read this first)

Everything below was written before the second working session. Three things changed.

**YNLT was never looking at the high-resolution image.** Table 1 came from
`n26_validate_ood_ynlt.ipynb` -> `PredictPlants.predict` -> `critical_review`, which patches
the `224x224` tensor the dataloader produced, not the source file. The clean repo copied
that faithfully. It is now a switch: `--ynlt-source original | input224 | both`
(default `both`). `original` re-opens the file and patches the full-resolution photograph,
which is what the paper describes; `input224` reproduces Table 1. The paper text, Table 1
caption and limitations now say which one produced the published numbers.

**Efficiency is measured, not estimated.** The draft's "322 M multiply-adds" was wrong by
about 14x. Measured on the RTX 3070 Ti with `scripts/benchmark_efficiency.py`:
4.47 GMACs and 5.0 ms for one 224 pass, 1.8 ms per patch in batches of 16, 99 patches for a
reviewed 3000x4000 image (111 for a 224 one), so about 187 ms per reviewed image. These are
in `Paper/main_v3.tex` as Table `tab:cost`.

**Two bugs were fixed.** `src/patching.py` raised on the second scale because it stacked
different-sized patches into one array, so YNLT would have crashed on any real image. And
the `convnext` preset used a timm tag that does not exist; the correct ConvNeXt-V1
counterpart is `convnext_tiny.fb_in22k_ft_in1k`.

New tooling, all smoke-tested end to end on synthetic images:

| Script | What it does |
|---|---|
| `scripts/run_matrix.py` | ablation x backbone x seed grid, sequential, skips finished runs |
| `scripts/aggregate_results.py` | mean +/- std tables from `results.json`, plus a LaTeX tabular body |
| `scripts/benchmark_efficiency.py` | params, MACs, latency, patch counts; needs no images |

Every training run now writes `results.json` next to `best.pth` with accuracies, binary
FNR, confusion counts, YNLT trigger rate, patches per reviewed image and ms per image.

Still blocked on the same thing: **the images are not on this machine**, and `Paper/figs/`
is missing so the draft cannot be compiled. Sections 6 and 9 below are still accurate
about what that blocks.

---

## 1. Project goal (advisor feedback)

**Paper contribution (reframed):** not “design a new network,” but **small-data aquatic computer vision + lab-to-field domain shift + confidence-aware adaptive inference (YNLT)**.

**Task:** classify 21 Michigan aquatic plant species (5 invasive) from limited lab images, and generalize to crowd-sourced field photos.

**Reported results (single split, not yet repeated):**

| Setting | Baseline | With full method |
|---|---|---|
| Laboratory (21-way) | 84.78% | 93.48% |
| OOD field (21-way) | 47.06% | 64.71% |

**Advisor priorities (in rough order):**

1. Paper LaTeX updates (intro, YNLT math, efficiency, ablation text)
2. Stronger baselines (ResNet, EfficientNet, ViT, ConvNeXt) under same frozen-backbone recipe
3. Repeated stratified splits (mean ± std) — **not** 5-fold CV (some species have only 2 lab images)
4. Ablation: isolate ConvNeXt-V2 backbone vs gated head vs YNLT
5. Computational efficiency / underwater-robot claim (measure or remove)
6. Reference verification (`ref.bib` was missing; partially recreated)
7. Paper reading (BioCLIP, etc.) — **deprioritized** by user in favor of paper + experiments

---

## 2. Repository layout (after cleanup)

### This repo — **active training workspace**

```
Yaman-Aquatic_Plants_Classification/
  train.py              # main entry: train frozen backbone + head
  evaluate.py           # score checkpoint with/without YNLT
  config.py             # paths, hyperparams, YNLT thresholds
  requirements.txt
  README.md             # quick start
  HANDOFF.md            # this file
  src/
    model.py            # timm backbones + linear / gated / MLP heads
    data.py             # lab + OOD splits, folder-name remapping
    ynlt.py             # confidence-gated second look
    patching.py         # multi-scale high-res patches
    metrics.py          # accuracy + invasive binary FNR
  scripts/
    run_matrix.py           # experiment grid, skips finished runs
    aggregate_results.py    # mean +/- std tables, LaTeX fragment
    benchmark_efficiency.py # params, MACs, latency, patch counts
  runs/models/          # checkpoints + results.json (created on first train)
```

**Images are not in this repo.** By default, training reads sibling folders (adjust paths in `config.py` or env vars):

- Lab: `../Aquatic Plant lab` or `AQUATIC_LAB_DATA`
- Field/OOD: `../Aquatic Plant outdoor` or `AQUATIC_OOD_DATA`

### Sibling folders under `YamanA/` (may or may not be in the same GitHub repo)

| Path | Role |
|---|---|
| `Paper/main_v3.tex` | Updated LaTeX draft |
| `Paper/ref.bib` | Recreated bibliography (ecology entries need verification against original) |
| `Work_Notes/YNLT_math_and_literature.md` | Working notes tied to paper edits |
| `AquaticPlantsClassification/` | **Archive only** — original notebooks, figures, Linux paths; do not train here |

---

## 3. What was changed in the paper (`Paper/main_v3.tex`)

**No new training was run.** Table numbers were kept; text was corrected to match the code.

### Introduction

- Shortened and reframed around: small lab data, field shift, confidence-gated second look.
- Added literature contrast: RAM, RA-CNN, multi-crop, MSDNet, selective prediction, OHEM, calibration papers.
- Removed unsupported claim that the model is ready for underwater robots (moved to limitations / future work).

### Abstract

- Fixed lab count: **225 → 230** images (matches training notebook split 138/46/46).

### Gated attention head (methods)

- Documented actual implementation from code:
  - \(u = \mathrm{GELU}(W_u h + b_u)\)
  - \(g = \sigma(\tanh(W_g h + b_g))\)
  - logits from \(u \odot g\) → dropout → linear
  - attention dim **168**, **261,933** trainable params, backbone frozen

### YNLT math (critical fix)

The draft previously had incomplete/wrong math. **Ground truth is the code** (`AquaticPlantsClassification/.../predict.py` + `config.py`, now mirrored in `src/ynlt.py` + `config.py`):

**Softmax probabilities** from logits \(z\): \(p_i = \mathrm{softmax}(z)_i\).

**Metrics (natural log, not log₂):**

- Confidence = \(p_{(1)}\)
- Margin = \(p_{(1)} - p_{(2)}\)
- Entropy = \(-\sum_i p_i \log p_i\)

**Confusion count** \(c \in \{0,1,2,3\}\):

- +1 if confidence **< 0.5**
- +1 if margin **< 0.2**
- +1 if entropy **> 2.0**

**Trigger YNLT** if \(c \ge 1\) (`critical_confusion_level = 1`).

**Skip YNLT** if first prediction is already **invasive** (`ignore_invasive_predictions = True`).

**Patching:** scales `[0.2, 0.3, 0.4, 0.5]`, overlaps `[0.0, 0.1, 0.2, 0.3]`.

**Vote weight:** \(w = (3 - c_{\text{patch}}) / 3\), class scores start at 1.0, sum weights. **Tie-break:** prefer invasive class (not a top-k override).

### Training / efficiency section

- Documented: PyTorch, frozen ConvNeXt-V2 Tiny, AdamW, cosine LR, 250 epochs, label smoothing 0.05, batch 4 on paper laptop (4 GB A500).
- Stated ~28.1M params, ~322M MACs per 224×224 forward; YNLT multiplies cost by number of patches.
- Robot deployment = **future work** until measured on embedded hardware.

### Ablation (Results text)

- Table 1 already had 4 rows (linear, linear+YNLT, gated, gated+YNLT on shared ConvNeXt-V2 Tiny).
- **Fixed narrative:** old text wrongly attributed gated-attention gain to the “second row” (which is linear+YNLT).
- Added honesty: single split, 46 lab test images → ~2.2% per one flip; need repeated splits + other backbones.

### Conclusion / broader impact

- Softened deployment claims; framed as laptop/field screening until latency is measured.

---

## 4. Bibliography (`Paper/ref.bib`)

- **Was missing**; created so `\bibliography{ref}` compiles.
- Added CV-related keys used in the new intro (Mnih RAM, Fu RA-CNN, Geifman selective, Huang MSDNet, etc.).
- **Several ecology/AIS entries are placeholders** reconstructed from cite keys — replace with Aaryan’s original `.bib` when available.

---

## 5. Clean training repo (`Yaman-Aquatic_Plants_Classification`)

Built to avoid 40+ notebooks in `AquaticPlantsClassification`. Only the pipeline needed for paper experiments.

### Design choices

- **Frozen timm backbone**, train classifier head only (same as paper).
- **Head types:** `linear`, `gated_attention`, `mlp`.
- **Backbone presets** in `config.py`: `convnextv2`, `convnext`, `resnet50`, `efficientnet`, `vit`.
- **OOD model selection:** save checkpoint with best **OOD validation accuracy** (matches phase-7 notebook behavior).
- **Folder names:** strips parentheticals, e.g. `Brasenia schreberi (watershield)` → `Brasenia schreberi`.

### Hardware defaults (user machine)

- **GPU:** RTX 3070 Ti (8 GB)
- **CPU:** i9-12900K
- Defaults vs paper laptop: `batch_size=32`, `num_workers=4`, `pin_memory=True`, **AMP on**
- Fallbacks: `--batch-size 16`, `--workers 0`, `--no-amp`

### Commands

```powershell
cd Yaman-Aquatic_Plants_Classification
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Ablation-style runs (seed 8)
python train.py --head linear --backbone convnextv2
python train.py --head gated_attention --backbone convnextv2
python train.py --head gated_attention --backbone convnextv2 --with-ynlt

# Stronger backbones (linear head, same recipe)
python train.py --head linear --backbone resnet50
python train.py --head linear --backbone efficientnet
python train.py --head linear --backbone vit
python train.py --head linear --backbone convnext

# Repeated splits (advisor request) — NOT 5-fold
python train.py --head gated_attention --backbone convnextv2 --seed 8
python train.py --head gated_attention --backbone convnextv2 --seed 9
# ... through seed 12

# Or run the whole grid at once (skips anything already finished)
python scripts\run_matrix.py --suite all --dry-run
python scripts\run_matrix.py --suite all
python scripts\aggregate_results.py --latex

# Evaluate saved checkpoint
python evaluate.py runs\models\<timestamp_folder>\best.pth --with-ynlt

# Cost numbers for the paper; needs no images
python scripts\benchmark_efficiency.py
```

Checkpoints: `runs/models/<timestamp>_<head>_<backbone>_seed<N>/best.pth`

---

## 6. Dataset caveat (important)

| Source | Lab images | OOD/field |
|---|---|---|
| Paper / training notebook | **230** (138 train / 46 val / 46 test) | **33** (16 val / 17 test) |
| Folders counted on disk earlier | **203** | **42** |

The clean repo **will load** the 203/42 folders, but those are **not** the exact list behind Table 1 in the paper. To replicate 84.78% → 93.48% / 47.06% → 64.71%, you need the **original preprocessed set** (likely `squared_reflected` + `hand` from the Linux project), or confirm with Aaryan which image list is official.

Rare classes (problem for stratified splits): e.g. *Hydrocharis morsus-ranae* (2), *Nymphaea odorata* (2), *Nuphar variegata* (3), *Potamogeton praelongus* (3).

---

## 7. Ablation table (paper Table 1 — existing numbers, single split)

All rows use **frozen ConvNeXt-V2 Tiny** (`convnextv2_tiny.fcmae_ft_in22k_in1k`).

| Model | Lab class acc | Lab bin acc | Lab bin FNR | OOD class acc | OOD bin acc | OOD bin FNR |
|---|---|---|---|---|---|---|
| Linear classifier | 84.78 | 93.48 | 30.0 | 47.06 | 58.82 | 62.5 |
| Linear + YNLT | 91.30 | 93.48 | 30.0 | 52.94 | 58.82 | 62.5 |
| Gated attention | 86.96 | 97.83 | 10.0 | 52.94 | 76.47 | 12.5 |
| Gated + YNLT | **93.48** | **97.83** | **10.0** | **64.71** | **76.47** | **12.5** |

**Interpretation (for paper):**

- YNLT on linear head: big lab gain, modest OOD gain; FNR unchanged on reported split.
- Gated head alone: helps OOD and strongly cuts invasive FNR.
- Full system: best 21-way lab and OOD accuracy.

**Still missing for advisor:**

- Same table with **ResNet / EfficientNet / ViT / ConvNeXt** backbones
- **Mean ± std** over seeds 8–12 (or 5 repeated 60/20/20 splits)
- Optional: BioCLIP linear probe as biology-specific baseline
- % of test images that trigger YNLT + latency (ms) with and without YNLT

---

## 8. YNLT implementation reference (config.py)

```python
critical_confusion_level = 1
ignore_invasive_predictions = True
margin_threshold = 0.2
entropy_threshold = 2.0
confidence_threshold = 0.5
patching_method = "multi-scale"
ms_scale = [0.2, 0.3, 0.4, 0.5]
ms_overlap = [0.0, 0.1, 0.2, 0.3]

ynlt_patch_source = "original"   # "original" = full-res file, "input224" = Table 1 behaviour
ynlt_patch_batch_size = 16
```

Training hyperparams (paper): `num_epochs=250`, `lr=0.0005`, `weight_decay=2e-5`, `label_smoothing=0.05`, `gated_attention_dim=168`, `classifier_head_dropout=0.1`.

---

## 9. TODO list for next session

### Experiments (highest impact) — all blocked on the images

- [ ] Obtain official **230 lab + 33 OOD** image paths (or confirm 203/42 is acceptable)
- [ ] `python scripts/run_matrix.py --suite all` covers the next four items in one command
- [ ] Run 4-row ablation with `train.py` on seed 8; compare to Table 1
- [ ] Run linear head on `resnet50`, `efficientnet`, `vit`, `convnext`
- [ ] Repeat main config for seeds **8, 9, 10, 11, 12**; report mean ± std
- [ ] Compare `--ynlt-source original` against `input224` (the new, most informative row)
- [x] Log **YNLT trigger rate** and wall-clock time per image — in every `results.json`
- [ ] (Optional) BioCLIP frozen encoder + linear head

### Paper

- [ ] Merge new results into Table 1 / add backbone comparison table
      (`python scripts/aggregate_results.py --latex` prints the tabular body)
- [x] Efficiency subsection now has measured numbers (Table `tab:cost` in `main_v3.tex`)
- [x] Document which patch source produced Table 1
- [ ] Replace the 11 `% UNVERIFIED` entries in `ref.bib` with Aaryan's originals
      (11 others are now verified with DOIs)
- [ ] Recover `Paper/figs/` — it is **not** in this repository, so the draft will not build
- [ ] Compile `main_v3.tex` with **XeLaTeX or LuaLaTeX** (it uses `fontspec` and Times New
      Roman); the `svg` package also needs Inkscape on `PATH`

### Code / repo

- [x] Aggregation script for multi-seed results (`scripts/aggregate_results.py`)
- [x] Root `Readme.md`, root `.gitignore`, and removal of the stale `D:\Aquatic_Research` paths
- [ ] Consider copying `Paper/` into the training repo, or leave it at the repo root as now

### Deprioritized

- [ ] Paper reading (BioCLIP, BioCLIP 2) — user chose to focus on paper + code first
- [ ] Full notebook archive in `AquaticPlantsClassification` — reference only

---

## 10. Key files to open in a new chat

| File | Why |
|---|---|
| `config.py` | paths, batch size, YNLT thresholds |
| `train.py` | training entry point |
| `src/ynlt.py` | second-look logic |
| `Paper/main_v3.tex` | draft with advisor-driven edits |
| `Paper/ref.bib` | citations (needs verification) |
| `../Work_Notes/YNLT_math_and_literature.md` | extra notes on math + literature |

---

## 11. Suggested prompt for the next conversation

Copy-paste something like:

> I'm continuing work on the aquatic plant YNLT paper. Read `Yaman-Aquatic_Plants_Classification/HANDOFF.md` first. My repo is the clean training folder; paper is in `Paper/main_v3.tex`. I have an RTX 3070 Ti. Next I want to [run ablation / add backbone baselines / repeated splits / update LaTeX with new numbers]. Help me with [specific task].

---

## 12. What was **not** done

- No training on real data, so **no new accuracy numbers** are in the paper tables. The
  only runs executed were synthetic smoke tests, and their output was deleted.
- `Paper/figs/` was not recovered, so `main_v3.tex` has not been compiled
- The 11 `% UNVERIFIED` bibliography entries were not resolved
- `AquaticPlantsClassification` was not deleted (intentionally kept as archive); the
  patching and preset bugs were fixed only in the clean repo, not in the archive
- GitHub remote / CI / release tags not configured

---

*Last updated 21 Sep 2026: YNLT patch-source fix, measured efficiency numbers, experiment
scripts, bibliography verification. Earlier session: paper revision + clean repo extraction
for GitHub. User: Yaman. Advisor context: Fengying Dang, Michigan Tech.*
