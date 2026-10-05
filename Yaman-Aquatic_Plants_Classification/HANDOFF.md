# Session handoff — aquatic plant classification (YNLT paper + training repo)

Use this file to start a **new conversation** and continue the work. It summarizes what was
done, what is where, and what still needs to be done. Newest update first.

---

## 0. Update, 4 Oct 2026 (read this first)

Outcome of the meeting with Dr. Dang and what was built in response.

### What the advisor asked for

1. **Weights vs. samples.** "How many parameters does the model train, and how many samples
   do you need for that many weights?" Answered in `Reports/Parameters_vs_Samples.md`
   (plain language) with the numbers from `scripts/parameter_budget.py`
   (`Reports/parameter_budget_tables.md`). Short version: the backbone is frozen
   (27.87 M weights not trained); the heads train 16,149 (linear), 33,201 (MLP) or
   261,933 (gated) weights on 768-d features. Lab alone gives 5.8 training images per
   class; lab+web gives 38.4. The textbook "10 samples per weight" rule does not apply to
   a frozen-feature probe, so the measured answer is the **learning curve** (`curve`
   suite) plus the 95% CI table for each test set.
2. **Train on the new, larger dataset combined with the old one.** Done: `--data lab+web`.
3. **Full-resolution vs. 224 YNLT showed no difference → drop it.** Done: `input224` is
   the default patch source, `original` is an ablation option, the paper's full-res rows
   are gone (text keeps one sentence saying they matched within a point at 5–10x cost).
4. **Find the best combination.** The `data` suite (3 data configs x 3 heads x 5 seeds)
   plus `backbones`, `small` and `curve` suites; `aggregate_results.py` builds the tables.
5. **Mention the underwater-rover goal** as the reason for ConvNeXt-V2 Tiny. Added to the
   abstract, introduction, backbone section, cost table and discussion of `main_v3.tex`,
   and Nano/Pico/Atto presets were added as smaller rover candidates.

Decisions taken with the user: the 42 outdoor photographs are a **pure held-out field
test** (checkpoint selection on the lab+web validation split, outdoor never seen); and the
1,140 GBIF-mediated images are used now, a fresh iNaturalist download (the earlier one was
75-px thumbnails) is optional later work.

### The web dataset

`Larger_Plants_Dataset/aquatic_dataset_report/aquatic_plant_images/<species>/*.jpg`,
1,140 images from 532 iNaturalist observations (up to 15 photos each), downloaded through
GBIF; `download_manifest.csv` has the occurrence key, licence and rights holder per file.
Licences: 763 CC BY-NC, 124 BY-NC-ND, 110 BY, 77 CC0, 26 BY-ND, 6 BY-NC-SA, rest other.
Sizes 354x265 to 2048x2048, median width 1536. Per-species counts are in `tab:data` of the
paper and in `parameter_budget_tables.md`.

### Protocol (new)

- `scripts/build_splits.py` writes `data/splits/<data>_seed<N>.csv` (`lab`, `web`,
  `lab_web`; seeds 8–12; all 15 exist). Columns: path, label, label_idx, source, group,
  split, license, rights_holder.
- Stratified 60/20/20 per species. **Web images are grouped by observation** so photos of
  one plant never straddle train and test. The split RNG is seeded per (source, species),
  so the lab partition is identical in `lab` and `lab+web` for the same seed.
- Every run is scored on the **same three test sets** of its seed: lab test (39), web test
  (210), field test (42 outdoor). For `lab`/`web`-only runs the test rows are read from the
  `lab_web` list of the same seed; `get_split_loaders` asserts there is no overlap with the
  run's train/val rows.
- Checkpoint = best **validation accuracy** on the training data's val split. OOD-based
  selection is gone.
- `--train-fraction f` keeps a fraction of the training *groups* per class (≥1) for the
  learning curve.
- `results.json` keys: `lab_test`, `web_test`, `field_test`, and `<name>_ynlt_input224`.
  Run folders: `<stamp>_<data>_<head>_<backbone>_seed<N>[_frac<f>]`. Fourteen archived
  runs from the old protocol (no `data` key) are ignored by `aggregate_results.py`.

### GPU data path (4 Oct, afternoon)

Even after the worker fixes below the GPU sat at 5 % utilisation: eight CPU workers
decoding 1500-px JPEGs and running the torchvision augmentations could feed only ~90
images/s. `src/gpu_data.py` now decodes every image once (PIL `draft` + resize to 320x320,
cached in `data/cache/images_320_<hash>.pt`, 425 MB for 1,385 images), keeps the bank on
the GPU as uint8, and applies `cfg.aug_tf` as batched GPU kernels with per-image random
parameters (crop/flip/rotation/perspective/TrivialAugment-geometry composed into one
homography + `grid_sample`; ColorJitter, TrivialAugment colour ops, Normalize,
RandomErasing as tensor arithmetic). Validation features are computed once per run
(frozen backbone). Result: **1.2 s/epoch on lab+web, ~5 min per run, ~1 min for lab-only;
the 93-job grid takes ~7–8 h.** `train.py --loader cpu` restores the old path. The 27 runs
made with the CPU path before the switch are archived in `runs/models_cpu_loader/` and are
not read by `aggregate_results.py`; the grid was restarted from zero on the GPU path so all
paper numbers come from one pipeline. Known small deviations from torchvision are listed
in the module docstring (bilinear vs bicubic, Equalize→AutoContrast, crop from the 320-px
squashed image).

### Windows DataLoader fixes (CPU path; keep these)

Workers were being respawned every epoch (`persistent_workers=True` fixes it), and with 8
workers OpenBLAS allocated per-thread buffers in each worker until one died with "Memory
allocation still failed after 10 retries" and the run hung. Fix: `OPENBLAS_NUM_THREADS=1`
is set before numpy is imported (`train.py`, `evaluate.py`, `config.py`), each worker calls
`torch.set_num_threads(1)` and `cv2.setNumThreads(0)`, eval loaders use 2 workers. Timing:
loader ~4 s/epoch, GPU ~1.1 s/epoch, observed ~9–10 s/epoch on lab+web (834 train images)
→ ~35–40 min per 250-epoch run; lab-only ~6 min. 12 workers gave no speedup.

### Experiment grid

```powershell
# launched 4 Oct 2026 in this order; logs in runs\logs\matrix_<suite>.log
foreach ($suite in @('data','curve','small','backbones')) { python -u scripts\run_matrix.py --suite $suite }
```

93 distinct jobs (45 + 18 + 9 + 21), about 7–8 hours on the GPU path (relaunched 09:30 on
4 Oct). `run_matrix.py` skips jobs that already have a `results.json`, so the loop can be
re-run after an interruption.

### Grid results (finished 17:36, 4 Oct; 93/93 runs, no failures)

All numbers are in the paper (`tab:datagrid`, `tab:backbones`, `tab:cost`, `fig:curve`;
no `\tbd` or `TODO(results)` left). Headlines, mean over seeds, frozen ConvNeXt-V2 Tiny:

| Training data | Head | Lab acc | Web acc | Outdoor acc | Lab FNR |
|---|---|---|---|---|---|
| Lab | linear | 80.0 | 17.5 | 25.7 | 11.1 |
| Web | linear | 40.0 | 54.3 | 36.2 | 37.8 |
| Lab + web | linear | 81.5 | 57.7 | 35.2 | 13.3 |
| Lab + web | gated | 83.6 | 56.3 | 31.4 | 6.7 |
| Lab + web | gated + YNLT | 81.0 | 55.5 | 32.4 | 2.2 |

- Adding the web images is the biggest effect: keeps lab accuracy, triples web accuracy,
  +10 points outdoors. Heads are within one std of each other at matched data.
- YNLT costs 1–6 points of 21-way accuracy and cuts invasive FNR everywhere (outdoor
  63–70 % → 37 %). Trigger rate 14–21 % lab, 31–34 % web, 38–41 % outdoor.
- Encoders (lab+web, linear): V2 Tiny 82.9/56.0/34.9; V1 Tiny 78.6/53.6/**47.6**;
  ViT-B 74.4/51.5/32.5; Nano 75.2/52.9/33.3; Pico 61.5/40.2/22.2; Atto 56.4/38.5/27.8;
  ResNet-50 59.0/40.4/22.2; EffNetV2-B0 60.7/36.2/14.3. Nano is the rover fallback.
- Learning curve (6/12/22/40 images per class): lab and web still rising ~10 points per
  doubling; outdoor flat at ~33 % from 22 images/class. Gated head is not more
  data-hungry than linear.
- GPU latencies (batch 1 / reviewed image): Tiny 5.2/209 ms, Nano 4.2/129, Pico 3.9/94,
  Atto 3.9/61, V1 Tiny 3.8/153, ResNet-50 3.6/129, EffNetV2-B0 5.7/59, ViT-B 5.0/383.

Caveat for the paper: ConvNeXt-V1 Tiny beating V2 on the 42 outdoor images (47.6 vs
34.9, ~5 images) is flagged in the text as needing a larger outdoor set.

### Remaining

1. Compile `main_v3.tex` (XeLaTeX/LuaLaTeX; `Paper/figs/` still lacks map, data_dist,
   augmentations and the qualitative figures) and proofread the new Results/Discussion.
2. Optional: re-download iNaturalist originals as a fourth source; BioCLIP encoder;
   a larger outdoor test set (the V1-vs-V2 outdoor gap and all outdoor FNRs are noisy).

### Files touched this session

| File | Change |
|---|---|
| `config.py` | `WEB_DATA_DIR`, `WEB_MANIFEST`, `SPLITS_DIR`, `DATA_CONFIGS`, `split_csv_path`; `num_workers=8`, `eval_workers=2`, `persistent_workers`, `prefetch_factor=4`; `ynlt_patch_source="input224"`; Nano/Pico/Atto presets |
| `scripts/build_splits.py` | new; deterministic observation-grouped splits |
| `scripts/parameter_budget.py` | new; capacity vs. samples tables |
| `scripts/plot_learning_curve.py` | new |
| `scripts/fill_paper_tables.py` | new; fills `tab:datagrid` / `tab:backbones` rows from `results.json` |
| `scripts/run_matrix.py` | rewritten around `Job(data, head, backbone, seed, fraction)` and the four suites |
| `scripts/aggregate_results.py` | rewritten; groups by data config, three test sets, hides `ynlt_original` unless `--show-original` |
| `scripts/benchmark_efficiency.py` | patch-source sizes now 224 / 1500x1000 / 1536x2048 / 3000x4000 |
| `src/data.py` | `ManifestDataset`, `read_split_records`, `subsample_train_groups`, `get_split_loaders(..., test_csv=)`, worker init |
| `src/gpu_data.py` | new; `ImageBank`, `gpu_augment`, `gpu_eval_transform`, `GpuSplit`, `get_gpu_splits` |
| `train.py`, `evaluate.py` | `--data`, `--split-csv`, `--train-fraction`; val-accuracy selection; three test sets |
| `Paper/main_v3.tex` | abstract, intro, dataset (`tab:data`), capacity subsection, rover motivation, 8-row `tab:cost`, results restructured into `tab:datagrid` / `tab:backbones` / `fig:curve`, discussion and conclusion rewritten; `\tbd` placeholders |
| `Reports/Parameters_vs_Samples.md`, `Reports/parameter_budget_tables.md` | new |
| `Reports/WeeklyReport_Week6_2026-10-05.md` | new |
| `README.md`, root `Readme.md` | updated for the new protocol |

---

## 0b. Update, 21 Sep 2026

**YNLT was never looking at the high-resolution image.** Table 1 came from
`n26_validate_ood_ynlt.ipynb` -> `PredictPlants.predict` -> `critical_review`, which patches
the `224x224` tensor the dataloader produced, not the source file. It became a switch,
`--ynlt-source original | input224 | both`. (4 Oct: `original` matched `input224` within a
point at 5–10x the cost, so `input224` is now the default and the paper drops the ablation.)

**Efficiency is measured, not estimated.** The draft's "322 M multiply-adds" was wrong by
about 14x. Measured on the RTX 3070 Ti with `scripts/benchmark_efficiency.py`:
4.47 GMACs and 5.0 ms for one 224 pass, 1.8 ms per patch in batches of 16, 111 patches for
a reviewed 224 input, about 205 ms per reviewed image. These are in `tab:cost`.

**Two bugs were fixed.** `src/patching.py` raised on the second scale because it stacked
different-sized patches into one array. And the `convnext` preset used a timm tag that does
not exist; the correct ConvNeXt-V1 counterpart is `convnext_tiny.fb_in22k_ft_in1k`.

Every training run writes `results.json` next to `best.pth` and TensorBoard event files
(`tensorboard --logdir runs\models`; `--no-tensorboard` disables it).

---

## 1. Project goal (advisor feedback)

**Paper contribution:** small-data aquatic computer vision + lab-to-field domain shift +
confidence-aware adaptive inference (YNLT), with a frozen encoder chosen for deployment on
an underwater rover.

**Task:** classify 21 Michigan aquatic plant species (5 invasive) from limited lab images
plus web-sourced field photographs, and generalize to unseen outdoor photographs.

**Original single-split numbers (archive notebooks, 230/33 image lists, OOD-selected
checkpoint; kept in the paper as `tab:ablation` for reference):**

| Setting | Baseline | With full method |
|---|---|---|
| Laboratory (21-way) | 84.78% | 93.48% |
| OOD field (21-way) | 47.06% | 64.71% |

**Advisor priorities (in rough order):**

1. Paper LaTeX updates — done, numbers pending the grid
2. Stronger baselines under the same frozen-backbone recipe — `backbones` suite
3. Repeated stratified splits (mean ± std), not 5-fold — seeds 8–12
4. Ablation: backbone vs. head vs. YNLT — `data` suite rows
5. Computational efficiency / rover claim — measured, `tab:cost`
6. Reference verification (`ref.bib`) — 11 `% UNVERIFIED` entries remain
7. Weights vs. samples — `Reports/Parameters_vs_Samples.md` + learning curve

---

## 2. Repository layout

```
Yaman-Aquatic_Plants_Classification/
  train.py  evaluate.py  config.py  requirements.txt  README.md  HANDOFF.md
  data/splits/            15 split CSVs (lab, web, lab_web x seeds 8-12)
  src/                    model.py data.py ynlt.py patching.py metrics.py
  scripts/                build_splits.py run_matrix.py aggregate_results.py
                          plot_learning_curve.py parameter_budget.py benchmark_efficiency.py
  runs/models/            checkpoints + results.json + TensorBoard events
  runs/logs/              matrix logs
```

Images live one level up (`config.py` or env vars): `Aquatic Plant lab` (203),
`Larger_Plants_Dataset/aquatic_dataset_report/aquatic_plant_images` (1,140 +
`download_manifest.csv`), `Aquatic Plant outdoor` (42).

Sibling folders at the repository root: `Paper/` (`main_v3.tex`, `ref.bib`; `figs/` is
still missing except what `plot_learning_curve.py` will create), `Reports/`,
`Work_Notes/`, `AquaticPlantsClassification/` (archive only).

---

## 3. YNLT implementation reference (config.py)

```python
critical_confusion_level = 1        # trigger if >= 1 of the three flags fires
ignore_invasive_predictions = True  # skip review when the first guess is invasive
confidence_threshold = 0.5
margin_threshold = 0.2
entropy_threshold = 2.0             # natural log
patching_method = "multi-scale"
ms_scale = [0.2, 0.3, 0.4, 0.5]
ms_overlap = [0.0, 0.1, 0.2, 0.3]
ynlt_patch_source = "input224"      # "original" = full-res file (ablation only)
ynlt_patch_batch_size = 16
```

Vote weight per patch \(w = (3 - c)/3\), class scores start at 1.0; tie-break prefers the
invasive class. Training: 250 epochs, AdamW, lr 5e-4, weight decay 2e-5, cosine schedule,
label smoothing 0.05, gated dim 168, head dropout 0.1, batch 32, AMP.

---

## 4. Paper status (`Paper/main_v3.tex`)

Structure after 4 Oct: Introduction (rover motivation, 4 contributions) → Materials and
Methods (study area; Dataset with lab / web / outdoor subsections and `tab:data`;
preprocessing; backbone with rover paragraph; `subsec:capacity`; gated head; YNLT math;
`tab:ynlt_outcomes`; `tab:cost` with eight backbones; training protocol) → Results
(`tab:ablation` original single split; `subsec:datagrid` / `tab:datagrid`;
`subsec:backbones` / `tab:backbones`; `subsec:curve` / `fig:curve`; qualitative figures)
→ Discussion → Conclusion.

All tables and the learning-curve figure are filled from the 93-run grid; the `\tbd`
macro is unused and can be deleted. All `\ref`s resolve.
The draft needs XeLaTeX/LuaLaTeX (`fontspec`) and Inkscape for `svg`; `Paper/figs/` is
still absent apart from the learning-curve figure.

---

## 5. What was **not** done

- `Paper/figs/` (map, data distribution, augmentations, qualitative figures) not recovered.
- 11 `% UNVERIFIED` bibliography entries unresolved.
- iNaturalist originals not re-downloaded (the earlier rows were 75-px thumbnails).
- `AquaticPlantsClassification` kept as archive, untouched.

---

## 6. Suggested prompt for the next conversation

> I'm continuing the aquatic plant YNLT paper. Read
> `Yaman-Aquatic_Plants_Classification/HANDOFF.md` first (section 0). The experiment grid
> from `scripts/run_matrix.py` has [finished / is at job N]. Help me run
> `aggregate_results.py --latex`, `plot_learning_curve.py` and `benchmark_efficiency.py`,
> fill the `\tbd` cells and `% TODO(results)` comments in `Paper/main_v3.tex`, and rewrite
> the Discussion around the actual numbers.

---

*Last updated 4 Oct 2026: combined lab+web protocol, observation-grouped splits, held-out
field test, capacity-vs-samples report, rover motivation, paper restructured, grid
launched. Earlier: 21 Sep 2026 YNLT patch-source fix and measured efficiency; first session
paper revision + clean repo. User: Yaman. Advisor: Fengying Dang, Michigan Tech.*
