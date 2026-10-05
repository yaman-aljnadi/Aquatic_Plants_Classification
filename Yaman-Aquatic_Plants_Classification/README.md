# Yaman aquatic plant classification — clean training workspace

Use this folder for new runs. The original notebooks stay in `AquaticPlantsClassification/`
as an archive. `HANDOFF.md` has the full project status.

## Layout

```
Yaman-Aquatic_Plants_Classification/
  train.py          start training here; writes results.json per run
  evaluate.py       score a saved checkpoint on the three test sets
  config.py         paths, batch size, YNLT thresholds and patch source, backbone presets
  data/splits/      one CSV per (data config, seed): which image is train/val/test/field_test
  data/cache/       decoded 320x320 uint8 image bank (built automatically, not tracked)
  src/
    model.py        timm backbones + linear / gated / MLP heads
    gpu_data.py     GPU-resident image bank + batched GPU augmentation (default training path)
    data.py         ManifestDataset + get_split_loaders (CPU DataLoader path, --loader cpu)
    ynlt.py         confidence-aware second look
    patching.py     multi-scale crops
    metrics.py      accuracy and invasive FNR
  scripts/
    build_splits.py         deterministic, observation-grouped splits -> data/splits/*.csv
    run_matrix.py           experiment suites (data, backbones, small, curve), skips finished runs
    aggregate_results.py    mean +/- std tables (Markdown and LaTeX)
    fill_paper_tables.py    writes those rows into the placeholder tables of Paper/main_v3.tex
    plot_learning_curve.py  accuracy vs. number of training images -> Paper/figs/learning_curve.pdf
    parameter_budget.py     trainable weights per head vs. images per class, CI tables
    benchmark_efficiency.py params, MACs, latency, patch counts (no images needed)
  runs/models/      checkpoints, results.json and TensorBoard events written here
  runs/logs/        run_matrix logs
```

## Images

Images are **not** copied here. Three sources are read from the repository root (one level
up), or from the environment variables in parentheses:

| Source | Default folder | Count | Role |
|---|---|---|---|
| lab | `../Aquatic Plant lab` (`AQUATIC_LAB_DATA`) | 203 | train / val / test (60/20/20 per species) |
| web | `../Larger_Plants_Dataset/aquatic_dataset_report/aquatic_plant_images` (`AQUATIC_WEB_DATA`) | 1,140 | train / val / test, grouped by iNaturalist observation |
| field | `../Aquatic Plant outdoor` (`AQUATIC_OOD_DATA`) | 42 | **held-out field test only**, never used for training or model selection |

The web folder must contain `download_manifest.csv` (written by the GBIF download script);
`build_splits.py` uses its `gbif_occurrence_key` column so that photographs of the same
observation never straddle train and test.

## Setup (RTX 3070 Ti + i9-12900K)

```powershell
cd Yaman-Aquatic_Plants_Classification
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Defaults: batch 32, mixed precision, channels-last, and the **GPU data path**
(`src/gpu_data.py`): every image is decoded once, squashed to 320x320 and cached in
`data/cache/` (~28 s the first time, 425 MB), the whole set lives on the GPU as uint8, and
the training augmentation (`cfg.aug_tf`) runs as batched GPU kernels. Validation features
are computed once because the backbone is frozen. A 250-epoch run takes about **1 min on
lab-only data and 5 min on lab+web** (1.2 s/epoch); the old DataLoader path
(`--loader cpu`, 8 workers, `OPENBLAS_NUM_THREADS=1`) took 35–40 min because the GPU sat at
5 % utilisation waiting for JPEG decoding. If you run out of VRAM: `--batch-size 16`.

## Commands

Build the split lists once (deterministic; re-running is a no-op unless `--force`):

```powershell
python scripts\build_splits.py --data all --seeds 8 9 10 11 12
```

Train one model. `--data` is `lab`, `web` or `lab+web` (default). Every run is scored on
the same lab test (39), web test (210) and field test (42) images of that seed:

```powershell
python train.py --data lab+web --head gated_attention --backbone convnextv2 --seed 8 --with-ynlt
python train.py --data lab     --head linear          --backbone convnextv2 --seed 8
python train.py --data lab+web --head linear --backbone convnextv2 --seed 8 --train-fraction 0.25
```

Whole suites (each skips jobs that already have a `results.json`):

| Suite | Runs | What it answers |
|---|---|---|
| `data` | 3 data configs x 3 heads x seeds 8–12 = 45 | does the web data help, which head is best |
| `backbones` | ConvNeXt-V1, ResNet-50, EfficientNet-B0, ViT-B/16, ConvNeXt-V2 Nano/Pico/Atto x seeds 8–10, linear head, lab+web = 21 | is ConvNeXt-V2 Tiny the right encoder (Tiny rows come from `data`) |
| `small` | Nano / Pico / Atto x gated x seeds 8–10 = 9 | how small can the encoder be for the rover |
| `curve` | train fractions 0.1 / 0.25 / 0.5 / 1.0 x linear, gated x seeds 8–10 = 24 (18 new; fraction 1.0 is shared with `data`) | how many images do the heads need |

`--suite all` runs the union, 93 jobs, about 7–8 hours on this GPU; lab-only runs finish first.

```powershell
python scripts\run_matrix.py --suite all --dry-run
python scripts\run_matrix.py --suite data
python scripts\aggregate_results.py            # Markdown tables
python scripts\aggregate_results.py --latex    # tabular bodies for the paper
python scripts\fill_paper_tables.py            # writes complete rows into tab:datagrid / tab:backbones (--partial, --dry-run)
python scripts\plot_learning_curve.py          # after the curve suite
python scripts\parameter_budget.py             # weights vs samples, no GPU needed
python scripts\benchmark_efficiency.py         # GPU latencies for the cost table
```

Score a checkpoint later:

```powershell
python evaluate.py runs\models\<run>\best.pth --data lab+web --seed 8 --with-ynlt
```

## Protocol in one paragraph

Splits are stratified per species and seeded per (source, species), so the lab partition
of seed 8 is the same whether you train on `lab` or `lab+web`. Web images are split by
observation, not by photo. The checkpoint is the epoch with the best accuracy on the
validation set of the training data; the 42 outdoor photographs are an unseen field test.
Runs are named `<timestamp>_<data>_<head>_<backbone>_seed<N>[_frac<f>]`, and
`results.json` holds `lab_test`, `web_test`, `field_test` metrics plus `*_ynlt_input224`
when YNLT was on.

## TensorBoard

```powershell
tensorboard --logdir runs\models
```

Per-epoch `loss/*`, `accuracy/*` and `lr`, the final metrics under `final/...`, the results
JSON on the Text tab, and an HParams tab (data, head, backbone, seed, train fraction).
`--no-tensorboard` turns logging off.

## YNLT patch source

`--with-ynlt` re-examines uncertain images with multi-scale patches cut from the 224x224
network input (`--ynlt-source input224`, the default). The alternative `original` cuts
patches from the full-resolution file; on the 1500x1000 laboratory photographs it matched
`input224` within one point at 5–10x the cost, so it is kept only as an ablation option
(`--ynlt-source both` reports both). Every run records the trigger rate, patches per
reviewed image and milliseconds per image.

## What this does not include

Notebooks, figure-drawing scripts, the Potamogeton XAI side project, and Linux paths from
the original repo. Those remain in `AquaticPlantsClassification`.
