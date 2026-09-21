# Yaman aquatic plant classification — clean training workspace

Use this folder for new runs. The original notebooks stay in `AquaticPlantsClassification/`
as an archive. `HANDOFF.md` has the full project status.

## Layout

```
Yaman-Aquatic_Plants_Classification/
  train.py          start training here; writes results.json per run
  evaluate.py       score a saved checkpoint
  config.py         paths, batch size, YNLT thresholds and patch source
  src/
    model.py        ConvNeXt / linear / gated / MLP heads
    data.py         lab + OOD splits
    ynlt.py         confidence-aware second look
    patching.py     multi-scale crops
    metrics.py      accuracy and invasive FNR
  scripts/
    run_matrix.py           ablation x backbone x seed, skips finished runs
    aggregate_results.py    mean +/- std tables (Markdown and LaTeX)
    benchmark_efficiency.py params, MACs, latency, patch counts (no images needed)
  runs/models/      checkpoints and results.json written here
```

Images are **not** copied here. Training reads:

- lab: `../Aquatic Plant lab`
- field: `../Aquatic Plant outdoor`

Override with environment variables `AQUATIC_LAB_DATA` and `AQUATIC_OOD_DATA` if the
original 230-image set lives somewhere else.

## Setup (RTX 3070 Ti + i9-12900K)

```powershell
cd Yaman-Aquatic_Plants_Classification
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

This GPU has 8 GB, vs 4 GB on the paper laptop, so the default batch size is **32** (paper
used 4). Mixed precision is on. DataLoader uses 4 workers.

If you run out of VRAM: `python train.py --batch-size 16`. If Windows workers misbehave:
`--workers 0`.

## Commands

Everything at once (skips jobs that already have a `results.json`):

```powershell
python scripts\run_matrix.py --suite all --dry-run   # see the plan first
python scripts\run_matrix.py --suite all
python scripts\aggregate_results.py --latex
```

Individual runs:

```powershell
python train.py --head linear --backbone convnextv2
python train.py --head gated_attention --backbone convnextv2 --with-ynlt
```

Stronger backbones (same frozen-encoder recipe):

```powershell
python train.py --head linear --backbone resnet50
python train.py --head linear --backbone efficientnet
python train.py --head linear --backbone vit
python train.py --head linear --backbone convnext
```

Repeated splits (do this instead of 5-fold; some species have only 2 images):

```powershell
python train.py --head gated_attention --backbone convnextv2 --seed 8
# ... through --seed 12
```

Score a checkpoint later:

```powershell
python evaluate.py runs\models\<run>\best.pth --with-ynlt
```

## YNLT patch source

`--with-ynlt` re-examines uncertain images with multi-scale patches. Where those patches
come from is a real experimental choice, controlled by `--ynlt-source`:

| Value | Patches cut from | Note |
|---|---|---|
| `original` | the full-resolution source file | what the paper describes |
| `input224` | the 224x224 network input | what the archive notebooks did for Table 1 |
| `both` (default) | both, reported as separate rows | gives the resolution ablation for free |

`original` is much slower: an uncertain 3000x4000 photograph produces tens of large
patches, each a separate forward pass. Unflagged images cost one normal forward pass
either way. Every run records the trigger rate, patches per reviewed image, and
milliseconds per image so the cost can be reported instead of guessed.

## What this does not include

Notebooks, figure-drawing scripts, the Potamogeton XAI side project, and Linux paths from
the original repo. Those remain in `AquaticPlantsClassification`.
