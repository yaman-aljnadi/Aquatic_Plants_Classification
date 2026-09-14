# Yaman aquatic plant classification — clean training workspace

Use this folder for new runs. The original notebooks stay in `AquaticPlantsClassification/` as an archive.

## Layout

```
Yaman-Aquatic_Plants_Classification/
  train.py          start training here
  evaluate.py       score a saved checkpoint
  config.py         paths, batch size, YNLT thresholds
  src/
    model.py        ConvNeXt / linear / gated / MLP heads
    data.py         lab + OOD splits
    ynlt.py         confidence-aware second look
    patching.py     multi-scale crops
    metrics.py      accuracy and invasive FNR
  runs/models/      checkpoints written here
```

Images are **not** copied here. Training reads:

- lab: `../Aquatic Plant lab`
- field: `../Aquatic Plant outdoor`

Override with environment variables `AQUATIC_LAB_DATA` and `AQUATIC_OOD_DATA` if the original 230-image set lives somewhere else.

## Setup (RTX 3070 Ti + i9-12900K)

```powershell
cd D:\Aquatic_Research\YamanA\Yaman-Aquatic_Plants_Classification
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

This GPU has 8 GB, vs 4 GB on the paper laptop, so the default batch size is **32** (paper used 4). Mixed precision is on. DataLoader uses 4 workers.

If you run out of VRAM: `python train.py --batch-size 16`. If Windows workers misbehave: `--workers 0`.

## Commands

Paper ablation on seed 8:

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
python train.py --head gated_attention --backbone convnextv2 --seed 9
python train.py --head gated_attention --backbone convnextv2 --seed 10
python train.py --head gated_attention --backbone convnextv2 --seed 11
python train.py --head gated_attention --backbone convnextv2 --seed 12
```

Score a checkpoint later:

```powershell
python evaluate.py runs\models\<run>\best.pth --with-ynlt
```

`--with-ynlt` is slow: every uncertain image is split into multi-scale patches.

## What this does not include

Notebooks, figure-drawing scripts, the Potamogeton XAI side project, and Linux paths from the original repo. Those remain in `AquaticPlantsClassification`.
