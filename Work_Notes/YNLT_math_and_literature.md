# Working notes: YNLT math (from the code) + next experiments

Confirmed in `cnn_utils/predict.py` and `a1_cnn_ynlt/config.py`.

## Uncertainty (as implemented)

Natural log, not log2:

```
entropy = -sum(p * log(p))
margin  = p_(1) - p_(2)
confidence = p_(1)
```

Thresholds:

- `confidence_threshold = 0.5`  (flag if confidence < 0.5)
- `margin_threshold = 0.2`      (flag if margin < 0.2)
- `entropy_threshold = 2.0`     (flag if entropy > 2.0)
- each flag adds weight 1
- `critical_confusion_level = 1` so **any one flag** triggers YNLT
- if the first prediction is already invasive, YNLT is skipped

Patch vote: `w = (3 - c) / 3`, class scores start at 1.0 then add `w`. Ties prefer invasive. Not `W = 4 - c`.

Patching: scales `[0.2, 0.3, 0.4, 0.5]`, overlap `[0.0, 0.1, 0.2, 0.3]`.

Gated head: gate = sigmoid(tanh(Linear(h))), features = GELU(Linear(h)), then multiply. Attention dim 168. 261,933 trainable params. Backbone frozen.

These formulas are now in `Paper/main_v3.tex`.

## Dataset

Training notebook: **230 lab** (138/46/46), **33 OOD** (16 val / 17 test). Abstract 225 was a typo and is fixed.

Your current folders have 203 lab and 42 outdoor files and extra parentheticals in folder names. The code now strips `(common name)` so those folders can load, but they are **not** the 230-image list used in the paper. Replicating Table 1 requires the original `squared_reflected` + `hand` directories.

## Why not 5-fold

Several species have 2 images. Stratified 5-fold needs 5 examples per class. Use **five repeated 60/20/20 splits** (`--seed 8` through `12`) instead.

## What to run for stronger results

```
cd AquaticPlantsClassification/cnn-approaches
$env:PYTHONPATH = (Get-Location).Path

python a1_cnn_ynlt/train_run.py --head linear --backbone convnextv2
python a1_cnn_ynlt/train_run.py --head gated_attention --backbone convnextv2 --with-ynlt
python a1_cnn_ynlt/train_run.py --head linear --backbone resnet50
python a1_cnn_ynlt/train_run.py --head linear --backbone efficientnet
python a1_cnn_ynlt/train_run.py --head linear --backbone vit
python a1_cnn_ynlt/train_run.py --head linear --backbone convnext
```

Do not invent new table numbers until these runs finish.

## Remaining paper debt

`Paper/ref.bib` was missing and has been recreated. Entries that were in the original (lost) bib, especially the AIS ecology citations, should be replaced with the recovered originals when Aaryan sends that file.
