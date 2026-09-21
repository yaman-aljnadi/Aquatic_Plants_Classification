# Aquatic plant classification (YNLT) — project root

Confidence-aware classification of 21 Michigan aquatic plant species (5 invasive) from a
small laboratory image set, evaluated under laboratory-to-field domain shift. Student:
Yaman. Advisor: Dr. Fengying Dang, Michigan Tech. Paper first-authored by Aaryan Panigrahi.

## What is in this repository

| Folder | Role |
|---|---|
| `Yaman-Aquatic_Plants_Classification/` | **Active training workspace.** New runs happen here. See its README. |
| `Paper/` | `main_v3.tex` (current draft) and `ref.bib`. |
| `Work_Notes/` | Working notes tying the code to the paper text. |
| `AquaticPlantsClassification/` | **Archive only** — original notebooks and Linux paths. Do not train here. |
| `PaperReading.md` | Weekly paper reading log in the advisor's format. |
| `WeeklyReport_*.md` | Weekly progress reports. |

`Yaman-Aquatic_Plants_Classification/HANDOFF.md` is the detailed status document: what has
been done, what the open questions are, and what to run next.

## What is missing from this checkout

Neither the images nor the paper figures are tracked here.

- **Images.** Training reads two folders of `ImageFolder`-style per-species subdirectories:
  - laboratory: `Aquatic Plant lab/` at this level, or the path in `AQUATIC_LAB_DATA`
  - field / out-of-distribution: `Aquatic Plant outdoor/`, or `AQUATIC_OOD_DATA`

  The paper reports 230 laboratory and 33 field images. The folders seen on the student's
  machine earlier held roughly 203 and 42, so they are **not** the set behind Table 1.
  Confirm the official list before publishing new numbers.

- **Figures.** `Paper/main_v3.tex` includes `figs/map.pdf`, `figs/data_dist.pdf`,
  `figs/augmentations.pdf`, `figs/sample_prediction_*.pdf` and `figs/figs26/*.pdf`.
  `Paper/figs/` does not exist here, so the draft will not compile until those are restored.
  The draft uses `fontspec`, so build it with XeLaTeX or LuaLaTeX, and the `svg` package
  needs Inkscape on `PATH`.

## Quick start

```powershell
cd Yaman-Aquatic_Plants_Classification
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# cost numbers for the paper; needs no images
python scripts/benchmark_efficiency.py

# experiments, once the image folders are in place
python scripts/run_matrix.py --suite all
python scripts/aggregate_results.py --latex
```
