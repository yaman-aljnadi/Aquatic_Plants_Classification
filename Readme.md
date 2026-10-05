# Aquatic plant classification (YNLT) — project root

Confidence-aware classification of 21 Michigan aquatic plant species (5 invasive) from a
small laboratory image set plus web-sourced field photographs, evaluated on unseen outdoor
photographs. The frozen ConvNeXt-V2 Tiny encoder is chosen with deployment on an
underwater rover in mind. Student: Yaman. Advisor: Dr. Fengying Dang, Michigan Tech.
Paper first-authored by Aaryan Panigrahi.

## What is in this repository

| Folder | Role |
|---|---|
| `Yaman-Aquatic_Plants_Classification/` | **Active training workspace.** New runs happen here. See its README and `HANDOFF.md`. |
| `Paper/` | `main_v3.tex` (current draft) and `ref.bib`. |
| `Reports/` | Weekly reports, paper-reading log, `Parameters_vs_Samples.md` (answer to the weights-vs-samples question). |
| `Work_Notes/` | Working notes tying the code to the paper text. |
| `Larger_Plants_Dataset/aquatic_dataset_report/` | GBIF download script, manifest and report for the web-sourced images. |
| `AquaticPlantsClassification/` | **Archive only** — original notebooks and Linux paths. Do not train here. |

`Yaman-Aquatic_Plants_Classification/HANDOFF.md` is the detailed status document.

## Image sources (not tracked in git)

| Folder at this level | Env override | Images | Role |
|---|---|---|---|
| `Aquatic Plant lab/` | `AQUATIC_LAB_DATA` | 203 (1500x1000, 21 species) | train / val / test |
| `Larger_Plants_Dataset/aquatic_dataset_report/aquatic_plant_images/` | `AQUATIC_WEB_DATA` | 1,140 (532 iNaturalist observations via GBIF, CC licences in `download_manifest.csv`) | train / val / test, split by observation |
| `Aquatic Plant outdoor/` | `AQUATIC_OOD_DATA` | 42 (9 species) | held-out field test only |

Split lists are deterministic and committed under
`Yaman-Aquatic_Plants_Classification/data/splits/` (`lab`, `web`, `lab_web` x seeds 8–12).
The paper's original Table 1 used a 230/33 image list from the archive notebooks that is
not on this machine; it is kept in the paper only as the single-split reference.

## Figures

`Paper/main_v3.tex` includes `figs/map.pdf`, `figs/data_dist.pdf`, `figs/augmentations.pdf`,
`figs/learning_curve.pdf`, `figs/sample_prediction_*.pdf` and `figs/figs26/*.pdf`. Only
`learning_curve.pdf` is produced here (`scripts/plot_learning_curve.py`); the rest must be
restored before the draft compiles. Build with XeLaTeX or LuaLaTeX (`fontspec`); the `svg`
package needs Inkscape on `PATH`.

## Quick start

```powershell
cd Yaman-Aquatic_Plants_Classification
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/build_splits.py --data all --seeds 8 9 10 11 12   # split lists
python scripts/parameter_budget.py                               # weights vs samples, no GPU
python scripts/run_matrix.py --suite all --dry-run               # 93 jobs
python scripts/run_matrix.py --suite data                        # start with the 45 data x head runs
python scripts/aggregate_results.py --latex
python scripts/plot_learning_curve.py
python scripts/benchmark_efficiency.py                           # GPU latencies for the cost table
```
