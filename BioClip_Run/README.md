# BioCLIP runs (aquatic plant classification)

Biology-trained ViT-B/16 encoder (`hf-hub:imageomics/bioclip`) on the same `lab+web+field` splits and seeds 8–12 as the ConvNeXt work. Training still uses `Yaman-Aquatic_Plants_Classification/` for data, config, and GPU loaders.

## Plain-language result

A frozen BioCLIP encoder was **not** enough on lab photos (77.9% vs ConvNeXt 80%). Teaching the **last ViT block** on our photos is what won: lab **89.2%**, web **67.1%**, mixed **72.5%**. That beats the best ConvNeXt mixed score (65.1%). **Keep BioCLIP (last block).**

One sentence for a meeting: *A biology-trained encoder was not enough when frozen; adapting its last block on our aquatic-plant photos is what beat the ConvNeXt baselines.*

| Setup | Lab | Web | Mixed (pooled) |
|---|---|---|---|
| ConvNeXt frozen | 80.0% ± 8.0 | 53.0% ± 4.7 | 59.5% ± 4.6 |
| BioCLIP frozen | 77.9% ± 3.9 | 60.7% ± 4.7 | 64.9% ± 4.0 |
| **BioCLIP last block** | **89.2% ± 4.9** | **67.1% ± 5.4** | **72.5% ± 4.5** |
| ConvNeXt last stage | 85.6% ± 9.7 | 58.1% ± 4.8 | 65.1% ± 5.0 |

Invasive miss rate (pooled) for last-block BioCLIP: **23.0% ± 5.0**. Cost: ~86M visual params, ~10.4 ms/image vs ConvNeXt-V2 Tiny ~28M and ~5.2 ms.

Open the figure: [`figs/bioclip_meeting.png`](figs/bioclip_meeting.png).

## Layout

| Path | Role |
|---|---|
| `scripts/train_bioclip.py` | Trainer (frozen linear probe or `--unfreeze-last-block`) |
| `scripts/plot_bioclip_results.py` | Meeting bar chart |
| `reports/bioclip_taxonomy.md` | 21-name match against TreeOfLife / iNat21 |
| `runs/models/20261118_154447_bioclip_linear_lab_web_field/` | Official frozen run |
| `runs/models/20261118_175109_bioclip_lastblock_lab_web_field/` | Official last-block run |
| `runs/logs/` | Console logs for those runs |
| `figs/` | `bioclip_meeting.png` / `.pdf` |

## How to run

From this folder, using the existing project venv:

```powershell
cd BioClip_Run
..\Yaman-Aquatic_Plants_Classification\.venv\Scripts\python.exe -u scripts\train_bioclip.py --seeds 8 9 10 11 12
..\Yaman-Aquatic_Plants_Classification\.venv\Scripts\python.exe -u scripts\train_bioclip.py --unfreeze-last-block --skip-zeroshot --seeds 8 9 10 11 12
..\Yaman-Aquatic_Plants_Classification\.venv\Scripts\python.exe scripts\plot_bioclip_results.py
```

New checkpoints write under `BioClip_Run/runs/models/`. The old path `Yaman-Aquatic_Plants_Classification/scripts/train_bioclip.py` is a stub that points here.
