# Prototypical Network work — handoff for another AI

Read this before changing the aquatic-plant ProtoNet code or quoting its numbers. It covers why the work was done, the exact protocol, every official number, which run folders to trust, and what was deliberately not done.

Student: Yaman Aljnadi. Advisor for this thread: Dr. Fengying Dang, Michigan Tech. Repo: https://github.com/yaman-aljnadi/Aquatic_Plants_Classification. Training code lives in `Yaman-Aquatic_Plants_Classification/`. Dates of this work: 12–13 November 2026.

The older project (linear / MLP / gated-attention heads, YNLT, last-stage ConvNeXt fine-tune, paper draft) is summarized in `Yaman-Aquatic_Plants_Classification/HANDOFF.md` and `Readme.md`. This file is only the Prototypical Network continuation.

---

## 1. What the professor asked for

Email from Dr. Dang, then confirmed with "Yes, try them":

- Few-shot learning with a Prototypical Network, plus data augmentation.
- A domain-specific Prototypical Network that integrates shape-aware and texture-aware embeddings for aquatic plant images.
- A second idea, background suppression, was listed in the same email. It was **not** implemented. Do not claim that it was.

The question she approved is: how well can the existing 21 Michigan aquatic species be recognized from fewer training images? It is **not** the classic meta-learning setup where some species are held out and never seen until test time. Every run keeps all 21 species in train, val, and test. Say that explicitly if you write a paper paragraph. Do not call these "novel-class" few-shot results.

---

## 2. What already existed, and the numbers ProtoNet is compared against

Same collection the current paper uses: `lab+web+field`, about 1,030 images (laboratory + cleaned GBIF/iNaturalist web + local outdoor). Outdoor photos are **inside** the pool, not a pure held-out field test. Splits are in `Yaman-Aquatic_Plants_Classification/data/splits/lab_web_field_seed{8-12}.csv`, built by `scripts/build_splits.py`. Web images are grouped by observation so photos of one plant do not cross train and test. Seeds 8–12 are the paper seeds. Do not invent new seeds if you want a comparison.

Per-seed train size is about 648–655 images, val about 189–193. All 21 classes are present in every train split. The smallest class is *Elodea canadensis* with 9 training images, which is why training episodes use 5 support + 3 query (8 images) rather than 5+5.

Backbone everywhere: `convnextv2_tiny.fcmae_ft_in22k_in1k` (preset key `convnextv2` in `config.py`). ImageNet-22k then ImageNet-1k. 768-d pooled embedding. Chosen because the longer-term target is an underwater rover, so the encoder stays small.

Earlier supervised baselines on this same pooled collection, frozen or last-stage ConvNeXt-V2 Tiny, linear head, seeds 8–12. Source: `Reports/last_stage_comparison.md`.

| Setup | Pooled accuracy | Invasive miss rate (FNR) | Lab | Web |
|---|---|---|---|---|
| Frozen encoder, linear head | 59.5% ± 4.6 | 31.3% ± 7.6 | 80.0% ± 8.0 | 53.0% ± 4.7 |
| Last ConvNeXt stage trained, linear head, backbone lr 1e-4 | 65.1% ± 5.0 | 28.8% ± 9.6 | 85.6% ± 9.7 | 58.1% ± 4.8 |

Three frozen heads (linear, MLP, gated attention) were tied on pooled accuracy near 59%. YNLT plus the gated head lowered invasive FNR from 32.6% ± 3.9 to 17.4% ± 5.7 without raising accuracy. YNLT was **not** rerun on ProtoNet.

A separate learning-curve point, frozen linear head, 10% of training images, pooled accuracy 32.9% ± 1.6, was computed from 3 seeds only (`train_fraction` 0.1). It is the comparison used for the frozen 5-shot ProtoNet (33.4%). Do not treat it as a five-seed number.

Known error pattern before ProtoNet, frozen linear seed 9, 126/189 correct: 28 of 63 mistakes stay inside the same genus (44%). Weak species are pondweeds (*Potamogeton praelongus* 2/10, *P. richardsonii* 2/7, *P. crispus* 6/14) and feathery plants (*Myriophyllum* vs *Cabomba*). Write-up: `Reports/confusion_seed9.md`. That file is one seed of the linear model, not a ProtoNet result.

Invasive species indices used for FNR (`config.py`): *Hydrocharis morsus-ranae*, *Myriophyllum spicatum*, *Nitellopsis obtusa*, *Potamogeton crispus*, *Cabomba*. FNR is the fraction of truly invasive test images predicted as a native species.

---

## 3. Two experiments, in order

### 3.1 Frozen prototype baseline (12 Nov 2026)

Script: `Yaman-Aquatic_Plants_Classification/scripts/protonet_baseline.py`.

No episodic training. The encoder stays the pretrained ConvNeXt. For each seed and each K in {1, 5, 10, all}:

1. Sample K images per class from that seed's **training** split. `all` uses every training image. If a class has fewer than K images, it uses all of them (*Elodea* has 9, so 10-shot is 9 for that class).
2. Prototype = mean of those images' 768-d backbone features. Features are **not** L2-normalized in this baseline.
3. Assign each test image to the nearest prototype by squared Euclidean distance. Cosine exists as `--distance cosine` but was not the reported run.
4. Score `lab_test`, `web_test`, `field_test`, `pooled_test`.

Support sampling RNG, shared later by the trainer: `numpy.random.RandomState(seed * 10007 + k)` with `k = 0` for the all-support case. Same formula in both scripts, so a given seed and K draws the same support indices when the train list order matches.

Augmentation ablation: `--aug-views 0` (one deterministic eval resize) vs `--aug-views 8` (average 8 augmented embeddings per support image, using `gpu_augment`). Query/test images are never augmented.

The script builds a linear head via `create_model_from_config` and `freeze_backbone`, then **ignores the head** and reads `model.backbone`. The "Trainable parameters: 16,149" line is that unused linear layer. Do not describe the frozen baseline as training 16k weights.

Official runs (five seeds, K = 1, 5, 10, all):

- No augmentation: `Yaman-Aquatic_Plants_Classification/runs/models/20261112_135057_protonet_frozen_lab_web_field_convnextv2_noaug_euclidean/`
- 8 augmented support views: `.../20261112_135334_protonet_frozen_lab_web_field_convnextv2_aug8_euclidean/`

Ignore `20261112_134958_...` (seed 9 smoke test only).

Mean ± sample std (ddof=1) over seeds 8–12. Accuracy, then invasive FNR in parentheses.

No augmentation:

| Shot | Lab | Web | Field | Pooled |
|---|---|---|---|---|
| 1 | 20.0 ± 4.9 (FNR 73.3) | 15.8 ± 3.7 (68.8) | 17.5 ± 6.8 (80.0) | 16.8 ± 2.6 (70.0) |
| 5 | 39.0 ± 6.1 (62.2) | 31.9 ± 5.1 (53.9) | 30.0 ± 19.0 (60.0) | 33.4 ± 3.8 (55.7) |
| 10 | 48.2 ± 6.1 (37.8) | 31.0 ± 2.4 (55.9) | 35.0 ± 5.6 (100.0) | 34.7 ± 2.1 (54.4) |
| all | 49.7 ± 3.9 (51.1) | 40.4 ± 2.2 (50.1) | 40.0 ± 18.5 (70.0) | 42.3 ± 2.3 (51.1) |

8 augmented views per support image, all-support pooled accuracy **38.6% ± 4.2** (FNR 56.5). Augmenting frozen support embeddings did not help. Full per-shot table is in that run's `results.json`.

Conclusion that drove the next experiment: a nearest-mean readout on frozen ImageNet features cannot beat a trained linear head (42.3% vs 59.5%). The 5-shot number (33.4%) only matches the old linear model when that model is also starved to ~10% of the data. Augmentation has to sit inside training, not on a frozen mean.

### 3.2 Episodic ProtoNet, then shape + texture (13 Nov 2026)

Script: `Yaman-Aquatic_Plants_Classification/scripts/train_protonet.py`.

This is the Snell et al. training loop, adapted to 21-way episodes on these species.

**Episode.** Every class, every episode. From the training split, draw 5 support images and 3 query images, disjoint. If a class is too small, query count is reduced first, then support count; a class must have at least 2 training images. Both sets go through the existing GPU augmentation (`src/gpu_data.py` `gpu_augment`, the same recipe as `cfg.aug_tf`: random resized crop scale 0.1–1, horizontal and vertical flip, rotation ±90, perspective, color jitter, TrivialAugmentWide, normalize, random erasing). Known deviations from torchvision are listed in the `gpu_data.py` module docstring (bilinear vs bicubic, crop taken from the 320-px cache, Equalize replaced by AutoContrast).

**Embedding.**

- `final`: global average pool after `norm_pre` on ConvNeXt-V2 Tiny. 768-d.
- `shape_texture`: that 768-d vector concatenated with a global average pool of `stages[1]` (0-based stage 1, 192 channels, 28×28). 960-d. Stage 1 was chosen as an early texture stage. Stage 0 (96-d at 56×56) was not used. `--texture-stage` can change it; every reported shape run used the default, 1.
- Then a bias-free `Linear` to 256-d, orthogonally initialized, then L2-normalize. No GELU, no BatchNorm, no dropout. Normalization keeps squared Euclidean distance in [0, 4].

**Loss.** Prototype = mean of the support embeddings in the 256-d space. The mean is **not** re-normalized. Logits = negative squared Euclidean distance to the 21 prototypes, divided by `--temperature` (default 1). Cross-entropy against the query labels. No label smoothing (the older `train.py` uses 0.05; this script does not).

**What is trained.**

- Frozen encoder: only the projection. 196,608 weights for `final` (768×256). 245,760 for `shape_texture` (960×256).
- `--unfreeze-last-stage`: also `stages[-1]` and `norm_pre`. Trainable counts were 15,683,328 (`final`) and 15,732,480 (`shape_texture`). Projection lr 5e-4 (`cfg.lr_initial`). Backbone lr 1e-4. AdamW, weight decay 2e-5, cosine schedule over 20 epochs down to 2e-5, grad clip 1.0, AMP on CUDA.
- Frozen stages run under `torch.no_grad` and in eval mode (no drop-path). The last stage, when trainable, stays in train mode so its drop-path is on. That matches `train.py`.

**Checkpoint.** Each epoch: build prototypes from **every training image** under the deterministic eval transform (no augmentation), score **every validation image**, keep the weights with the highest validation accuracy. Test images are never used for prototypes or for selection. This is train-support / val-query, not prototypes built from the validation images themselves (that would leak). After training, the best checkpoint is reloaded and the test protocol of section 3.1 is repeated at K = 1, 5, 10, and all, with no test-time augmentation.

**Budget.** 20 epochs × 30 episodes. About 13–16 minutes per five-seed run on an RTX 3070 Ti. Each run writes `results.json`, `results.partial.json`, and `seed{8-12}_best.pth`.

Official runs, all seeds 8–12, all on `lab+web+field`:

| Run folder under `runs/models/` | Features | Last stage | Wall time |
|---|---|---|---|
| `20261113_001948_protonet_episodic_lab_web_field_final_frozen` | final 768-d | frozen | 13.4 min |
| `20261113_003317_protonet_episodic_lab_web_field_final_laststage` | final 768-d | trained | 16.1 min |
| `20261113_004931_protonet_episodic_lab_web_field_shape_texture_frozen` | stage-1 + final | frozen | 14.4 min |
| `20261113_010359_protonet_episodic_lab_web_field_shape_texture_laststage` | stage-1 + final | trained | 16.1 min |

Best validation epoch (val accuracy) per seed:

| Seed | final, frozen | final, last stage | shape, frozen | shape, last stage |
|---|---|---|---|---|
| 8 | 18 (0.556) | 11 (0.582) | 16 (0.577) | 20 (0.608) |
| 9 | 11 (0.554) | 15 (0.622) | 10 (0.539) | 19 (0.632) |
| 10 | 5 (0.516) | 13 (0.578) | 3 (0.526) | 18 (0.589) |
| 11 | 9 (0.562) | 16 (0.568) | 8 (0.573) | 20 (0.568) |
| 12 | 9 (0.531) | 20 (0.651) | 7 (0.510) | 20 (0.625) |

Seed 12 of `final_laststage` hit its best val accuracy on the last epoch (0.651). The cosine schedule had not clearly plateaued for every seed. A longer run could move the last-stage numbers. It was not done.

**Do not use these smoke folders as results:** `20261113_001703` (1 epoch, 2 episodes), `20261113_001740` (5 epochs, seed 9 only), `20261113_001850` (last-stage smoke), `20261113_001906` (shape smoke, 1 epoch).

---

## 4. Official test numbers

Mean ± std over seeds 8–12. Pooled accuracy is the number to quote first. Field accuracy is noisy: only about eight outdoor images fall in a given pooled test split. Do not rank models by field accuracy.

### 4.1 Pooled test accuracy (%)

| Setup | 1-shot | 5-shot | 10-shot | All supports | Invasive FNR, all supports |
|---|---|---|---|---|---|
| Frozen prototypes, no aug | 16.8 ± 2.6 | 33.4 ± 3.8 | 34.7 ± 2.1 | 42.3 ± 2.3 | 51.1 ± 4.8 |
| Frozen prototypes, 8 aug views | 16.6 ± 2.4 | 31.5 ± 1.9 | 33.0 ± 3.5 | 38.6 ± 4.2 | 56.5 |
| Episodic, final, encoder frozen | 33.3 ± 3.8 | 50.2 ± 4.5 | 49.4 ± 4.4 | 50.8 ± 4.0 | 40.9 ± 3.2 |
| Episodic, final, last stage | 55.6 ± 3.7 | 56.7 ± 3.6 | 57.4 ± 4.3 | 57.5 ± 4.3 | 35.5 ± 9.3 |
| Episodic, shape+texture, encoder frozen | 31.5 ± 3.0 | 49.1 ± 5.3 | 49.7 ± 3.8 | 52.2 ± 3.6 | 38.0 ± 4.5 |
| Episodic, shape+texture, last stage | 58.0 ± 3.7 | 60.4 ± 5.1 | 59.3 ± 4.5 | 59.3 ± 4.8 | 33.4 ± 11.2 |
| Linear head, encoder frozen (earlier) |  |  |  | 59.5 ± 4.6 | 31.3 ± 7.6 |
| Linear head, last stage (earlier) |  |  |  | 65.1 ± 5.0 | 28.8 ± 9.6 |

The linear-head FNR column is the invasive miss rate of those earlier runs, not a shot setting. Frozen-prototype FNR for the 8-view all-support cell is 56.5% (std was computed; see `results.json` if you need it).

### 4.2 Best episodic model, all four splits

Shape + texture, last stage trained (`20261113_010359_...`):

| Shot | Lab | Web | Field | Pooled | Pooled invasive FNR |
|---|---|---|---|---|---|
| 1 | 72.3 ± 5.8 | 53.1 ± 5.1 | 72.5 ± 10.5 | 58.0 ± 3.7 | 37.2 |
| 5 | 76.4 ± 11.8 | 55.2 ± 5.7 | 77.5 ± 5.6 | 60.4 ± 5.1 | 32.9 |
| 10 | 74.9 ± 12.1 | 54.2 ± 5.0 | 75.0 ± 15.3 | 59.3 ± 4.5 | 33.3 |
| all | 75.4 ± 10.4 | 54.0 ± 5.4 | 75.0 ± 8.8 | 59.3 ± 4.8 | 33.4 |

Lab is high and web is still near 54%, same pattern as the linear head (lab ~80–86%, web ~53–58%). The ProtoNet lab number (75%) is below the last-stage linear lab number (85.6%). The web number (54%) is close to the frozen linear web number (53%) and a bit under the last-stage linear web number (58.1%).

### 4.3 What the numbers mean

- Training the embedding is the gain. Frozen means: 42.3% all-support pooled. A 256-d projection on a frozen encoder: 50.8%. Unfreezing the last stage: 57.5%.
- Shape + texture is a smaller gain on top of that: 50.8% → 52.2% (encoder frozen) and 57.5% → 59.3% (last stage trained).
- 59.3% matches the frozen linear head (59.5%) and is about 6 points under the best older model (65.1% ± 5.0). The gap is similar in size to the seed std, but it is in the same direction on both lab and web, so do not call ProtoNet a win over the last-stage linear head.
- Once the last stage is trained, 5-shot pooled accuracy is about as high as all-support (60.4% vs 59.3% for shape + texture; 56.7% vs 57.5% for final-only). More support images past 5 are not the missing piece for this embedding.
- Invasive FNR fell from 51% (frozen prototypes, all supports) to 33–36% for the trained models. The last-stage linear head was 28.8% ± 9.6. The intervals overlap. Do not claim an invasive-detection win.
- 10-shot is sometimes slightly below 5-shot. That is real in the tables (for example shape+texture last stage, pooled 59.3% at 10-shot vs 60.4% at 5-shot). Do not smooth it away. *Elodea* cannot contribute a true 10th support image.

### 4.4 Look-alike species

`train_protonet.py` stores, for pooled test at each shot, recall of the pondweeds and the feathery species, plus the top confusions. Counts below are **summed across the five test splits**, so the denominators are about five times one seed. They are not "108 images in one test set."

All-support pooled test:

| Species | Final embedding, last stage | Shape + texture, last stage |
|---|---|---|
| *Potamogeton gramineus* | 52/108 (48%) | 59/108 (55%) |
| *Potamogeton crispus* | 25/71 (35%) | 30/71 (42%) |
| *Potamogeton richardsonii* | 16/30 (53%) | 16/30 (53%) |
| *Potamogeton praelongus* | 22/52 (42%) | 21/52 (40%) |
| *Potamogeton illinoensis* | 38/61 (62%) | 39/61 (64%) |
| *Potamogeton natans* | 24/39 (62%) | 25/39 (64%) |
| *Potamogeton robbinsii* | 22/26 (85%) | 21/26 (81%) |
| *Myriophyllum spicatum* | 17/53 (32%) | 16/53 (30%) |
| *Myriophyllum sibiricum* | 13/29 (45%) | 15/29 (52%) |
| *Cabomba* | 26/47 (55%) | 27/47 (57%) |
| *Ceratophyllum demersum* | 24/49 (49%) | 27/49 (55%) |

Shape + texture helps *P. gramineus* and *P. crispus* a little. It does not help *P. praelongus* or *M. spicatum*.

Most common confusions, shape + texture, last stage, all supports, summed over seeds (true → predicted, count):

- *P. gramineus* → *P. illinoensis*, 13
- *M. spicatum* → *M. sibiricum*, 12
- *P. crispus* → *M. spicatum*, 9
- *C. demersum* → *M. sibiricum*, 9
- *P. richardsonii* → *P. crispus*, 9
- *P. praelongus* → *P. richardsonii*, 8

Same story as the linear-head error analysis: pondweed vs pondweed, and feathery plant vs feathery plant. Full per-seed blocks are under `lookalikes` and `lookalike_summary` in each episodic `results.json`.

---

## 5. How to reproduce

From `Yaman-Aquatic_Plants_Classification/`, with the existing `.venv` (PyTorch 2.6 + CUDA, RTX 3070 Ti on the machine that produced these runs):

```powershell
.\.venv\Scripts\python.exe -u scripts\protonet_baseline.py --data lab+web+field --seeds 8 9 10 11 12 --shots 1 5 10 --aug-views 0
.\.venv\Scripts\python.exe -u scripts\protonet_baseline.py --data lab+web+field --seeds 8 9 10 11 12 --shots 1 5 10 --aug-views 8
.\.venv\Scripts\python.exe -u scripts\train_protonet.py --features final --seeds 8 9 10 11 12
.\.venv\Scripts\python.exe -u scripts\train_protonet.py --features final --unfreeze-last-stage --seeds 8 9 10 11 12
.\.venv\Scripts\python.exe -u scripts\train_protonet.py --features shape_texture --seeds 8 9 10 11 12
.\.venv\Scripts\python.exe -u scripts\train_protonet.py --features shape_texture --unfreeze-last-stage --seeds 8 9 10 11 12
.\.venv\Scripts\python.exe scripts\plot_protonet_meeting.py
```

Image banks are cached under `Yaman-Aquatic_Plants_Classification/data/cache/` (320×320 uint8). The first seed of a new split list rebuilds the bank; later seeds load it. Logs from the four official episodic runs: `Yaman-Aquatic_Plants_Classification/runs/logs/protonet_final_frozen.log`, `protonet_final_laststage.log`, `protonet_shape_frozen.log`, `protonet_shape_laststage.log`.

Meeting figure, plain-language labels, rounded to the nearest percent: `Reports/protonet_meeting.png`. Left panel is all-support pooled accuracy for the six setups. Right panel is the best new model (shape + texture, last stage) at 1, 5, 10, and all photos, with reference lines at 59.5% and 65.1%. The script hardcodes the published means; it does not re-read `results.json`. If you rerun experiments, update `scripts/plot_protonet_meeting.py` or the figure will lie.

`print_summary_table` in `protonet_baseline.py` takes an optional `title` argument. The frozen script still prints the old title by default.

---

## 6. Files a new session should open

| Path | Role |
|---|---|
| `Yaman-Aquatic_Plants_Classification/scripts/protonet_baseline.py` | Frozen nearest-prototype readout |
| `Yaman-Aquatic_Plants_Classification/scripts/train_protonet.py` | Episodic trainer, shape/texture flag, look-alike report |
| `Yaman-Aquatic_Plants_Classification/scripts/plot_protonet_meeting.py` | Meeting bar chart |
| `Yaman-Aquatic_Plants_Classification/src/gpu_data.py` | GPU image bank and augmentation the episodes use |
| `Yaman-Aquatic_Plants_Classification/src/model.py` | Older heads. `freeze_backbone(..., unfreeze_last_stage=True)` is the pattern the trainer copies for the last stage |
| `Yaman-Aquatic_Plants_Classification/config.py` | Class list, invasive indices, lr 5e-4, weight decay 2e-5, augmentation |
| `Reports/episodic_protonet_results.md` | Short results note already shown to the student |
| `Reports/WeeklyReport_Week12_2026-11-12.md` | Week 12 report text, includes both the frozen baseline and the episodic/shape results |
| `Reports/protonet_meeting.png` | Figure for the professor meeting |
| `Reports/last_stage_comparison.md` | 59.5% and 65.1% linear-head baselines |
| `Reports/confusion_seed9.md` | Pre-ProtoNet error analysis, one linear checkpoint |

The paper draft `Paper/main_v4.tex` does **not** contain these ProtoNet numbers. Do not add them unless asked. The draft still says Grad-CAM errors are on the plant rather than the background. That sentence is based on a six-image panel of mostly clean photos. A later look at `Reports/heatmaps/` found 53 of 61 wrong maps were web photos, and a sample of 12 of those 53 included several maps whose hottest region was not the plant (a rope and arm, glare, a corner of water). That review was incomplete and was not turned into a table. Do not upgrade it into a paper claim, and do not delete the paper sentence without a full recount.

---

## 7. Decisions that were intentional

- Same 21 species in every split. Not novel-class few-shot.
- Support images at test time come from the training split only, not train+val, so the comparison with a linear head trained only on the train split stays fair.
- Checkpoint metric is validation accuracy with train-set prototypes, not validation loss, and not a few-shot subsample of val.
- Euclidean distance, not cosine, in every reported number. Embeddings are unit-length; prototypes are means of those unit vectors.
- One linear projection. A two-layer MLP was considered and not used, so a gain could not be blamed on a bigger head.
- Shape/texture was trained only after the final-embedding episodic runs existed, so the two effects can be separated: episodic training vs the extra feature.
- Background suppression was left undone. It needs a mask on messy web photos. It is the other idea in the professor's email, not a completed ablation.
- Field/outdoor accuracy is reported because the split code emits it. It is not a ranking metric.

---

## 8. What is still open

The professor had not, at the time of this file, picked the next step. Reasonable continuations, in the order the results support:

1. The remaining errors are look-alike pondweeds and feathery plants, especially *Potamogeton praelongus* (~40% recall) and *Myriophyllum spicatum* (~30% recall), even in the best ProtoNet. More photos of those taxa, or a feature aimed at leaf margin rather than a global pool, is the direct response to the error table. Global-pooling stage 1 throws away the spatial layout of the leaf.
2. The best ProtoNet (59.3% ± 4.8 pooled) is still short of the last-stage linear head (65.1% ± 5.0). A longer last-stage run is justified because several seeds peaked at epoch 18–20. That is a small experiment, not a new method.
3. Background suppression is still only a hypothesis. A cheap test that does not need a segmenter: hand-crop a few dozen wrong web test images to the plant and rescore them. If accuracy jumps, suppression is worth building. If it does not, the errors are morphological.
4. Do not spend time on another frozen-feature ablation. Section 3.1 already answered that.
5. YNLT was not combined with ProtoNet. The invasive FNR of the best ProtoNet (33.4% ± 11.2) is in the same range as the linear heads and well above the old gated+YNLT number (17.4% ± 5.7). Combining them is a possible follow-up, not something these runs measured.

Week 12 report text the student was going to paste is in `Reports/WeeklyReport_Week12_2026-11-12.md`. It already includes the episodic and shape results. An earlier shorter draft that stopped at "next step: train an episodic ProtoNet" is obsolete.
