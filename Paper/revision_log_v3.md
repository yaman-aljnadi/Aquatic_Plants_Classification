# Revision log for `main_v3.tex`

This note lists the substantive changes made so the manuscript reports the cleaned 1,030-image collection (handheld photographs excluded). Each entry gives the location, the reason, the previous wording, and the wording now in the paper.

The numerical results come from the five-seed pooled runs (frozen ConvNeXt-V2 Tiny; linear, MLP, and gated-attention heads, with and without YNLT) and from the five-seed last-stage runs (linear head, last ConvNeXt stage trained). They were checked against `results.json` before they were written into the tables.

## 1. Title

**Location.** Title.

**Why.** The previous title described a controlled-training, out-of-distribution experiment. That is not the experiment being reported.

**Previous text.** Improving Out-of-Distribution Aquatic Plant Classification from Limited Controlled Training Data Using Confidence-Aware Re-Evaluation

**Revised text.** Confidence-Aware Identification of Michigan Aquatic Plants from a Mixed Photograph Collection

## 2. Abstract

**Location.** Abstract.

**Why.** The abstract cited 1,140 web photographs, a single 230-image laboratory split (84.78% to 93.48%), and a 17-image field split (47.06% to 64.71%). Those numbers are from an earlier protocol. The reported collection is 1,030 images, and the reported scores are five-seed means.

**Previous text.**

> Aquatic invasive species (AIS) threaten the resilience of fisheries and aquaculture systems, causing severe ecological damage and economic losses. Accurate early identification is necessary to manage these species, and the long-term goal of this work is a classifier that runs on a camera-equipped underwater rover. Collecting high-quality images in underwater environments is difficult and expensive, so training sets are small and usually captured in the laboratory, and computer vision models trained on them often fail to generalize to the visual variations found in natural settings. This study addresses that gap with three ingredients. A compact ConvNeXt-V2 Tiny backbone (28 M parameters, 4.5 G multiply-adds per image) is kept frozen and only a parameter-efficient Gated Attention head (0.26 M weights) is trained, which suits the compute budget of an onboard computer. The You Need To Look Twice (YNLT) module, inspired by the human behavior of closely re-examining uncertain details, re-evaluates an image with multi-scale patches only when confidence, margin and entropy indicate an unreliable first look. Finally, the 203 laboratory photographs are complemented with 1,140 openly licensed field photographs of the same 21 Michigan species, drawn from GBIF-mediated iNaturalist observations and split by observation so that no plant appears in both training and test data. On a single laboratory split the gated head with YNLT raised accuracy from 84.78% to 93.48% in the laboratory and from 47.06% to 64.71% on 17 crowd-sourced field images. The combined protocol reports mean and standard deviation over five seeds on a laboratory test, a 210-image web test, and 42 held-out outdoor photographs, together with a learning curve that relates accuracy to the number of training images per class. These results indicate how laboratory and web imagery can be combined to identify invasive plants in complex environments with a model small enough for field deployment.

**Revised text.**

> Aquatic invasive species threaten fisheries and aquaculture through ecological damage and economic loss. The long-term aim of this work is a classifier that can run on a camera-equipped underwater rover. Underwater photographs are costly to collect, so the study uses a mixed collection of 1,030 photographs of 21 Michigan aquatic plant species, five of them invasive: 203 laboratory photographs, 785 openly licensed field photographs drawn from Global Biodiversity Information Facility (GBIF) records of iNaturalist observations, and 42 outdoor photographs. Handheld photographs were excluded. Photographs from the same observation are placed in a single split. A ConvNeXt-V2 Tiny backbone is paired with a linear, multilayer-perceptron, or gated-attention head. The You Need to Look Twice (YNLT) module re-evaluates an image with multi-scale patches only when confidence, margin, and entropy indicate an unreliable first prediction. Over five seeds, with the encoder frozen, the three heads reach 58.6–59.5% top-1 accuracy on the pooled test. Training the last ConvNeXt stage, with earlier stages left frozen, raises pooled accuracy to 65.1% ± 5.0%. YNLT with the gated-attention head lowers the invasive false-negative rate (FNR) from 32.6% ± 3.9% to 17.4% ± 5.7%, while pooled accuracy changes by less than one point. Gradient-weighted class activation maps and a species-level error analysis show that the remaining mistakes concentrate on morphologically similar pondweeds.

**Keywords.** “Domain generalization” and “precision agriculture” were removed. The keyword list is now: Aquatic invasive species; Computer vision; Fine-grained recognition; Environmental monitoring.

## 3. Introduction and contributions

**Location.** Last three paragraphs of the Introduction, including the contribution list.

**Why.** The introduction framed the paper as laboratory-to-field transfer, stated that the gated head suppresses laboratory background, and described a 1,140-image web set with a laboratory / web / both ablation. On the 1,030-image runs the three heads are tied on accuracy, and the outdoor photographs are inside the pool.

**Previous text.**

> This paper addresses that gap. We train only a small classification head on the frozen ConvNeXt-V2 backbone using laboratory images of 21 Michigan aquatic species, five of them invasive, and we evaluate on crowd-sourced field photographs that were never used for training. [...] Because the laboratory set alone provides only about six training images per species, we additionally assemble 1,140 openly licensed field photographs [...] and measure what training on laboratory images, web images, or both does to laboratory, web and outdoor test accuracy.
>
> Contributions:
> 1. A small-data aquatic classification setting that explicitly measures controlled-to-field domain shift, rather than reporting only laboratory accuracy.
> 2. YNLT, a test-time module that spends extra computation only when confidence, margin, and entropy indicate an unreliable first look, instead of always cropping or abstaining.
> 3. A gated attention classification head, trained with a frozen backbone, that improves out-of-distribution accuracy relative to a linear classifier on the same features.
> 4. A web-sourced field image set for the same 21 species, with observation-grouped splits, and a data-source ablation (laboratory, web, both) with a learning curve that relates accuracy to the number of training images, answering how many samples the trained head needs.

**Revised text.**

> This paper evaluates that design on one mixed collection of 21 Michigan aquatic species, five of them invasive. The collection contains 1,030 photographs: laboratory images, cleaned web images from GBIF-mediated iNaturalist observations, and a small number of outdoor photographs. Handheld photographs are excluded. [...] Results are means and standard deviations over five seeds, with a learning curve, class-activation maps, and a species-level account of the remaining errors.
>
> Contributions:
> 1. A 1,030-image collection of 21 Michigan aquatic species, combining laboratory, cleaned web, and outdoor photographs, with handheld photographs excluded and with splits grouped by observation.
> 2. YNLT, a test-time module that spends extra computation only when confidence, margin, and entropy indicate an unreliable first look, instead of always cropping or abstaining.
> 3. A five-seed comparison of linear, multilayer-perceptron, and gated-attention heads on a frozen ConvNeXt-V2 Tiny encoder, and of training the last encoder stage, on this collection.
> 4. A learning curve, gradient-weighted class activation maps, and a species-level error analysis relating accuracy to sample size and to look-alike taxa.

The sentence that said models “fail on handheld or in situ images” was rewritten to say that morphologically similar species remain difficult when photographs differ in background, scale, and viewpoint. Pl@ntNet was added as a related-work citation.

## 4. Dataset

**Location.** Dataset details, including Table `tab:data`.

**Why.** The table still listed 1,140 web images and treated 42 outdoor photographs as a held-out field test. After cleaning, the web source is 785 images, handheld photographs (307) are unused, and the three remaining sources are pooled (203 + 785 + 42 = 1,030).

**Previous text.**

> This study therefore uses three image collections of the same 21 species: photographs captured in a controlled laboratory setting, openly licensed field photographs gathered from biodiversity databases, and a small set of outdoor photographs that is used only for testing.
>
> The 42 outdoor photographs of hand-held plants (9 species, 16 of them *Nuphar variegata*) are never used for training or model selection in the new protocol. They are the held-out field test.
>
> Table total: Laboratory 203; Web 1,140 (532 observations); Outdoor 42.

**Revised text.**

> This study therefore uses one mixed collection of the same 21 species. The collection contains laboratory photographs, openly licensed field photographs from biodiversity databases, and outdoor photographs. These three sources are pooled and split together. Handheld photographs, 307 images, are stored separately and are not used for training, validation, or testing.
>
> The outdoor source contains 42 photographs of 9 species, 16 of them *Nuphar variegata*. In the experiments reported here these photographs are part of the same pool as the laboratory and web images. They are not held out as a separate test. Because a 60/20/20 split leaves about eight outdoor images in the test portion, source-wise outdoor accuracy is reported only with that sample size in view and is not used to rank models.
>
> For seed 8 the pooled split is 655 training, 189 validation, and 186 test images. Other seeds move a few web images between splits, so the pooled test contains 186 to 190 images.
>
> Table total: Laboratory 203; Web 785; Outdoor 42. *Elodea canadensis* is 15 / 0 / 0.

The per-species web counts in the table were replaced with the cleaned counts (for example *Potamogeton gramineus* 106 to 94, *Potamogeton praelongus* 87 to 52, *Myriophyllum sibiricum* 70 to 26).

## 5. Methods paragraphs that carried old counts

**Location.** Augmentation; model capacity; patch source; training procedure; ConvNeXt-V2 Tiny paragraph.

**Why.** These paragraphs still referred to 834 training images, a 230/33 ablation, held-out outdoor photographs, and a backbone-accuracy section that was not rerun on the cleaned collection.

**Previous text (representative).**

> A 250-epoch run on 834 images takes about five minutes.
>
> Every training set in this study is far smaller than any of these counts (122 laboratory training images, 834 with the web images). [...] The combined set about 38 [images per species].
>
> On the 203-image laboratory and 42-image outdoor collections [...] the two sources agreed within one accuracy point [...] All YNLT results in this paper therefore use the downsampled source, which is also what produced Table [ablation].
>
> In the new protocol the checkpoint with the highest accuracy on the validation split of the training collection is kept, and the outdoor photographs are never seen before testing.
>
> Section [backbone comparison] reports what accuracy each of [Nano, Pico, Atto] gives up.

**Revised text (representative).**

> A 250-epoch run on about 650 training images takes about five minutes on an NVIDIA RTX 3070 Ti. At evaluation, images are resized to 224×224 and normalized, with no random transform.
>
> The pooled training split contains about 650 images. [...] What governs such a probe is the number of examples per class, here on the order of 30 in the full training split. When the last ConvNeXt stage is trained, the trainable count rises to 15.5 M parameters.
>
> All YNLT results in this paper use the downsampled source, which yields 111 patches per reviewed image.
>
> The checkpoint with the highest validation accuracy is kept. [...] In the last-stage experiment the last ConvNeXt stage is optimized at 1×10⁻⁴ while the head keeps the learning rate above, and earlier stages stay frozen.
>
> Table [cost] records their cost on the workstation used for this study. Those smaller encoders were not re-trained on the 1,030-image collection, so they are not ranked by accuracy here.

AdamW, label smoothing, and random erasing now have citations (see Section 9).

## 6. Results tables that were removed

**Location.** Former Tables `tab:ablation`, `tab:datagrid`, and `tab:backbones`, and the “Best model performance” subsection that used the old example figures.

**Why.** Those tables were not recomputed on the cleaned 1,030-image pool. Leaving them in the paper would mix two datasets. The single-split claim that gated attention plus YNLT is the best model (93.48% laboratory, 64.71% field) does not describe the current runs.

**Previous text.**

> Ablation caption: Results are from one stratified 60/20/20 laboratory split (46 test images) and a 17-image OOD test split.
>
> Headline numbers in that table: linear 84.78% laboratory / 47.06% OOD; gated attention + YNLT 93.48% laboratory / 64.71% OOD.
>
> Data-grid caption: laboratory test (39 images), web test (about 210 images), and 42 held-out outdoor photographs. Training rows were laboratory only, web only, and laboratory + web. Representative laboratory + web numbers: linear 81.5% ± 8.0 laboratory, 57.7% ± 4.4 web, 35.2% ± 2.6 outdoor.
>
> Backbone table: frozen encoders on the laboratory + web collection, three seeds, with outdoor photographs held out. Representative row: ConvNeXt-V2 Tiny linear 82.9% ± 9.7 laboratory, 56.0% ± 3.3 web, 34.9% ± 2.7 outdoor.
>
> Best-model subsection: “The last row in [the ablation] shows that the best performing model is a combination of a Gated Attention based classification head [...] along with the proposed YNLT block,” illustrated by `sample_prediction_initial.pdf` and `sample_prediction_final.pdf` on an out-of-distribution example (*Potamogeton gramineus* first called *Heteranthera dubia*).

**Revised text.** Those three tables, that subsection, and those two example figures are no longer in the manuscript. The cost table is kept, because it is a workstation measurement, and the text now says the smaller encoders were not retrained on this collection.

## 7. Results that replaced them

**Location.** Results: Classification head and YNLT; Training the last ConvNeXt stage.

**Why.** These are the completed experiments on the 1,030-image collection.

**Revised text.** Table `tab:heads` (frozen encoder, five seeds):

| Head | Laboratory acc. | Laboratory FNR | Web acc. | Web FNR | Pooled acc. | Pooled FNR |
|---|---|---|---|---|---|---|
| Linear | 80.0 ± 8.0 | 15.6 ± 9.9 | 53.0 ± 4.7 | 35.4 ± 9.9 | 59.5 ± 4.6 | 31.3 ± 7.6 |
| Linear + YNLT | 75.9 ± 7.2 | 6.7 ± 9.9 | 50.9 ± 2.8 | 25.7 ± 9.4 | 57.1 ± 2.6 | 22.2 ± 6.9 |
| MLP | 77.4 ± 10.0 | 11.1 ± 13.6 | 53.7 ± 3.5 | 32.6 ± 5.8 | 59.4 ± 4.3 | 28.4 ± 5.5 |
| MLP + YNLT | 77.4 ± 11.2 | 6.7 ± 9.9 | 52.5 ± 2.7 | 20.7 ± 6.8 | 58.5 ± 3.6 | 18.4 ± 4.4 |
| Gated attention | 80.0 ± 5.8 | 8.9 ± 9.3 | 52.0 ± 5.2 | 37.6 ± 6.5 | 58.6 ± 4.6 | 32.6 ± 3.9 |
| Gated attention + YNLT | 79.0 ± 7.6 | 2.2 ± 5.0 | 52.5 ± 3.8 | 19.9 ± 5.7 | 58.8 ± 4.0 | 17.4 ± 5.7 |

The prose states that the accuracy gaps among the three heads are smaller than the seed-to-seed spread, and that the outdoor slice (about eight images, 70–75% for these frozen heads) is not used to rank models.

Table `tab:laststage` (linear head, five seeds):

| Encoder | Pooled acc. | Laboratory acc. | Web acc. | Pooled FNR |
|---|---|---|---|---|
| Frozen | 59.5 ± 4.6 | 80.0 ± 8.0 | 53.0 ± 4.7 | 31.3 ± 7.6 |
| Last stage trained | 65.1 ± 5.0 | 85.6 ± 9.7 | 58.1 ± 4.8 | 28.8 ± 9.6 |

The prose states that the FNR change (31.3% ± 7.6 to 28.8% ± 9.6) is smaller than the standard deviation, so it is not described as an improvement in invasive detection. The last-stage linear model has the highest pooled accuracy. The lowest pooled invasive FNR remains gated attention with YNLT.

## 8. Figures

**Location.** Learning curve, and the new loss, recall, and Grad-CAM figures.

**Why.** The previous learning curve was fit on the older laboratory + web protocol and included a held-out outdoor panel. The example prediction figures belonged to that protocol. The meeting asked for the figures to be redone and for heatmaps and the error analysis to be added.

**Previous text.**

> Caption: Test accuracy against the number of training images [...] Panels: laboratory test, web test, outdoor test.
>
> Prose: laboratory accuracy rises from 30% (linear) and 36% (gated) at 6 images per class to 82% and 84% at 40, and web accuracy from 32–35% to 56–58%. [...] for the unseen hand-held photographs the limit is the domain of the training images, not their number.

**Revised text.**

> Caption: Top-1 accuracy against the number of training images for the linear and gated-attention heads on a frozen ConvNeXt-V2 Tiny encoder. Mean ± standard deviation; three seeds at the smaller fractions and five seeds at the full split. YNLT is not applied. Panels: pooled test, laboratory test, web test.
>
> Prose: On the pooled test, linear-head accuracy rises from 32.9% ± 1.6% at 95 training images to 43.6% ± 1.2% at 188, 51.6% ± 3.5% at 353, and 59.5% ± 4.6% at 650. The gated head follows the same path, from 32.4% ± 2.9% to 58.6% ± 4.6%. Neither curve has flattened.

New figures:

- `figs/loss_curves.pdf`. Mean training and validation loss over five seeds. Validation loss falls early and then stays near 1.5 through epoch 250. The checkpoint is the epoch of highest validation accuracy.
- `figs/species_recall.pdf`. Per-species recall for the frozen linear head, seed 9 (189 images, 126 correct, 66.7%). The dashed line is that checkpoint’s accuracy.
- `figs/gradcam_panel.pdf`. Grad-CAM pairs for the same seed-9 checkpoint: correct laboratory, web, and outdoor examples, and incorrect look-alike examples. The caption states the checkpoint. It does not claim that the maps are from the last-stage model.

The seed-9 error paragraph reports: 28 of 63 mistakes (44%) stay in the same genus; *Potamogeton praelongus* is 2/10 and is called *Potamogeton crispus* four times; *Potamogeton richardsonii* is 2/7; *Potamogeton crispus* is 6/14; *Myriophyllum spicatum* is called *Cabomba* twice. The invasive FNR on this one split is 18.8% (9 of 48). The text says these counts are not the five-seed means.

Figures kept, with no change to the graphic: the Michigan map, the augmentation plate, the pipeline diagram, and the ensemble diagram. Their captions do not describe a controlled-to-field experiment.

**Note on the heatmaps.** The maps were produced earlier for the frozen linear seed-9 checkpoint on this same pooled test. They were not recomputed for the last-stage checkpoint in this revision, because that checkpoint’s training environment was not available here. The caption names the frozen seed-9 checkpoint so the figure is not attributed to the last-stage model.

## 9. Discussion and conclusion

**Location.** Discussion and Conclusion.

**Why.** Both sections still concluded from the single-split 230/33 experiment and from the 1,140-image, outdoor-held-out protocol, including the claim that the gated head produced 64.71% accuracy on unseen crowd-sourced images and that outdoor accuracy had risen from about 25% to about 35%.

**Previous text (conclusion, shortened).**

> A frozen ConvNeXt-V2 Tiny backbone [...], a gated attention head [...], and [...] YNLT raised laboratory accuracy from 84.78% to 93.48% and field accuracy from 47.06% to 64.71% on a single stratified split of 230 laboratory and 33 field images. [...] 1,140 openly licensed field photographs [...] Training on laboratory and web images together is the best combination: it keeps laboratory accuracy at 82–84%, triples web accuracy to 56–58% and raises outdoor accuracy from about 25% to about 35% [...] YNLT cuts the invasive false negative rate [...] to 2% on the laboratory test and from 63–70% to 37% outdoors. [...] Nano keeps more than 90% of its accuracy at 55% of the cost.

**Revised text (conclusion).**

> This study classifies 21 Michigan aquatic plant species, five of them invasive, from a mixed collection of 1,030 photographs. [...] With a frozen ConvNeXt-V2 Tiny encoder, a linear head, an MLP, and a gated-attention head reach 58.6% to 59.5% pooled top-1 accuracy over five seeds. The differences are smaller than the seed-to-seed spread. YNLT [...] lowers the pooled invasive FNR of the gated head from 32.6% ± 3.9% to 17.4% ± 5.7%. Training the last ConvNeXt stage raises pooled accuracy from 59.5% ± 4.6% to 65.1% ± 5.0% and does not clearly change the invasive FNR. Accuracy is still increasing at about 650 training images. The mistakes that remain are concentrated on look-alike pondweeds, and class activation maps place the network’s response on the plant rather than on the background.

The discussion now says explicitly that the paper does not measure transfer from laboratory photographs to a separate field test, and that it does not measure underwater rover imagery.

## 10. References

**Location.** `ref.bib`, cited from the new paragraphs.

**Why.** The new dataset statement, the training recipe, and the heatmaps needed sources that were not cited. Existing unverified introduction entries were left unchanged.

**Previous keys used in the new passages.** ConvNeXt-V2 only (`Wooetal2023`). Label smoothing cited Müller et al. only. No Grad-CAM, GBIF, iNaturalist, Pl@ntNet, AdamW, or random-erasing entries.

**Added and cited.**

| Key | Work |
|---|---|
| `liu2022convnet` | Liu et al., A ConvNet for the 2020s, CVPR 2022 (entry was already in the file; it is now cited) |
| `joly2016plantnet` | Joly et al., A look inside the Pl@ntNet experience, Multimedia Systems, 2016 |
| `gbif2026` | GBIF occurrence download used for the web photographs |
| `inaturalist2026` | iNaturalist research-grade observations, accessed through GBIF |
| `selvaraju2017gradcam` | Selvaraju et al., Grad-CAM, ICCV 2017 |
| `zhong2020random` | Zhong et al., Random Erasing Data Augmentation, AAAI 2020 |
| `loshchilov2019adamw` | Loshchilov and Hutter, Decoupled Weight Decay Regularization, ICLR 2019 |
| `szegedy2016rethinking` | Szegedy et al., Rethinking the Inception Architecture for Computer Vision, CVPR 2016 (label smoothing) |

## 11. What was checked and what was not rerun

Checked against the saved run files: the six frozen-head rows, the last-stage means (65.1 ± 5.0 pooled accuracy, 28.8 ± 9.6 pooled FNR, 85.6 ± 9.7 laboratory, 58.1 ± 4.8 web), and the learning-curve points printed while redrawing Figure `fig:curve`.

Not rerun, and therefore not given new numbers: the laboratory-only versus web-only grid, the Nano / Pico / Atto / ResNet / EfficientNet / ViT comparison, and Grad-CAM on the last-stage checkpoint. The first two are omitted. The heatmaps are labeled as the frozen linear seed-9 checkpoint.
