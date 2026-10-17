Week 7 - 10/05/2026 - 10/12/2026

Summary of work:

1- Went through the web photographs and removed images that do not belong. After that cleanup the web set is 785 images (it was 1,140). Handheld photos were not put back into training. Elodea canadensis has no web photos left.

2- Moved the handheld images into their own folder (`Larger_Plants_Dataset/aquatic_dataset_report/Hand_Held_Images`) and kept them out of the training pool.

3- Trained one combined set: lab + cleaned web + outdoor (1,030 images; 655 train / 189 val / 186 test). Web photos stay grouped by observation so the same plant is not in both train and test. Frozen ConvNeXt-V2 Tiny, linear / MLP / gated heads, five seeds, plus the learning-curve fractions. 33 runs, no failures. Runs are in `runs/models/latest_pooled_20261009`; the 4 October grid is in `runs/models/previous`.

4- Grad-CAM heatmaps for every test image of the best single checkpoint (linear head, seed 9, 128/189 correct). Saved under `Reports/heatmaps/correct` (128) and `Reports/heatmaps/wrong` (61).

5- Saved train and validation loss every epoch and plotted the curves (`Reports/pooled_loss_curves.png`) and the learning curve (`Reports/pooled_learning_curve.png`).

Dataset in the current training:

Three kinds of photograph are in the model. Handheld photos are not.

- Lab (indoor, controlled white background): 203 images. All are used.
- Web (field photographs from GBIF / iNaturalist, after cleanup): 785 images. These are the web photos that are not handheld.
- Outdoor: 42 images. These used to be a held-out test. They are now in the same pool as the lab and web photos.
- Handheld: 307 images, in `Hand_Held_Images`. Not in train, validation, or test.

The original web download was 1,140 images. 307 of those were moved to the handheld folder. Another 48 were removed because they did not belong (39 of those were every web photo of Elodea canadensis). Elodea is lab-only now: 15 indoor photos, no web, no outdoor, no handheld.

Split is 60% train / 20% validation / 20% test, per species. Web photos of the same plant stay in one split. Counts below are seed 8. Other seeds move a few web images between splits. Train / val / test already mix lab, web, and outdoor. Handheld is extra and unused.

| Species | Lab (indoor) | Web (field) | Outdoor | Handheld (not used) | Train | Val | Test |
|---|---:|---:|---:|---:|---:|---:|---:|
| Brasenia schreberi | 8 | 45 | 0 | 15 | 35 | 9 | 9 |
| Cabomba | 5 | 45 | 0 | 7 | 30 | 10 | 10 |
| Ceratophyllum demersum | 21 | 31 | 1 | 20 | 33 | 10 | 10 |
| Elodea canadensis | 15 | 0 | 0 | 0 | 9 | 3 | 3 |
| Heteranthera dubia | 7 | 28 | 0 | 4 | 23 | 6 | 6 |
| Hydrocharis morsus-ranae | 2 | 27 | 0 | 9 | 18 | 5 | 6 |
| Myriophyllum sibiricum | 4 | 26 | 0 | 44 | 18 | 6 | 6 |
| Myriophyllum spicatum | 13 | 42 | 3 | 6 | 35 | 12 | 11 |
| Najas flexilis | 5 | 20 | 0 | 28 | 16 | 4 | 5 |
| Nitellopsis obtusa | 5 | 35 | 2 | 19 | 31 | 5 | 6 |
| Nuphar variegata | 3 | 34 | 16 | 0 | 31 | 11 | 11 |
| Nymphaea odorata | 2 | 33 | 1 | 0 | 22 | 6 | 8 |
| Potamogeton crispus | 20 | 56 | 1 | 11 | 51 | 14 | 12 |
| Potamogeton gramineus | 17 | 94 | 7 | 11 | 75 | 23 | 20 |
| Potamogeton illinoensis | 36 | 33 | 0 | 23 | 44 | 14 | 11 |
| Potamogeton natans | 8 | 37 | 0 | 1 | 29 | 8 | 8 |
| Potamogeton praelongus | 3 | 52 | 0 | 35 | 36 | 9 | 10 |
| Potamogeton richardsonii | 6 | 24 | 6 | 39 | 24 | 6 | 6 |
| Potamogeton robbinsii | 8 | 22 | 0 | 21 | 20 | 5 | 5 |
| Ranunculus aquatilis | 5 | 62 | 5 | 4 | 44 | 14 | 14 |
| Vallisneria americana | 10 | 39 | 0 | 10 | 31 | 9 | 9 |
| Total | 203 | 785 | 42 | 307 | 655 | 189 | 186 |

Augmentation:

Used on the training images only. Every photo is trained at 224×224. Validation and test images are only resized to 224×224 and normalized. They are not randomly changed.

- Random resized crop, keeping 10% to 100% of the image, then resized to 224×224
- Horizontal flip, probability 0.5
- Vertical flip, probability 0.5
- Rotation up to 90 degrees
- Random perspective, distortion 0.15, probability 0.5
- Color jitter: brightness 0.3, contrast 0.3, saturation 0.3, hue 0.1
- TrivialAugmentWide
- ImageNet normalization (mean 0.485, 0.456, 0.406; std 0.229, 0.224, 0.225)
- Random erasing, probability 0.5, covering 2% to 10% of the image

Important Findings:

1- The three heads are tied on the mixed test: linear 59.5%, MLP 59.4%, gated attention 58.6% (mean over five seeds). The second look (YNLT) on the gated head is 58.8% accuracy and cuts the invasive miss rate from about 31% to 17.4%. Accuracy is not where the heads differ; missed invasive plants are.

2- Validation loss falls for about 40 epochs and then stays flat near 1.5 through epoch 250 for every head. It does not climb away from the training loss. The saved checkpoint is the best validation accuracy on that flat part, so the training recipe was left as it is. The model is not overfitting.

3- The learning curve is still rising: about 33% pooled accuracy at 95 training images, about 59% at 650. More photographs would still help.

4- Against the 4 October runs (lab + web only, all 42 outdoor photos held out): lab accuracy is about the same (~80% now, 81–84% then) and web accuracy is a few points lower (~53% vs 56–58%) after the bad and handheld photos were removed. Outdoor accuracy looks higher (~70% vs ~35%), but that is not a fair comparison. Outdoor photos are now inside training, and the outdoor test slice is only 8 images. The old 35% was a real unseen field test.

5- On the heatmaps, correct predictions light up the plant (leaf, flower, floating leaves), not the coin, the label card, or the background. The mistakes are look-alike species (Myriophyllum read as Cabomba or Ceratophyllum), and the highlight is still on the leaves.

Next Steps:

1- Put these numbers into the paper. `main_v3.tex` was not edited this week.

2- The outdoor test is only 8 images now. A larger held-out outdoor set is needed before outdoor accuracy means much.

3- Collect more photographs. The learning curve has not flattened.

4- The 75x75 iNaturalist download is still deferred.

Meeting summary:

Asked to drop images that do not belong, separate handheld photos, train lab, web, and outdoor together, and check that the model does not overfit. That is what this week did. The combined model holds laboratory accuracy, the heads are tied near 59% on the mixed test, and the second look is what lowers missed invasive plants. Validation loss is flat, so there is no overfitting. Outdoor accuracy cannot be compared with last week because those photos are no longer unseen. Heatmaps show the model looks at the plant. More images would still help.
