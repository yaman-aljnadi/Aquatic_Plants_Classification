Week 8 - 10/13/2026 - 10/19/2026

Summary of work:

1- Scored the existing linear model (seed 9) on the full held-out test set, 189 images. No new training for this step. 126 images are correct (66.7%).

2- Trained one new comparison on the same pooled data, the same augmentations, and the same linear head. The only change is that the last ConvNeXt-V2 stage is trained at a lower learning rate (1e-4). The earlier stages stay frozen. Five seeds (8–12).

3- Compared those runs with the frozen linear runs from Week 7 and checked the validation loss. The last-stage runs are kept because validation loss does not climb after the saved checkpoint.

Important Findings:

1- On the seed-9 test set, 63 of 189 images are wrong. 28 of those 63 mistakes (44%) stay inside the same genus. The weakest species are pondweeds: Potamogeton praelongus is 2/10 correct and is called Potamogeton crispus 4 times. Potamogeton crispus is 6/14 correct. Potamogeton richardsonii is 2/7 correct. Myriophyllum spicatum is confused with Cabomba. Hydrocharis morsus-ranae, Potamogeton robbinsii, and Ranunculus aquatilis are fully correct on this split.

2- Training the last stage, five seeds, mean ± standard deviation:

| | Frozen encoder | Last stage trained |
|---|---|---|
| Mixed test accuracy | 59.5% ± 4.6 | 65.1% ± 5.0 |
| Lab accuracy | 80.0% ± 8.0 | 85.6% ± 9.7 |
| Web accuracy | 53.0% ± 4.7 | 58.1% ± 4.8 |
| Invasive miss rate | 31.3% ± 7.6 | 28.8% ± 9.6 |

3- Mixed-test accuracy is about 5 points higher. The invasive miss rate is only a little lower, and that drop is smaller than the seed-to-seed spread, so it is not a clear improvement. Validation loss stays flat, so the extra stage is not overfitting. Handheld photos are still out of training.

Next Steps:

1- Use the last-stage linear model as the current best accuracy result (65.1% on the mixed test).

2- The species that still fail are pondweeds confused with other pondweeds. More photos of those species, especially Potamogeton praelongus, Potamogeton crispus, and Potamogeton richardsonii, are the next place accuracy can move. The learning curve from Week 7 was still rising at 650 training images.

3- The outdoor test slice is still only 8 images, so outdoor accuracy should not be compared with the older held-out field test.

Meeting summary:

This week I stopped changing the head and looked at which species the model gets wrong. The mistakes are look-alike pondweeds and feathery plants. Training the last ConvNeXt stage, with everything else frozen, raises mixed-test accuracy by about 5 points (59.5% to 65.1%) without the validation loss rising. The invasive miss rate is about the same. I would like to know whether to put this result into the paper next, or to collect more photographs of the pondweed species the model still confuses.
