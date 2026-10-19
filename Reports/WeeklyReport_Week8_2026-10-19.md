Week 8 - 10/13/2026 - 10/20/2026

Summary of work:

1- Scored the existing linear model (seed 9) on the full held-out test set and wrote a per-class recall table plus the most common confusions. No new training for this step. 126 of 189 test images are correct (66.7%).

2- Trained one new comparison on the same pooled data, the same augmentations, and the same linear head. The only change is that the last ConvNeXt-V2 stage is trained at a lower learning rate (1e-4). The earlier stages stay frozen. Five seeds (8–12). Validation loss was checked the same way as last week.

3- Compared those runs with the frozen linear runs from Week 7. The last-stage runs are kept because validation loss does not climb after the saved checkpoint.

4- Updated the paper (`Paper/main_v3.tex`) so it reports the cleaned 1,030-image collection and these results. Handheld photographs stay out. The old single-split numbers (84.78% to 93.48% in the laboratory, 47.06% to 64.71% on 17 field images) and the held-out outdoor tables were removed, because those runs used the earlier 1,140-image web set. Wording that described a controlled laboratory-to-field test was removed. The outdoor photographs are described as part of the same pool, not as a separate field test.

5- Redrew the result figures from the 1,030-image runs and added them to the paper: the learning curve, the training and validation loss curves, a per-species recall chart for the seed-9 linear model, and a Grad-CAM panel from that same checkpoint. A short change log for review is in `Paper/revision_log_v3.md`. The manuscript and the figure files are being uploaded to Overleaf.

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

4- On this collection the three frozen heads are tied on mixed-test accuracy (linear 59.5%, MLP 59.4%, gated attention 58.6%). YNLT with the gated head does not raise accuracy, but it lowers the invasive miss rate from 32.6% ± 3.9 to 17.4% ± 5.7. The paper now states both results: last-stage training for accuracy, and YNLT for missed invasive plants. The learning curve is still rising at about 650 training images.

Next Steps:

I might require some guidance on what I should investigate next, should I keep trying to achieve better results or change my focus into something else?

Meeting summary:

This week I scored the frozen linear model image by image and trained the last ConvNeXt stage. Mixed-test accuracy rose by about 5 points (59.5% to 65.1%) without the validation loss rising. The invasive miss rate is about the same. The mistakes are look-alike pondweeds and feathery plants. I put these results into the paper, replaced the old 1,140-image and held-out outdoor numbers, and added the learning curve, loss curves, species recall chart, and Grad-CAM panel. I would like guidance on whether the next step is more photographs of the pondweed species the model still confuses, or a change of focus.
