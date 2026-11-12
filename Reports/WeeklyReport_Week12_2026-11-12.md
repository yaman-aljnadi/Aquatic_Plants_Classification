Week 12 - 11/09/2026 - 11/16/2026

Summary of work:

After the last discussion, Dr. Dang suggested few-shot learning with Prototypical Networks and data augmentation, including a longer-term idea of shape- and texture-aware plant embeddings. I chose to start with a frozen Prototypical Network baseline on our existing aquatic plant collection, rather than jumping straight into a new embedding design or background suppression.

The reason is that our earlier work already showed where the difficulty is. On the pooled 1,030-image set, a frozen ConvNeXt-V2 Tiny linear head reaches about 59.5% accuracy, and training the last encoder stage raises that to about 65.1%. The remaining mistakes are concentrated on look-alike species, especially pondweeds, and several classes still have few images. Grad-CAM also showed that the model often looks at the plant, but still confuses similar morphology. That made few-shot classification a natural next question: how well can we recognize a species when only a handful of support images are available?

I therefore reused the same encoder, the same `lab+web+field` splits, and the same seeds (8–12), and implemented a Prototypical Network readout in `scripts/protonet_baseline.py`. For each seed, K support images per class are taken from the training split, a class prototype is the mean ConvNeXt feature of those supports, and each test image is assigned to the nearest prototype by Euclidean distance. I evaluated K = 1, 5, 10, and all training images, with and without support-time augmentation (averaging 8 augmented views per support image). This gives a clean few-shot baseline before spending time on episodic training or shape/texture branches.

Important Findings:

1- Accuracy rises with more support images, as expected for a prototype classifier:

| Support | Lab | Web | Field | Pooled | Invasive FNR |
|---|---|---|---|---|---|
| 1-shot | 20.0% ± 4.9 | 15.8% ± 3.7 | 17.5% ± 6.8 | 16.8% ± 2.6 | 70.0% ± 9.1 |
| 5-shot | 39.0% ± 6.1 | 31.9% ± 5.1 | 30.0% ± 19.0 | 33.4% ± 3.8 | 55.7% ± 12.0 |
| 10-shot | 48.2% ± 6.1 | 31.0% ± 2.4 | 35.0% ± 5.6 | 34.7% ± 2.1 | 54.4% ± 7.0 |
| all train | 49.7% ± 3.9 | 40.4% ± 2.2 | 40.0% ± 18.5 | 42.3% ± 2.3 | 51.1% ± 4.8 |

2- The frozen ProtoNet is a useful few-shot baseline, but it is not yet competitive with a trained head. Using all training images as supports reaches 42.3% ± 2.3 pooled accuracy, below the frozen linear head (59.5% ± 4.6). At the low-data end, 5-shot ProtoNet (33.4% ± 3.8) is close to the linear learning-curve point at 10% of the training set (32.9% ± 1.6). So the current ImageNet features already support a prototype readout, but the embedding still needs to be adapted for this aquatic plant task.

3- Support-time augmentation alone did not help. Averaging 8 augmented views per support image lowered all-support pooled accuracy slightly to 38.6% ± 4.2. This suggests that, for the next step, augmentation should be used inside episodic ProtoNet training rather than only when building frozen prototypes.

4- Field numbers remain noisy because only about eight outdoor photographs fall in a given pooled test split, so I am treating lab, web, and pooled accuracy as the main comparisons for now.

Next Steps:

1- Train an episodic Prototypical Network (support/query episodes and ProtoNet loss), optionally unfreezing the last ConvNeXt stage, so the embedding is learned for nearest-prototype classification instead of only being read out from a frozen encoder.

2- After that baseline is in place, add shape- and texture-aware embeddings aimed at the look-alike pondweed and feathery-plant confusions from the earlier error analysis.

3- Keep the frozen ProtoNet table above as the reference few-shot baseline for this new direction.
