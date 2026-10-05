# Model weights vs. number of training samples

**Question from Dr. Dang (meeting of 3 Oct 2026):** what are the weights/parameters of the
model, and how many samples do we need to train them?

Numbers below come from `Yaman-Aquatic_Plants_Classification/scripts/parameter_budget.py`
(full tables in `Reports/parameter_budget_tables.md`). Re-run it after any change to the
heads, backbones, or image folders.

## 1. Which weights are trained

The network has two parts.

| Part | Weights | Trained on our images? |
|---|---|---|
| ConvNeXt-V2 Tiny backbone | 27.87 M | No. Frozen at the published ImageNet-22k/1k checkpoint. |
| Linear head | 16,149 | Yes |
| MLP head (42 hidden units) | 33,201 | Yes |
| Gated attention head (168 units) | 261,933 | Yes |

So the number that matters for the sample-size question is not 28 M, it is 16 k to 262 k,
depending on the head. The backbone contributes a fixed 768-dimensional feature vector per
image; the head is a classifier fitted on those 768 numbers.

Closed forms with D = 768 features, K = 21 classes, d = 168, m = 42:

- linear: D*K + K = 16,149
- MLP: D*m + m + m*K + K = 33,201
- gated attention: 2*(D*d + d) + d*K + K = 261,933

The same heads on the smaller ConvNeXt-V2 variants (rover candidates) have fewer weights
because the feature vector is shorter: Nano (D = 640) 218,925 gated weights, Pico (512)
175,917, Atto (320) 111,405.

## 2. How many images stand behind those weights

| Data configuration | Images | Train images (60%) | Train images per class | Train images per gated weight |
|---|---|---|---|---|
| Lab only (what the paper used) | 203 | 122 | 5.8 | 0.0005 |
| Web (GBIF/iNaturalist) only | 1,140 | 684 | 32.6 | 0.0026 |
| Lab + web | 1,343 | 806 | 38.4 | 0.0031 |

The 42 outdoor photographs are never trained on; they are the held-out field test.

### Why "10 samples per parameter" is the wrong rule here

The classical rule (about 10 samples per free parameter) describes an unregularised model
fitted from scratch. By that rule the gated head would need about 2.6 M images and even the
linear head about 160 k. No transfer-learning classifier meets this, and it is not how
these models are evaluated in the literature. Three things change the picture:

1. **The features are fixed.** The 27.87 M backbone weights were fitted on 14 M ImageNet
   images, not ours. We only fit a classifier on a 768-dimensional representation.
2. **The head is regularised.** Weight decay (2e-5), dropout (0.1), label smoothing (0.05),
   heavy augmentation (random resized crops of 10-100% of the image, flips, rotation,
   colour jitter, TrivialAugment, random erasing) and checkpoint selection on validation
   accuracy all limit the effective capacity. A linear probe on frozen features is, in
   effect, a 21-class logistic regression on 768 inputs, which is routinely fitted with a
   few tens of examples per class.
3. **What governs a probe is images per class, not images per weight.** Published linear
   probes on frozen ImageNet encoders reach most of their accuracy by roughly 20-50 images
   per class and saturate somewhere between 50 and a few hundred. Our lab-only set has
   about 6 training images per class (several species have one or two). The combined set
   has about 38, which is in the range where probes normally stabilise.

### The honest answer: measure it

The defensible answer to "how many samples do we need" is a learning curve: train the same
head on 10%, 25%, 50% and 100% of the combined training split (keeping validation and test
fixed) and plot test accuracy against the number of training images. Where the curve
flattens is where more images stop paying off. This is run by
`python scripts/run_matrix.py --suite curve` and plotted by
`python scripts/plot_learning_curve.py` (figure: `Paper/figs/learning_curve.pdf`).

Running the linear, MLP and gated heads side by side on the same data also answers the
related question "does the larger head need more data": if gated attention beats linear at
every fraction the extra 246 k weights are not hurting; if it only wins at 100% the head is
data-hungry.

## 3. How big the test sets need to be

Even a perfect model cannot be judged on a tiny test set. The 95% confidence half-width of an
accuracy estimate is 1.96 * sqrt(p(1-p)/n):

| Test images n | half-width at 80% accuracy |
|---|---|
| 17 (paper field test) | +/-19 points |
| 21 (current field test) | +/-17 points |
| 41-46 (lab test) | +/-12 points |
| ~230 (web test, 20% of 1,140) | +/-5 points |
| 500 | +/-3.5 points |

Two single runs can only be called different when the gap exceeds about
1.96 * sqrt(2p(1-p)/n): 17 points for n = 41, 7 points for n = 230. One flipped image is
2.4 points on a 41-image lab test and 4.8 points on a 21-image field test. This is why the
seed-to-seed spread in the current Tables 2-3 is so large, and why the new protocol reports
mean +/- standard deviation over five seeds and adds a 228-image web test split.

## 4. Summary for the meeting

- Trainable weights: 16 k (linear), 33 k (MLP), 262 k (gated). Backbone (27.9 M) is frozen.
- Lab-only training gave about 6 images per class; lab + web gives about 38.
- Samples-per-weight rules do not apply to a regularised probe on frozen features; images
  per class does, and the learning curve measures it directly.
- Test sets of 21-46 images cannot separate models closer than 12-19 points; the 228-image
  web test and five-seed averaging bring that down to a few points.
