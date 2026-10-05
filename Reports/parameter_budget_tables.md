## 1. Weights: frozen backbone vs. trained head

Only the classification head is trained. Backbone weights are the published timm
checkpoints and never change, so they are not fitted to our images at all.

| Backbone preset | timm name | Feature dim | Frozen params (M) | Linear trainable | MLP (42 hidden) trainable | Gated attention (168) trainable |
|---|---|---|---|---|---|---|
| convnextv2 | `convnextv2_tiny.fcmae_ft_in22k_in1k` | 768 | 27.87 | 16,149 | 33,201 | 261,933 |
| convnextv2_nano | `convnextv2_nano.fcmae_ft_in22k_in1k` | 640 | 14.98 | 13,461 | 27,825 | 218,925 |
| convnextv2_pico | `convnextv2_pico.fcmae_ft_in1k` | 512 | 8.55 | 10,773 | 22,449 | 175,917 |
| convnextv2_atto | `convnextv2_atto.fcmae_ft_in1k` | 320 | 3.39 | 6,741 | 14,385 | 111,405 |
| convnext | `convnext_tiny.fb_in22k_ft_in1k` | 768 | 27.82 | 16,149 | 33,201 | 261,933 |
| resnet50 | `resnet50.a1_in1k` | 2048 | 23.51 | 43,029 | 86,961 | 692,013 |
| efficientnet | `tf_efficientnetv2_b0.in1k` | 1280 | 5.86 | 26,901 | 54,705 | 433,965 |
| vit | `vit_base_patch16_224.augreg_in21k_ft_in1k` | 768 | 85.80 | 16,149 | 33,201 | 261,933 |

Closed forms (D = feature dim, K = 21 classes, d = 168, m = 42):
linear = D*K + K; MLP = D*m + m + m*K + K; gated = 2*(D*d + d) + d*K + K.

## 2. Samples behind those weights

Training uses 60% of each configuration (the rest is validation and test).

| Data | Images | Classes | Min / median / max per class | Train images (60%) | Train images per class | train img per Linear weight | train img per MLP (42 hidden) weight | train img per Gated attention (168) weight |
|---|---|---|---|---|---|---|---|---|
| lab | 203 | 21 | 2 / 7 / 36 | 122 | 5.8 | 0.0076 | 0.0037 | 0.0005 |
| web | 1140 | 21 | 32 / 52 / 106 | 684 | 32.6 | 0.0424 | 0.0206 | 0.0026 |
| lab+web | 1343 | 21 | 37 / 61 / 123 | 806 | 38.4 | 0.0499 | 0.0243 | 0.0031 |
| field (outdoor, held-out test only) | 42 | 9 | 1 / 3 / 16 | 0 | -- | -- | -- | -- |

Per-class image counts:

| Species | Invasive | lab | web | lab+web | field |
|---|---|---|---|---|---|
| Brasenia schreberi |  | 8 | 60 | 68 | 0 |
| Cabomba | yes | 5 | 53 | 58 | 0 |
| Ceratophyllum demersum |  | 21 | 52 | 73 | 1 |
| Elodea canadensis |  | 15 | 39 | 54 | 0 |
| Heteranthera dubia |  | 7 | 32 | 39 | 0 |
| Hydrocharis morsus-ranae | yes | 2 | 36 | 38 | 0 |
| Myriophyllum sibiricum |  | 4 | 70 | 74 | 0 |
| Myriophyllum spicatum | yes | 13 | 48 | 61 | 3 |
| Najas flexilis |  | 5 | 48 | 53 | 0 |
| Nitellopsis obtusa | yes | 5 | 54 | 59 | 2 |
| Nuphar variegata |  | 3 | 34 | 37 | 16 |
| Nymphaea odorata |  | 2 | 36 | 38 | 1 |
| Potamogeton crispus | yes | 20 | 67 | 87 | 1 |
| Potamogeton gramineus |  | 17 | 106 | 123 | 7 |
| Potamogeton illinoensis |  | 36 | 56 | 92 | 0 |
| Potamogeton natans |  | 8 | 38 | 46 | 0 |
| Potamogeton praelongus |  | 3 | 87 | 90 | 0 |
| Potamogeton richardsonii |  | 6 | 63 | 69 | 6 |
| Potamogeton robbinsii |  | 8 | 43 | 51 | 0 |
| Ranunculus aquatilis |  | 5 | 66 | 71 | 5 |
| Vallisneria americana |  | 10 | 52 | 62 | 0 |

Reading the ratio column: every configuration has far fewer images than trainable
weights, so the classical '10 samples per parameter' rule is violated by orders of
magnitude for every head. That rule describes an unregularised model fitted from
scratch. Here the 768-d features are fixed, and the head is a regularised probe
(weight decay 2e-5, dropout 0.1, label smoothing 0.05, heavy augmentation, checkpoint
selection on validation accuracy). The quantity that governs a probe like this is
images per class, which is what the learning curve measures directly.

## 3. What a test split of size n can resolve

95% confidence half-width of an accuracy estimate, 1.96 * sqrt(p(1-p)/n):

| True accuracy | n=17 | n=21 | n=41 | n=46 | n=100 | n=150 | n=230 | n=300 | n=500 |
|---|---|---|---|---|---|---|---|---|---|
| 50% | +/-23.8 | +/-21.4 | +/-15.3 | +/-14.4 | +/-9.8 | +/-8.0 | +/-6.5 | +/-5.7 | +/-4.4 |
| 65% | +/-22.7 | +/-20.4 | +/-14.6 | +/-13.8 | +/-9.3 | +/-7.6 | +/-6.2 | +/-5.4 | +/-4.2 |
| 80% | +/-19.0 | +/-17.1 | +/-12.2 | +/-11.6 | +/-7.8 | +/-6.4 | +/-5.2 | +/-4.5 | +/-3.5 |
| 90% | +/-14.3 | +/-12.8 | +/-9.2 | +/-8.7 | +/-5.9 | +/-4.8 | +/-3.9 | +/-3.4 | +/-2.6 |

Test images needed for a given half-width at 80% accuracy:

| Half-width | n |
|---|---|
| +/-10 pts | 62 |
| +/-5 pts | 246 |
| +/-3 pts | 683 |
| +/-2 pts | 1537 |

Two independent accuracies differ significantly (alpha 0.05, two-sided) only when the
gap exceeds about 1.96 * sqrt(2 * p(1-p)/n):

| n per model | Minimum detectable gap at p=0.8 |
|---|---|
| 21 | 24.2 pts |
| 41 | 17.3 pts |
| 230 | 7.3 pts |
| 500 | 5.0 pts |

One image in a 41-image test set is 2.4 points; in a 21-image field test it is 4.8
points. A 230-image web test set resolves differences of about 7 points between two
single runs, and smaller ones once five seeds are averaged.
