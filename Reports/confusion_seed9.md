# Confusion on the pooled test, linear head, seed 9

Checkpoint: the frozen ConvNeXt-V2 Tiny linear model trained on lab + web + outdoor.
No new training. The test split is the held-out 20% of that same pool.

Images: 189. Correct: 126 (66.7%).
Invasive miss rate: 18.8% (9 missed of 48 invasive test images).

| Species | Test images | Correct | Recall | Most common mistake |
|---|---:|---:|---:|---|
| Brasenia schreberi | 9 | 7 | 78% | Hydrocharis morsus-ranae (1) |
| Cabomba | 9 | 8 | 89% | Ranunculus aquatilis (1) |
| Ceratophyllum demersum | 9 | 7 | 78% | Myriophyllum spicatum (1) |
| Elodea canadensis | 3 | 2 | 67% | Potamogeton gramineus (1) |
| Heteranthera dubia | 6 | 5 | 83% | Potamogeton natans (1) |
| Hydrocharis morsus-ranae | 6 | 6 | 100% |  |
| Myriophyllum sibiricum | 6 | 3 | 50% | Ceratophyllum demersum (1) |
| Myriophyllum spicatum | 10 | 6 | 60% | Cabomba (2) |
| Najas flexilis | 4 | 3 | 75% | Ranunculus aquatilis (1) |
| Nitellopsis obtusa | 9 | 6 | 67% | Potamogeton gramineus (1) |
| Nuphar variegata | 11 | 9 | 82% | Potamogeton crispus (1) |
| Nymphaea odorata | 7 | 5 | 71% | Potamogeton gramineus (1) |
| Potamogeton crispus | 14 | 6 | 43% | Myriophyllum spicatum (3) |
| Potamogeton gramineus | 24 | 13 | 54% | Potamogeton crispus (3) |
| Potamogeton illinoensis | 12 | 8 | 67% | Potamogeton crispus (2) |
| Potamogeton natans | 7 | 5 | 71% | Nuphar variegata (2) |
| Potamogeton praelongus | 10 | 2 | 20% | Potamogeton crispus (4) |
| Potamogeton richardsonii | 7 | 2 | 29% | Potamogeton crispus (2) |
| Potamogeton robbinsii | 5 | 5 | 100% |  |
| Ranunculus aquatilis | 13 | 13 | 100% |  |
| Vallisneria americana | 8 | 5 | 62% | Heteranthera dubia (1) |

## Most common confusions

| True species | Predicted | Count |
|---|---|---:|
| Potamogeton praelongus | Potamogeton crispus | 4 |
| Potamogeton crispus | Myriophyllum spicatum | 3 |
| Potamogeton gramineus | Potamogeton crispus | 3 |
| Myriophyllum spicatum | Cabomba | 2 |
| Potamogeton crispus | Potamogeton gramineus | 2 |
| Potamogeton gramineus | Potamogeton praelongus | 2 |
| Potamogeton gramineus | Potamogeton natans | 2 |
| Potamogeton illinoensis | Potamogeton crispus | 2 |
| Potamogeton natans | Nuphar variegata | 2 |
| Potamogeton praelongus | Potamogeton gramineus | 2 |
| Potamogeton richardsonii | Potamogeton crispus | 2 |
| Brasenia schreberi | Hydrocharis morsus-ranae | 1 |
| Brasenia schreberi | Ceratophyllum demersum | 1 |
| Cabomba | Ranunculus aquatilis | 1 |
| Ceratophyllum demersum | Myriophyllum spicatum | 1 |

Of 63 mistakes, 28 stay inside the same genus name (44%).
The rest cross genera, mostly between plants with a similar growth form (feathery submerged leaves, or broad floating leaves).
