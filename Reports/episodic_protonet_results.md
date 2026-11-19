# Episodic ProtoNet results (11/13/2026)

Trained a Prototypical Network on the same lab+web+field splits and ConvNeXt-V2 Tiny encoder (seeds 8–12). Each training episode uses 5 support and 3 query images per species, with the usual augmentation. Test scores use 1, 5, 10, or all training images as supports. Checkpoint is the best validation accuracy. Prototypes for validation and test come from the training split only.

The frozen prototype baseline from 11/12 did not train the embedding (42.3% pooled, all supports). These runs do.

## Pooled accuracy (mean ± std, 5 seeds)

| Setup | 1-shot | 5-shot | 10-shot | All supports | Invasive FNR (all) |
|---|---|---|---|---|---|
| Frozen prototypes (11/12, no training) | 16.8% ± 2.6 | 33.4% ± 3.8 | 34.7% ± 2.1 | 42.3% ± 2.3 | 51.1% ± 4.8 |
| Episodic, encoder frozen | 33.3% ± 3.8 | 50.2% ± 4.5 | 49.4% ± 4.4 | 50.8% ± 4.0 | 40.9% ± 3.2 |
| Episodic, last stage trained | 55.6% ± 3.7 | 56.7% ± 3.6 | 57.4% ± 4.3 | 57.5% ± 4.3 | 35.5% ± 9.3 |
| Shape + texture, encoder frozen | 31.5% ± 3.0 | 49.1% ± 5.3 | 49.7% ± 3.8 | 52.2% ± 3.6 | 38.0% ± 4.5 |
| Shape + texture, last stage trained | 58.0% ± 3.7 | 60.4% ± 5.1 | 59.3% ± 4.5 | 59.3% ± 4.8 | 33.4% ± 11.2 |
| Linear head, encoder frozen (earlier) |  |  |  | 59.5% ± 4.6 |  |
| Linear head, last stage trained (earlier) |  |  |  | 65.1% ± 5.0 | 28.8% ± 9.6 |

## What changed

Episodic training is what moved the numbers. Frozen prototypes were 42.3%. Training only a 256-d projection raised all-support pooled accuracy to 50.8%. Training the last ConvNeXt stage as well raised it to 57.5%.

Shape + texture (global pool of ConvNeXt stage 1, concatenated with the final embedding) added a smaller gain: 50.8% → 52.2% with a frozen encoder, and 57.5% → 59.3% with the last stage trained. 59.3% matches the frozen linear head (59.5%) and is still below the last-stage linear head (65.1%).

With the last stage trained, 5-shot accuracy is about the same as using every training image (60.4% vs 59.3% for shape + texture). The trained embedding is the part that matters. Extra support images past 5 do not add much.

Invasive miss rate fell from 51% (frozen prototypes, all supports) to 33–36% for the trained models. That is in the same range as the earlier linear heads, not a clear improvement on that metric.

## Look-alike species

Counts below are summed over the 5 test splits (all supports). Shape + texture with the last stage trained, versus episodic training on the final embedding only:

| Species | Final embedding, last stage | Shape + texture, last stage |
|---|---|---|
| Potamogeton gramineus | 52/108 (48%) | 59/108 (55%) |
| Potamogeton crispus | 25/71 (35%) | 30/71 (42%) |
| Potamogeton richardsonii | 16/30 (53%) | 16/30 (53%) |
| Potamogeton praelongus | 22/52 (42%) | 21/52 (40%) |
| Myriophyllum spicatum | 17/53 (32%) | 16/53 (30%) |
| Cabomba | 26/47 (55%) | 27/47 (57%) |

The texture feature helps *P. gramineus* and *P. crispus* a little. It does not fix *P. praelongus* or *M. spicatum*. The common mistakes are still inside the pondweeds (*P. gramineus* called *P. illinoensis*) and between feathery plants (*M. spicatum* called *M. sibiricum* or *Cabomba*).

## Runs

- `runs/models/20261113_001948_protonet_episodic_lab_web_field_final_frozen`
- `runs/models/20261113_003317_protonet_episodic_lab_web_field_final_laststage`
- `runs/models/20261113_004931_protonet_episodic_lab_web_field_shape_texture_frozen`
- `runs/models/20261113_010359_protonet_episodic_lab_web_field_shape_texture_laststage`

Script: `Yaman-Aquatic_Plants_Classification/scripts/train_protonet.py`
