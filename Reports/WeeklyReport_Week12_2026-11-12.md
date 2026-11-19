Week 12 - 11/09/2026 - 11/16/2026

Summary of work:

Started the Prototypical Network direction. First built a frozen ProtoNet baseline on ConvNeXt-V2 Tiny and the lab+web+field splits (seeds 8–12): class prototype = mean feature of K support images, test images assigned to the nearest prototype. Then trained an episodic ProtoNet (5 support and 3 query images per class each episode, usual augmentation), with the encoder frozen and with the last ConvNeXt stage trained. Then added a shape-plus-texture embedding (early-stage pool concatenated with the final embedding) and trained it the same way. Code: `scripts/protonet_baseline.py`, `scripts/train_protonet.py`.

Important Findings:

1- Frozen prototypes (no training) did not beat the linear head. Pooled accuracy: 1-shot 16.8%, 5-shot 33.4%, 10-shot 34.7%, all supports 42.3%. The linear head is 59.5% with all training images. Support-time augmentation made the frozen prototypes worse (38.6%).

2- Episodic training is what raised accuracy. All-support pooled: 50.8% with a frozen encoder, 57.5% with the last stage trained. Shape + texture added a smaller gain, to 52.2% (encoder frozen) and 59.3% (last stage trained). 59.3% matches the frozen linear head (59.5%) and is still below the last-stage linear head (65.1%).

3- With the last stage trained, 5-shot is about as good as using every training image (60.4% vs 59.3% for shape + texture). Invasive miss rate fell from 51% (frozen prototypes) to about 33%, which is in the same range as the earlier linear heads.

4- Shape + texture helped Potamogeton gramineus (48% to 55% recall) and Potamogeton crispus (35% to 42%), summed over the five seeds. It did not help Potamogeton praelongus or Myriophyllum spicatum. Remaining mistakes are still inside the pondweeds and between feathery plants.

Next Steps:

1- The prototype loss is now close to the frozen linear head, but still about 6 points below the last-stage linear head (65.1%).

2- Look-alike pondweeds, especially Potamogeton praelongus and Myriophyllum spicatum, are still the main errors.
