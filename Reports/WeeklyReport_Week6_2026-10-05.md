Week 6 - 9/28/2026

Summary of work:

1- Weights vs. samples question: written up in `Reports/Parameters_vs_Samples.md`. The backbone is frozen (27.9 M); only the head is trained (16 k linear / 33 k MLP / 262 k gated). The answer is measured with a learning curve rather than a rule.

2- Added the new web dataset (1,140 GBIF/iNaturalist photos, 532 observations) and built leak-free splits (grouped by observation; same lab partition for every data config). The 42 outdoor photos are now a pure held-out field test.

3- Dropped the full-resolution YNLT patch source as discussed (matched 224x224 within one point at 5-10x the cost).

4- Moved the training pipeline onto the GPU (images decoded once, augmentation as GPU kernels): 35-40 min per run -> 5 min. Ran the full experiment grid, 93 runs: data x head x 5 seeds, learning curve, 8 encoders, small rover encoders.

5- Paper: added rover motivation, web-dataset section, capacity-vs-samples subsection, measured cost table; results and discussion rewritten around the new tables (all numbers filled).

Important Findings:

1- Lab + web is the best combination: lab accuracy unchanged (~82-84 %), web accuracy 17 -> 57 %, outdoor 25 -> 35 %. Lab-only models do not transfer to field photos.

2- At matched data the head barely matters (linear / MLP / gated within one std). YNLT trades 1-6 points of 21-way accuracy for much lower invasive FNR (lab 13 -> 2 %, outdoor 63-70 -> 37 %).

3- ConvNeXt-V2 Tiny is the best encoder per unit of compute (5.2 ms/image); Nano keeps >90 % of its accuracy at 55 % of the cost; ViT-B/16 is worse at 3.8x the cost.

4- Learning curve: lab and web accuracy still rise ~10 points per doubling at 40 images/class, so more data will still help. Outdoor accuracy flattens at ~33 % from 22 images/class: that gap is domain, not sample size.

Next Steps:

1- Compile the paper and proofread the new Results / Discussion.

2- Enlarge the outdoor test set (42 images; every number moves in 2.4-point steps).

3- Optional: re-download iNaturalist originals as a fourth source; BioCLIP encoder.

Questions:

1- ~80 % of the web images are CC BY-NC / ND. Fine for reporting aggregate results, not for redistributing the dataset. OK for the paper?

2- Keep the original single-split Table 1 (230/33 images) as a reference, or replace it with the new repeated-split tables?
