# Weekly Progress Report — Week 4

**Student:** Yaman  
**Advisor:** Dr. Fengying Dang  
**Dates covered:** 9/15/2026 – 9/21/2026  
**Report due:** Monday 9/21/2026 (midnight)  
**Project:** Confidence-aware aquatic plant classification (YNLT)

---

## Summary of work

1. **Found and fixed a mismatch between the paper and the code that produced Table 1**

   The paper states that when YNLT flags an uncertain prediction, patches are cut from the
   original high-resolution photograph. Tracing the evaluation notebook
   (`n26_validate_ood_ynlt.ipynb`) through `PredictPlants.predict` into
   `critical_review`, the patches are in fact cut from the 224x224 tensor the dataloader
   produced, not from the source file. At scales 0.2–0.5 those crops are 44 to 112 pixels
   wide and are then upsampled back to 224. The second look therefore re-weighted evidence
   the network had already seen rather than recovering lost detail.

   The training repository now implements both behaviours and reports them as separate
   rows (`--ynlt-source original | input224 | both`), so the resolution question becomes an
   ablation instead of an ambiguity. The paper text, the Table 1 caption, and the
   limitations section were corrected to say which one produced the published numbers.

2. **Measured computational cost instead of estimating it**

   The draft claimed roughly 322 M multiply-adds for one 224x224 forward pass. Measured
   with `fvcore`, the frozen ConvNeXt-V2 Tiny plus gated head is **4.47 G** multiply-adds,
   about fourteen times larger. Parameter counts in the draft were correct
   (28.13 M total, 261,933 trainable). Full measured table, on the RTX 3070 Ti:

   | Quantity | Value |
   |---|---|
   | Multiply-adds, one 224x224 pass | 4.47 G |
   | Latency, one 224x224 pass | 5.0 ms |
   | Latency per patch (batches of 16) | 1.8 ms |
   | Patches per reviewed image, 224 source | 111 |
   | Patches per reviewed image, 3000x4000 source | 99 |
   | Latency of one reviewed image, full resolution | 187 ms |

   A reviewed image costs about 37 times an unflagged one, which is the number the
   deployment discussion actually needs. The paper now carries this as a table and the
   robot-deployment claim stays in future work.

3. **Built the experiment infrastructure for the remaining requests**

   - `scripts/run_matrix.py` runs the full grid (ablation rows, the four extra backbones,
     seeds 8–12) one job at a time and skips anything already finished, so an interrupted
     session resumes safely.
   - Every run writes `results.json` with accuracies, binary FNR, confusion counts, YNLT
     trigger rate, patches per reviewed image, and milliseconds per image.
   - `scripts/aggregate_results.py` turns those files into mean ± standard deviation
     tables and emits a LaTeX tabular body that can be pasted straight into Table 1.
   - `scripts/benchmark_efficiency.py` produced the numbers in item 2 and needs no images.

4. **Fixed two defects found while testing**

   - The multi-scale patcher crashed on the second scale because it tried to stack patches
     of different sizes into one array. YNLT would have failed at runtime on any image.
   - The `convnext` baseline preset used the timm tag `convnext_tiny.in22k_ft_in1k`, which
     does not exist; the correct ConvNeXt-V1 counterpart is `convnext_tiny.fb_in22k_ft_in1k`.

   The whole pipeline was then run end to end on synthetic images (train, evaluate, both
   YNLT sources, results aggregation) to confirm it works before the real data arrives.

5. **Verified the reconstructed bibliography**

   Eleven ecology and aquatic-invasive-species entries were confirmed against the
   published record and now carry full author lists, volumes, pages and DOIs, including
   Cuthbert et al. 2021 (whose title in our file was wrong: it is *aquatic* invasive alien
   species, not biological invasions) and the Jochems et al. SSRN preprint, which has
   since appeared in *Journal of Environmental Management*. Eleven entries could not be
   matched to any publication and are now explicitly marked `% UNVERIFIED` in `ref.bib`
   with a note on what is uncertain.

---

## Important findings

1. The published YNLT results were obtained by re-examining the downsampled input, not the
   original photograph. This weakens the stated mechanism, and repeating the ablation with
   full-resolution patches is now the most informative experiment we have left.
2. The efficiency claim in the draft was off by roughly a factor of fourteen. It is now
   measured rather than estimated.
3. The image folders are still not resolved (paper: 230 lab / 33 field; machine: ~203 / ~42),
   and `Paper/figs/` is missing from the repository, so the draft cannot currently be
   compiled to PDF.

---

## Next steps

1. Obtain the official image set, then run `scripts/run_matrix.py --suite all`: the four
   ablation rows, four extra backbones, and seeds 8–12.
2. Fill Table 1 with mean ± standard deviation and add the full-resolution YNLT row.
3. Recover `Paper/figs/` and compile `main_v3.tex` with XeLaTeX.
4. Replace the eleven `% UNVERIFIED` bibliography entries with Aaryan's originals.
5. Finish the BioCLIP reading notes and continue at two papers per week.

---

## Questions

1. The reported YNLT gains come from patching the downsampled input. Should the
   full-resolution version become the method in the paper, with the downsampled one
   reported as an ablation, or should both be presented side by side as the contribution?
2. Which image list is official for the paper: the original 230 lab + 33 field set, or the
   folders currently on my machine? This still blocks every new results table.
3. Can Aaryan send the original `ref.bib`? Eleven citations cannot be verified from the
   cite keys alone, including the ones supporting the expertise and remote-sensing claims
   in the introduction.
