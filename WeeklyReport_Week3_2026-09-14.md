# Weekly Progress Report — Week 3

**Student:** Yaman  
**Advisor:** Dr. Fengying Dang  
**Dates covered:** 9/8/2026 – 9/14/2026  
**Report due:** Monday 9/14/2026 (midnight)  
**Submitted:** 9/15/2026 (with apology for the delay)  
**Project:** Confidence-aware aquatic plant classification (YNLT)

---

## Summary of work

1. **Paper revision (`Paper/main_v3.tex`) in response to your comments**
   - Reframed the contribution as small-data aquatic computer vision + lab-to-field domain shift + confidence-aware adaptive inference (YNLT).
   - Rewrote and shortened the introduction, and added discussion of related advanced methods (recurrent attention, coarse-to-fine recognition, uncertainty-aware / selective prediction, multi-crop / multi-resolution inference, hard-example mining, attention-based fine-grained classification).
   - Wrote the missing YNLT mathematics from the actual implementation: confidence, margin, entropy, thresholds, trigger rule, invasive skip, multi-scale patching, and vote weighting.
   - Corrected the gated-attention head description to match the code.
   - Clarified computational efficiency and softened the underwater-robot deployment claim until latency is measured.
   - Fixed the ablation discussion text so it matches Table 1.
   - Fixed the abstract laboratory image count typo (225 → 230).
   - Recreated the missing `ref.bib` file so the paper can compile citations.

2. **Code review of the existing training codebase**
   - Inspected ConvNeXt-V2 + gated head + YNLT implementation.
   - Confirmed Table 1 already isolates Linear / Linear+YNLT / Gated / Gated+YNLT on a shared frozen ConvNeXt-V2 Tiny backbone.
   - Confirmed the notebook split: 230 lab images (138/46/46) and 33 OOD images (16/17).

3. **Clean experiment repository for new training runs**
   - Created `Yaman-Aquatic_Plants_Classification/` (GitHub-ready) with a minimal training/evaluation pipeline.
   - Prepared scripts for: ablation rows, stronger baselines (ResNet / EfficientNet / ViT / ConvNeXt), and repeated stratified 60/20/20 splits (preferred over 5-fold because some species have only 2 images).
   - Configured defaults for my workstation (RTX 3070 Ti, i9-12900K).

4. **Paper reading file**
   - Added dates to the PaperReading file.
   - Week 2 papers remain: Swin Transformer and DINOv2 (dated 9/1/2026).
   - BioCLIP was started as draft notes; completion was delayed because this week prioritized paper revision and experiment infrastructure. I will finish BioCLIP next.

---

## Important findings

1. The draft YNLT section did not fully match the code; the paper text is now aligned with the implementation.
2. Minimum ablation isolation is already in Table 1; remaining experiment gaps are stronger backbones and repeated-split mean ± std.
3. Local image folders (~203 lab / ~42 outdoor) do not yet match the paper’s 230/33 list; this must be resolved before new result tables.

---

## Next steps

1. Confirm the official image list (230/33 vs current local folders).
2. Run repeated-split and stronger-baseline experiments in the new repo.
3. Update paper tables with mean ± std and backbone comparisons.
4. Finish BioCLIP reading notes (dated) and continue 2 papers/week.
5. Verify remaining bibliography entries against original sources.

---

## Questions

1. Which image list should be treated as official for the paper: the original 230 lab + 33 OOD set, or the folders currently on my machine?
2. Should OOD remain a fixed holdout while lab data uses repeated 60/20/20 seeds?
3. Is BioCLIP acceptable as a stronger biology-specific baseline, or should baselines stay limited to ImageNet models for now?

---

## Note on delay

I apologize for missing the Monday midnight deadline. There was no emergency; I underestimated time while revising the paper and setting up the clean training repository, and I failed to communicate in advance. I will send future weekly reports on time.
