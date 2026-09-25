# Weekly Progress Report — Week 4

**Student:** Yaman  
**Advisor:** Dr. Fengying Dang  
**Dates covered:** 9/15/2026 – 9/21/2026  
**Report due:** Monday 9/21/2026  
**Project:** Confidence-aware aquatic plant classification (YNLT)

---

## Summary of work

- Found that Table 1 YNLT patches the 224×224 input, not the original high-resolution photo.
- Code now supports both (`original` and `input224`); paper text and Table 1 caption updated.
- Measured cost on RTX 3070 Ti: 4.47 G MACs / 5.0 ms per 224 pass (draft said ~322 M).
- One reviewed image: ~99–111 patches, ~187 ms.
- Built run-matrix, results aggregation, and TensorBoard logging.
- Fixed multi-scale patcher crash and invalid ConvNeXt-V1 timm tag.
- Verified 11 bibliography entries; 11 remain `% UNVERIFIED`.
- Training matrix is running; results not ready this week.

---

## Important findings

- Published YNLT numbers come from downsampled patches, not full-resolution photos.
- Efficiency claim in the draft was off by ~14×; now measured.
- Image list still unresolved (paper: 230/33 vs local: ~203/42). `Paper/figs/` is still missing.

---

## Next steps

- Finish the training matrix (ablations, extra backbones, seeds 8–12).
- Fill Table 1 with mean ± std and the full-resolution YNLT row.
- Recover `Paper/figs/` and compile the draft.
- Replace the 11 unverified bibliography entries.

---

## Questions

- Should full-resolution YNLT become the main method, with 224-input as an ablation?
- Which image list is official: 230/33 or the current local folders?
