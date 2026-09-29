# Paper reading log

Advisor format for each paper: **title**, **simple summary**, **understanding**, **thoughts**.
Target: 2 papers per week.

---

## Week of 1 Sep 2026

### Swin Transformer

**Title:** Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (Liu et al., ICCV 2021).

**Summary:** I read about a general-purpose vision backbone that computes self-attention locally inside non-overlapping windows. By shifting those windows between consecutive layers and gradually merging image patches, the model gets linear computational complexity with respect to image size.

**Understanding:** Standard Vision Transformers struggle with high-resolution images because global self-attention scales quadratically. Swin’s hierarchical design builds feature maps at several scales, more like a CNN, which makes it usable for dense tasks such as segmentation and detection.

**Thoughts:** Swin is relevant as a *backbone baseline*, not as a replacement for YNLT. It still downsamples to a fixed input size, so fine pondweed details can disappear before classification. If we later compare ViT-style models, Swin-Tiny is a fairer high-resolution candidate than vanilla ViT-B/16.

### DINOv2

**Title:** DINOv2: Learning Robust Visual Features without Supervision (Oquab et al., 2023).

**Summary:** Self-supervised training on a carefully curated 142-million-image set can produce general visual features. The authors train a large model and distill it into smaller ones that work with a simple linear classifier.

**Understanding:** The features transfer across distributions without task-specific labels. Because of the patch-level training objective, the model often separates foreground from background without being taught to do so.

**Thoughts:** This is close to our problem: few labels, and field photos that do not look like the lab. A frozen DINOv2 + linear head is a stronger baseline than a random CNN. It still does not decide *when* to look at a high-resolution crop. That decision is the YNLT claim.

---

## Week of 8 Sep 2026 (this week) — BioCLIP pair

Assigned reading: BioCLIP. Read both papers below. Fill any line marked “add after reading” in your own words before Friday.

### Paper 1 — BioCLIP

**Title:** BioCLIP: A Vision Foundation Model for the Tree of Life (Stevens et al., CVPR 2024).

**PDF:** https://openaccess.thecvf.com/content/CVPR2024/papers/Stevens_BioCLIP_A_Vision_Foundation_Model_for_the_Tree_of_Life_CVPR_2024_paper.pdf

**Summary:** The authors release TreeOfLife-10M (about 10 million images, 454 thousand taxa) and train a CLIP-style model that matches organism photos to taxonomic name strings (kingdom → species), plus scientific and common names. On 10 fine-grained biology datasets, including a rare-species set unseen in training, BioCLIP beats CLIP/OpenCLIP by about 16–17% absolute in zero-shot and few-shot classification.

**Understanding:** Ordinary ImageNet models know “plant vs animal,” but not *Onoclea sensibilis* vs *Onoclea hintonii*. BioCLIP uses the tree of life as text: even if a species is rare, the model has usually seen the genus or family. That is why it is strong in the low-data regime. The vision backbone is a ViT-B/16 trained with the CLIP contrastive loss, not with a 21-way aquatic-plant softmax.

**Thoughts (draft — add a personal sentence after you finish the PDF):**

- For *our* paper, BioCLIP is the example of a **stronger baseline**. The advisor asked for more than a linear ConvNeXt head. A frozen BioCLIP encoder + linear probe on 138 lab images is a legitimate comparison.
- BioCLIP does **not** solve lab-to-field shift by looking twice. It solves it by pretraining on many biological photos.
- Our claimed contribution should be written against this: we are not proposing a new foundation model. We propose a **confidence-gated high-resolution second look** when the training set is tiny and the test photos are from another domain.
- Question to check in the paper: how well does BioCLIP cover freshwater macrophytes / Great Lakes taxa? If coverage is thin, that is a reason our task is still hard.

**Add after reading:** one thing that surprised you, and one limitation you would mention in our introduction.

### Paper 2 — BioCLIP 2

**Title:** BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning (Gu et al., NeurIPS 2025).

**PDF:** https://arxiv.org/abs/2505.23883

**Summary:** Scale the same idea to TreeOfLife-200M (214 million images, about 952 thousand taxa). BioCLIP 2 improves species classification by about 18% over BioCLIP. After this scale-up, the embedding space shows extra structure that was not an explicit training target: related species line up with ecological traits (for example beak size in Darwin’s finches), and within-species variation (age, sex) stays in a direction almost orthogonal to the species direction.

**Understanding:** Hierarchical taxonomic text is not just a label. At scale it leaks functional biology into the embedding. That is useful for habitat/trait tasks even though the loss only asked for species matching.

**Thoughts (draft — add a personal sentence after you finish the PDF):**

- If BioCLIP 2 embeddings already separate fine species, a second-look module may help less on in-distribution photos and more on *our* OOD handheld/field set, where backgrounds and imaging conditions change.
- Using BioCLIP 2 as a frozen backbone later would test whether YNLT still helps once the features are biology-specific.
- Keep the distinction clear in the introduction: foundation-model pretraining (BioCLIP) vs adaptive inference (YNLT). They can be combined; they are not the same idea.

**Add after reading:** would you rather use BioCLIP as a baseline, a backbone, or both? Write one sentence.

---

## Next week preview (literature for the introduction)

Read one paper from each cluster, then write 4 lines in this file. Do not skip the “how YNLT differs” line.

1. Mnih et al., Recurrent Models of Visual Attention, 2014.
2. Fu et al., Look Closer to See Better (RACNN), 2017.
3. Guo et al., On Calibration of Modern Neural Networks, 2017 (already cited in the draft).
4. Geifman & El-Yaniv, Selective Classification for Deep Neural Networks, 2017.
5. Shrivastava et al., Training Region-based Object Detectors with Online Hard Example Mining, 2016.
6. Huang et al., Multi-Scale Dense Networks for Resource Efficient Image Classification (MSDNet), 2018.
7. Krizhevsky et al., ImageNet Classification with Deep Convolutional Neural Networks — 10-crop testing.
8. One attention fine-grained paper: RACNN, MA-CNN, WS-DAN, or TransFG.

---

## How to use this file

- Keep Word (`PaperReading.docx`) in sync if the advisor wants that copy. This markdown file is the working version.
- Every entry needs all four headings. Week 1 notes were missing **Thoughts**.
- After each paper, add one sentence: *what this changes in our introduction or experiments*.
