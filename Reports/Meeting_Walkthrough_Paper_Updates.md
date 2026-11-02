# Meeting walkthrough — paper updates (about 30–40 minutes)

**Do not open `revision_log_v3.md` first.** That file is for email or as backup if she asks “what exactly changed.” In the meeting, open **Overleaf `main_v3_t.tex` (PDF)** and scroll. Have the Week 8 report ready if she wants the experiment story first.

**Goal of the meeting:** show that the paper now matches the 1,030-image experiments, then ask what to do next.

---

## Opening (1–2 minutes)

Say something close to this:

> I updated the paper to match the cleaned 1,030-image dataset and the runs we discussed. Handheld photos are still out. I removed the old single-split numbers. I can walk through the model results first, then show where the paper changed.

If she wants the experiment story before the PDF, use the Week 8 report for 3 minutes, then open Overleaf.

---

## Part A — Models and results (10–15 minutes) — lead with this

She likes models and numbers. Stay here until she moves you.

### 1. What was trained (one sentence)

> Same ConvNeXt-V2 Tiny backbone. Same 1,030-image pool. Three heads with the encoder frozen: linear, MLP, gated attention. Then one more setup: linear head, last ConvNeXt stage trained, earlier stages frozen. Five seeds each.

### 2. Open the head table in the PDF (Results → Classification head and YNLT)

Point at **pooled accuracy** and **pooled FNR**:

| What to say | Number |
|---|---|
| Three heads are tied on accuracy | ~59% (linear 59.5, MLP 59.4, gated 58.6) |
| Gaps are smaller than seed noise | “not a real head win” |
| YNLT + gated attention | FNR 32.6% → **17.4%**, accuracy almost unchanged |
| Lab vs web | Lab ~80%, web ~53% — lab is easier backgrounds |

**If she asks “which model is best?”**

> For 21-way accuracy: last-stage linear, 65.1%. For missing fewer invasive plants: gated attention + YNLT, FNR 17.4%. They answer different questions.

### 3. Open the last-stage table

| | Frozen | Last stage |
|---|---|---|
| Mixed / pooled accuracy | 59.5% ± 4.6 | **65.1% ± 5.0** |
| Lab | 80.0 ± 8.0 | 85.6 ± 9.7 |
| Web | 53.0 ± 4.7 | 58.1 ± 4.8 |
| Invasive miss rate | 31.3 ± 7.6 | 28.8 ± 9.6 |

Say:

> About 5 points higher on the mixed test. The invasive miss rate barely moved, and that drop is smaller than the seed spread, so I do not call it an improvement. Validation loss stayed flat, so the extra stage is not overfitting.

### 4. Show the error figure (species recall) and Grad-CAM

> Seed 9 linear: 126/189 correct (66.7%). Weakest species are pondweeds — *P. praelongus*, *P. crispus*, *P. richardsonii*. About 44% of mistakes stay in the same genus. Grad-CAM is on the plant, not the coin or background, so the model is looking at the right place and still confusing look-alikes.

### 5. Learning curve (one line)

> Still rising at ~650 training images. More photos of the hard pondweeds would still help.

---

## Part B — What changed in the paper (8–10 minutes)

Do this as **four scrolls**, not a full read of the revision log.

### Scroll 1 — Abstract / title (30 seconds)

> Title and abstract now say 1,030 images, handheld excluded, and the new numbers (59% frozen heads, 65.1% last stage, YNLT FNR 17.4%). The old 84.78% → 93.48% and 47% → 64.71% are gone.

### Scroll 2 — Dataset table (1–2 minutes)

> Lab 203 + cleaned web 785 + outdoor 42 = 1,030. Handheld 307 are out. Outdoor photos are **inside** the pool, not a held-out field test. Observation-grouped splits. Elodea is lab-only.

### Scroll 3 — Contributions (1 minute)

> No more “controlled-to-field domain shift.” Contributions are: the 1,030 collection, YNLT, the five-seed head and last-stage comparison, and the learning curve / Grad-CAM / species errors.

### Scroll 4 — New figures (2–3 minutes)

Show only: learning curve → loss curves → species recall → Grad-CAM. Say they replace the old OOD example figures.

**If she asks why accuracy looks lower than before:**

> The old high numbers were one small laboratory split. This paper reports five seeds on the cleaned mixed test. The numbers are lower but they match the dataset we agreed on.

**If she asks about the revision log:**

> I wrote `revision_log_v3.md` so you can see old text next to new text for every major change. I can email it after the meeting.

---

## Part C — Ask her what to do next (3–5 minutes)

End with **one clear ask** (you already put this in the Week 8 report):

> Should I keep pushing accuracy — for example more photographs of the pondweeds the model confuses — or shift focus to something else, for example writing, more analysis, or a different experiment?

Optional options if she wants choices:

1. Collect / add more *Potamogeton praelongus*, *crispus*, *richardsonii* photos  
2. Leave accuracy as is and polish the paper for submission  
3. Something she prefers (underwater data, handheld as a separate test later, BioCLIP, etc.)

---

## Cheat sheet — numbers to have memorized

| Fact | Number |
|---|---|
| Dataset | 1,030 (203 + 785 + 42); handheld out |
| Frozen heads, pooled acc. | ~59% |
| Last stage, pooled acc. | **65.1% ± 5.0** |
| YNLT gated FNR | **17.4% ± 5.7** (from 32.6%) |
| Seed-9 linear | 126/189 = 66.7% |
| Same-genus mistakes | 28/63 = 44% |

---

## What **not** to do

- Do not walk the revision log line by line in the meeting.  
- Do not defend the old 93% / 64% numbers. Those were a different experiment.  
- Do not call outdoor accuracy (~8 test images) a field result.  
- Do not say “best model” without naming the metric (accuracy vs invasive FNR).

---

## After the meeting

Email her: Overleaf link (or PDF of `main_v3_t`), Week 8 report, and `revision_log_v3.md`. One sentence: “Paper now matches the 1,030-image / last-stage / YNLT results we discussed.”
