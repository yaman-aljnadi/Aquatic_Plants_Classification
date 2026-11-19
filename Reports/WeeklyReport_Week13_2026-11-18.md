Week 13 - 11/16/2026 - 11/18/2026

Summary of work:

1- Met with Dr. Dang on the Prototypical Network results. From her notes: form an own direction, test pieces one at a time, drop what does not work, look at BioCLIP, and keep text work different from Tristan.

2- Taxonomy check against TreeOfLife / BioCLIP name tables (`Reports/bioclip_taxonomy.md`): 20 of 21 names present at species; Cabomba present as genus only. Coverage is not the blocker.

3- Frozen BioCLIP ViT-B/16 + linear head on `lab+web+field`, seeds 8–12 (`scripts/train_bioclip.py`). Name-only zero-shot in the same run. Lab and web reported separately.

4- Last-block BioCLIP: same splits, train the last ViT residual block + ln_post + visual proj + linear head (`--unfreeze-last-block`). No zero-shot rerun.

Important Findings:

1- Frozen BioCLIP vs frozen ConvNeXt-V2 Tiny linear head:

| Split | BioCLIP frozen | ConvNeXt frozen |
|---|---|---|
| Lab | 77.9% ± 3.9 | 80.0% ± 8.0 |
| Web | 60.7% ± 4.7 | 53.0% ± 4.7 |
| Pooled | 64.9% ± 4.0 | 59.5% ± 4.6 |

Lab was below the keep rule (must beat 80%), so the frozen encoder alone was marked drop. Web and pooled were higher than ConvNeXt. Name-only zero-shot was weak (lab 52.3%, pooled 45.9%).

2- Last-block BioCLIP vs the same baselines (and vs last-stage ConvNeXt pooled 65.1%):

| Setup | Lab | Web | Pooled | Invasive FNR (pooled) |
|---|---|---|---|---|
| ConvNeXt frozen linear | 80.0% ± 8.0 | 53.0% ± 4.7 | 59.5% ± 4.6 | 31.3% ± 7.6 |
| BioCLIP frozen linear | 77.9% ± 3.9 | 60.7% ± 4.7 | 64.9% ± 4.0 | 28.7% ± 11.5 |
| BioCLIP last block + linear | **89.2% ± 4.9** | **67.1% ± 5.4** | **72.5% ± 4.5** | **23.0% ± 5.0** |
| ConvNeXt last-stage linear | 85.6% ± 9.7 | 58.1% ± 4.8 | 65.1% ± 5.0 | 28.8% ± 9.6 |

Decision: **keep BioCLIP**. Lab 89.2% > 80.0% and pooled 72.5% > 65.1%.

3- Rover cost: ConvNeXt-V2 Tiny is about 27.9M params and ~5.2 ms/image. BioCLIP ViT-B/16 is 86.2M visual params and ~10.4 ms/image here (~2× slower, ~3× more visual weights).

Runs: `runs/models/20261118_154447_bioclip_linear_lab_web_field/`, `runs/models/20261118_175109_bioclip_lastblock_lab_web_field/`.

Next Steps:

1- Keep last-block BioCLIP as the new encoder baseline.

2- Look-alike pondweeds and feathery plants are still the open error pattern from ProtoNet; check whether last-block BioCLIP fixes them or only raises overall accuracy.

3- Finish BioCLIP / BioCLIP 2 notes in `Reports/PaperReading.md` and add Snell et al. as the third paper.
