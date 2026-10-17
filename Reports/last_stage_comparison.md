# Last ConvNeXt stage versus a frozen encoder

Same data, same augmentations, same linear head. The only change is that the last ConvNeXt-V2 stage is trained (learning rate 1e-4) while the earlier stages stay frozen. Five seeds, 8–12. Checkpoint is still the best validation accuracy.

Validation loss does not climb away from training loss after that checkpoint, so these runs are kept.

| Setup | Pooled accuracy | Invasive miss rate | Lab accuracy | Web accuracy |
|---|---|---|---|---|
| Frozen encoder | 59.5% ± 4.6 | 31.3% ± 7.6 | 80.0% ± 8.0 | 53.0% ± 4.7 |
| Last stage trained | 65.1% ± 5.0 | 28.8% ± 9.6 | 85.6% ± 9.7 | 58.1% ± 4.8 |

Pooled accuracy is about 5 points higher. The invasive miss rate is a little lower, but the seed-to-seed spread is larger than that gap, so the miss rate is not a clear win. The species the frozen model already confuses, especially pondweeds with other pondweeds, are listed in `Reports/confusion_seed9.md`.

Runs: `Yaman-Aquatic_Plants_Classification/runs/models/*_lab_web_field_linear_convnextv2_laststage_seed*`.
