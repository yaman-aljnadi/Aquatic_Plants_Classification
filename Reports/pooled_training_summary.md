# Pooled training (lab + web + outdoor)

Trained 9 October 2026. Handheld photographs stayed in `Larger_Plants_Dataset/aquatic_dataset_report/Hand_Held_Images` and were not used. `Paper/main_v3.tex` was not edited.

## Data

Counts are in `Reports/pooled_split_counts.md`. Seed 8, observation-grouped 60/20/20:

| Source | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Lab | 126 | 38 | 39 | 203 |
| Web (cleaned) | 502 | 144 | 139 | 785 |
| Outdoor | 27 | 7 | 8 | 42 |
| Pooled | 655 | 189 | 186 | 1030 |

`Elodea canadensis` has no remaining web photographs after the handheld images were removed. The outdoor test slice is 8 images, so that column moves in large steps.

## Results

Frozen ConvNeXt-V2 Tiny, five seeds (8–12), checkpoint = best validation accuracy. Mean ± standard deviation. Accuracy and invasive false-negative rate are percentages.

| Head | Lab acc | Lab FNR | Web acc | Web FNR | Outdoor acc | Outdoor FNR | Pooled acc | Pooled FNR |
|---|---|---|---|---|---|---|---|---|
| Linear | 80.0 ± 8.0 | 15.6 ± 9.9 | 53.0 ± 4.7 | 35.4 ± 9.9 | 72.5 ± 10.5 | 30.0 ± 27.4 | 59.5 ± 4.6 | 31.3 ± 7.6 |
| Linear + YNLT | 75.9 ± 7.2 | 6.7 ± 9.9 | 50.9 ± 2.8 | 25.7 ± 9.4 | 75.0 ± 15.3 | 30.0 ± 27.4 | 57.1 ± 2.6 | 22.2 ± 6.9 |
| MLP | 77.4 ± 10.0 | 11.1 ± 13.6 | 53.7 ± 3.5 | 32.6 ± 5.8 | 72.5 ± 10.5 | 30.0 ± 27.4 | 59.4 ± 4.3 | 28.4 ± 5.5 |
| MLP + YNLT | 77.4 ± 11.2 | 6.7 ± 9.9 | 52.5 ± 2.7 | 20.7 ± 6.8 | 72.5 ± 13.7 | 30.0 ± 27.4 | 58.5 ± 3.6 | 18.4 ± 4.4 |
| Gated attention | 80.0 ± 5.8 | 8.9 ± 9.3 | 52.0 ± 5.2 | 37.6 ± 6.5 | 70.0 ± 6.8 | 50.0 ± 0.0 | 58.6 ± 4.6 | 32.6 ± 3.9 |
| Gated attention + YNLT | 79.0 ± 7.6 | 2.2 ± 5.0 | 52.5 ± 3.8 | 19.9 ± 5.7 | 72.5 ± 10.5 | 40.0 ± 22.4 | 58.8 ± 4.0 | 17.4 ± 5.7 |

The three heads are within one standard deviation of each other on pooled accuracy (about 59%). YNLT lowers the invasive false-negative rate on the pooled test (about 31% to 17–22%) and costs a point or two of accuracy.

Learning curve (linear and gated, seeds 8–10, plus the full-data seeds above): pooled accuracy rises from about 33% at 95 training images to about 59% at 650, with no plateau. More photographs would still help. Figure: `Reports/pooled_learning_curve.png`.

## Overfitting

`Reports/pooled_loss_curves.png` is the mean train and validation loss over the five full-data seeds. For every head, validation loss falls for the first 30–40 epochs and then stays flat near 1.5 through epoch 250. Training loss keeps easing down, which is expected under heavy augmentation and label smoothing. The saved checkpoint is the epoch with the best validation accuracy, which is on that flat stretch. The recipe (250 epochs, dropout 0.1, weight decay 2e-5, label smoothing 0.05) was left unchanged.

## Where the runs are

`Yaman-Aquatic_Plants_Classification/runs/models/*_lab_web_field_*` (33 runs, 0 failures, 159 minutes). Each folder has `results.json`, `history.csv`, and `best.pth`. Log: `runs/logs/matrix_pooled.log`.
