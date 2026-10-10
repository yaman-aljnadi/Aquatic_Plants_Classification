"""Plot train and validation loss for the pooled (lab+web+field) runs.

    python scripts/plot_loss_curves.py

Reads ``history.csv`` next to each ``results.json`` with ``data == lab+web+field`` and
``train_fraction == 1``. Writes ``Reports/pooled_loss_curves.png`` and prints whether
validation loss climbs while training loss falls after the selected checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

HEAD_LABELS = {"linear": "Linear", "mlp": "MLP", "gated_attention": "Gated attention"}
HEAD_ORDER = ["linear", "mlp", "gated_attention"]


def load_histories(runs_dir, data="lab+web+field", fraction=1.0):
    """head -> list of {epoch, train_loss, val_loss, train_acc, val_acc, best_epoch, seed}."""
    grouped = defaultdict(list)
    for path in glob.glob(os.path.join(runs_dir, "**", "results.json"), recursive=True):
        with open(path, encoding="utf-8") as handle:
            try:
                record = json.load(handle)
            except json.JSONDecodeError:
                continue
        if record.get("data") != data or float(record.get("train_fraction", 1.0)) != fraction:
            continue
        history_path = os.path.join(os.path.dirname(path), "history.csv")
        if not os.path.isfile(history_path):
            continue
        with open(history_path, encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        grouped[record["head"]].append({
            "seed": record.get("seed"),
            "best_epoch": int(record.get("best_epoch") or 0),
            "rows": rows,
        })
    return grouped


def series(rows, key):
    return [float(row[key]) for row in rows]


def diagnose(run):
    """True when validation loss rises after the best checkpoint while training loss falls."""
    rows = run["rows"]
    best = run["best_epoch"]
    if best < 1 or best > len(rows):
        return False, "no checkpoint epoch"
    train_best = float(rows[best - 1]["train_loss"])
    val_best = float(rows[best - 1]["val_loss"])
    train_last = float(rows[-1]["train_loss"])
    val_last = float(rows[-1]["val_loss"])
    climbed = val_last > val_best + 0.05 and train_last < train_best - 0.02
    detail = (
        f"seed {run['seed']}: best epoch {best}, "
        f"train {train_best:.3f}->{train_last:.3f}, val {val_best:.3f}->{val_last:.3f}"
    )
    return climbed, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=os.path.join(ROOT, "runs", "models"))
    parser.add_argument("--out", default=os.path.join(ROOT, "..", "Reports", "pooled_loss_curves.png"))
    args = parser.parse_args()

    grouped = load_histories(args.runs)
    if not grouped:
        print("No pooled histories found. Train with: python scripts/run_matrix.py --suite pooled")
        return 1

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib")
        return 1

    heads = [head for head in HEAD_ORDER if head in grouped]
    fig, axes = plt.subplots(1, len(heads), figsize=(4.4 * len(heads), 3.6), sharey=True)
    if len(heads) == 1:
        axes = [axes]

    diverged = []
    print("Train / validation loss at the selected checkpoint and at the last epoch:")
    for ax, head in zip(axes, heads):
        runs = grouped[head]
        length = min(len(run["rows"]) for run in runs)
        epochs = list(range(1, length + 1))
        train = [series(run["rows"], "train_loss")[:length] for run in runs]
        val = [series(run["rows"], "val_loss")[:length] for run in runs]
        train_mean = [statistics.mean(values) for values in zip(*train)]
        val_mean = [statistics.mean(values) for values in zip(*val)]
        ax.plot(epochs, train_mean, label="Train")
        ax.plot(epochs, val_mean, label="Validation")
        ax.set_title(f"{HEAD_LABELS[head]} (n={len(runs)})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        for run in runs:
            climbed, detail = diagnose(run)
            print(f"  {HEAD_LABELS[head]}  {detail}")
            if climbed:
                diverged.append(f"{HEAD_LABELS[head]} {detail}")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")
    if diverged:
        print("VALIDATION LOSS DIVERGED:")
        for line in diverged:
            print(f"  {line}")
        return 2
    print("Validation loss does not climb away from training loss after the selected checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
