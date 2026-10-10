"""Plot test accuracy against the number of training images (the learning curve).

    python scripts/plot_learning_curve.py
    python scripts/plot_learning_curve.py --out ../Paper/figs/learning_curve.pdf --split web_test

Reads every ``runs/models/*/results.json`` trained on ``lab+web`` with ConvNeXt-V2 Tiny
(any ``train_fraction``), groups by head and fraction, and draws mean +/- std of the
first-pass accuracy per test split. This is the empirical answer to "how many samples
does the head need": where the curve flattens, more images stop paying off.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

HEAD_LABELS = {"linear": "Linear head", "mlp": "MLP head", "gated_attention": "Gated attention head"}
SPLIT_LABELS = {"lab_test": "Lab test", "web_test": "Web test", "field_test": "Field test (outdoor)"}


def load(runs_dir, data="lab+web", backbone="convnextv2"):
    """(head, split) -> fraction -> {"acc": [...], "n_train": [...]}.

    Runs are grouped by ``train_fraction`` (the number of training images differs by a
    few per seed because whole observations are kept); the x position is the mean count.
    """
    points = defaultdict(lambda: defaultdict(lambda: {"acc": [], "n_train": []}))
    for path in glob.glob(os.path.join(runs_dir, "**", "results.json"), recursive=True):
        with open(path, encoding="utf-8") as handle:
            try:
                record = json.load(handle)
            except json.JSONDecodeError:
                continue
        if record.get("data") != data or record.get("backbone_key") != backbone:
            continue
        n_train = record.get("splits", {}).get("train")
        if not n_train:
            continue
        fraction = float(record.get("train_fraction", 1.0))
        for split in SPLIT_LABELS:
            acc = record.get("metrics", {}).get(split, {}).get("accuracy")
            if acc is not None:
                bucket = points[(record["head"], split)][fraction]
                bucket["acc"].append(acc)
                bucket["n_train"].append(n_train)
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=os.path.join(ROOT, "runs", "models"))
    parser.add_argument("--data", default="lab+web")
    parser.add_argument("--out", default=os.path.join(ROOT, "..", "Paper", "figs", "learning_curve.pdf"))
    parser.add_argument("--splits", nargs="+", default=["lab_test", "web_test", "field_test"])
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib")
        return 1

    points = load(args.runs, data=args.data)
    if not points:
        print("No lab+web ConvNeXt-V2 runs found. Run: python scripts/run_matrix.py --suite curve")
        return 1

    fig, axes = plt.subplots(1, len(args.splits), figsize=(4.2 * len(args.splits), 3.4), sharey=True)
    if len(args.splits) == 1:
        axes = [axes]
    markers = {"linear": "o", "mlp": "s", "gated_attention": "^"}
    print("| Head | Split | Fraction | Train images | n runs | Accuracy (%) |")
    print("|---|---|---|---|---|---|")
    for ax, split in zip(axes, args.splits):
        for head in ("linear", "gated_attention"):
            series = points.get((head, split))
            if not series or len(series) < 2:
                continue  # only heads that were run at several fractions make a curve
            fractions = sorted(series)
            xs = [statistics.mean(series[f]["n_train"]) for f in fractions]
            means = [100 * statistics.mean(series[f]["acc"]) for f in fractions]
            stds = [100 * (statistics.stdev(series[f]["acc"]) if len(series[f]["acc"]) > 1 else 0.0) for f in fractions]
            ax.errorbar(xs, means, yerr=stds, marker=markers[head], capsize=3, label=HEAD_LABELS[head])
            for f, x, m, s in zip(fractions, xs, means, stds):
                print(f"| {HEAD_LABELS[head]} | {SPLIT_LABELS[split]} | {f:.0%} | {x:.0f} | {len(series[f]['acc'])} | {m:.1f} +/- {s:.1f} |")
        ax.set_title(SPLIT_LABELS[split])
        ax.set_xlabel("Training images")
        ax.set_xscale("log")
        ax.set_xticks([125, 250, 500, 1000])
        ax.set_xticks([], minor=True)
        ax.set_xticklabels(["125", "250", "500", "1000"])
        ax.set_xlim(100, 1100)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out)
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=150)
    print(f"\nSaved {args.out} and {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
