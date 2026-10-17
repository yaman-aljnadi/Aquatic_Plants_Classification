"""Per-class recall and confusion pairs for one checkpoint.

    python scripts/confusion_report.py runs/models/.../best.pth --seed 9
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

import config as cfg
from scripts.build_splits import ensure_split
from src.gpu_data import get_gpu_splits
from src.metrics import binary_fnr, collect_predictions
from src.model import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--data", default="lab+web+field")
    parser.add_argument("--out", default=os.path.join(cfg.REPO_ROOT, "Reports", "confusion_seed9.md"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint).to(device).eval()
    split_csv = ensure_split(args.data, args.seed)
    loaders = get_gpu_splits(cfg, split_csv, device, seed=args.seed, test_csv=split_csv)
    y_true, y_pred = collect_predictions(model, loaders["pooled_test"], device)
    names = cfg.CANONICAL_CLASSNAMES_LIST
    binary = binary_fnr(y_true, y_pred, cfg.INVASIVE_INDICES)
    n = len(y_true)
    correct = int((y_true == y_pred).sum())

    rows = []
    pairs = Counter()
    for label in range(len(names)):
        mask = y_true == label
        support = int(mask.sum())
        hits = int((y_pred[mask] == label).sum()) if support else 0
        mistakes = Counter(int(p) for p in y_pred[mask] if int(p) != label)
        top = mistakes.most_common(1)
        top_txt = f"{names[top[0][0]]} ({top[0][1]})" if top else ""
        rows.append((names[label], support, hits, (hits / support) if support else None, top_txt))
        for pred, count in mistakes.items():
            pairs[(names[label], names[pred])] += count

    lines = [
        "# Confusion on the pooled test, linear head, seed 9",
        "",
        "Checkpoint: the frozen ConvNeXt-V2 Tiny linear model trained on lab + web + outdoor.",
        "No new training. The test split is the held-out 20% of that same pool.",
        "",
        f"Images: {n}. Correct: {correct} ({100 * correct / n:.1f}%).",
        f"Invasive miss rate: {100 * binary['FNR']:.1f}% "
        f"({binary['FN']} missed of {binary['TP'] + binary['FN']} invasive test images).",
        "",
        "| Species | Test images | Correct | Recall | Most common mistake |",
        "|---|---:|---:|---:|---|",
    ]
    for name, support, hits, recall, top_txt in rows:
        recall_txt = "n/a" if recall is None else f"{100 * recall:.0f}%"
        lines.append(f"| {name} | {support} | {hits} | {recall_txt} | {top_txt} |")

    lines.extend(["", "## Most common confusions", "", "| True species | Predicted | Count |", "|---|---|---:|"])
    for (true_name, pred_name), count in pairs.most_common(15):
        lines.append(f"| {true_name} | {pred_name} | {count} |")

    same_genus = sum(count for (a, b), count in pairs.items() if a.split()[0] == b.split()[0])
    total_errors = n - correct
    lines.extend([
        "",
        f"Of {total_errors} mistakes, {same_genus} stay inside the same genus name "
        f"({100 * same_genus / max(total_errors, 1):.0f}%).",
        "The rest cross genera, mostly between plants with a similar growth form "
        "(feathery submerged leaves, or broad floating leaves).",
        "",
    ])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Wrote {args.out}")
    print(f"accuracy {correct / n:.3f}  invasive FNR {binary['FNR']:.3f}  same-genus errors {same_genus}/{total_errors}")


if __name__ == "__main__":
    main()
