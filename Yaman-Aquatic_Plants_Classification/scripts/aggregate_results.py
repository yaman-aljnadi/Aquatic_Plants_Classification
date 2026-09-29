"""Aggregate run results into mean +/- std tables.

    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --runs runs/models --latex

Groups every ``runs/models/*/results.json`` by (backbone, head, variant), where a
variant is the plain model, ``ynlt_original`` or ``ynlt_input224``. Prints a Markdown
table and, with ``--latex``, a tabular body ready to paste into ``Paper/main_v3.tex``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

HEAD_LABELS = {
    "linear": "Linear classifier",
    "gated_attention": "Gated attention",
    "mlp": "MLP head",
}
VARIANT_LABELS = {
    "plain": "",
    "ynlt_original": " + YNLT (full-res patches)",
    "ynlt_input224": " + YNLT (224 patches)",
}
COLUMNS = [
    ("lab", "accuracy", "Lab class acc"),
    ("lab", "binary_accuracy", "Lab bin acc"),
    ("lab", "binary_FNR", "Lab bin FNR"),
    ("ood", "accuracy", "OOD class acc"),
    ("ood", "binary_accuracy", "OOD bin acc"),
    ("ood", "binary_FNR", "OOD bin FNR"),
]


def load_runs(runs_dir):
    records = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "results.json"))):
        with open(path, encoding="utf-8") as handle:
            try:
                records.append(json.load(handle))
            except json.JSONDecodeError:
                print(f"Skipping unreadable {path}")
    return records


def metric_key(split, variant):
    if variant == "plain":
        return split
    return f"{split}_{variant}"


def collect(records):
    """{(backbone, head, variant): {column_name: [values...], 'seeds': [...]}}"""
    groups = {}
    for record in records:
        metrics = record.get("metrics", {})
        variants = ["plain"]
        variants += [
            f"ynlt_{source}"
            for source in ("original", "input224")
            if f"lab_ynlt_{source}" in metrics
        ]
        for variant in variants:
            key = (record.get("backbone_key", record.get("backbone", "?")), record["head"], variant)
            bucket = groups.setdefault(key, {"seeds": []})
            bucket["seeds"].append(record.get("seed"))
            for split, field, name in COLUMNS:
                value = metrics.get(metric_key(split, variant), {}).get(field)
                if value is not None:
                    bucket.setdefault(name, []).append(value)
            for extra in ("trigger_rate", "mean_patches_per_reviewed", "ms_per_image"):
                for split in ("lab", "ood"):
                    value = metrics.get(metric_key(split, variant), {}).get(extra)
                    if value is not None:
                        bucket.setdefault(f"{split} {extra}", []).append(value)
    return groups


def fmt(values, percent=True, decimals=2):
    if not values:
        return "--"
    scale = 100.0 if percent else 1.0
    mean = statistics.mean(values) * scale
    if len(values) == 1:
        return f"{mean:.{decimals}f}"
    std = statistics.stdev(values) * scale
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def row_label(backbone, head, variant):
    return f"{HEAD_LABELS.get(head, head)} ({backbone}){VARIANT_LABELS[variant]}"


def sort_key(key):
    backbone, head, variant = key
    return (backbone, ["linear", "mlp", "gated_attention"].index(head) if head in ("linear", "mlp", "gated_attention") else 9, variant)


def print_markdown(groups):
    names = [name for _, _, name in COLUMNS]
    print("| Model | n | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for key in sorted(groups, key=sort_key):
        bucket = groups[key]
        cells = [fmt(bucket.get(name, [])) for name in names]
        print(f"| {row_label(*key)} | {len(bucket['seeds'])} | " + " | ".join(cells) + " |")


def print_cost(groups):
    rows = [
        (key, bucket)
        for key, bucket in groups.items()
        if any(k.endswith("trigger_rate") for k in bucket)
    ]
    if not rows:
        return
    print()
    print("YNLT cost:")
    print("| Model | Split | Trigger rate | Patches / reviewed | ms / image |")
    print("|---|---|---|---|---|")
    for key, bucket in sorted(rows, key=lambda item: sort_key(item[0])):
        for split in ("lab", "ood"):
            if f"{split} trigger_rate" not in bucket:
                continue
            print(
                f"| {row_label(*key)} | {split.upper()} "
                f"| {fmt(bucket[f'{split} trigger_rate'])}% "
                f"| {fmt(bucket.get(f'{split} mean_patches_per_reviewed', []), percent=False, decimals=1)} "
                f"| {fmt(bucket.get(f'{split} ms_per_image', []), percent=False, decimals=1)} |"
            )


def print_latex(groups):
    print()
    print("% paste into the tabular body of Table 1 in Paper/main_v3.tex")
    for key in sorted(groups, key=sort_key):
        bucket = groups[key]
        cells = [fmt(bucket.get(name, [])).replace("+/-", r"$\pm$") for _, _, name in COLUMNS]
        print(f"{row_label(*key)} & " + " & ".join(cells) + r" \\")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=os.path.join(ROOT, "runs", "models"))
    parser.add_argument("--latex", action="store_true", help="also print a LaTeX tabular body")
    args = parser.parse_args()

    records = load_runs(args.runs)
    if not records:
        print(f"No results.json found under {args.runs}. Train something first.")
        return
    print(f"{len(records)} run(s) from {args.runs}")
    print()
    groups = collect(records)
    print_markdown(groups)
    print_cost(groups)
    if args.latex:
        print_latex(groups)


if __name__ == "__main__":
    main()
