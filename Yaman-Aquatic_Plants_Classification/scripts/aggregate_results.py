"""Aggregate run results into mean +/- std tables.

    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --latex
    python scripts/aggregate_results.py --data lab+web --full

Groups every ``runs/models/*/results.json`` written by the current protocol (those with a
``data`` field) by (data, backbone, head, variant). A variant is the plain first pass or
``ynlt_input224``; ``ynlt_original`` rows are hidden unless ``--show-original`` is given,
because on the 203/42 collection they matched the 224 vote within one point at 5-10x the
cost. Learning-curve runs (train_fraction < 1) go to their own table.

Prints Markdown, and with ``--latex`` a tabular body ready to paste into
``Paper/main_v3.tex``.
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

HEAD_LABELS = {"linear": "Linear", "mlp": "MLP", "gated_attention": "Gated attention"}
HEAD_ORDER = ["linear", "mlp", "gated_attention"]
VARIANT_LABELS = {"plain": "", "ynlt_input224": " + YNLT", "ynlt_original": " + YNLT (full-res)"}
VARIANT_ORDER = ["plain", "ynlt_input224", "ynlt_original"]
DATA_ORDER = ["lab", "web", "lab+web"]
DATA_LABELS = {"lab": "Lab", "web": "Web", "lab+web": "Lab + web"}
BACKBONE_LABELS = {
    "convnextv2": "ConvNeXt-V2 Tiny",
    "convnextv2_nano": "ConvNeXt-V2 Nano",
    "convnextv2_pico": "ConvNeXt-V2 Pico",
    "convnextv2_atto": "ConvNeXt-V2 Atto",
    "convnext": "ConvNeXt-V1 Tiny",
    "resnet50": "ResNet-50",
    "efficientnet": "EfficientNetV2-B0",
    "vit": "ViT-B/16",
}
SPLITS = [("lab_test", "Lab"), ("web_test", "Web"), ("field_test", "Field")]
COMPACT_FIELDS = [("accuracy", "acc"), ("binary_FNR", "FNR")]
FULL_FIELDS = [("accuracy", "acc"), ("binary_accuracy", "bin acc"), ("binary_FNR", "FNR")]


def load_runs(runs_dir):
    records, legacy = [], 0
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "results.json"))):
        with open(path, encoding="utf-8") as handle:
            try:
                record = json.load(handle)
            except json.JSONDecodeError:
                print(f"Skipping unreadable {path}")
                continue
        if "data" not in record:
            legacy += 1
            continue
        records.append(record)
    if legacy:
        print(f"({legacy} archived lab-only run(s) from the old protocol ignored)")
    return records


def collect(records, show_original):
    """{(data, backbone, head, variant, fraction): {column: [values], 'seeds': [...]}}"""
    groups = {}
    for record in records:
        metrics = record.get("metrics", {})
        variants = ["plain"]
        for source in ("input224", "original"):
            if source == "original" and not show_original:
                continue
            if any(key.endswith(f"_ynlt_{source}") for key in metrics):
                variants.append(f"ynlt_{source}")
        key_base = (
            record["data"],
            record.get("backbone_key", record.get("backbone", "?")),
            record["head"],
        )
        fraction = float(record.get("train_fraction", 1.0))
        for variant in variants:
            bucket = groups.setdefault((*key_base, variant, fraction), {"seeds": [], "n_train": []})
            bucket["seeds"].append(record.get("seed"))
            bucket["n_train"].append(record.get("splits", {}).get("train"))
            for split, _ in SPLITS:
                name = split if variant == "plain" else f"{split}_{variant}"
                entry = metrics.get(name, {})
                for field in ("accuracy", "binary_accuracy", "binary_FNR", "trigger_rate", "ms_per_image"):
                    value = entry.get(field)
                    if value is not None:
                        bucket.setdefault((split, field), []).append(value)
    return groups


def fmt(values, percent=True, decimals=1):
    if not values:
        return "--"
    scale = 100.0 if percent else 1.0
    mean = statistics.mean(values) * scale
    if len(values) == 1:
        return f"{mean:.{decimals}f}"
    std = statistics.stdev(values) * scale
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def sort_key(key):
    data, backbone, head, variant, fraction = key
    return (
        DATA_ORDER.index(data) if data in DATA_ORDER else 9,
        backbone != "convnextv2",
        backbone,
        HEAD_ORDER.index(head) if head in HEAD_ORDER else 9,
        VARIANT_ORDER.index(variant),
        fraction,
    )


def row_label(key, with_data=True, with_backbone=True, with_fraction=False):
    data, backbone, head, variant, fraction = key
    parts = []
    if with_data:
        parts.append(DATA_LABELS.get(data, data))
    if with_backbone:
        parts.append(BACKBONE_LABELS.get(backbone, backbone))
    label = HEAD_LABELS.get(head, head) + VARIANT_LABELS[variant]
    if with_fraction:
        label += f" ({fraction:.0%} of train)"
    parts.append(label)
    return " / ".join(parts)


def columns(full):
    fields = FULL_FIELDS if full else COMPACT_FIELDS
    return [(split, field, f"{label} {short}") for split, label in SPLITS for field, short in fields]


def print_table(title, keys, groups, cols, latex=False, with_data=True, with_backbone=True, with_fraction=False):
    if not keys:
        return
    print()
    print(f"### {title}")
    print()
    header = ["Model", "n"] + (["Train images"] if with_fraction else []) + [name for _, _, name in cols]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    rows = []
    for key in keys:
        bucket = groups[key]
        lead = [row_label(key, with_data, with_backbone, with_fraction), str(len(bucket["seeds"]))]
        if with_fraction:
            n_train = [n for n in bucket["n_train"] if n is not None]
            lead.append(str(round(statistics.mean(n_train))) if n_train else "--")
        cells = [fmt(bucket.get((split, field), [])) for split, field, _ in cols]
        rows.append((lead, cells))
        print("| " + " | ".join(lead + cells) + " |")
    if latex:
        print()
        print(f"% LaTeX body: {title}  (the 'Model' label is split on ' / ' into separate columns)")
        previous_group = None
        for lead, cells in rows:
            label_cells = lead[0].split(" / ")
            if with_data and previous_group is not None and label_cells[0] != previous_group:
                print(r"\hline")  # rule between training-data blocks of tab:datagrid
            previous_group = label_cells[0]
            latex_cells = [c.replace("+/-", r"$\pm$") for c in cells]
            print(" & ".join(label_cells + lead[2:] + latex_cells) + r" \\")


def print_cost(keys, groups):
    rows = [key for key in keys if any(k[1] == "trigger_rate" for k in groups[key] if isinstance(k, tuple))]
    if not rows:
        return
    print()
    print("### YNLT cost (trigger rate %, ms per image including unreviewed images)")
    print()
    print("| Model | " + " | ".join(f"{label} trigger | {label} ms" for _, label in SPLITS) + " |")
    print("|---|" + "---|---|" * len(SPLITS))
    for key in rows:
        bucket = groups[key]
        cells = []
        for split, _ in SPLITS:
            cells.append(fmt(bucket.get((split, "trigger_rate"), [])))
            cells.append(fmt(bucket.get((split, "ms_per_image"), []), percent=False))
        print(f"| {row_label(key)} | " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=os.path.join(ROOT, "runs", "models"))
    parser.add_argument("--data", choices=DATA_ORDER, default=None, help="only this training data")
    parser.add_argument("--full", action="store_true", help="also show binary accuracy")
    parser.add_argument("--show-original", action="store_true", help="include full-resolution YNLT rows")
    parser.add_argument("--latex", action="store_true", help="also print LaTeX tabular bodies")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="only these seeds (e.g. --seeds 8 9 10 to match the three-seed encoder table)")
    args = parser.parse_args()

    records = load_runs(args.runs)
    if args.seeds:
        records = [r for r in records if r.get("seed") in args.seeds]
    if not records:
        print(f"No results.json from the current protocol under {args.runs}. Train something first.")
        return
    print(f"{len(records)} run(s) from {args.runs}" + (f" (seeds {args.seeds})" if args.seeds else ""))
    groups = collect(records, args.show_original)
    cols = columns(args.full)
    keys = sorted(groups, key=sort_key)
    if args.data:
        keys = [k for k in keys if k[0] == args.data]

    main_keys = [k for k in keys if k[1] == "convnextv2" and k[4] >= 1.0]
    backbone_keys = [k for k in keys if k[1] != "convnextv2" and k[4] >= 1.0]
    curve_keys = [
        k for k in keys
        if k[0] == "lab+web" and k[1] == "convnextv2" and k[3] == "plain" and k[2] in ("linear", "gated_attention")
    ]
    curve_keys.sort(key=lambda k: (HEAD_ORDER.index(k[2]), k[4]))

    print_table("Training data x head, frozen ConvNeXt-V2 Tiny", main_keys, groups, cols, args.latex, with_backbone=False)
    print_table("Other encoders (lab+web)", backbone_keys, groups, cols, args.latex, with_data=False)
    if any(k[4] < 1.0 for k in curve_keys):
        print_table("Learning curve (lab+web, first pass only)", curve_keys, groups, cols, args.latex,
                    with_data=False, with_backbone=False, with_fraction=True)
    print_cost([k for k in main_keys + backbone_keys if k[3] != "plain"], groups)


if __name__ == "__main__":
    main()
