"""How many weights are trained, and how many images stand behind them.

    python scripts/parameter_budget.py
    python scripts/parameter_budget.py --markdown Reports/parameter_budget_tables.md

Answers the advisor's question "what are the weights/parameters of the model and how many
samples do you need for them" with numbers rather than rules of thumb:

1. parameter counts (frozen backbone vs. trainable head) for every backbone preset and head;
2. images available per data configuration (lab / web / lab+web), per class and per
   trainable parameter;
3. the statistical resolution of each test split: the 95% confidence half-width of an
   accuracy estimate and the test-set size needed to tell two models apart.

Random weights are used, so nothing is downloaded. Image folders are optional: without
them the script prints the capacity tables and the confidence-interval table only.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as cfg  # noqa: E402
from src.model import build_model  # noqa: E402

HEADS = ("linear", "mlp", "gated_attention")
HEAD_LABELS = {"linear": "Linear", "mlp": "MLP (42 hidden)", "gated_attention": "Gated attention (168)"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def head_param_formula(head, feat, k=cfg.CANONICAL_NUM_CLASSES, d=cfg.gated_attention_dim, m=cfg.mlp_hidden_dim):
    """Closed-form trainable parameter count, used to cross-check the module count."""
    if head == "linear":
        return feat * k + k
    if head == "mlp":
        return feat * m + m + m * k + k
    if head == "gated_attention":
        return 2 * (feat * d + d) + d * k + k
    raise ValueError(head)


def count_images(root):
    """{canonical class name: n images} for an ImageFolder-style root, or None."""
    if not root or not os.path.isdir(root):
        return None
    counts = Counter()
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        name = entry.name.split("(")[0].strip().replace("_", " ")
        n = sum(1 for f in os.scandir(entry.path) if f.is_file() and os.path.splitext(f.name)[1].lower() in IMAGE_EXT)
        if n:
            counts[name] += n
    return counts


def ci_half_width(p, n, z=1.96):
    return z * math.sqrt(p * (1 - p) / n)


def n_for_half_width(p, half, z=1.96):
    return math.ceil(z * z * p * (1 - p) / (half * half))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+", default=list(cfg.BACKBONE_PRESETS))
    parser.add_argument("--markdown", default=None, help="also write the report to this file")
    args = parser.parse_args()

    lines = []
    out = lines.append

    # ------------------------------------------------------------------ capacity
    out("## 1. Weights: frozen backbone vs. trained head")
    out("")
    out("Only the classification head is trained. Backbone weights are the published timm")
    out("checkpoints and never change, so they are not fitted to our images at all.")
    out("")
    out("| Backbone preset | timm name | Feature dim | Frozen params (M) | " + " | ".join(f"{HEAD_LABELS[h]} trainable" for h in HEADS) + " |")
    out("|---|---|---|---|" + "---|" * len(HEADS))
    feature_dims = {}
    trainable = {}
    for key in args.backbones:
        name = cfg.BACKBONE_PRESETS.get(key, key)
        row = []
        frozen = None
        for head in HEADS:
            model = build_model(
                backbone_name=name,
                classifier_head_type=head,
                num_classes=cfg.CANONICAL_NUM_CLASSES,
                attention_dim=cfg.gated_attention_dim,
                mlp_hidden_dim=cfg.mlp_hidden_dim,
                dropout_p=cfg.classifier_head_dropout,
                pretrained=False,
            )
            n_head = sum(p.numel() for p in model.classifier.parameters())
            n_total = sum(p.numel() for p in model.parameters())
            feat = model.backbone.num_features
            assert n_head == head_param_formula(head, feat), (key, head, n_head)
            frozen = n_total - n_head
            feature_dims[key] = feat
            trainable[(key, head)] = n_head
            row.append(f"{n_head:,}")
            del model
        out(f"| {key} | `{name}` | {feature_dims[key]} | {frozen / 1e6:.2f} | " + " | ".join(row) + " |")
    out("")
    out("Closed forms (D = feature dim, K = 21 classes, d = 168, m = 42):")
    out("linear = D*K + K; MLP = D*m + m + m*K + K; gated = 2*(D*d + d) + d*K + K.")
    out("")

    # ------------------------------------------------------------------ data
    lab = count_images(cfg.LAB_DATA_DIR)
    web = count_images(cfg.WEB_DATA_DIR)
    field = count_images(cfg.OOD_DATA_DIR)
    configs = {}
    if lab:
        configs["lab"] = lab
    if web:
        configs["web"] = web
    if lab and web:
        configs["lab+web"] = lab + web

    out("## 2. Samples behind those weights")
    out("")
    if not configs:
        out("Image folders not found; set AQUATIC_LAB_DATA / AQUATIC_WEB_DATA to fill this section.")
        out("")
    else:
        out("Training uses 60% of each configuration (the rest is validation and test).")
        out("")
        out("| Data | Images | Classes | Min / median / max per class | Train images (60%) | Train images per class | "
            + " | ".join(f"train img per {HEAD_LABELS[h]} weight" for h in HEADS) + " |")
        out("|---|---|---|---|---|---|" + "---|" * len(HEADS))
        for label, counts in configs.items():
            values = sorted(counts.values())
            n = sum(values)
            n_train = round(0.6 * n)
            cells = [f"{n_train / trainable[('convnextv2', h)]:.4f}" for h in HEADS]
            out(f"| {label} | {n} | {len(counts)} | {values[0]} / {values[len(values) // 2]} / {values[-1]} "
                f"| {n_train} | {n_train / len(counts):.1f} | " + " | ".join(cells) + " |")
        if field:
            out(f"| field (outdoor, held-out test only) | {sum(field.values())} | {len(field)} | "
                f"{min(field.values())} / {sorted(field.values())[len(field) // 2]} / {max(field.values())} | 0 | -- | " + " | ".join("--" for _ in HEADS) + " |")
        out("")
        out("Per-class image counts:")
        out("")
        names = cfg.CANONICAL_CLASSNAMES_LIST
        out("| Species | Invasive | " + " | ".join(configs) + (" | field |" if field else " |"))
        out("|---|---|" + "---|" * (len(configs) + (1 if field else 0)))
        for name in names:
            cells = [str(counts.get(name, 0)) for counts in configs.values()]
            if field:
                cells.append(str(field.get(name, 0)))
            out(f"| {name} | {'yes' if name in cfg.INVASIVE_SPECIES_NAMES else ''} | " + " | ".join(cells) + " |")
        out("")
    out("Reading the ratio column: every configuration has far fewer images than trainable")
    out("weights, so the classical '10 samples per parameter' rule is violated by orders of")
    out("magnitude for every head. That rule describes an unregularised model fitted from")
    out("scratch. Here the 768-d features are fixed, and the head is a regularised probe")
    out("(weight decay 2e-5, dropout 0.1, label smoothing 0.05, heavy augmentation, checkpoint")
    out("selection on validation accuracy). The quantity that governs a probe like this is")
    out("images per class, which is what the learning curve measures directly.")
    out("")

    # ------------------------------------------------------------------ statistics
    out("## 3. What a test split of size n can resolve")
    out("")
    out("95% confidence half-width of an accuracy estimate, 1.96 * sqrt(p(1-p)/n):")
    out("")
    sizes = [17, 21, 41, 46, 100, 150, 230, 300, 500]
    out("| True accuracy | " + " | ".join(f"n={n}" for n in sizes) + " |")
    out("|---|" + "---|" * len(sizes))
    for p in (0.5, 0.65, 0.8, 0.9):
        out(f"| {p:.0%} | " + " | ".join(f"+/-{100 * ci_half_width(p, n):.1f}" for n in sizes) + " |")
    out("")
    out("Test images needed for a given half-width at 80% accuracy:")
    out("")
    out("| Half-width | n |")
    out("|---|---|")
    for half in (0.10, 0.05, 0.03, 0.02):
        out(f"| +/-{100 * half:.0f} pts | {n_for_half_width(0.8, half)} |")
    out("")
    out("Two independent accuracies differ significantly (alpha 0.05, two-sided) only when the")
    out("gap exceeds about 1.96 * sqrt(2 * p(1-p)/n):")
    out("")
    out("| n per model | Minimum detectable gap at p=0.8 |")
    out("|---|---|")
    for n in (21, 41, 230, 500):
        out(f"| {n} | {100 * 1.96 * math.sqrt(2 * 0.8 * 0.2 / n):.1f} pts |")
    out("")
    out("One image in a 41-image test set is 2.4 points; in a 21-image field test it is 4.8")
    out("points. A 230-image web test set resolves differences of about 7 points between two")
    out("single runs, and smaller ones once five seeds are averaged.")

    report = "\n".join(lines)
    print(report)
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(report + "\n")
        print(f"\nWritten to {args.markdown}")


if __name__ == "__main__":
    main()
