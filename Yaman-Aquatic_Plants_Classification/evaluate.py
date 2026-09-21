"""Evaluate a saved checkpoint with and without YNLT.

    python evaluate.py runs\\models\\<run>\\best.pth --with-ynlt
    python evaluate.py runs\\models\\<run>\\best.pth --with-ynlt --ynlt-source input224
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as cfg
from src.data import get_3_dataloaders, get_ood_dataloaders
from src.metrics import binary_fnr, evaluate_loader
from src.model import load_model
from src.ynlt import PATCH_SOURCES, evaluate_with_ynlt


def report(name, acc, binary):
    fnr = binary.get("FNR")
    fnr_txt = "n/a" if fnr is None else f"{fnr:.4f}"
    print(f"{name:24s}  accuracy={acc:.4f}  binary_FNR={fnr_txt}")


def plain(model, loader, device, name, out):
    acc, binary = evaluate_loader(model, loader, cfg, device)
    report(name, acc, binary)
    out[name] = {"accuracy": acc, "binary_FNR": binary["FNR"]}


def with_ynlt(model, loader, device, name, source, out):
    stats = evaluate_with_ynlt(model, loader, device, patch_source=source)
    y_true = np.array(stats["y_true"])
    y_pred = np.array(stats["y_pred"])
    acc = float(accuracy_score(y_true, y_pred))
    binary = binary_fnr(y_true, y_pred, cfg.INVASIVE_INDICES)
    label = f"{name}+YNLT[{stats['patch_source']}]"
    report(label, acc, binary)
    print(
        f"  reviewed {stats['n_review']}/{stats['n_total']} images ({stats['trigger_rate']:.1%}),"
        f" {stats['mean_patches_per_reviewed']:.1f} patches/reviewed,"
        f" {stats['ms_per_image']:.1f} ms/image vs {stats['ms_per_image_first_pass']:.1f} ms first pass"
    )
    out[label] = {
        "accuracy": acc,
        "binary_FNR": binary["FNR"],
        "patch_source": stats["patch_source"],
        "trigger_rate": stats["trigger_rate"],
        "mean_patches_per_reviewed": stats["mean_patches_per_reviewed"],
        "ms_per_image": stats["ms_per_image"],
        "ms_per_image_first_pass": stats["ms_per_image_first_pass"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--seed", type=int, default=cfg.random_seed)
    parser.add_argument("--with-ynlt", action="store_true")
    parser.add_argument("--ynlt-source", choices=[*PATCH_SOURCES, "both"], default="both")
    parser.add_argument("--no-ood", action="store_true")
    parser.add_argument("--save", default=None, help="write the numbers to this JSON file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint).to(device)
    sources = list(PATCH_SOURCES) if args.ynlt_source == "both" else [args.ynlt_source]
    out = {"checkpoint": args.checkpoint, "seed": args.seed}

    _, _, test_loader = get_3_dataloaders(cfg, random_seed=args.seed)
    plain(model, test_loader, device, "LAB", out)
    if args.with_ynlt:
        for source in sources:
            with_ynlt(model, test_loader, device, "LAB", source, out)

    if not args.no_ood:
        _, test_ood = get_ood_dataloaders(cfg, random_seed=args.seed)
        plain(model, test_ood, device, "OOD", out)
        if args.with_ynlt:
            for source in sources:
                with_ynlt(model, test_ood, device, "OOD", source, out)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2)
        print(f"Saved {args.save}")


if __name__ == "__main__":
    main()
