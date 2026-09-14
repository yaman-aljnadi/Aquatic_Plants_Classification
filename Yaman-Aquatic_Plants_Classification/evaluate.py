"""Evaluate a saved checkpoint with and without YNLT."""

from __future__ import annotations

import argparse
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
from src.ynlt import evaluate_with_ynlt


def report(name, acc, binary):
    fnr = binary.get("FNR")
    fnr_txt = "n/a" if fnr is None else f"{fnr:.4f}"
    print(f"{name:12s}  accuracy={acc:.4f}  binary_FNR={fnr_txt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--seed", type=int, default=cfg.random_seed)
    parser.add_argument("--with-ynlt", action="store_true")
    parser.add_argument("--no-ood", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint).to(device)
    _, _, test_loader = get_3_dataloaders(cfg, random_seed=args.seed)
    acc, binary = evaluate_loader(model, test_loader, cfg, device)
    report("LAB", acc, binary)

    if args.with_ynlt:
        y_true, y_pred, n_review = evaluate_with_ynlt(model, test_loader, device)
        acc = float(accuracy_score(y_true, y_pred))
        binary = binary_fnr(np.array(y_true), np.array(y_pred), cfg.INVASIVE_INDICES)
        report("LAB+YNLT", acc, binary)
        print(f"YNLT reviewed {n_review}/{len(y_true)} lab images")

    if not args.no_ood:
        _, test_ood = get_ood_dataloaders(cfg, random_seed=args.seed)
        acc, binary = evaluate_loader(model, test_ood, cfg, device)
        report("OOD", acc, binary)
        if args.with_ynlt:
            y_true, y_pred, n_review = evaluate_with_ynlt(model, test_ood, device)
            acc = float(accuracy_score(y_true, y_pred))
            binary = binary_fnr(np.array(y_true), np.array(y_pred), cfg.INVASIVE_INDICES)
            report("OOD+YNLT", acc, binary)
            print(f"YNLT reviewed {n_review}/{len(y_true)} OOD images")


if __name__ == "__main__":
    main()
