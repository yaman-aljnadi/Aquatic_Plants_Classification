"""Train a frozen-backbone aquatic plant classifier.

Examples (from this folder):

    python train.py --head linear --backbone convnextv2
    python train.py --head gated_attention --backbone convnextv2 --with-ynlt
    python train.py --head linear --backbone resnet50 --seed 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as cfg
from src.data import get_3_dataloaders, get_ood_dataloaders
from src.metrics import evaluate_loader
from src.model import create_model_from_config, freeze_backbone, load_model, save_model_weights
from src.ynlt import evaluate_with_ynlt


def parse_args():
    parser = argparse.ArgumentParser(description="Yaman aquatic plant training")
    parser.add_argument("--head", choices=["linear", "gated_attention", "mlp"], default=cfg.classification_head_type)
    parser.add_argument("--backbone", default="convnextv2", help="preset key or a full timm name")
    parser.add_argument("--seed", type=int, default=cfg.random_seed)
    parser.add_argument("--epochs", type=int, default=cfg.num_epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr_initial)
    parser.add_argument("--workers", type=int, default=cfg.num_workers)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-ood", action="store_true")
    parser.add_argument("--with-ynlt", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
    model.train(mode=train)
    total_loss, n_correct, n_seen = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, labels)
            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * labels.size(0)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_seen += labels.size(0)
    return total_loss / max(n_seen, 1), n_correct / max(n_seen, 1)


def report_split(name, acc, binary):
    fnr = binary.get("FNR")
    fnr_txt = "n/a" if fnr is None else f"{fnr:.4f}"
    print(f"{name:12s}  accuracy={acc:.4f}  binary_FNR={fnr_txt}")


def main():
    args = parse_args()
    backbone = cfg.BACKBONE_PRESETS.get(args.backbone, args.backbone)
    cfg.train_timm_model_name = backbone
    cfg.classification_head_type = args.head
    cfg.random_seed = args.seed
    cfg.num_epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr_initial = args.lr
    cfg.num_workers = args.workers

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and not args.cpu:
        torch.backends.cudnn.benchmark = True

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    use_amp = bool(cfg.use_amp and device.type == "cuda" and not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp else None

    print(f"Device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"AMP: {use_amp}  batch={cfg.batch_size}  workers={cfg.num_workers}")
    print(f"Lab data: {cfg.main_data_dir}")
    print(f"OOD data: {cfg.hand_test_data_dir}")
    print(f"Backbone: {cfg.train_timm_model_name}  Head: {cfg.classification_head_type}  Seed: {args.seed}")

    train_loader, val_loader, test_loader = get_3_dataloaders(cfg, random_seed=args.seed)
    val_ood_loader = test_ood_loader = None
    if not args.no_ood:
        val_ood_loader, test_ood_loader = get_ood_dataloaders(cfg, random_seed=args.seed)

    model = create_model_from_config(cfg)
    freeze_backbone(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr_initial,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=cfg.lr_cosine_minimum)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(cfg.MODELS_DIR, f"{stamp}_{args.head}_{args.backbone}_seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best.pth")
    best_ood_acc = -1.0

    t0 = time.time()
    for epoch in range(1, cfg.num_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
        scheduler.step()
        msg = f"Epoch {epoch:03d}/{cfg.num_epochs}  train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}"
        score = va_acc
        if val_ood_loader is not None:
            _, ood_acc = run_epoch(model, val_ood_loader, criterion, optimizer, scaler, device, train=False)
            msg += f"  ood_val {ood_acc:.3f}"
            score = ood_acc
        print(msg)
        if score >= best_ood_acc:
            best_ood_acc = score
            save_model_weights(model, best_path, verbose=False)

    print(f"Training wall time: {(time.time() - t0) / 60:.1f} min")
    model = load_model(best_path).to(device)

    lab_acc, lab_bin = evaluate_loader(model, test_loader, cfg, device)
    report_split("LAB", lab_acc, lab_bin)
    if test_ood_loader is not None:
        ood_acc, ood_bin = evaluate_loader(model, test_ood_loader, cfg, device)
        report_split("OOD", ood_acc, ood_bin)

    if args.with_ynlt:
        y_true, y_pred, n_review = evaluate_with_ynlt(model, test_loader, device)
        from sklearn.metrics import accuracy_score
        from src.metrics import binary_fnr
        acc = float(accuracy_score(y_true, y_pred))
        binary = binary_fnr(np.array(y_true), np.array(y_pred), cfg.INVASIVE_INDICES)
        report_split("LAB+YNLT", acc, binary)
        print(f"YNLT reviewed {n_review}/{len(y_true)} lab test images")
        if test_ood_loader is not None:
            y_true, y_pred, n_review = evaluate_with_ynlt(model, test_ood_loader, device)
            acc = float(accuracy_score(y_true, y_pred))
            binary = binary_fnr(np.array(y_true), np.array(y_pred), cfg.INVASIVE_INDICES)
            report_split("OOD+YNLT", acc, binary)
            print(f"YNLT reviewed {n_review}/{len(y_true)} OOD test images")

    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
