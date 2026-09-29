"""Train a frozen-backbone aquatic plant classifier.

Examples (from this folder):

    python train.py --head linear --backbone convnextv2
    python train.py --head gated_attention --backbone convnextv2 --with-ynlt
    python train.py --head linear --backbone resnet50 --seed 8

Every run writes ``results.json`` and TensorBoard event files next to its checkpoint.
Aggregate several runs with ``python scripts/aggregate_results.py``; compare them visually
with ``tensorboard --logdir runs/models``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as cfg
from src.data import get_3_dataloaders, get_ood_dataloaders
from src.metrics import binary_fnr, evaluate_loader
from src.model import create_model_from_config, freeze_backbone, load_model, save_model_weights
from src.ynlt import PATCH_SOURCES, evaluate_with_ynlt

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard is optional
    SummaryWriter = None


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
    parser.add_argument(
        "--ynlt-source",
        choices=[*PATCH_SOURCES, "both"],
        default="both",
        help="patch the full-resolution file ('original'), the 224 input ('input224'), or report both",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true", help="skip writing TensorBoard event files")
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
    print(f"{name:16s}  accuracy={acc:.4f}  binary_FNR={fnr_txt}")


def score_split(model, loader, device):
    acc, binary = evaluate_loader(model, loader, cfg, device)
    return {
        "accuracy": acc,
        "binary_accuracy": (binary["TP"] + binary["TN"]) / max(sum(binary[k] for k in ("TP", "TN", "FP", "FN")), 1),
        "binary_FNR": binary["FNR"],
        "confusion": {k: binary[k] for k in ("TP", "TN", "FP", "FN")},
    }, acc, binary


def open_writer(save_dir, disabled):
    """TensorBoard writer for one run, or None when unavailable/disabled."""
    if disabled:
        return None
    if SummaryWriter is None:
        print("TensorBoard not installed (pip install tensorboard) - skipping event logs.")
        return None
    return SummaryWriter(log_dir=save_dir)


def log_final_metrics(writer, results, step):
    """Mirror the numbers in results.json into scalars and the HParams tab."""
    if writer is None:
        return
    hparam_metrics = {}
    for split, record in results["metrics"].items():
        for key, value in record.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                writer.add_scalar(f"final/{split}/{key}", value, step)
        hparam_metrics[f"hparam/{split}_accuracy"] = record["accuracy"]
        if record.get("binary_FNR") is not None:
            hparam_metrics[f"hparam/{split}_binary_FNR"] = record["binary_FNR"]
    hparams = {
        "head": results["head"],
        "backbone": results["backbone_key"],
        "seed": results["seed"],
        "epochs": results["epochs"],
        "batch_size": results["batch_size"],
        "lr": results["lr"],
        "trainable_params": results["trainable_params"],
    }
    # run_name="." keeps the hparams in this run's directory instead of a timestamped child.
    writer.add_hparams(hparams, hparam_metrics, run_name=".")
    writer.add_text("results", "```json\n" + json.dumps(results, indent=2) + "\n```", step)


def score_split_ynlt(model, loader, device, patch_source):
    stats = evaluate_with_ynlt(model, loader, device, patch_source=patch_source)
    y_true = np.array(stats["y_true"])
    y_pred = np.array(stats["y_pred"])
    acc = float(accuracy_score(y_true, y_pred))
    binary = binary_fnr(y_true, y_pred, cfg.INVASIVE_INDICES)
    record = {
        "accuracy": acc,
        "binary_accuracy": (binary["TP"] + binary["TN"]) / max(len(y_true), 1),
        "binary_FNR": binary["FNR"],
        "confusion": {k: binary[k] for k in ("TP", "TN", "FP", "FN")},
        "patch_source": stats["patch_source"],
        "n_reviewed": stats["n_review"],
        "n_total": stats["n_total"],
        "trigger_rate": stats["trigger_rate"],
        "mean_patches_per_reviewed": stats["mean_patches_per_reviewed"],
        "ms_per_image": stats["ms_per_image"],
        "ms_per_image_first_pass": stats["ms_per_image_first_pass"],
    }
    return record, acc, binary, stats


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
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

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
    n_trainable, n_total_params = freeze_backbone(model)
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
    best_epoch = 0
    writer = open_writer(save_dir, args.no_tensorboard)

    t0 = time.time()
    for epoch in range(1, cfg.num_epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
        scheduler.step()
        msg = f"Epoch {epoch:03d}/{cfg.num_epochs}  train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}"
        score = va_acc
        ood_acc = None
        if val_ood_loader is not None:
            ood_loss, ood_acc = run_epoch(model, val_ood_loader, criterion, optimizer, scaler, device, train=False)
            msg += f"  ood_val {ood_acc:.3f}"
            score = ood_acc
        print(msg)
        if writer is not None:
            writer.add_scalar("loss/train", tr_loss, epoch)
            writer.add_scalar("loss/val", va_loss, epoch)
            writer.add_scalar("accuracy/train", tr_acc, epoch)
            writer.add_scalar("accuracy/val", va_acc, epoch)
            if ood_acc is not None:
                writer.add_scalar("loss/ood_val", ood_loss, epoch)
                writer.add_scalar("accuracy/ood_val", ood_acc, epoch)
            writer.add_scalar("lr", lr_now, epoch)
            writer.add_scalar("accuracy/best_selection", max(score, best_ood_acc), epoch)
        if score >= best_ood_acc:
            best_ood_acc = score
            best_epoch = epoch
            save_model_weights(model, best_path, verbose=False)

    train_minutes = (time.time() - t0) / 60
    print(f"Training wall time: {train_minutes:.1f} min")
    model = load_model(best_path).to(device)

    results = {
        "run": os.path.basename(save_dir),
        "timestamp": stamp,
        "head": args.head,
        "backbone_key": args.backbone,
        "backbone": backbone,
        "seed": args.seed,
        "epochs": cfg.num_epochs,
        "best_epoch": best_epoch,
        "selection_metric": "ood_val_accuracy" if val_ood_loader is not None else "lab_val_accuracy",
        "best_selection_score": best_ood_acc,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr_initial,
        "amp": use_amp,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "trainable_params": int(n_trainable),
        "total_params": int(n_total_params),
        "train_minutes": train_minutes,
        "lab_data_dir": cfg.main_data_dir,
        "ood_data_dir": cfg.hand_test_data_dir,
        "splits": {
            "lab_train": len(train_loader.dataset),
            "lab_val": len(val_loader.dataset),
            "lab_test": len(test_loader.dataset),
            "ood_val": len(val_ood_loader.dataset) if val_ood_loader is not None else 0,
            "ood_test": len(test_ood_loader.dataset) if test_ood_loader is not None else 0,
        },
        "checkpoint": best_path,
        "metrics": {},
    }

    record, acc, binary = score_split(model, test_loader, device)
    results["metrics"]["lab"] = record
    report_split("LAB", acc, binary)
    if test_ood_loader is not None:
        record, acc, binary = score_split(model, test_ood_loader, device)
        results["metrics"]["ood"] = record
        report_split("OOD", acc, binary)

    if args.with_ynlt:
        sources = list(PATCH_SOURCES) if args.ynlt_source == "both" else [args.ynlt_source]
        for source in sources:
            record, acc, binary, stats = score_split_ynlt(model, test_loader, device, source)
            results["metrics"][f"lab_ynlt_{record['patch_source']}"] = record
            report_split(f"LAB+YNLT[{record['patch_source']}]", acc, binary)
            print(
                f"  reviewed {stats['n_review']}/{stats['n_total']} images"
                f"  ({stats['trigger_rate']:.1%}), {stats['mean_patches_per_reviewed']:.1f} patches/reviewed,"
                f" {stats['ms_per_image']:.1f} ms/image vs {stats['ms_per_image_first_pass']:.1f} ms first pass"
            )
            if test_ood_loader is not None:
                record, acc, binary, stats = score_split_ynlt(model, test_ood_loader, device, source)
                results["metrics"][f"ood_ynlt_{record['patch_source']}"] = record
                report_split(f"OOD+YNLT[{record['patch_source']}]", acc, binary)
                print(
                    f"  reviewed {stats['n_review']}/{stats['n_total']} images"
                    f"  ({stats['trigger_rate']:.1%}), {stats['mean_patches_per_reviewed']:.1f} patches/reviewed,"
                    f" {stats['ms_per_image']:.1f} ms/image vs {stats['ms_per_image_first_pass']:.1f} ms first pass"
                )

    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    log_final_metrics(writer, results, cfg.num_epochs)
    if writer is not None:
        writer.close()

    print(f"Best checkpoint: {best_path}")
    print(f"Results: {results_path}")
    if writer is not None:
        print(f"TensorBoard: tensorboard --logdir {cfg.MODELS_DIR}")


if __name__ == "__main__":
    main()
