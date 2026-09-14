"""Train one frozen-backbone classifier with the paper recipe.

Examples (from cnn-approaches/, with PYTHONPATH set to this folder):

    python a1_cnn_ynlt/train_run.py --head linear --backbone convnextv2_tiny.fcmae_ft_in22k_in1k
    python a1_cnn_ynlt/train_run.py --head gated_attention --seed 8
    python a1_cnn_ynlt/train_run.py --head linear --backbone resnet50 --epochs 100

Repeated splits: run the same command with --seed 8,9,10,11,12 and average the test numbers.
Do not use 5-fold CV: some species have only two laboratory images.
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

HERE = os.path.dirname(os.path.abspath(__file__))
CNN_APPROACHES = os.path.abspath(os.path.join(HERE, ".."))
if CNN_APPROACHES not in sys.path:
    sys.path.insert(0, CNN_APPROACHES)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import a1_cnn_ynlt.config as cfg
import cnn_utils.data as data_utils
from cnn_utils.evaluate import EvalClassification
from cnn_utils.model import create_model_from_config, save_model_weights
from cnn_utils.predict import PredictPlants


BACKBONE_PRESETS = {
    "convnextv2": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "convnext": "convnext_tiny.in22k_ft_in1k",
    "resnet50": "resnet50.a1_in1k",
    "efficientnet": "tf_efficientnetv2_b0.in1k",
    "vit": "vit_base_patch16_224.augreg_in21k_ft_in1k",
}


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_train:,} / {n_total:,}")
    return n_train, n_total


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, n_correct, n_seen = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_seen += labels.size(0)
    return total_loss / max(n_seen, 1), n_correct / max(n_seen, 1)


def evaluate_split(model, loader, device):
    evaluator = EvalClassification(cfg, model, loader)
    evaluator.evaluate()
    acc = evaluator.get_accuracy(verbose=False)
    binary = evaluator.get_binary_metrics(display=False)
    return acc, binary.get("FNR")


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen-backbone aquatic plant training")
    parser.add_argument("--head", choices=["linear", "gated_attention", "mlp"], default="gated_attention")
    parser.add_argument("--backbone", default=None, help="timm model name, or a key in BACKBONE_PRESETS")
    parser.add_argument("--seed", type=int, default=cfg.random_seed)
    parser.add_argument("--epochs", type=int, default=cfg.num_epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr_initial)
    parser.add_argument("--no-ood", action="store_true")
    parser.add_argument("--with-ynlt", action="store_true", help="Also report YNLT accuracy after training")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.backbone in BACKBONE_PRESETS:
        args.backbone = BACKBONE_PRESETS[args.backbone]
    if args.backbone:
        cfg.train_timm_model_name = args.backbone
    cfg.classification_head_type = args.head
    cfg.random_seed = args.seed
    cfg.num_epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr_initial = args.lr

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    print(f"Lab data: {cfg.main_data_dir}")
    print(f"OOD data: {cfg.hand_test_data_dir}")
    print(f"Backbone: {cfg.train_timm_model_name}  Head: {cfg.classification_head_type}  Seed: {args.seed}")

    train_loader, val_loader, test_loader = data_utils.get_3_dataloaders(cfg, random_seed=args.seed)
    val_ood_loader = test_ood_loader = None
    if not args.no_ood:
        val_ood_loader, test_ood_loader = data_utils.get_ood_dataloaders(cfg, random_seed=args.seed)

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

    run_name = datetime.now().strftime("%m-%d_%H-%M--%S")
    save_dir = os.path.join(cfg.models_path, run_name)
    os.makedirs(save_dir, exist_ok=True)
    best_ood_acc = -1.0
    best_path = os.path.join(save_dir, f"best_{args.head}_{cfg.train_timm_model_name.replace('.', '-')}.pth")

    t0 = time.time()
    for epoch in range(1, cfg.num_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        msg = f"Epoch {epoch:03d}/{cfg.num_epochs}  train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}"
        ood_acc = None
        if val_ood_loader is not None:
            _, ood_acc = run_epoch(model, val_ood_loader, criterion, optimizer, device, train=False)
            msg += f"  ood_val_acc {ood_acc:.3f}"
            if ood_acc >= best_ood_acc:
                best_ood_acc = ood_acc
                save_model_weights(model, best_path, verbose=False)
        elif va_acc >= best_ood_acc:
            best_ood_acc = va_acc
            save_model_weights(model, best_path, verbose=False)
        print(msg)

    print(f"Training wall time: {(time.time() - t0) / 60:.1f} min")
    if os.path.isfile(best_path):
        from cnn_utils.model import load_model
        model = load_model(best_path, verbose=True).to(device)

    lab_acc, lab_fnr = evaluate_split(model, test_loader, device)
    print(f"LAB test accuracy={lab_acc:.4f}  binary FNR={lab_fnr}")
    if test_ood_loader is not None:
        ood_acc, ood_fnr = evaluate_split(model, test_ood_loader, device)
        print(f"OOD test accuracy={ood_acc:.4f}  binary FNR={ood_fnr}")

    if args.with_ynlt:
        predictor = PredictPlants(model, model_name=args.head)
        ynlt_eval = predictor.evaluate_with_predict_fn(test_loader)
        print(f"LAB + YNLT accuracy={ynlt_eval.get_accuracy(verbose=False):.4f}")
        if test_ood_loader is not None:
            ynlt_ood = predictor.evaluate_with_predict_fn(test_ood_loader)
            print(f"OOD + YNLT accuracy={ynlt_ood.get_accuracy(verbose=False):.4f}")

    print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
