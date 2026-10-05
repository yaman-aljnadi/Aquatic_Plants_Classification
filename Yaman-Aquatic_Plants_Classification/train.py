"""Train a frozen-backbone aquatic plant classifier.

Examples (from this folder):

    python train.py --data lab+web --head gated_attention --backbone convnextv2 --with-ynlt
    python train.py --data lab --head linear --backbone convnextv2
    python train.py --data lab+web --head linear --backbone convnextv2_nano --seed 9
    python train.py --data lab+web --head gated_attention --train-fraction 0.25   # learning curve

``--data`` selects which image collection is trained on (``lab``: 203 laboratory photos,
``web``: 1,140 GBIF/iNaturalist field photos, ``lab+web``: both). The split list for
(data, seed) is read from ``data/splits`` and built by ``scripts/build_splits.py`` if it
does not exist yet. The checkpoint with the best accuracy on the combined validation split
is kept and scored on the lab test, web test and the 42 held-out outdoor field photos.

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

# Must precede the numpy import: DataLoader workers re-import this module on Windows.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as cfg
from scripts.build_splits import ensure_split
from src.data import get_split_loaders
from src.gpu_data import get_gpu_splits
from src.metrics import binary_fnr, evaluate_loader
from src.model import create_model_from_config, freeze_backbone, load_model, save_model_weights
from src.ynlt import PATCH_SOURCES, evaluate_with_ynlt

TEST_SPLITS = ("lab_test", "web_test", "field_test")

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard is optional
    SummaryWriter = None


def parse_args():
    parser = argparse.ArgumentParser(description="Yaman aquatic plant training")
    parser.add_argument("--data", choices=list(cfg.DATA_CONFIGS), default="lab+web", help="which image collection to train on")
    parser.add_argument("--split-csv", default=None, help="explicit split list; default data/splits/<data>_seed<N>.csv")
    parser.add_argument(
        "--train-fraction", type=float, default=1.0,
        help="keep this fraction of the training images per class (whole observations); for the learning curve",
    )
    parser.add_argument("--head", choices=["linear", "gated_attention", "mlp"], default=cfg.classification_head_type)
    parser.add_argument("--backbone", default="convnextv2", help="preset key or a full timm name")
    parser.add_argument("--seed", type=int, default=cfg.random_seed)
    parser.add_argument("--epochs", type=int, default=cfg.num_epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr_initial)
    parser.add_argument("--workers", type=int, default=cfg.num_workers)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--with-ynlt", action="store_true")
    parser.add_argument(
        "--ynlt-source",
        choices=[*PATCH_SOURCES, "both"],
        default=cfg.ynlt_patch_source,
        help="patch the 224 input ('input224', default), the full-resolution file ('original'), or report both",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--loader", choices=["gpu", "cpu"], default=None,
        help="gpu (default on CUDA): decode once, keep images on the GPU, augment in batched kernels; "
             "cpu: the torchvision DataLoader path",
    )
    parser.add_argument("--no-tensorboard", action="store_true", help="skip writing TensorBoard event files")
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
    model.train(mode=train)
    total_loss, n_correct, n_seen = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
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


@torch.no_grad()
def cache_features(model, split, device, use_amp):
    """Backbone features of a GpuSplit under the evaluation transform.

    The backbone is frozen and the evaluation transform is deterministic, so the
    validation images only need one pass through the encoder; every epoch then scores
    the head on the cached features.
    """
    model.eval()
    feats = []
    for images, _ in split:
        with torch.autocast(device_type=device.type, enabled=use_amp):
            feats.append(model.backbone(images.to(device, non_blocking=True)).float())
    return torch.cat(feats, dim=0), split.labels.to(device)


@torch.no_grad()
def validate_on_features(model, criterion, feats, labels):
    model.eval()
    logits = model.classifier(feats)
    loss = criterion(logits, labels)
    return loss.item(), (logits.argmax(1) == labels).float().mean().item()


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
        "data": results["data"],
        "train_fraction": results["train_fraction"],
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

    loader_kind = args.loader or ("gpu" if device.type == "cuda" else "cpu")
    print(f"Device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"AMP: {use_amp}  batch={cfg.batch_size}  loader={loader_kind}"
          + (f"  workers={cfg.num_workers}" if loader_kind == "cpu" else f"  cache={cfg.gpu_cache_size}px"))
    split_csv = args.split_csv or ensure_split(args.data, args.seed)
    # Every run is scored on the same three test sets, taken from the lab+web list of this
    # seed (identical per-source partitions, verified leak-free inside get_split_loaders).
    test_csv = split_csv if args.split_csv else ensure_split("lab+web", args.seed)
    print(f"Data: {args.data}  split list: {split_csv}")
    print(f"Backbone: {cfg.train_timm_model_name}  Head: {cfg.classification_head_type}  Seed: {args.seed}")

    if loader_kind == "gpu":
        loaders = get_gpu_splits(cfg, split_csv, device, train_fraction=args.train_fraction, seed=args.seed, test_csv=test_csv)
    else:
        loaders = get_split_loaders(cfg, split_csv, train_fraction=args.train_fraction, seed=args.seed, test_csv=test_csv)
    train_loader, val_loader = loaders["train"], loaders["val"]
    if val_loader is None:
        raise ValueError("The split list has no validation rows; checkpoint selection needs them.")
    test_loaders = {name: loaders[name] for name in TEST_SPLITS if loaders.get(name) is not None}

    model = create_model_from_config(cfg)
    n_trainable, n_total_params = freeze_backbone(model)
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)  # faster convolutions under AMP; no-op for ViT

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    val_cache = cache_features(model, val_loader, device, use_amp) if loader_kind == "gpu" else None
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr_initial,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=cfg.lr_cosine_minimum)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_tag = args.data.replace("+", "_")
    frac_tag = "" if args.train_fraction >= 1.0 else f"_frac{args.train_fraction:g}"
    save_dir = os.path.join(cfg.MODELS_DIR, f"{stamp}_{data_tag}_{args.head}_{args.backbone}_seed{args.seed}{frac_tag}")
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best.pth")
    best_val_acc = -1.0
    best_epoch = 0
    writer = open_writer(save_dir, args.no_tensorboard)

    t0 = time.time()
    for epoch in range(1, cfg.num_epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        if val_cache is not None:
            va_loss, va_acc = validate_on_features(model, criterion, *val_cache)
        else:
            va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
        scheduler.step()
        print(f"Epoch {epoch:03d}/{cfg.num_epochs}  train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
        if writer is not None:
            writer.add_scalar("loss/train", tr_loss, epoch)
            writer.add_scalar("loss/val", va_loss, epoch)
            writer.add_scalar("accuracy/train", tr_acc, epoch)
            writer.add_scalar("accuracy/val", va_acc, epoch)
            writer.add_scalar("lr", lr_now, epoch)
            writer.add_scalar("accuracy/best_selection", max(va_acc, best_val_acc), epoch)
        if va_acc >= best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            save_model_weights(model, best_path, verbose=False)

    train_minutes = (time.time() - t0) / 60
    print(f"Training wall time: {train_minutes:.1f} min  (best epoch {best_epoch}, val acc {best_val_acc:.4f})")
    model = load_model(best_path).to(device)

    results = {
        "run": os.path.basename(save_dir),
        "timestamp": stamp,
        "data": args.data,
        "train_fraction": args.train_fraction,
        "split_csv": os.path.relpath(split_csv, ROOT).replace("\\", "/"),
        "test_csv": os.path.relpath(test_csv, ROOT).replace("\\", "/"),
        "head": args.head,
        "backbone_key": args.backbone,
        "backbone": backbone,
        "seed": args.seed,
        "epochs": cfg.num_epochs,
        "best_epoch": best_epoch,
        "selection_metric": "val_accuracy",
        "best_selection_score": best_val_acc,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr_initial,
        "amp": use_amp,
        "loader": loader_kind,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "trainable_params": int(n_trainable),
        "total_params": int(n_total_params),
        "train_minutes": train_minutes,
        "splits": loaders["sizes"],
        "checkpoint": best_path,
        "metrics": {},
    }

    for name, loader in test_loaders.items():
        record, acc, binary = score_split(model, loader, device)
        results["metrics"][name] = record
        report_split(name.upper(), acc, binary)

    if args.with_ynlt:
        sources = list(PATCH_SOURCES) if args.ynlt_source == "both" else [args.ynlt_source]
        for source in sources:
            for name, loader in test_loaders.items():
                record, acc, binary, stats = score_split_ynlt(model, loader, device, source)
                results["metrics"][f"{name}_ynlt_{record['patch_source']}"] = record
                report_split(f"{name.upper()}+YNLT[{record['patch_source']}]", acc, binary)
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
