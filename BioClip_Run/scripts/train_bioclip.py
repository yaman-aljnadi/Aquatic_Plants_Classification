"""BioCLIP ViT-B/16 on the same lab+web+field splits as the ConvNeXt linear head.

Default: vision tower frozen, linear head trained with the train.py recipe
(AdamW, label smoothing, cosine schedule, the GPU augmentation). Images use
OpenAI CLIP mean/std. Lab and web are scored separately because TreeOfLife-10M
includes iNaturalist and the web photos are iNaturalist images.

``--unfreeze-last-block`` also trains the last ViT residual block, ``ln_post``,
and the visual projection (backbone lr 1e-4; head lr 5e-4). Zero-shot is skipped
in that mode. Keep BioCLIP if mean lab > frozen ConvNeXt lab (80%) or mean
pooled > last-stage ConvNeXt pooled (65.1%).

Zero-shot (frozen mode only) uses the scientific name ("a photo of ...").

Examples (from BioClip_Run/, using the project venv):

    ..\\Yaman-Aquatic_Plants_Classification\\.venv\\Scripts\\python.exe -u scripts\\train_bioclip.py --seeds 8 9 10 11 12
    ..\\Yaman-Aquatic_Plants_Classification\\.venv\\Scripts\\python.exe -u scripts\\train_bioclip.py --unfreeze-last-block --skip-zeroshot --seeds 8 9 10 11 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# This file lives in BioClip_Run/scripts/. Training still uses config/src/splits
# from the main aquatic-plant workspace next door.
BIOCLIP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_ROOT = os.path.abspath(os.path.join(BIOCLIP_ROOT, "..", "Yaman-Aquatic_Plants_Classification"))
if TRAIN_ROOT not in sys.path:
    sys.path.insert(0, TRAIN_ROOT)

import config as cfg
from scripts.build_splits import ensure_split
from scripts.train_protonet import score_from_features
from src.data import read_split_records, test_record_groups
from src.gpu_data import get_gpu_splits, gpu_eval_transform
from src.metrics import binary_fnr, evaluate_loader

# OpenAI CLIP image stats. BioCLIP is a fine-tune of that checkpoint.
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# ConvNeXt-V2 Tiny linear head on this same collection (Reports/last_stage_comparison.md).
CONVNEXT_LAB = 0.800
CONVNEXT_POOLED = 0.595
CONVNEXT_LASTSTAGE_POOLED = 0.651

TEST_SPLITS = ("lab_test", "web_test", "field_test", "pooled_test")
MODEL_ID = "hf-hub:imageomics/bioclip"


def parse_args():
    parser = argparse.ArgumentParser(description="BioCLIP linear probe (frozen or last-block)")
    parser.add_argument("--data", default="lab+web+field")
    parser.add_argument("--seeds", type=int, nargs="+", default=[8, 9, 10, 11, 12])
    parser.add_argument("--epochs", type=int, default=cfg.num_epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr_initial)
    parser.add_argument("--backbone-lr", type=float, default=1e-4, help="lr for unfrozen last ViT block")
    parser.add_argument(
        "--unfreeze-last-block", action="store_true",
        help="train the last residual block, ln_post, and visual proj with the linear head",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-zeroshot", action="store_true")
    parser.add_argument("--skip-linear", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


class BioClipProbe(nn.Module):
    """BioCLIP image encoder plus a linear head. Text tower is unused for the linear probe."""

    def __init__(self, clip_model, num_classes, unfreeze_last_block=False):
        super().__init__()
        self.clip = clip_model
        self.unfreeze_last_block = bool(unfreeze_last_block)
        for param in self.clip.parameters():
            param.requires_grad = False
        dim = int(self.clip.visual.output_dim)
        self.classifier = nn.Linear(dim, num_classes)
        if self.unfreeze_last_block:
            visual = self.clip.visual
            for param in visual.transformer.resblocks[-1].parameters():
                param.requires_grad = True
            for param in visual.ln_post.parameters():
                param.requires_grad = True
            if visual.proj is not None:
                visual.proj.requires_grad = True
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {n_train:,} / {n_total:,}")

    def train(self, mode=True):
        super().train(mode)
        # Frozen CLIP stays in eval. The last block (if trained) follows ``mode``.
        self.clip.eval()
        if self.unfreeze_last_block and mode:
            self.clip.visual.transformer.resblocks[-1].train()
            self.clip.visual.ln_post.train()
        return self

    def encode_trainable(self, images):
        """Last residual block + ln_post + proj under grad; earlier ViT under no_grad."""
        visual = self.clip.visual
        with torch.no_grad():
            x = visual._embeds(images)
            for block in visual.transformer.resblocks[:-1]:
                x = block(x)
            x = x.detach()
        x = visual.transformer.resblocks[-1](x)
        pooled, _tokens = visual._pool(x)
        if visual.proj is not None:
            pooled = pooled @ visual.proj
        return pooled.float()

    def encode(self, images):
        if self.unfreeze_last_block and any(p.requires_grad for p in self.clip.visual.parameters()):
            if torch.is_grad_enabled():
                return self.encode_trainable(images)
            with torch.no_grad():
                return self.encode_trainable(images)
        with torch.no_grad():
            return self.clip.encode_image(images).float()

    def forward(self, images):
        return self.classifier(self.encode(images))


# Back-compat alias used nowhere critical.
FrozenBioClip = BioClipProbe


def load_bioclip(device):
    import open_clip

    clip_model, _, _ = open_clip.create_model_and_transforms(MODEL_ID)
    tokenizer = open_clip.get_tokenizer(MODEL_ID)
    clip_model = clip_model.to(device)
    clip_model.eval()
    return clip_model, tokenizer


def apply_clip_normalization():
    """Point the GPU augment/eval transforms at CLIP stats for this process."""
    cfg.normalization_parameters = {"mean": list(CLIP_MEAN), "std": list(CLIP_STD)}


def score_split(model, loader, device):
    acc, binary = evaluate_loader(model, loader, cfg, device)
    return {
        "accuracy": acc,
        "n": int(len(loader.records)) if hasattr(loader, "records") else None,
        "binary_FNR": binary["FNR"],
        "confusion": {k: binary[k] for k in ("TP", "TN", "FP", "FN")},
    }


def mean_std(values):
    values = [float(v) for v in values]
    if len(values) == 1:
        return values[0], 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))


def summarize(per_seed, key_path):
    """Mean and std of one metric. key_path is ('metrics', 'lab_test', 'accuracy')."""
    found = []
    for seed_rec in per_seed:
        node = seed_rec
        ok = True
        for key in key_path:
            if not isinstance(node, dict) or key not in node or node[key] is None:
                ok = False
                break
            node = node[key]
        if ok:
            found.append(node)
    if not found:
        return None
    m, s = mean_std(found)
    return {"mean": m, "std": s, "n": len(found)}


@torch.no_grad()
def time_forward(model, device, batch=1, repeats=30, warmup=10):
    images = torch.randn(batch, 3, 224, 224, device=device)
    model.eval()
    for _ in range(warmup):
        model.encode(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.encode(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / repeats
    return ms / batch


def param_counts(model):
    visual = sum(p.numel() for p in model.clip.visual.parameters())
    total = sum(p.numel() for p in model.clip.parameters())
    head = sum(p.numel() for p in model.classifier.parameters())
    return {"visual": int(visual), "clip_total": int(total), "linear_head": int(head)}


def run_epoch(model, loader, criterion, optimizer, scaler, device, train):
    model.train(mode=train)
    total_loss, n_correct, n_seen = 0.0, 0, 0
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


@torch.no_grad()
def cache_features(model, split, device, use_amp):
    model.eval()
    feats = []
    labels = []
    for images, y in split:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            feats.append(model.encode(images).float())
        labels.append(y.to(device))
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def validate_on_features(model, criterion, feats, labels):
    model.eval()
    logits = model.classifier(feats)
    loss = criterion(logits, labels)
    return loss.item(), (logits.argmax(1) == labels).float().mean().item()


def trainable_state_dict(model):
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters() if param.requires_grad}


def load_trainable_state(model, state):
    own = dict(model.named_parameters())
    for name, tensor in state.items():
        own[name].data.copy_(tensor.to(own[name].device, dtype=own[name].dtype))


def train_one_seed(model, loaders, device, use_amp, args, seed, save_dir, visual_init=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = model.classifier
    nn.init.kaiming_uniform_(head.weight, a=5 ** 0.5)
    nn.init.zeros_(head.bias)
    if args.unfreeze_last_block and visual_init is not None:
        # Restore the pretrained last block so seeds do not continue from each other.
        load_trainable_state(model, visual_init)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    head_params = [p for p in head.parameters() if p.requires_grad]
    visual_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("classifier.")]
    if visual_params:
        optimizer = torch.optim.AdamW(
            [
                {"params": head_params, "lr": args.lr},
                {"params": visual_params, "lr": args.backbone_lr},
            ],
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=cfg.lr_cosine_minimum)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    # Feature cache is only valid while the vision tower does not change.
    val_cache = None if args.unfreeze_last_block else cache_features(model, loaders["val"], device, use_amp)

    best_acc = -1.0
    best_epoch = 0
    best_state = None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, optimizer, scaler, device, train=True)
        if val_cache is not None:
            va_loss, va_acc = validate_on_features(model, criterion, *val_cache)
        else:
            va_loss, va_acc = run_epoch(model, loaders["val"], criterion, optimizer, scaler, device, train=False)
        scheduler.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"  seed {seed}  epoch {epoch:03d}/{args.epochs}  "
                f"train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}"
            )
        if va_acc >= best_acc:
            best_acc = va_acc
            best_epoch = epoch
            best_state = trainable_state_dict(model)
    load_trainable_state(model, best_state)
    minutes = (time.time() - t0) / 60.0
    metrics = {}
    for name in TEST_SPLITS:
        loader = loaders.get(name)
        if loader is None or len(loader.records) == 0:
            continue
        metrics[name] = score_split(model, loader, device)
        fnr = metrics[name]["binary_FNR"]
        fnr_txt = "n/a" if fnr is None else f"{fnr:.3f}"
        print(f"  seed {seed}  {name:12s}  acc={metrics[name]['accuracy']:.4f}  FNR={fnr_txt}")
    ckpt = {
        "trainable": best_state,
        "seed": seed,
        "best_epoch": best_epoch,
        "unfreeze_last_block": bool(args.unfreeze_last_block),
    }
    torch.save(ckpt, os.path.join(save_dir, f"seed{seed}_best.pth"))
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_acc,
        "train_minutes": minutes,
        "metrics": metrics,
        "unfreeze_last_block": bool(args.unfreeze_last_block),
    }


@torch.no_grad()
def zeroshot_one_seed(clip_model, tokenizer, loaders, device, use_amp, seed):
    prompts = [f"a photo of {name}" for name in cfg.CANONICAL_CLASSNAMES_LIST]
    tokens = tokenizer(prompts).to(device)
    with torch.autocast(device_type=device.type, enabled=use_amp):
        text = clip_model.encode_text(tokens)
    text = F.normalize(text.float(), dim=-1)
    scale = clip_model.logit_scale.exp().float().clamp(max=100)
    metrics = {}
    for name in TEST_SPLITS:
        loader = loaders.get(name)
        if loader is None or len(loader.records) == 0:
            continue
        y_true, y_pred = [], []
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                image = clip_model.encode_image(images)
            image = F.normalize(image.float(), dim=-1)
            pred = (scale * image @ text.T).argmax(1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        binary = binary_fnr(y_true, y_pred, cfg.INVASIVE_INDICES)
        acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
        metrics[name] = {
            "accuracy": acc,
            "n": int(len(y_true)),
            "binary_FNR": binary["FNR"],
            "confusion": {k: binary[k] for k in ("TP", "TN", "FP", "FN")},
        }
        fnr = metrics[name]["binary_FNR"]
        fnr_txt = "n/a" if fnr is None else f"{fnr:.3f}"
        print(f"  zeroshot seed {seed}  {name:12s}  acc={acc:.4f}  FNR={fnr_txt}")
    return {"seed": seed, "prompts": prompts, "metrics": metrics}


@torch.no_grad()
def encode_eval(model, split, device, use_amp):
    """Evaluation-transform features, including for the training split (no augmentation)."""
    feats = []
    bs = split.batch_size
    for start in range(0, len(split.records), bs):
        raw = split.bank.images[split.rows[start:start + bs]]
        images = gpu_eval_transform(cfg, raw)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            feats.append(model.encode(images).float())
    return torch.cat(feats, dim=0), split.labels


def five_shot(model, loaders, device, use_amp, seed):
    """Same readout as train_protonet.score_from_features, on frozen BioCLIP features."""
    train_feats, train_labels = encode_eval(model, loaders["train"], device, use_amp)
    train_feats = F.normalize(train_feats, dim=1)
    test_feats, test_labels = {}, {}
    for name in TEST_SPLITS:
        loader = loaders.get(name)
        if loader is None or len(loader.records) == 0:
            continue
        feats, labels = encode_eval(model, loader, device, use_amp)
        test_feats[name] = F.normalize(feats, dim=1)
        test_labels[name] = labels.cpu().numpy()
    shot_key, payload, _pred = score_from_features(
        train_feats,
        train_labels.cpu().numpy(),
        test_feats,
        test_labels,
        5,
        seed,
        cfg.CANONICAL_NUM_CLASSES,
    )
    for name, record in payload["metrics"].items():
        fnr = record["binary_FNR"]
        fnr_txt = "n/a" if fnr is None else f"{fnr:.3f}"
        print(f"  5-shot seed {seed}  {name:12s}  acc={record['accuracy']:.4f}  FNR={fnr_txt}")
    return {"seed": seed, "shot": shot_key, **payload}


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def decide_keep(lab_mean, pooled_mean, unfreeze_last_block):
    """Keep rule: lab > frozen ConvNeXt lab, or (last-block) pooled > last-stage ConvNeXt pooled."""
    beat_lab = lab_mean is not None and lab_mean > CONVNEXT_LAB
    beat_pooled = (
        unfreeze_last_block
        and pooled_mean is not None
        and pooled_mean > CONVNEXT_LASTSTAGE_POOLED
    )
    keep = bool(beat_lab or beat_pooled)
    if beat_lab and beat_pooled:
        reason = (
            f"lab {100 * lab_mean:.1f}% > frozen ConvNeXt lab {100 * CONVNEXT_LAB:.1f}% "
            f"and pooled {100 * pooled_mean:.1f}% > last-stage ConvNeXt pooled "
            f"{100 * CONVNEXT_LASTSTAGE_POOLED:.1f}%; keep BioCLIP"
        )
    elif beat_lab:
        reason = (
            f"lab {100 * lab_mean:.1f}% > frozen ConvNeXt lab {100 * CONVNEXT_LAB:.1f}%; keep BioCLIP"
        )
    elif beat_pooled:
        reason = (
            f"pooled {100 * pooled_mean:.1f}% > last-stage ConvNeXt pooled "
            f"{100 * CONVNEXT_LASTSTAGE_POOLED:.1f}%; keep BioCLIP"
        )
    else:
        lab_txt = "n/a" if lab_mean is None else f"{100 * lab_mean:.1f}%"
        pooled_txt = "n/a" if pooled_mean is None else f"{100 * pooled_mean:.1f}%"
        if unfreeze_last_block:
            reason = (
                f"lab {lab_txt} not above frozen ConvNeXt lab {100 * CONVNEXT_LAB:.1f}% "
                f"and pooled {pooled_txt} not above last-stage ConvNeXt pooled "
                f"{100 * CONVNEXT_LASTSTAGE_POOLED:.1f}%; BioCLIP dropped"
            )
        else:
            reason = (
                f"lab {lab_txt} is not above frozen ConvNeXt lab {100 * CONVNEXT_LAB:.1f}%; "
                f"BioCLIP dropped"
            )
    return keep, reason


def main():
    args = parse_args()
    if args.unfreeze_last_block:
        args.skip_zeroshot = True
    apply_clip_normalization()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    use_amp = bool(cfg.use_amp and device.type == "cuda" and not args.no_amp)
    stage_tag = "lastblock" if args.unfreeze_last_block else "frozen"
    print(f"Device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(
        f"BioCLIP {MODEL_ID}  data={args.data}  seeds={args.seeds}  epochs={args.epochs}  "
        f"mode={stage_tag}"
        + (f"  backbone_lr={args.backbone_lr}" if args.unfreeze_last_block else "")
    )

    clip_model, tokenizer = load_bioclip(device)
    probe = BioClipProbe(
        clip_model, cfg.CANONICAL_NUM_CLASSES, unfreeze_last_block=args.unfreeze_last_block,
    ).to(device)
    visual_init = None
    if args.unfreeze_last_block:
        visual_init = {
            name: param.detach().cpu().clone()
            for name, param in probe.named_parameters()
            if param.requires_grad and not name.startswith("classifier.")
        }
    counts = param_counts(probe)
    ms_one = time_forward(probe, device, batch=1)
    print(
        f"Visual params: {counts['visual']:,}  full CLIP: {counts['clip_total']:,}  "
        f"linear head: {counts['linear_head']:,}"
    )
    print(f"Forward: {ms_one:.1f} ms/image (batch 1, 224 px, vision tower only)")
    print(
        "Rover line: ConvNeXt-V2 Tiny was chosen for the rover "
        "(HANDOFF.md: about 5.2 ms forward, 27.9M params). "
        f"BioCLIP ViT-B/16 is {counts['visual'] / 1e6:.1f}M visual params and {ms_one:.1f} ms/image here."
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = os.path.join(BIOCLIP_ROOT, "runs", "models")
    os.makedirs(models_dir, exist_ok=True)
    save_dir = os.path.join(models_dir, f"{stamp}_bioclip_{stage_tag}_lab_web_field")
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "model": MODEL_ID,
        "data": args.data,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "backbone_lr": args.backbone_lr if args.unfreeze_last_block else None,
        "unfreeze_last_block": bool(args.unfreeze_last_block),
        "label_smoothing": cfg.label_smoothing,
        "weight_decay": cfg.weight_decay,
        "normalization": "openai_clip",
        "prompt": "a photo of {scientific name}",
        "comparison": {
            "frozen_convnext_linear_lab": CONVNEXT_LAB,
            "frozen_convnext_linear_pooled": CONVNEXT_POOLED,
            "laststage_convnext_linear_pooled": CONVNEXT_LASTSTAGE_POOLED,
            "frozen_bioclip_lab": 0.779,
            "frozen_bioclip_web": 0.607,
            "frozen_bioclip_pooled": 0.649,
            "keep_if": (
                "lab > frozen ConvNeXt lab (80%), or pooled > last-stage ConvNeXt pooled (65.1%)"
                if args.unfreeze_last_block
                else "bioclip lab mean > frozen convnext lab mean"
            ),
        },
        "params": counts,
        "ms_per_image_batch1": ms_one,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "zeroshot": [],
        "linear": [],
        "five_shot": None,
    }

    for seed in args.seeds:
        split_csv = ensure_split(args.data, seed)
        print(f"\nSeed {seed}  split={split_csv}")
        loaders = get_gpu_splits(
            cfg, split_csv, device, batch_size=args.batch_size, seed=seed, test_csv=split_csv,
        )
        if not args.skip_zeroshot:
            results["zeroshot"].append(zeroshot_one_seed(clip_model, tokenizer, loaders, device, use_amp, seed))
            write_json(os.path.join(save_dir, "results.partial.json"), results)
        if not args.skip_linear:
            results["linear"].append(
                train_one_seed(probe, loaders, device, use_amp, args, seed, save_dir, visual_init=visual_init)
            )
            write_json(os.path.join(save_dir, "results.partial.json"), results)
        del loaders
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if results["linear"]:
        lab = summarize(results["linear"], ("metrics", "lab_test", "accuracy"))
        web = summarize(results["linear"], ("metrics", "web_test", "accuracy"))
        pooled = summarize(results["linear"], ("metrics", "pooled_test", "accuracy"))
        fnr = summarize(results["linear"], ("metrics", "pooled_test", "binary_FNR"))
        results["linear_summary"] = {"lab_test": lab, "web_test": web, "pooled_test": pooled, "pooled_FNR": fnr}
        keep, reason = decide_keep(
            None if lab is None else lab["mean"],
            None if pooled is None else pooled["mean"],
            args.unfreeze_last_block,
        )
        results["decision"] = {"keep_bioclip": keep, "reason": reason}
        print(f"\nLinear lab {100 * lab['mean']:.1f}% +/- {100 * lab['std']:.1f}")
        print(f"Linear web {100 * web['mean']:.1f}% +/- {100 * web['std']:.1f}")
        print(f"Linear pooled {100 * pooled['mean']:.1f}% +/- {100 * pooled['std']:.1f}")
        print(reason)
        if keep and not args.unfreeze_last_block:
            results["five_shot"] = []
            for seed in args.seeds:
                split_csv = ensure_split(args.data, seed)
                loaders = get_gpu_splits(
                    cfg, split_csv, device, batch_size=args.batch_size, seed=seed, test_csv=split_csv, verbose=False,
                )
                results["five_shot"].append(five_shot(probe, loaders, device, use_amp, seed))
                del loaders
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        elif keep:
            results["five_shot"] = {
                "ran": False,
                "reason": "Last-block BioCLIP kept; 5-shot readout left for a follow-up if needed.",
            }
        else:
            results["five_shot"] = {
                "ran": False,
                "reason": reason,
            }
    if results["zeroshot"]:
        results["zeroshot_summary"] = {
            "lab_test": summarize(results["zeroshot"], ("metrics", "lab_test", "accuracy")),
            "web_test": summarize(results["zeroshot"], ("metrics", "web_test", "accuracy")),
            "pooled_test": summarize(results["zeroshot"], ("metrics", "pooled_test", "accuracy")),
            "pooled_FNR": summarize(results["zeroshot"], ("metrics", "pooled_test", "binary_FNR")),
        }

    write_json(os.path.join(save_dir, "results.json"), results)
    print(f"Results: {os.path.join(save_dir, 'results.json')}")


if __name__ == "__main__":
    main()
