"""Episodic Prototypical Network on the aquatic-plant splits.

Trains a projection (and optionally the last ConvNeXt stage) so class means in the
embedding separate the 21 species. Each episode draws K support and Q query images
from every class in the training split, augments them, and applies the Snell et al.
loss: cross-entropy on negative squared Euclidean distance to the class means.

The checkpoint is the best accuracy on the full validation split, using prototypes
built from every training image (eval transform, no augmentation). Test images are
not used for prototypes or for checkpoint selection. After training, each seed is
scored at 1/5/10-shot and all-support, same protocol as scripts/protonet_baseline.py.

Examples (from Yaman-Aquatic_Plants_Classification/):

    python scripts/train_protonet.py --features final --seeds 8 9 10 11 12
    python scripts/train_protonet.py --features final --unfreeze-last-stage --seeds 8 9 10 11 12
    python scripts/train_protonet.py --features shape_texture --seeds 8 9 10 11 12
    python scripts/train_protonet.py --features shape_texture --unfreeze-last-stage --seeds 9
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as cfg
from scripts.build_splits import ensure_split
from scripts.protonet_baseline import (
    build_prototypes,
    mean_std,
    nearest_prototype,
    print_summary_table,
    sample_support_indices,
    score_predictions,
)
from src.data import read_split_records, test_record_groups
from src.gpu_data import ImageBank, gpu_augment, gpu_eval_transform

TEST_SPLITS = ("lab_test", "web_test", "field_test", "pooled_test")

# Species the earlier linear-head error analysis mixed up: pondweeds, and the
# feathery submerged plants that share a growth form.
LOOKALIKE_NAMES = (
    "Cabomba",
    "Ceratophyllum demersum",
    "Myriophyllum sibiricum",
    "Myriophyllum spicatum",
    "Potamogeton crispus",
    "Potamogeton gramineus",
    "Potamogeton illinoensis",
    "Potamogeton natans",
    "Potamogeton praelongus",
    "Potamogeton richardsonii",
    "Potamogeton robbinsii",
)
LOOKALIKE_INDICES = {cfg.CANONICAL_CLASS_TO_INDEX[name] for name in LOOKALIKE_NAMES}


def parse_args():
    parser = argparse.ArgumentParser(description="Episodic Prototypical Network")
    parser.add_argument("--data", choices=list(cfg.DATA_CONFIGS), default="lab+web+field")
    parser.add_argument("--backbone", default="convnextv2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[8, 9, 10, 11, 12])
    parser.add_argument("--features", choices=["final", "shape_texture"], default="final")
    parser.add_argument(
        "--texture-stage", type=int, default=1,
        help="0-based ConvNeXt stage whose global pool is the texture vector (shape_texture only)",
    )
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--unfreeze-last-stage", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--episodes-per-epoch", type=int, default=30)
    parser.add_argument("--k-shot", type=int, default=5, help="support images per class inside a training episode")
    parser.add_argument("--q-query", type=int, default=3, help="query images per class inside a training episode")
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 5, 10], help="K values for the test table")
    parser.add_argument("--no-include-all", action="store_true", help="skip the all-support test column")
    parser.add_argument("--lr", type=float, default=cfg.lr_initial)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0, help="divide squared distance by this before softmax")
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size, help="eval encoding batch, not the episode")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def abs_path(record):
    path = record["path"]
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(cfg.REPO_ROOT, path))


class EpisodicProtoNet(nn.Module):
    """ConvNeXt encoder plus a linear projection onto the prototype space.

    ``final`` uses the usual 768-d pooled embedding. ``shape_texture`` concatenates
    that vector with a global pool of an earlier stage (leaf texture) before the
    projection. Embeddings are L2-normalised so the squared Euclidean loss stays bounded.
    """

    def __init__(self, backbone_name, proj_dim, feature_mode, texture_stage, pretrained=True):
        super().__init__()
        self.feature_mode = feature_mode
        self.texture_stage = int(texture_stage)
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        n_stages = len(self.backbone.stages)
        if not 0 <= self.texture_stage < n_stages:
            raise ValueError(f"texture stage {self.texture_stage} is outside 0..{n_stages - 1}")
        with torch.no_grad():
            dummy = self._forward_all(torch.zeros(2, 3, 224, 224))
        self.proj = nn.Linear(dummy.size(1), proj_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        self.proj_dim = proj_dim

    def _set_stage_modes(self):
        """Drop-path only on stages that are actually training."""
        self.backbone.stem.eval()
        for stage in self.backbone.stages:
            if any(p.requires_grad for p in stage.parameters()):
                stage.train()
            else:
                stage.eval()
        if any(p.requires_grad for p in self.backbone.norm_pre.parameters()):
            self.backbone.norm_pre.train()
        else:
            self.backbone.norm_pre.eval()

    def _forward_all(self, images):
        x = self.backbone.stem(images)
        texture = None
        for i, stage in enumerate(self.backbone.stages):
            x = stage(x)
            if self.feature_mode == "shape_texture" and i == self.texture_stage:
                texture = x.mean(dim=(2, 3))
        x = self.backbone.norm_pre(x)
        shape = x.mean(dim=(2, 3))
        if self.feature_mode != "shape_texture":
            return shape
        if texture is None:
            raise RuntimeError(f"texture stage {self.texture_stage} was not applied")
        return torch.cat([texture, shape], dim=1)

    def _forward_train(self, images):
        """Frozen prefix under no_grad; gradient runs only from the first trainable stage."""
        stages = list(self.backbone.stages)
        train_from = len(stages)
        for i, stage in enumerate(stages):
            if any(p.requires_grad for p in stage.parameters()):
                train_from = i
                break
        texture = None

        def run(x, start, end):
            found = None
            for i in range(start, end):
                x = stages[i](x)
                if self.feature_mode == "shape_texture" and i == self.texture_stage:
                    found = x.mean(dim=(2, 3))
            return x, found

        with torch.no_grad():
            x = self.backbone.stem(images)
            x, texture = run(x, 0, train_from)
        if train_from < len(stages):
            x = x.detach()
            x, texture_later = run(x, train_from, len(stages))
            if texture_later is not None:
                texture = texture_later
            x = self.backbone.norm_pre(x)
        else:
            with torch.no_grad():
                x = self.backbone.norm_pre(x)
        shape = x.mean(dim=(2, 3))
        if self.feature_mode != "shape_texture":
            return shape
        return torch.cat([texture, shape], dim=1)

    def embed(self, images):
        if self.training:
            self._set_stage_modes()
        if torch.is_grad_enabled() and any(p.requires_grad for p in self.backbone.parameters()):
            raw = self._forward_train(images)
        else:
            # Frozen encoder, or eval under no_grad. Detach so the projection can still train.
            with torch.no_grad():
                raw = self._forward_all(images)
            raw = raw.detach()
        return F.normalize(self.proj(raw.float()), dim=1)


def freeze_for_protonet(model, unfreeze_last_stage):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.proj.parameters():
        param.requires_grad = True
    if unfreeze_last_stage:
        for param in model.backbone.stages[-1].parameters():
            param.requires_grad = True
        for param in model.backbone.norm_pre.parameters():
            param.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_train:,} / {n_total:,}")
    return n_train, n_total


def prepare_images(raw, augment):
    images = gpu_augment(cfg, raw) if augment else gpu_eval_transform(cfg, raw)
    if images.device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def split_support_query(indices, k, q, rng):
    """Disjoint support and query index lists. Query is reduced first if the class is small."""
    order = list(indices)
    rng.shuffle(order)
    n = len(order)
    if n < 2:
        raise RuntimeError(f"class has {n} training image(s); episodes need at least 2")
    q_c = min(q, n - 1)
    k_c = min(k, n - q_c)
    return order[:k_c], order[k_c:k_c + q_c]


def sample_episode(by_class, k, q, rng):
    support, query = [], []
    support_y, query_y = [], []
    for label in sorted(by_class):
        s_idx, q_idx = split_support_query(by_class[label], k, q, rng)
        support.extend(s_idx)
        query.extend(q_idx)
        support_y.extend([label] * len(s_idx))
        query_y.extend([label] * len(q_idx))
    return support, query, support_y, query_y


def protonet_logits(support_z, support_y, query_z, num_classes):
    proto = support_z.new_zeros(num_classes, support_z.size(1))
    proto.index_add_(0, support_y, support_z)
    counts = torch.bincount(support_y, minlength=num_classes).clamp(min=1).unsqueeze(1).to(support_z.dtype)
    proto = proto / counts
    dist = torch.cdist(query_z, proto, p=2).pow(2)
    return -dist, proto


def lookalike_report(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    per_species = {}
    for name in LOOKALIKE_NAMES:
        idx = cfg.CANONICAL_CLASS_TO_INDEX[name]
        mask = y_true == idx
        per_species[name] = {
            "n": int(mask.sum()),
            "correct": int((y_pred[mask] == idx).sum()) if mask.any() else 0,
        }
    pairs = Counter()
    for true_i, pred_i in zip(y_true.tolist(), y_pred.tolist()):
        if true_i == pred_i or true_i not in LOOKALIKE_INDICES:
            continue
        pairs[(cfg.CANONICAL_INDEX_TO_CLASS[true_i], cfg.CANONICAL_INDEX_TO_CLASS[pred_i])] += 1
    top_pairs = [
        {"true": true_name, "pred": pred_name, "n": int(n)}
        for (true_name, pred_name), n in pairs.most_common(8)
    ]
    return {"per_species": per_species, "top_confusions": top_pairs}


@torch.no_grad()
def encode_records(model, bank, records, device, use_amp, batch_size):
    if not records:
        return torch.empty(0, model.proj_dim, device=device), np.zeros(0, dtype=int)
    paths = [abs_path(r) for r in records]
    labels = np.array([int(r["label_idx"]) for r in records], dtype=int)
    rows = bank.rows(paths)
    chunks = []
    model.eval()
    for start in range(0, len(paths), batch_size):
        raw = bank.images[rows[start:start + batch_size]]
        images = prepare_images(raw, augment=False)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            chunks.append(model.embed(images).float())
    return torch.cat(chunks, dim=0), labels


def score_from_features(train_feats, train_labels, test_feats, test_labels, shot, seed, num_classes):
    k = None if shot == "all" else int(shot)
    shot_key = "all" if k is None else str(k)
    rng = np.random.RandomState(seed * 10007 + (0 if k is None else k))
    chosen, support_counts = sample_support_indices(train_labels.tolist(), k, rng)
    chosen_t = torch.tensor(chosen, dtype=torch.long, device=train_feats.device)
    prototypes, _counts = build_prototypes(
        train_feats[chosen_t],
        torch.tensor(train_labels[chosen], dtype=torch.long, device=train_feats.device),
        num_classes,
    )
    del _counts
    metrics = {}
    predictions = {}
    for name, feats in test_feats.items():
        pred = nearest_prototype(feats, prototypes, distance="euclidean")
        pred_np = pred.cpu().numpy()
        metrics[name] = score_predictions(test_labels[name], pred_np)
        predictions[name] = pred_np
    return shot_key, {
        "k": k,
        "n_support": int(len(chosen)),
        "support_counts": {str(c): int(n) for c, n in support_counts.items()},
        "metrics": metrics,
    }, predictions


def evaluate_shots(model, bank, train_records, test_groups, device, use_amp, batch_size, seed, shots, include_all):
    train_feats, train_labels = encode_records(model, bank, train_records, device, use_amp, batch_size)
    test_feats, test_labels = {}, {}
    for name in TEST_SPLITS:
        recs = test_groups.get(name) or []
        if not recs:
            continue
        feats, labels = encode_records(model, bank, recs, device, use_amp, batch_size)
        if len(labels) == 0:
            continue
        test_feats[name] = feats
        test_labels[name] = labels

    shot_specs = list(shots) + (["all"] if include_all else [])
    shots_out = {}
    lookalikes = {}
    for shot in shot_specs:
        shot_key, payload, predictions = score_from_features(
            train_feats, train_labels, test_feats, test_labels, shot, seed, cfg.CANONICAL_NUM_CLASSES,
        )
        shots_out[shot_key] = payload
        if "pooled_test" in predictions:
            lookalikes[shot_key] = lookalike_report(test_labels["pooled_test"], predictions["pooled_test"])
        for name, record in payload["metrics"].items():
            fnr = record["binary_FNR"]
            fnr_txt = "n/a" if fnr is None else f"{fnr:.3f}"
            print(
                f"  seed {seed}  shot={shot_key:<4}  {name:12s}  "
                f"acc={record['accuracy']:.4f}  FNR={fnr_txt}"
            )
    return shots_out, lookalikes


def validation_accuracy(model, bank, train_records, val_records, device, use_amp, batch_size):
    train_feats, train_labels = encode_records(model, bank, train_records, device, use_amp, batch_size)
    val_feats, val_labels = encode_records(model, bank, val_records, device, use_amp, batch_size)
    prototypes, _counts = build_prototypes(
        train_feats,
        torch.tensor(train_labels, dtype=torch.long, device=device),
        cfg.CANONICAL_NUM_CLASSES,
    )
    pred = nearest_prototype(val_feats, prototypes, distance="euclidean")
    y_true = torch.tensor(val_labels, dtype=torch.long, device=device)
    return float((pred == y_true).float().mean().item()), int(len(val_labels))


def run_epoch(model, bank, by_class, optimizer, scaler, device, use_amp, args, rng):
    model.train()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    for _ in range(args.episodes_per_epoch):
        support, query, support_y, query_y = sample_episode(by_class, args.k_shot, args.q_query, rng)
        raw = bank.local_images[support + query]
        images = prepare_images(raw, augment=True)
        labels_s = torch.tensor(support_y, dtype=torch.long, device=device)
        labels_q = torch.tensor(query_y, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            embedded = model.embed(images)
            n_s = len(support)
            logits, _proto = protonet_logits(embedded[:n_s], labels_s, embedded[n_s:], cfg.CANONICAL_NUM_CLASSES)
            loss = F.cross_entropy(logits / args.temperature, labels_q)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
        total_loss += float(loss.item()) * labels_q.size(0)
        total_correct += int((logits.argmax(1) == labels_q).sum().item())
        total_seen += int(labels_q.size(0))
    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


def load_split(data, seed, device):
    split_csv = ensure_split(data, seed)
    records = read_split_records(split_csv)
    for rec in records:
        rec["_abs_path"] = abs_path(rec)
    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "val"]
    test_groups = test_record_groups(records)
    needed = sorted({
        abs_path(r)
        for r in train_records + val_records + sum((test_groups[n] for n in TEST_SPLITS), [])
    })
    bank = ImageBank(cfg, needed, device, verbose=True)
    # Local row order matches train_records so episode indices gather the right images.
    # The bank itself is keyed by path; keep a stacked tensor in train-record order.
    train_rows = bank.rows([abs_path(r) for r in train_records])
    bank.local_images = bank.images[train_rows]
    by_class = defaultdict(list)
    for i, rec in enumerate(train_records):
        by_class[int(rec["label_idx"])].append(i)
    return split_csv, bank, train_records, val_records, test_groups, by_class


def train_one_seed(model, bank, train_records, val_records, by_class, device, use_amp, args, seed, save_dir):
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    groups = [{"params": [p for p in model.proj.parameters() if p.requires_grad], "lr": args.lr}]
    stage_params = [p for p in model.backbone.parameters() if p.requires_grad]
    if stage_params:
        groups.append({"params": stage_params, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=cfg.lr_cosine_minimum)
    rng = np.random.RandomState(seed)
    best_val = -1.0
    best_epoch = 0
    best_path = os.path.join(save_dir, f"seed{seed}_best.pth")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, bank, by_class, optimizer, scaler, device, use_amp, args, rng)
        va_acc, n_val = validation_accuracy(
            model, bank, train_records, val_records, device, use_amp, args.batch_size,
        )
        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_acc": va_acc,
            "seconds": time.time() - t0,
        })
        print(
            f"Seed {seed}  epoch {epoch:02d}/{args.epochs}  "
            f"episode {tr_loss:.3f}/{tr_acc:.3f}  val {va_acc:.3f} (n={n_val})  "
            f"{history[-1]['seconds']:.0f}s"
        )
        if va_acc > best_val:
            best_val = va_acc
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_acc": va_acc}, best_path)

    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return best_epoch, best_val, history, best_path


def aggregate_lookalikes(per_seed, shot_key):
    totals = {name: {"n": 0, "correct": 0} for name in LOOKALIKE_NAMES}
    pairs = Counter()
    for seed_rec in per_seed:
        block = seed_rec.get("lookalikes", {}).get(shot_key)
        if not block:
            continue
        for name, counts in block["per_species"].items():
            totals[name]["n"] += counts["n"]
            totals[name]["correct"] += counts["correct"]
        for pair in block["top_confusions"]:
            pairs[(pair["true"], pair["pred"])] += pair["n"]
    species = []
    for name in LOOKALIKE_NAMES:
        n = totals[name]["n"]
        correct = totals[name]["correct"]
        species.append({
            "species": name,
            "n": n,
            "correct": correct,
            "recall": (correct / n) if n else None,
        })
    top_pairs = [
        {"true": a, "pred": b, "n": int(n)}
        for (a, b), n in pairs.most_common(8)
    ]
    return {"per_species": species, "top_confusions": top_pairs}


def summarise(per_seed, shot_keys, test_names):
    summary = {}
    for shot_key in shot_keys:
        summary[shot_key] = {}
        for name in test_names:
            accs, fnrs = [], []
            for seed_rec in per_seed:
                metrics = seed_rec["shots"].get(shot_key, {}).get("metrics", {})
                if name not in metrics:
                    continue
                accs.append(metrics[name]["accuracy"])
                if metrics[name].get("binary_FNR") is not None:
                    fnrs.append(metrics[name]["binary_FNR"])
            if not accs:
                continue
            mean_acc, std_acc = mean_std(accs)
            summary[shot_key][name] = {
                "accuracy_mean": mean_acc,
                "accuracy_std": std_acc,
                "binary_FNR_mean": mean_std(fnrs)[0] if fnrs else None,
                "binary_FNR_std": mean_std(fnrs)[1] if fnrs else None,
            }
    return summary


def main():
    args = parse_args()
    backbone = cfg.BACKBONE_PRESETS.get(args.backbone, args.backbone)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    use_amp = bool(cfg.use_amp and device.type == "cuda" and not args.no_amp)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(
        f"Episodic ProtoNet  data={args.data}  features={args.features}  "
        f"texture_stage={args.texture_stage}  proj={args.proj_dim}  "
        f"last_stage={args.unfreeze_last_stage}  "
        f"episodes={args.episodes_per_epoch}x{args.epochs}  K={args.k_shot} Q={args.q_query}"
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_tag = args.data.replace("+", "_")
    stage_tag = "laststage" if args.unfreeze_last_stage else "frozen"
    save_dir = os.path.join(
        cfg.MODELS_DIR,
        f"{stamp}_protonet_episodic_{data_tag}_{args.features}_{stage_tag}",
    )
    os.makedirs(save_dir, exist_ok=True)

    per_seed = []
    t0 = time.time()
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        print(f"\n=== seed {seed} ===")
        split_csv, bank, train_records, val_records, test_groups, by_class = load_split(args.data, seed, device)
        print(f"Split {split_csv}  train={len(train_records)}  val={len(val_records)}")
        model = EpisodicProtoNet(
            backbone, args.proj_dim, args.features, args.texture_stage, pretrained=True,
        )
        n_trainable, n_total = freeze_for_protonet(model, args.unfreeze_last_stage)
        model.to(device)
        if device.type == "cuda":
            model.to(memory_format=torch.channels_last)
        best_epoch, best_val, history, best_path = train_one_seed(
            model, bank, train_records, val_records, by_class, device, use_amp, args, seed, save_dir,
        )
        print(f"Best epoch {best_epoch}  val acc {best_val:.4f}")
        shots_out, lookalikes = evaluate_shots(
            model, bank, train_records, test_groups, device, use_amp, args.batch_size,
            seed, args.shots, include_all=not args.no_include_all,
        )
        per_seed.append({
            "seed": seed,
            "n_train": len(train_records),
            "n_val": len(val_records),
            "best_epoch": best_epoch,
            "best_val_acc": best_val,
            "trainable_params": int(n_trainable),
            "total_params": int(n_total),
            "checkpoint": best_path,
            "history": history,
            "shots": shots_out,
            "lookalikes": lookalikes,
        })
        del model, bank
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Keep a recoverable copy if a later seed fails.
        with open(os.path.join(save_dir, "results.partial.json"), "w", encoding="utf-8") as handle:
            json.dump({"per_seed": per_seed}, handle)

    minutes = (time.time() - t0) / 60.0
    shot_keys = [str(s) for s in args.shots] + ([] if args.no_include_all else ["all"])
    test_names = [
        name for name in TEST_SPLITS
        if any(name in seed_rec["shots"][shot_keys[0]]["metrics"] for seed_rec in per_seed)
    ]
    summary = summarise(per_seed, shot_keys, test_names)
    lookalike_summary = {key: aggregate_lookalikes(per_seed, key) for key in ("5", "all") if key in shot_keys}

    results = {
        "run": os.path.basename(save_dir),
        "timestamp": stamp,
        "method": "protonet_episodic",
        "data": args.data,
        "backbone_key": args.backbone,
        "backbone": backbone,
        "features": args.features,
        "texture_stage": args.texture_stage if args.features == "shape_texture" else None,
        "proj_dim": args.proj_dim,
        "unfreeze_last_stage": bool(args.unfreeze_last_stage),
        "epochs": args.epochs,
        "episodes_per_epoch": args.episodes_per_epoch,
        "k_shot": args.k_shot,
        "q_query": args.q_query,
        "lr": args.lr,
        "backbone_lr": args.backbone_lr if args.unfreeze_last_stage else None,
        "temperature": args.temperature,
        "seeds": args.seeds,
        "shots": args.shots,
        "include_all": not args.no_include_all,
        "minutes": minutes,
        "per_seed": per_seed,
        "summary": summary,
        "lookalike_summary": lookalike_summary,
        "prior_pooled_accuracy": {
            "frozen_protonet_all_support": 0.423,
            "frozen_linear_head": 0.595,
            "last_stage_linear_head": 0.651,
            "note": "Means from the earlier frozen ProtoNet run and the linear-head papers runs on this same pooled collection. Not recomputed here.",
        },
    }
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    title = f"Episodic ProtoNet {args.features} {stage_tag} (mean +/- std over seeds)"
    print_summary_table(per_seed, shot_keys, test_names, title=title)
    if "all" in summary and "pooled_test" in summary["all"]:
        got = summary["all"]["pooled_test"]["accuracy_mean"]
        print(
            f"\nAll-support pooled {100 * got:.1f}%  |  "
            f"frozen ProtoNet 42.3%  |  linear head 59.5%  |  last-stage linear 65.1%"
        )
    print(f"\nWall time: {minutes:.1f} min")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
