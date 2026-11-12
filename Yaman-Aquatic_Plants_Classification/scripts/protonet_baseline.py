"""Frozen-feature Prototypical Network baseline (few-shot).

Classifies by nearest class prototype in ConvNeXt-V2 feature space. A prototype is the
mean embedding of K support images sampled from the training split of a seed. This is the
Snell et al. Prototypical Network readout without episodic training yet.

Examples (from Yaman-Aquatic_Plants_Classification/):

    python scripts/protonet_baseline.py --data lab+web+field --seeds 8 9 10 11 12
    python scripts/protonet_baseline.py --data lab+web+field --shots 1 5 10 --seeds 9
    python scripts/protonet_baseline.py --data lab+web+field --aug-views 8   # support-time aug

Writes one results.json under runs/models/ and prints a mean+/-std table for the weekly report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as cfg
from scripts.build_splits import ensure_split
from src.data import read_split_records, test_record_groups
from src.gpu_data import ImageBank, gpu_augment, gpu_eval_transform
from src.metrics import binary_fnr
from src.model import create_model_from_config, freeze_backbone


TEST_SPLITS = ("lab_test", "web_test", "field_test", "pooled_test")


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen-feature Prototypical Network baseline")
    parser.add_argument("--data", choices=list(cfg.DATA_CONFIGS), default="lab+web+field")
    parser.add_argument("--backbone", default="convnextv2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[8, 9, 10, 11, 12])
    parser.add_argument(
        "--shots", type=int, nargs="+", default=[1, 5, 10],
        help="K support images per class; use 0 or omit 'all' via --include-all",
    )
    parser.add_argument(
        "--include-all", action="store_true", default=True,
        help="also evaluate prototypes built from every training image (default on)",
    )
    parser.add_argument("--no-include-all", action="store_false", dest="include_all")
    parser.add_argument(
        "--aug-views", type=int, default=0,
        help="if >0, average this many augmented embeddings per support image",
    )
    parser.add_argument(
        "--distance", choices=["euclidean", "cosine"], default="euclidean",
        help="euclidean is the classic ProtoNet distance; cosine uses L2-normalised features",
    )
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def abs_path(record):
    path = record["path"]
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(cfg.REPO_ROOT, path))


@torch.no_grad()
def encode_uint8(model, cfg_obj, images_uint8, device, use_amp, augment=False):
    """Encode a uint8 bank batch [B,3,S,S] under eval or training augmentation."""
    if augment:
        images = gpu_augment(cfg_obj, images_uint8)
    else:
        images = gpu_eval_transform(cfg_obj, images_uint8)
    images = images.to(device, non_blocking=True)
    if device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    with torch.autocast(device_type=device.type, enabled=use_amp):
        feats = model.backbone(images).float()
    return feats


@torch.no_grad()
def encode_paths(model, bank, paths, device, use_amp, batch_size, aug_views=0):
    """Feature matrix [N, D] for absolute paths, optionally averaging aug_views supports."""
    rows = bank.rows(paths)
    feats = []
    for start in range(0, len(paths), batch_size):
        idx = rows[start:start + batch_size]
        raw = bank.images[idx]
        if aug_views and aug_views > 0:
            acc = None
            for _ in range(aug_views):
                view = encode_uint8(model, cfg, raw, device, use_amp, augment=True)
                acc = view if acc is None else acc + view
            batch_feats = acc / float(aug_views)
        else:
            batch_feats = encode_uint8(model, cfg, raw, device, use_amp, augment=False)
        feats.append(batch_feats)
    return torch.cat(feats, dim=0)


def sample_support_indices(labels, k, rng):
    """Per-class indices into the train list. k=None means use every image of the class."""
    by_class = defaultdict(list)
    for i, label in enumerate(labels):
        by_class[int(label)].append(i)
    chosen = []
    support_counts = {}
    for label in sorted(by_class):
        idxs = list(by_class[label])
        rng.shuffle(idxs)
        if k is None:
            take = idxs
        else:
            take = idxs[: min(k, len(idxs))]
        support_counts[label] = len(take)
        chosen.extend(take)
    return chosen, support_counts


def build_prototypes(feats, labels, num_classes):
    """Mean feature per class. Missing classes get a NaN prototype (never nearest)."""
    dim = feats.size(1)
    prototypes = feats.new_full((num_classes, dim), float("nan"))
    counts = torch.zeros(num_classes, dtype=torch.long, device=feats.device)
    for label in labels.unique().tolist():
        mask = labels == label
        prototypes[label] = feats[mask].mean(dim=0)
        counts[label] = int(mask.sum().item())
    return prototypes, counts


def nearest_prototype(query_feats, prototypes, distance="euclidean"):
    valid = ~torch.isnan(prototypes[:, 0])
    proto = prototypes.clone()
    proto[~valid] = 0.0
    if distance == "cosine":
        q = F.normalize(query_feats, dim=1)
        p = F.normalize(proto, dim=1)
        scores = q @ p.T  # higher is better
        scores[:, ~valid] = -1e9
        return scores.argmax(dim=1)
    # Squared Euclidean: ||q - p||^2 = ||q||^2 + ||p||^2 - 2 q·p ; drop ||q||^2
    q2 = (query_feats ** 2).sum(dim=1, keepdim=True)
    p2 = (proto ** 2).sum(dim=1).unsqueeze(0)
    dist = q2 + p2 - 2.0 * (query_feats @ proto.T)
    dist[:, ~valid] = 1e9
    return dist.argmin(dim=1)


def score_predictions(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    binary = binary_fnr(y_true, y_pred, cfg.INVASIVE_INDICES)
    return {
        "accuracy": acc,
        "n": int(len(y_true)),
        "binary_FNR": binary["FNR"],
        "confusion": {k: binary[k] for k in ("TP", "TN", "FP", "FN")},
    }


def mean_std(values):
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))


def print_summary_table(all_seed_results, shots_keys, test_names):
    print("\n=== ProtoNet frozen baseline (mean +/- std over seeds) ===")
    header = f"{'shot':>8}  " + "  ".join(f"{name:>14}" for name in test_names) + f"  {'inv_FNR':>10}"
    print(header)
    for shot_key in shots_keys:
        cells = []
        fnrs = []
        for name in test_names:
            accs = []
            for seed_rec in all_seed_results:
                metrics = seed_rec["shots"].get(shot_key, {}).get("metrics", {})
                if name in metrics:
                    accs.append(metrics[name]["accuracy"])
                    if metrics[name].get("binary_FNR") is not None:
                        fnrs.append(metrics[name]["binary_FNR"])
            if accs:
                m, s = mean_std(accs)
                cells.append(f"{100 * m:5.1f}% +/- {100 * s:4.1f}")
            else:
                cells.append(f"{'n/a':>14}")
        if fnrs:
            # Prefer pooled_test FNR when present; else average whatever was collected last.
            pooled_fnrs = []
            for seed_rec in all_seed_results:
                metrics = seed_rec["shots"].get(shot_key, {}).get("metrics", {})
                if "pooled_test" in metrics and metrics["pooled_test"].get("binary_FNR") is not None:
                    pooled_fnrs.append(metrics["pooled_test"]["binary_FNR"])
            use = pooled_fnrs or fnrs
            m, s = mean_std(use)
            fnr_cell = f"{100 * m:5.1f}%+/-{100 * s:4.1f}"
        else:
            fnr_cell = f"{'n/a':>10}"
        print(f"{shot_key:>8}  " + "  ".join(f"{c:>14}" for c in cells) + f"  {fnr_cell:>10}")


def run_seed(model, bank, train_records, test_groups, device, use_amp, args, seed):
    train_paths = [abs_path(r) for r in train_records]
    train_labels = [int(r["label_idx"]) for r in train_records]
    train_label_t = torch.tensor(train_labels, dtype=torch.long, device=device)

    # Cache deterministic train features once; re-encode only when using support aug.
    train_feats_eval = encode_paths(
        model, bank, train_paths, device, use_amp, args.batch_size, aug_views=0,
    )

    test_feats = {}
    test_labels = {}
    for name, recs in test_groups.items():
        if not recs:
            continue
        paths = [abs_path(r) for r in recs]
        test_feats[name] = encode_paths(model, bank, paths, device, use_amp, args.batch_size, aug_views=0)
        test_labels[name] = np.array([int(r["label_idx"]) for r in recs], dtype=int)

    shot_specs = list(args.shots)
    if args.include_all:
        shot_specs = shot_specs + ["all"]

    seed_out = {
        "seed": seed,
        "n_train": len(train_records),
        "shots": {},
    }

    for shot in shot_specs:
        k = None if shot == "all" else int(shot)
        shot_key = "all" if k is None else f"{k}"
        rng = np.random.RandomState(seed * 10007 + (0 if k is None else k))
        chosen, support_counts = sample_support_indices(train_labels, k, rng)

        if args.aug_views > 0:
            support_paths = [train_paths[i] for i in chosen]
            support_feats = encode_paths(
                model, bank, support_paths, device, use_amp, args.batch_size, aug_views=args.aug_views,
            )
            support_y = train_label_t[chosen]
        else:
            support_feats = train_feats_eval[chosen]
            support_y = train_label_t[chosen]

        prototypes, counts = build_prototypes(support_feats, support_y, cfg.CANONICAL_NUM_CLASSES)
        metrics = {}
        for name, feats in test_feats.items():
            pred = nearest_prototype(feats, prototypes, distance=args.distance)
            metrics[name] = score_predictions(test_labels[name], pred.cpu().numpy())
            print(
                f"  seed {seed}  shot={shot_key:<4}  {name:12s}  "
                f"acc={metrics[name]['accuracy']:.4f}  "
                f"FNR={metrics[name]['binary_FNR']}"
            )

        seed_out["shots"][shot_key] = {
            "k": k,
            "n_support": int(len(chosen)),
            "support_counts": {str(c): int(n) for c, n in support_counts.items()},
            "prototype_counts": {str(i): int(counts[i].item()) for i in range(cfg.CANONICAL_NUM_CLASSES)},
            "metrics": metrics,
        }
    return seed_out


def main():
    args = parse_args()
    backbone = cfg.BACKBONE_PRESETS.get(args.backbone, args.backbone)
    cfg.train_timm_model_name = backbone
    cfg.classification_head_type = "linear"  # unused; we only need the backbone

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    use_amp = bool(cfg.use_amp and device.type == "cuda" and not args.no_amp)
    print(f"Device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(
        f"ProtoNet frozen  data={args.data}  backbone={args.backbone}  "
        f"shots={args.shots}{' +all' if args.include_all else ''}  "
        f"aug_views={args.aug_views}  distance={args.distance}"
    )

    model = create_model_from_config(cfg, pretrained=True)
    freeze_backbone(model, unfreeze_last_stage=False)
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    model.eval()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_tag = args.data.replace("+", "_")
    aug_tag = f"_aug{args.aug_views}" if args.aug_views > 0 else "_noaug"
    save_dir = os.path.join(
        cfg.MODELS_DIR,
        f"{stamp}_protonet_frozen_{data_tag}_{args.backbone}{aug_tag}_{args.distance}",
    )
    os.makedirs(save_dir, exist_ok=True)

    all_seed_results = []
    t0 = time.time()
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        split_csv = ensure_split(args.data, seed)
        records = read_split_records(split_csv)
        for rec in records:
            rec["_abs_path"] = abs_path(rec)
        train_records = [r for r in records if r["split"] == "train"]
        test_groups = test_record_groups(records)
        needed = sorted({
            abs_path(r) for r in train_records + sum((test_groups[n] for n in TEST_SPLITS), [])
        })
        print(f"\nSeed {seed}  split={split_csv}  train={len(train_records)}  bank={len(needed)}")
        bank = ImageBank(cfg, needed, device, verbose=True)
        seed_out = run_seed(model, bank, train_records, test_groups, device, use_amp, args, seed)
        all_seed_results.append(seed_out)
        del bank
        if device.type == "cuda":
            torch.cuda.empty_cache()

    minutes = (time.time() - t0) / 60.0
    shot_keys = [str(s) for s in args.shots] + (["all"] if args.include_all else [])
    # Keep only splits that appeared for every seed.
    test_names = []
    for name in TEST_SPLITS:
        if all(
            name in seed_rec["shots"][shot_keys[0]]["metrics"]
            for seed_rec in all_seed_results
            if shot_keys[0] in seed_rec["shots"]
        ):
            # include if any seed has it with n>0
            if any(
                seed_rec["shots"][shot_keys[0]]["metrics"].get(name, {}).get("n", 0) > 0
                for seed_rec in all_seed_results
            ):
                test_names.append(name)

    summary = {}
    for shot_key in shot_keys:
        summary[shot_key] = {}
        for name in test_names:
            accs = [s["shots"][shot_key]["metrics"][name]["accuracy"] for s in all_seed_results]
            fnrs = [
                s["shots"][shot_key]["metrics"][name]["binary_FNR"]
                for s in all_seed_results
                if s["shots"][shot_key]["metrics"][name].get("binary_FNR") is not None
            ]
            m, sd = mean_std(accs)
            summary[shot_key][name] = {
                "accuracy_mean": m,
                "accuracy_std": sd,
                "binary_FNR_mean": mean_std(fnrs)[0] if fnrs else None,
                "binary_FNR_std": mean_std(fnrs)[1] if fnrs else None,
            }

    results = {
        "run": os.path.basename(save_dir),
        "timestamp": stamp,
        "method": "protonet_frozen",
        "data": args.data,
        "backbone_key": args.backbone,
        "backbone": backbone,
        "seeds": args.seeds,
        "shots": args.shots,
        "include_all": bool(args.include_all),
        "aug_views": int(args.aug_views),
        "distance": args.distance,
        "minutes": minutes,
        "per_seed": all_seed_results,
        "summary": summary,
    }
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print_summary_table(all_seed_results, shot_keys, test_names)
    print(f"\nWall time: {minutes:.1f} min")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
