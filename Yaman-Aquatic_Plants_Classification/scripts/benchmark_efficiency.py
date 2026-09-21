"""Measure the numbers the paper's efficiency subsection needs. No dataset required.

    python scripts/benchmark_efficiency.py
    python scripts/benchmark_efficiency.py --backbones convnextv2 --pretrained

Reports, per backbone: parameter counts, multiply-adds for one 224x224 forward pass,
batch-1 latency, patch-batch throughput, and how many patches the multi-scale YNLT
schedule produces at each candidate patch source. Random weights are used by default:
parameter counts, MACs and latency do not depend on the weight values, and this avoids
downloading pretrained checkpoints.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as cfg
from src.model import build_model, freeze_backbone
from src.patching import ImagePatchedDataset

PATCH_SOURCE_SIZES = {
    "224 network input (Table 1 behaviour)": (224, 224),
    "3000x4000 original photograph": (3000, 4000),
    "4000x4000 squared_reflected": (4000, 4000),
}


def count_macs(model, device):
    """Multiply-adds for one 224x224 forward pass, or None if no counter is installed."""
    example = torch.randn(1, 3, 224, 224, device=device)
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None, "fvcore not installed (pip install fvcore)"
    analysis = FlopCountAnalysis(model, example)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    # fvcore reports multiply-adds, not 2x multiply-adds.
    return int(analysis.total()), "fvcore (multiply-adds)"


@torch.no_grad()
def time_forward(model, device, batch_size, iters=50, warmup=10):
    example = torch.randn(batch_size, 3, 224, 224, device=device)
    for _ in range(warmup):
        model(example)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        model(example)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / iters


def count_patches(height, width, seed=0):
    """Patches the multi-scale schedule yields for an image of this size."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    dataset = ImagePatchedDataset(image=image)
    dataset.extract_patches_multi_scale(scales=cfg.ms_scale, overlap=cfg.ms_overlap)
    sizes = sorted({patch.shape[0] for patch in dataset.patches})
    return len(dataset), sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+", default=list(cfg.BACKBONE_PRESETS))
    parser.add_argument("--head", default="gated_attention", choices=["linear", "gated_attention", "mlp"])
    parser.add_argument("--pretrained", action="store_true", help="download real weights (not needed for cost)")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--patch-batch", type=int, default=cfg.ynlt_patch_batch_size)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device: {device_name}")
    print(f"Head: {args.head}  pretrained: {args.pretrained}  timing over {args.iters} iterations")
    print()

    print("## Patch counts per reviewed image")
    print(f"Scales {cfg.ms_scale}, overlaps {cfg.ms_overlap}.")
    print()
    print("| Patch source | Patches per image | Patch sizes (px) |")
    print("|---|---|---|")
    patch_counts = {}
    for label, (height, width) in PATCH_SOURCE_SIZES.items():
        n_patches, sizes = count_patches(height, width)
        patch_counts[label] = n_patches
        print(f"| {label} | {n_patches} | {', '.join(str(s) for s in sizes)} |")
    print()

    rows = []
    for key in args.backbones:
        name = cfg.BACKBONE_PRESETS.get(key, key)
        model = build_model(
            backbone_name=name,
            classifier_head_type=args.head,
            num_classes=cfg.CANONICAL_NUM_CLASSES,
            attention_dim=cfg.gated_attention_dim,
            mlp_hidden_dim=cfg.mlp_hidden_dim,
            dropout_p=cfg.classifier_head_dropout,
            pretrained=args.pretrained,
        )
        n_trainable, n_total = freeze_backbone(model)
        model.eval().to(device)

        macs, macs_note = count_macs(model, device)
        ms_single = time_forward(model, device, 1, iters=args.iters)
        ms_batch = time_forward(model, device, args.patch_batch, iters=max(args.iters // 2, 5))
        rows.append(
            {
                "key": key,
                "name": name,
                "total": n_total,
                "trainable": n_trainable,
                "macs": macs,
                "ms_single": ms_single,
                "ms_per_patch": ms_batch / args.patch_batch,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"## Model cost ({device_name}, {args.head} head, MACs from {macs_note})")
    print()
    print("| Backbone | timm name | Total params (M) | Trainable params | MACs @224 (G) | ms / image (batch 1) | ms / patch (batch %d) |" % args.patch_batch)
    print("|---|---|---|---|---|---|---|")
    for row in rows:
        macs_txt = "--" if row["macs"] is None else f"{row['macs'] / 1e9:.2f}"
        print(
            f"| {row['key']} | `{row['name']}` | {row['total'] / 1e6:.2f} | {row['trainable']:,} "
            f"| {macs_txt} | {row['ms_single']:.2f} | {row['ms_per_patch']:.2f} |"
        )
    print()

    print("## Cost of one reviewed image (first pass + all patches)")
    print()
    print("| Backbone | " + " | ".join(f"{label} (ms)" for label in PATCH_SOURCE_SIZES) + " |")
    print("|---|" + "---|" * len(PATCH_SOURCE_SIZES))
    for row in rows:
        cells = [
            f"{row['ms_single'] + patch_counts[label] * row['ms_per_patch']:.1f}"
            for label in PATCH_SOURCE_SIZES
        ]
        print(f"| {row['key']} | " + " | ".join(cells) + " |")
    print()
    print("Images that are not flagged cost only the batch-1 first pass. Multiply the")
    print("reviewed-image cost by the measured YNLT trigger rate for a per-dataset average.")


if __name__ == "__main__":
    main()
