"""Grad-CAM overlays for one saved checkpoint.

    python scripts/gradcam.py runs/models/latest_pooled_20261009/<run>/best.pth --seed 9

Saves heatmaps under Reports/heatmaps/. The map is the last ConvNeXt stage,
weighted by how strongly each channel pushed the predicted class.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config as cfg
from src.model import load_model


def encoder(model):
    return model.backbone


def load_rows(seed):
    path = cfg.split_csv_path("lab+web+field", seed)
    with open(path, encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    for row in rows:
        row["label_idx"] = int(row["label_idx"])
        row["abs"] = os.path.normpath(os.path.join(cfg.REPO_ROOT, row["path"]))
    return rows


def prepare(path, device):
    image = Image.open(path).convert("RGB")
    tensor = cfg.basic_tf(image).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    return image, tensor


def grad_cam(model, layer, tensor):
    """Map for the class the model actually predicts."""
    captured = {}

    def forward_hook(_module, _inputs, output):
        captured["act"] = output

    def backward_hook(_module, _grad_in, grad_out):
        captured["grad"] = grad_out[0]

    handles = [
        layer.register_forward_hook(forward_hook),
        layer.register_full_backward_hook(backward_hook),
    ]
    try:
        logits = model(tensor)
        pred = int(logits.argmax(1).item())
        score = logits[0, pred]
        model.zero_grad(set_to_none=True)
        score.backward()
    finally:
        for handle in handles:
            handle.remove()
    activations = captured["act"]
    gradients = captured["grad"]
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    probs = torch.softmax(logits.detach(), dim=1)[0].cpu()
    return cam, pred, float(probs.max())


def overlay(image, cam):
    import matplotlib.pyplot as plt

    resized = image.resize((cam.shape[1], cam.shape[0]), Image.BILINEAR)
    base = np.asarray(resized).astype(np.float32) / 255.0
    heat = plt.get_cmap("jet")(cam)[..., :3]
    mixed = (0.55 * base + 0.45 * heat).clip(0, 1)
    return (mixed * 255).astype(np.uint8)


def save_pair(image, cam, path):
    """Original photograph beside the heatmap, both at the network input size."""
    heat = Image.fromarray(overlay(image, cam))
    photo = image.resize(heat.size, Image.BILINEAR)
    canvas = Image.new("RGB", (heat.size[0] * 2, heat.size[1]))
    canvas.paste(photo, (0, 0))
    canvas.paste(heat, (heat.size[0], 0))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--out", default=os.path.join(cfg.REPO_ROOT, "Reports", "heatmaps"))
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint).to(device).eval()
    layer = encoder(model).stages[-1]
    rows = load_rows(args.seed)

    os.makedirs(args.out, exist_ok=True)
    n_correct = 0
    for index, row in enumerate(rows, start=1):
        image, tensor = prepare(row["abs"], device)
        cam, pred, confidence = grad_cam(model, layer, tensor)
        correct = pred == row["label_idx"]
        n_correct += int(correct)
        mark = "correct" if correct else "wrong"
        pred_name = cfg.CANONICAL_INDEX_TO_CLASS[pred].replace(" ", "_")
        true_name = row["label"].replace(" ", "_")
        stem = f"{index:03d}_{row['source']}_{mark}_true-{true_name}_pred-{pred_name}_{confidence:.0%}"
        folder = os.path.join(args.out, mark)
        os.makedirs(folder, exist_ok=True)
        save_pair(image, cam, os.path.join(folder, stem + ".png"))
        if index % 25 == 0 or index == len(rows):
            print(f"  {index}/{len(rows)}", flush=True)
    print(f"Test images {len(rows)}, correct {n_correct} ({n_correct / max(len(rows), 1):.1%})")
    print(f"Wrote {len(rows)} heatmaps under {args.out}")


if __name__ == "__main__":
    main()
