"""YNLT: confidence-gated high-resolution second look. No plotting.

Two patch sources are supported, selected with ``cfg.ynlt_patch_source``:

``original``
    Re-open the source image and cut patches from the full-resolution photograph.
    This is what the paper describes.

``input224``
    Cut patches from the 224x224 network input. This is what the archive notebooks
    (``cnn_utils/predict.py::critical_review``) did, so it reproduces Table 1.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

import config as cfg
from src import data as data_utils
from src import patching as patching_utils

PATCH_SOURCES = ("original", "input224")


def _max_confusion():
    return cfg.margin_weight + cfg.entropy_weight + cfg.confidence_weight


def confusion_level(confidence, margin, entropy):
    level = 0
    if margin < cfg.margin_threshold:
        level += cfg.margin_weight
    if entropy > cfg.entropy_threshold:
        level += cfg.entropy_weight
    if confidence < cfg.confidence_threshold:
        level += cfg.confidence_weight
    return level


def is_critical(level, predicted_idx):
    if level < cfg.critical_confusion_level:
        return False
    if cfg.ignore_invasive_predictions and predicted_idx in cfg.INVASIVE_INDICES:
        return False
    return True


def _diagnostics(logits):
    """Per-row confidence, margin, entropy and confusion count for a batch of logits."""
    probs = F.softmax(logits, dim=1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)
    confidence = sorted_probs[:, 0]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
    level = (
        (margin < cfg.margin_threshold).long() * cfg.margin_weight
        + (entropy > cfg.entropy_threshold).long() * cfg.entropy_weight
        + (confidence < cfg.confidence_threshold).long() * cfg.confidence_weight
    )
    return probs, sorted_idx, confidence, margin, entropy, level


@torch.no_grad()
def predict_logits(model, image, device):
    tensor = data_utils.image_to_tensor(cfg, image, device=device)
    probs, sorted_idx, confidence, margin, entropy, level = _diagnostics(model(tensor))
    pred_idx = int(sorted_idx[0, 0].item())
    return {
        "predicted_class_idx": pred_idx,
        "predicted_class_name": cfg.CANONICAL_CLASSNAMES_LIST[pred_idx],
        "confidence": float(confidence[0].item()),
        "margin": float(margin[0].item()),
        "entropy": float(entropy[0].item()),
        "confusion_level": int(level[0].item()),
        "probs": probs[0],
    }


@torch.no_grad()
def review_patches(model, patch_image, device, label=None, batch_size=None):
    """Weighted vote over multi-scale patches of ``patch_image`` (a uint8 HxWx3 array)."""
    batch_size = int(batch_size or getattr(cfg, "ynlt_patch_batch_size", 16))
    loader = patching_utils.get_dataloader(
        image=patch_image, label=label, batch_size=batch_size, shuffle=False
    )
    scores = {}
    n_patches = 0
    max_c = _max_confusion()
    for patches, _ in loader:
        patches = patches.to(device, non_blocking=True)
        _, sorted_idx, _, _, _, level = _diagnostics(model(patches))
        for row in range(patches.size(0)):
            name = cfg.CANONICAL_CLASSNAMES_LIST[int(sorted_idx[row, 0].item())]
            weight = (max_c - int(level[row].item())) / max_c
            scores[name] = scores.get(name, 1.0) + weight
            n_patches += 1
    if not scores:
        return None
    max_score = max(scores.values())
    winners = [name for name, value in scores.items() if value == max_score]
    final_name = winners[0]
    if len(winners) > 1:
        for name in winners:
            if name in cfg.INVASIVE_SPECIES_NAMES:
                final_name = name
                break
    return {
        "final_prediction_class_name": final_name,
        "final_prediction_class_index": cfg.CANONICAL_CLASS_TO_INDEX[final_name],
        "scores": scores,
        "num_patches": n_patches,
    }


def _downsampled_patch_image(image, device):
    return data_utils.to_uint8_img(cfg, data_utils.image_to_tensor(cfg, image, device=device))


@torch.no_grad()
def predict_with_ynlt(model, image, device, true_label=None, patch_image=None):
    """Single-image prediction. ``patch_image`` defaults to the 224x224 input."""
    first = predict_logits(model, image, device)
    first["used_ynlt"] = False
    first["num_patches"] = 0
    if cfg.do_critical_review and is_critical(first["confusion_level"], first["predicted_class_idx"]):
        if patch_image is None:
            patch_image = _downsampled_patch_image(image, device)
        reviewed = review_patches(model, patch_image, device, label=true_label)
        if reviewed is not None:
            first["used_ynlt"] = True
            first["num_patches"] = reviewed["num_patches"]
            first["initial_prediction_idx"] = first["predicted_class_idx"]
            first["predicted_class_idx"] = reviewed["final_prediction_class_index"]
            first["predicted_class_name"] = reviewed["final_prediction_class_name"]
    return first


def _new_stats(patch_source):
    return {
        "patch_source": patch_source,
        "y_true": [],
        "y_pred": [],
        "n_total": 0,
        "n_review": 0,
        "total_patches": 0,
        "first_pass_seconds": 0.0,
        "total_seconds": 0.0,
    }


@torch.no_grad()
def _accumulate(model, device, stats, first_pass_input, patch_image, true_idx):
    t0 = time.perf_counter()
    first = predict_logits(model, first_pass_input, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_first = time.perf_counter()

    pred_idx = first["predicted_class_idx"]
    if cfg.do_critical_review and is_critical(first["confusion_level"], pred_idx):
        reviewed = review_patches(model, patch_image, device, label=true_idx)
        if reviewed is not None:
            pred_idx = reviewed["final_prediction_class_index"]
            stats["n_review"] += 1
            stats["total_patches"] += reviewed["num_patches"]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    stats["y_true"].append(int(true_idx))
    stats["y_pred"].append(int(pred_idx))
    stats["n_total"] += 1
    stats["first_pass_seconds"] += t_first - t0
    stats["total_seconds"] += t_end - t0


def _finalize(stats):
    n = max(stats["n_total"], 1)
    stats["trigger_rate"] = stats["n_review"] / n
    stats["mean_patches_per_reviewed"] = (
        stats["total_patches"] / stats["n_review"] if stats["n_review"] else 0.0
    )
    stats["ms_per_image"] = 1000.0 * stats["total_seconds"] / n
    stats["ms_per_image_first_pass"] = 1000.0 * stats["first_pass_seconds"] / n
    return stats


@torch.no_grad()
def evaluate_with_ynlt(model, loader, device, patch_source=None):
    """Score a loader with YNLT enabled.

    Returns a stats dict with ``y_true``, ``y_pred``, trigger rate, patches per
    reviewed image, and wall time per image with and without the second look.
    """
    model.eval()
    patch_source = patch_source or getattr(cfg, "ynlt_patch_source", "input224")
    if patch_source not in PATCH_SOURCES:
        raise ValueError(f"patch_source must be one of {PATCH_SOURCES}, got {patch_source!r}")

    samples = data_utils.loader_samples(loader) if patch_source == "original" else None
    if patch_source == "original" and not samples:
        print("YNLT: dataloader exposes no file paths; falling back to patch_source='input224'.")
        patch_source = "input224"

    stats = _new_stats(patch_source)
    if patch_source == "original":
        for path, label in samples:
            image = data_utils.load_original_image(path)
            _accumulate(model, device, stats, image, np.asarray(image), label)
    else:
        for images, labels in loader:
            if images.ndim == 3:
                images = images.unsqueeze(0)
                labels = labels.unsqueeze(0)
            for i in range(images.size(0)):
                tensor = images[i]
                _accumulate(
                    model,
                    device,
                    stats,
                    tensor,
                    _downsampled_patch_image(tensor, device),
                    int(labels[i].item()),
                )
    return _finalize(stats)
