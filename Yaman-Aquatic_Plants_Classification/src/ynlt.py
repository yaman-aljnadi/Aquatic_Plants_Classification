"""YNLT: confidence-gated high-resolution second look. No plotting."""

import torch
import torch.nn.functional as F

import config as cfg
from src import data as data_utils
from src import patching as patching_utils


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


@torch.no_grad()
def predict_logits(model, image, device):
    tensor = data_utils.image_to_tensor(cfg, image, device=device)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0]
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    pred_idx = int(sorted_idx[0].item())
    confidence = float(sorted_probs[0].item())
    margin = float(sorted_probs[0].item() - sorted_probs[1].item())
    entropy = float(-(probs * torch.log(probs + 1e-10)).sum().item())
    return {
        "predicted_class_idx": pred_idx,
        "predicted_class_name": cfg.CANONICAL_CLASSNAMES_LIST[pred_idx],
        "confidence": confidence,
        "margin": margin,
        "entropy": entropy,
        "confusion_level": confusion_level(confidence, margin, entropy),
        "probs": probs,
    }


@torch.no_grad()
def review_patches(model, image, device, label=None):
    img = data_utils.to_uint8_img(cfg, data_utils.image_to_tensor(cfg, image, device=device))
    loader = patching_utils.get_dataloader(image=img, label=label, batch_size=1, shuffle=False)
    scores = {}
    max_c = cfg.margin_weight + cfg.entropy_weight + cfg.confidence_weight
    for patch, _ in loader:
        pred = predict_logits(model, patch, device)
        name = pred["predicted_class_name"]
        weight = (max_c - pred["confusion_level"]) / max_c
        scores[name] = scores.get(name, 1.0) + weight
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
    }


@torch.no_grad()
def predict_with_ynlt(model, image, device, true_label=None):
    first = predict_logits(model, image, device)
    first["used_ynlt"] = False
    if cfg.do_critical_review and is_critical(first["confusion_level"], first["predicted_class_idx"]):
        reviewed = review_patches(model, image, device, label=true_label)
        if reviewed is not None:
            first["used_ynlt"] = True
            first["initial_prediction_idx"] = first["predicted_class_idx"]
            first["predicted_class_idx"] = reviewed["final_prediction_class_index"]
            first["predicted_class_name"] = reviewed["final_prediction_class_name"]
    return first


@torch.no_grad()
def evaluate_with_ynlt(model, loader, device):
    model.eval()
    y_true, y_pred, n_review = [], [], 0
    for images, labels in loader:
        if images.ndim == 3:
            images = images.unsqueeze(0)
            labels = labels.unsqueeze(0)
        for i in range(images.size(0)):
            true_idx = int(labels[i].item())
            pred = predict_with_ynlt(model, images[i], device, true_label=true_idx)
            y_true.append(true_idx)
            y_pred.append(pred["predicted_class_idx"])
            n_review += int(pred["used_ynlt"])
    return y_true, y_pred, n_review
