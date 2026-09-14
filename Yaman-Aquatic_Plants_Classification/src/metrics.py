"""Accuracy and invasive-vs-native false negative rate."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
        y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def binary_fnr(y_true, y_pred, invasive_indices):
    true_bin = np.isin(y_true, invasive_indices).astype(int)
    pred_bin = np.isin(y_pred, invasive_indices).astype(int)
    cm = confusion_matrix(true_bin, pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    denom = tp + fn
    fnr = float(fn / denom) if denom else None
    return {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn), "FNR": fnr}


def evaluate_loader(model, loader, cfg, device):
    y_true, y_pred = collect_predictions(model, loader, device)
    acc = float(accuracy_score(y_true, y_pred))
    binary = binary_fnr(y_true, y_pred, cfg.INVASIVE_INDICES)
    return acc, binary
