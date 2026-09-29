"""Frozen-backbone classifiers used in the paper."""

import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionHead(nn.Module):
    def __init__(self, input_dim, output_dim, attention_dim=None, dropout_p=0.5):
        super().__init__()
        if attention_dim is None:
            attention_dim = input_dim // 4
        self.attn_l1 = nn.Linear(input_dim, attention_dim)
        self.cls_l1 = nn.Linear(input_dim, attention_dim)
        self.cls_l2 = nn.Linear(attention_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        gate_weights = torch.sigmoid(torch.tanh(self.attn_l1(x)))
        cls_hidden = F.gelu(self.cls_l1(x))
        return self.cls_l2(self.dropout(cls_hidden * gate_weights))


class MLPHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout_p=0.5):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class GatedAttnModel(nn.Module):
    def __init__(self, backbone_model_name, num_classes, attention_dim, dropout_p, pretrained=True):
        super().__init__()
        self.config = {
            "backbone_name": backbone_model_name,
            "classifier_head_type": "gated_attention",
            "num_classes": num_classes,
            "mlp_hidden_dim": None,
            "attention_dim": attention_dim,
            "dropout_p": dropout_p,
        }
        self.backbone = timm.create_model(backbone_model_name, pretrained=pretrained, num_classes=0)
        self.classifier = GatedAttentionHead(
            self.backbone.num_features, num_classes, attention_dim, dropout_p
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


class MLPModel(nn.Module):
    def __init__(self, backbone_model_name, num_classes, mlp_hidden_dim, dropout_p, pretrained=True):
        super().__init__()
        self.config = {
            "backbone_name": backbone_model_name,
            "classifier_head_type": "mlp",
            "num_classes": num_classes,
            "mlp_hidden_dim": mlp_hidden_dim,
            "attention_dim": None,
            "dropout_p": dropout_p,
        }
        self.backbone = timm.create_model(backbone_model_name, pretrained=pretrained, num_classes=0)
        self.classifier = MLPHead(self.backbone.num_features, num_classes, mlp_hidden_dim, dropout_p)

    def forward(self, x):
        return self.classifier(self.backbone(x))


class LinearBaselineModel(nn.Module):
    def __init__(self, backbone_model_name, num_classes, pretrained=True):
        super().__init__()
        self.config = {
            "backbone_name": backbone_model_name,
            "classifier_head_type": "linear",
            "num_classes": num_classes,
            "attention_dim": None,
            "mlp_hidden_dim": None,
            "dropout_p": None,
        }
        self.model = timm.create_model(backbone_model_name, pretrained=pretrained, num_classes=0)
        self.classifier = nn.Linear(self.model.num_features, num_classes)

    @property
    def backbone(self):
        return self.model

    def forward(self, x):
        return self.classifier(self.model(x))


def build_model(
    backbone_name,
    classifier_head_type,
    num_classes,
    attention_dim=None,
    mlp_hidden_dim=None,
    dropout_p=0.0,
    pretrained=True,
):
    if classifier_head_type == "gated_attention":
        return GatedAttnModel(backbone_name, num_classes, attention_dim, dropout_p, pretrained)
    if classifier_head_type == "linear":
        return LinearBaselineModel(backbone_name, num_classes, pretrained)
    if classifier_head_type == "mlp":
        return MLPModel(backbone_name, num_classes, mlp_hidden_dim, dropout_p, pretrained)
    raise ValueError(f"Unsupported classifier_head_type: {classifier_head_type}")


def create_model_from_config(config, pretrained=True):
    return build_model(
        backbone_name=config.train_timm_model_name,
        classifier_head_type=config.classification_head_type,
        num_classes=config.CANONICAL_NUM_CLASSES,
        attention_dim=config.gated_attention_dim,
        mlp_hidden_dim=config.mlp_hidden_dim,
        dropout_p=config.classifier_head_dropout,
        pretrained=pretrained,
    )


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_train:,} / {n_total:,}")
    return n_train, n_total


def save_model_weights(model, save_path, verbose=True):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": model.config}, save_path)
    if verbose:
        print(f"Saved {save_path}")


def load_model(model_path, verbose=True):
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    model = build_model(**checkpoint["config"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if verbose:
        print(f"Loaded {model_path}")
    return model
