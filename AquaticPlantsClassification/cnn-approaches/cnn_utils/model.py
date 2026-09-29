# cnn_utils/model.py

"""Defines model architectures and provides factory functions for creation.

This module contains the building blocks for all models used in the project,
including custom classifier heads like GatedAttentionHead and MLPHead.

The primary user-facing function is `create_model_from_config`, which builds
a model based on the settings in a provided configuration object.

For reproducibility, the `load_model` function uses the lower-level `build_model`
factory to perfectly reconstruct a model from its saved configuration file.
"""

# Every model must 
#     1. Inherit from `torch.nn.Module`
#     2. have a self.classifier - this is the part that gets unfrozen and trained
#     3. have a self.config dictionary that contains the model configuration

# The save_model_weights function saves the model's self.config dictionary along with the model's state_dict.
# The load_model function returns a model instance with the config dictionary and the model's state_dict as 
#  saved by the save_model_weights function.

# NOTE: 
#      This module currently only supports timm_backcones with a classifier head
#      that is either a gated attention head or a linear classifier.

import os
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionHead(nn.Module):
    """
    This is a simple attention mechanism that computes a scalar attention score for each output feature.
    
    This is not a full attention mechanism like in transformers, but rather a gating mechanism based on learned weights.
    
    It is intended as a more sophisticated alternative to a simple linear classifier, that might possibly 
    generalize better to different types of unseen data. It should learn to focus on the most relevant features, 
    and suppress the noise.
    """
    def __init__(self, 
                input_dim: int, 
                output_dim: int, 
                attention_dim: int = None, 
                dropout_p: float = 0.5
                ):
        
        super(GatedAttentionHead, self).__init__()
        
        if attention_dim is None:
            attention_dim = input_dim // 4
                        
        self.attn_l1 = nn.Linear(input_dim, attention_dim)
        self.cls_l1 = nn.Linear(input_dim, attention_dim)
        self.cls_l2 = nn.Linear(attention_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
        gate_values = torch.tanh(self.attn_l1(x))
        gate_weights = torch.sigmoid(gate_values)
        
        cls_hidden = F.gelu(self.cls_l1(x))
        gated_hidden = cls_hidden * gate_weights
        
        gated_hidden_reg = self.dropout(gated_hidden)
        output_logits = self.cls_l2(gated_hidden_reg)
        return output_logits

class MLPHead(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) head with a single hidden layer.
    Designed for ablation studies against more complex heads like GatedAttentionHead.
    """
    def __init__(self,
                input_dim: int, 
                output_dim: int,
                hidden_dim: int = None,  
                dropout_p: float = 0.5
                ):
        
        super(MLPHead, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GatedAttnModel(nn.Module):
    def __init__(self, 
                backbone_model_name: str, 
                num_classes: int, 
                attention_dim: int, 
                dropout_p: float, 
                pretrained: bool = True
                ):
        
        super(GatedAttnModel, self).__init__()
        
        # The permanent record of this specific instance's parameters.
        self.config = {
            'backbone_name': backbone_model_name,
            'classifier_head_type': 'gated_attention',
            'num_classes': num_classes,
            'mlp_hidden_dim': None, # Not used by this model, but part of the universal config
            'attention_dim': attention_dim,
            'dropout_p': dropout_p
        }
        
        self.backbone = timm.create_model(backbone_model_name, pretrained=pretrained, num_classes=0)
        self.classifier = GatedAttentionHead(
            input_dim=self.backbone.num_features,
            attention_dim=self.config['attention_dim'],
            output_dim=self.config['num_classes'],
            dropout_p=self.config['dropout_p']
        )
        
    def forward(self, x):
        return self.classifier(self.backbone(x))

class MLPModel(nn.Module):
    """
    Combines a timm backbone with a custom MLPHead classifier.
    """
    def __init__(self, 
                backbone_model_name: str, 
                num_classes: int, 
                mlp_hidden_dim: int, 
                dropout_p: float, 
                pretrained: bool = True
                ):
        
        super(MLPModel, self).__init__()
        
        self.config = {
            'backbone_name': backbone_model_name,
            'classifier_head_type': 'mlp', # The key to identifying this model
            'num_classes': num_classes,
            'mlp_hidden_dim': mlp_hidden_dim,
            'attention_dim': None, # Not used by this model, but part of the universal config
            'dropout_p': dropout_p
        }
        
        self.backbone = timm.create_model(backbone_model_name, pretrained=pretrained, num_classes=0)
        self.classifier = MLPHead(
            input_dim=self.backbone.num_features,
            hidden_dim=self.config['mlp_hidden_dim'],
            output_dim=self.config['num_classes'],
            dropout_p=self.config['dropout_p']
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

class LinearBaselineModel(nn.Module):
    def __init__(self, 
                backbone_model_name: str, 
                num_classes: int, 
                pretrained: bool = True
                ):
        
        super().__init__()
        self.config = {
            'backbone_name': backbone_model_name,
            'classifier_head_type': 'linear',
            'num_classes': num_classes,
            'attention_dim': None,
            'mlp_hidden_dim': None,
            'dropout_p': None
        }
        
        self.model = timm.create_model(backbone_model_name, pretrained=pretrained, num_classes=0)
        self.classifier = nn.Linear(self.model.num_features, num_classes)

    @property
    def backbone(self):
        return self.model
        
    def forward(self, x):
        features = self.model(x)
        logits = self.classifier(features)
        return logits

# Helper functions to create models and save/load their weights

def build_model(
        backbone_name: str,
        classifier_head_type: str,
        num_classes: int,
        attention_dim: int = None,      
        mlp_hidden_dim: int = None,     
        dropout_p: float = 0.0,         
        pretrained: bool = True
        ):
    
    """Internal factory that builds a model from explicit parameters."""
    
    if classifier_head_type == 'gated_attention':
        return GatedAttnModel(
            backbone_model_name=backbone_name,
            num_classes=num_classes,
            attention_dim=attention_dim,
            dropout_p=dropout_p,
            pretrained=pretrained
        )
    elif classifier_head_type == 'linear':
        return LinearBaselineModel(
            backbone_model_name=backbone_name,
            num_classes=num_classes,
            pretrained=pretrained
        )
    elif classifier_head_type == 'mlp':
        return MLPModel(
            backbone_model_name=backbone_name,
            num_classes=num_classes,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout_p=dropout_p,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"Unsupported classifier_head_type: {classifier_head_type}")

def create_model_from_config(config, pretrained: bool = True):
    """
    User-facing factory that builds a model from a config object.
    """
    model = build_model(
        backbone_name=config.train_timm_model_name,
        classifier_head_type=config.classification_head_type,
        num_classes=config.CANONICAL_NUM_CLASSES,
        attention_dim=config.gated_attention_dim,
        mlp_hidden_dim=config.mlp_hidden_dim,
        dropout_p=config.classifier_head_dropout,
        pretrained=pretrained
    )
    
    return model

def save_model_weights(model, save_path, verbose=True):
    """
    Saves the model's state and its exact configuration.
    """
    # sanity checks
    if not hasattr(model, 'config'):
        raise AttributeError("Model must have a 'config' attribute to be saved.")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # saving it
    torch.save({
                'model_state_dict': model.state_dict(),
                'config': model.config
                }, 
                save_path)
    
    
    if verbose:
        print(f"Model weights and config saved to {save_path}")


def load_model(model_path, verbose=True):
    """
    Loads a model by using its saved config to call the universal `get_model` factory.
    This ensures perfect reconstruction regardless of the global config.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    model = build_model(**config, pretrained=False)
    
    # Now load the state dict into the correctly constructed model shell
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if verbose:
        print(f"Model loaded successfully from {model_path}")
    
    return model