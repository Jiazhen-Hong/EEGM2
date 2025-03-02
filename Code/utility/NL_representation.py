"""
Created on Tue Nov 14 18:12:39 2024
@author: jiazhen@emotiv.com
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
import numpy as np


def count_parameters_up_to(model, target_layer_name):
    total_params = 0
    found_layer = False

    for name, module in model.named_modules():
        if hasattr(module, "parameters"):  
            for param in module.parameters(recurse=False):  
                total_params += param.numel()
        
        if target_layer_name in name:  
            found_layer = True
            break  

    if found_layer:
        print(f"Corrected: Total trainable parameters up to {target_layer_name}: {total_params}")
    else:
        print(f"Warning: Layer {target_layer_name} not found in the model!")

    return total_params

def locate_submodule_by_name(model: torch.nn.Module, submodule_name: str):
    names = submodule_name.split(".")
    current_layer = model
    for n in names:
        if not hasattr(current_layer, n):
            raise ValueError(f"Submodule '{submodule_name}' not found in model.")
        current_layer = getattr(current_layer, n)
    return current_layer


def extract_pooled_representation(
    model: torch.nn.Module, 
    loader: DataLoader,
    target_submodule_name: str,
    device: str = "cuda"
):
    if isinstance(model, torch.nn.DataParallel):
        core_model = model.module
    else:
        core_model = model

    target_module = locate_submodule_by_name(core_model, target_submodule_name)
    captured_features = []

    def forward_hook(module, inp, out):
        captured_features.append(out)

    hook_handle = target_module.register_forward_hook(forward_hook)

    all_pooled = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            captured_features.clear()

            _ = model(X_batch)

            if len(captured_features) == 0:
                raise RuntimeError(
                    f"No features were captured from submodule '{target_submodule_name}'."
                )

            feats = captured_features[0]  

            if feats.dim() == 3:
                feats = compute_time_quantile_stats(feats)
                
            all_pooled.append(feats.cpu())
            all_labels.append(y_batch)

    hook_handle.remove()

    pooled_features = torch.cat(all_pooled, dim=0)  # [N, C]
    all_labels = torch.cat(all_labels, dim=0)       # [N]

    return pooled_features, all_labels


def compute_time_quantile_stats(features: torch.Tensor, q_list=[0.05, 0.25, 0.5, 0.75, 0.95]):
    min_ = features.min(dim=2).values    # [B, T]
    max_ = features.max(dim=2).values    # [B, T]
    mean_ = features.mean(dim=2)         # [B, T]
    std_ = features.std(dim=2)           # [B, T]
    quantiles = torch.quantile(
        features,
        torch.tensor(q_list, device=features.device),
        dim=2
    )  
    
    quantiles = quantiles.permute(1, 2, 0)  # [B, T, len(q_list)]
    quantiles = quantiles.reshape(features.size(0), -1)  # [B, T*len(q_list)]

    return torch.cat([min_, max_, mean_, std_, quantiles], dim=1)

