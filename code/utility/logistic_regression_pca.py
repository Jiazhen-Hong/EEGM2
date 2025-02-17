
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
    """
    统计目标层之前的参数数量，避免重复累加 DataParallel 结构。
    """
    total_params = 0
    found_layer = False

    for name, module in model.named_modules():
        if hasattr(module, "parameters"):  # 确保是可训练的模块
            for param in module.parameters(recurse=False):  # 只统计当前模块的参数
                total_params += param.numel()
        
        if target_layer_name in name:  
            found_layer = True
            break  # 找到目标层后停止，避免统计后续层

    if found_layer:
        print(f"Corrected: Total trainable parameters up to {target_layer_name}: {total_params}")
    else:
        print(f"Warning: Layer {target_layer_name} not found in the model!")

    return total_params

def locate_submodule_by_name(model: torch.nn.Module, submodule_name: str):
    """
    在给定模型中，通过层名称(可能是嵌套的)查找对应子模块并返回。
    比如 "encoder.layer2.conv" 等。

    Args:
        model (torch.nn.Module): 待搜索的模型实例。
        submodule_name (str): 目标子模块名称，例如 "bottleneck"、"layer3.0.conv" 等。

    Returns:
        torch.nn.Module: 找到的子模块。

    Raises:
        ValueError: 如果没有找到对应的子模块。
    """
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
    """
    在指定子模块处挂钩 (hook)，捕获前向传播输出的特征。
    如果捕获到的特征是 [B, C, T]，则进行全局平均池化到 [B, C]。

    Args:
        model (torch.nn.Module): 预训练的模型。
        loader (DataLoader): 提供输入数据和标签的 DataLoader。
        target_submodule_name (str): 要捕获特征的子模块名称。
        device (str): "cuda" 或 "cpu"。

    Returns:
        (torch.Tensor, torch.Tensor):
            - pooled_features: 捕获并池化后的特征 (shape: [N, C]) 或 [N, D]。
            - all_labels: 对应的标签 (shape: [N])。
    """
    # 如果是 DataParallel 包裹，取出实际模型
    if isinstance(model, torch.nn.DataParallel):
        core_model = model.module
    else:
        core_model = model

    target_module = locate_submodule_by_name(core_model, target_submodule_name)

    # 存放当前 batch 的捕获特征
    captured_features = []

    def forward_hook(module, inp, out):
        # out 可能是 [B, C, T] 或其他形状
        captured_features.append(out)

    # 在目标子模块注册 hook
    hook_handle = target_module.register_forward_hook(forward_hook)

    all_pooled = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            captured_features.clear()

            # 前向传播
            _ = model(X_batch)

            if len(captured_features) == 0:
                raise RuntimeError(
                    f"No features were captured from submodule '{target_submodule_name}'."
                )

            # 取当前 batch 输出
            feats = captured_features[0]  # shape 可能是 [B, T, C]

            # 如果是 3D [B, T, C]，对 time 维 (最后一维) 做平均池化
            if feats.dim() == 3:
                #feats = feats.mean(dim=-1)  # -> [B, T]
                #feats = compute_time_stats(feats)  # -> [B, T*4]
                #feats = segment_pool(feats, num_segments=4)  # -> [B, T*num_segments] or [B, T*C*num_segments]
                feats = compute_time_quantile_stats(feats)
                #feats = compute_time_quantile_stats(feats)
                


            all_pooled.append(feats.cpu())
            all_labels.append(y_batch)

    hook_handle.remove()

    pooled_features = torch.cat(all_pooled, dim=0)  # [N, C]
    all_labels = torch.cat(all_labels, dim=0)       # [N]

    return pooled_features, all_labels


def compute_time_stats(features: torch.Tensor):
    """
    features: [B, C, T]
    返回:  [B, C*4] (比如拼接 mean, std, max, min)
    """
    mean_ = features.mean(dim=-1)     # [B, C]
    std_  = features.std(dim=-1)      # [B, C]
    max_  = features.max(dim=-1).values  # [B, C]
    min_  = features.min(dim=-1).values  # [B, C]
    # 拼接在特征维度上: [B, 4*C]
    combined = torch.cat([mean_, std_, max_, min_], dim=1)
    return combined

def segment_pool(features: torch.Tensor, num_segments=5):
    """
    features: [B, C, T]
    将时间轴 T 分成 num_segments 段，然后每段做 mean 或者 mean+std
    
    返回 shape: [B, C * num_segments] (若只用 mean)
    或 [B, C * 2 * num_segments] (若拼接 mean+std)
    """
    B, C, T = features.shape
    segment_size = T // num_segments  # 向下取整

    pooled_list = []
    for seg_idx in range(num_segments):
        start = seg_idx * segment_size
        end   = (seg_idx+1) * segment_size if seg_idx < num_segments-1 else T
        seg_feats = features[:, :, start:end]  # [B, C, seg_length]
        
        # 例如：只做 mean
        seg_mean = seg_feats.mean(dim=-1)  # [B, C]
        pooled_list.append(seg_mean)
        
        # 如果再想加 std:
        # seg_std = seg_feats.std(dim=-1)
        # pooled_list.append(seg_std)

    # 拼起来: [B, C*num_segments] 或 [B, 2*C*num_segments]
    return torch.cat(pooled_list, dim=-1)


def compute_time_quantile_stats(features: torch.Tensor, q_list=[0.05, 0.25, 0.5, 0.75, 0.95]):
    """
    沿通道维度计算分位数（输入 [B, T, C]）
    返回形状：[B, T*(4 + len(q_list))]
    """
    # 基础统计量
    min_ = features.min(dim=2).values    # [B, T]
    max_ = features.max(dim=2).values    # [B, T]
    mean_ = features.mean(dim=2)         # [B, T]
    std_ = features.std(dim=2)           # [B, T]
    
    # 分位数计算（沿通道维度 dim=2）
    quantiles = torch.quantile(
        features,
        torch.tensor(q_list, device=features.device),
        dim=2
    )  # 形状 [len(q_list), B, T]
    
    # 调整维度顺序并展平
    quantiles = quantiles.permute(1, 2, 0)  # [B, T, len(q_list)]
    quantiles = quantiles.reshape(features.size(0), -1)  # [B, T*len(q_list)]
    
    # 拼接所有特征
    return torch.cat([min_, max_, mean_, std_, quantiles], dim=1)


def compute_advanced_eeg_stats(features: torch.Tensor, fs=128, q_list=[0.05, 0.25, 0.5, 0.75, 0.95]):
    """
    高级EEG特征计算（输入 [B, T, C]）
    返回形状：[B, T*(基础特征数 + 扩展特征数)]
    """
    B, T, C = features.shape
    
    # ===== 基础统计量 =====
    min_ = features.min(dim=2).values   # [B, T]
    max_ = features.max(dim=2).values   # [B, T]
    mean_ = features.mean(dim=2)        # [B, T]
    std_ = features.std(dim=2)          # [B, T]
    
    # ===== 高阶统计量 =====
    skew_ = torch.tensor([torch.skew(sample, dim=1) for sample in features])  # [B, T]
    kurt_ = torch.tensor([torch.kurtosis(sample, dim=1) for sample in features])  # [B, T]

    # ===== Hjorth参数 =====
    def compute_hjorth(x):
        x_deriv = x[:, 1:] - x[:, :-1]
        activity = x.std(dim=1)**2
        mobility = (x_deriv.std(dim=1) / x.std(dim=1))
        complexity = (x_deriv[:, 1:] - x_deriv[:, :-1]).std(dim=1) / x_deriv.std(dim=1)
        return activity, mobility, complexity
    
    hjorth_act, hjorth_mob, hjorth_comp = compute_hjorth(features.permute(0,2,1))  # [B, T]

    # ===== 非线性特征 =====
    def approximate_entropy(x, m=2, r=0.2):
        # 近似熵计算（简化版）
        N = x.shape[1]
        phi = torch.zeros(B, T)
        for t in range(T-m+1):
            template = x[:, t:t+m]
            diff = torch.abs(x[:, None, :] - template[:, :, None])
            matches = (diff < r).all(dim=1).sum(dim=1)
            phi += torch.log(matches / (N - m + 1.0))
        return phi / (N - m + 1.0)
    
    apen = approximate_entropy(features.permute(0,2,1))  # [B, T]

    # ===== 分位数特征 =====
    quantiles = torch.quantile(features, torch.tensor(q_list, device=features.device), dim=2)
    quantiles = quantiles.permute(1, 2, 0).reshape(B, -1)  # [B, T*len(q_list)]

    # ===== 特征拼接 =====
    return torch.cat([
        min_, max_, mean_, std_, 
        skew_, kurt_,
        hjorth_act, hjorth_mob, hjorth_comp,
        apen,
        quantiles
    ], dim=1)  # 总维度 T*(4+2+3+1+5) = 15T

