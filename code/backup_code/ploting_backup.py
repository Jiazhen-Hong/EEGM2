import matplotlib.pyplot as plt
import numpy as np
import torch
from models import BrainMamba1, BrainMamba2, BrainMamba1_multibranch, BrainMamba2_multibranch, UNet
from models.BrainAttn_multibranch import *
from BIOT_models.biot import UnsupervisedPretrain
from BIOT_models.cnn_transformer import CNNTransformer
from EEGPT_models.engine_pretraining import LitEEGPT
from EEGPT_models.configs import MODELS_CONFIGS, get_config
from EEGConformer_models.eeg_conformer import Conformer  
from EEG2Rep_models.models import EEG2Rep
import os
import json

# 读取配置文件
with open("config_downstream.json", "r") as f:  
    config = json.load(f)

# 读取 EEG2Rep 的配置
with open("./EEG2Rep_models/config_eeg2rep.json", "r") as f:  
    config_eeg2rep = json.load(f)

device = torch.device(config["GPU_device"])  # e.g. "cuda:0"

def measure_memory(model, batch_size, in_channels, seq_len, device):
    """测量模型的显存占用"""
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Conformer 需要特殊的输入形状
    if isinstance(model, Conformer):
        x = torch.randn(batch_size, 1, in_channels, seq_len, device=device)
    else:
        x = torch.randn(batch_size, in_channels, seq_len, device=device)

    try:
        with torch.no_grad():
            _ = model(x)  # Forward pass
        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(device)
            return peak_mem / 1024**2  # 转换为 MB
        else:
            return 0.0
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[OOM] seq_len={seq_len} => returning None")
            torch.cuda.empty_cache()
            return None
        else:
            raise e

def create_model(name, in_channels, config, seq_len):
    """
    工厂函数：根据 name 返回对应模型的实例，确保每个模型适配不同的 seq_len
    """
    if name == "BrainMamba1":
        return BrainMamba1(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"])
    elif name == "BrainMamba2":
        return BrainMamba2(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"])
    elif name == "BrainMamba1_multibranch":
        return BrainMamba1_multibranch(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"])
    elif name == "BrainMamba2_multibranch":
        return BrainMamba2_multibranch(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"])
    elif name == "BIOT":
        return UnsupervisedPretrain(emb_size=256, heads=8, depth=4, n_channels=in_channels, n_fft=200, hop_length=100)
    elif name == "CNNTransformer":
        return CNNTransformer(in_channels=in_channels, n_classes=2, fft=200, steps=5, dropout=0.2, nhead=64, emb_size=256, n_segments=5)
    elif name == "BrainAttn_multibranch":
        return BrainAttn_multibranch(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"], scale_factor=1)
    elif name == "EEGPT":
        tag = "base3"
        variant = "D"
        return LitEEGPT(get_config(**(MODELS_CONFIGS[tag])), USE_LOSS_A=(variant!="A"), USE_LN=(variant!="B"), USE_SKIP=(variant!="C"))
    elif name == "EEGConformer":
        return Conformer(emb_size=200, depth=6, n_classes=2, num_heads=5)
    elif name == "EEG2Rep":
        config_eeg2rep['Data_shape'] = [config["batch_size"], in_channels, seq_len]
        return EEG2Rep(config_eeg2rep, num_classes=in_channels)
    else:
        raise ValueError(f"Unknown model name {name}")

def main():
    model_names = [
        "BrainMamba1_multibranch",
        "BrainMamba2_multibranch",
        "BIOT",
        "CNNTransformer",
        "BrainAttn_multibranch",
        "EEG2Rep"
    ]

    sequence_lengths = [1024, 2048, 4096, 8192, 16384]  
    batch_size = 64
    in_channels = 1  

    # 确保 config 里有 batch_size
    config["batch_size"] = batch_size  

    # 存储显存数据
    memory_data = {"Sequence Length": sequence_lengths}
    for name in model_names:
        memory_data[name] = {seq_len: None for seq_len in sequence_lengths}

    # 存储参数量
    param_counts = {name: {} for name in model_names}

    for name in model_names:
        for seq_len in sequence_lengths:
            print(f"\n[INFO] Initializing {name} with seq_len={seq_len}")
            
            # **为每个 seq_len 重新实例化模型**
            model = create_model(name, in_channels, config, seq_len).to(device)

            # 统计参数量
            num_params = sum(p.numel() for p in model.parameters())
            param_counts[name][seq_len] = num_params

            # 计算显存占用
            mem_mb = measure_memory(model, batch_size, in_channels, seq_len, device)
            memory_data[name][seq_len] = mem_mb

    # 打印参数量
    print("\n=== Param Counts ===")
    for name in model_names:
        for seq_len in sequence_lengths:
            print(f"{name} (seq_len={seq_len}): {param_counts[name][seq_len]} params")

    # 打印显存使用情况
    print("\n=== Memory Usage (MB) ===")
    for name, data in memory_data.items():
        print(f"{name}: {data}")

    # 画图
    plt.figure(figsize=(8, 6))
    for name in model_names:
        y_arr = [memory_data[name][seq_len] for seq_len in sequence_lengths]
        plt.plot(sequence_lengths, y_arr, label=name, marker="o")

    plt.xlabel("Sequence Length")
    plt.ylabel("Peak Memory (MB)")
    plt.title("Memory Usage vs Sequence Length (Single Channel)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mem_usage_vs_seq_len_all.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()