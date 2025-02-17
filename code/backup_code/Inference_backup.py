import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from models import BrainMamba1, BrainMamba2, BrainMamba1_multibranch, BrainMamba2_multibranch, BrainUnet_multibranch, UNet
from models.BrainAttn_multibranch import *
from Github_models.EEG2Rep_models.models import EEG2Rep
from Github_models.EEGNet import EEGNet
from Github_models.MAEEG import *

from Github_models.BIOT_models.biot import UnsupervisedPretrain, BIOTClassifier
from Github_models.BIOT_models.cnn_transformer import CNNTransformer
from Github_models.BENDR.reproduction_JH_selfTransformer import BENDR

import os
import json

torch.backends.cuda.enable_mem_efficient_sdp(False)  # 关闭 PyTorch 内存优化
torch.backends.cuda.enable_flash_sdp(False)

# 读取配置文件
with open("config_downstream.json", "r") as f:  
    config = json.load(f)

# 读取 EEG2Rep 的配置
with open("./Github_models/EEG2Rep_models/config_eeg2rep.json", "r") as f:  
    config_eeg2rep = json.load(f)

device = torch.device(config["GPU_device"])  # e.g. "cuda:0"

def measure_memory(model, batch_size, in_channels, seq_len, device):
    """测量模型的显存占用"""
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

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

def measure_inference_time(model, batch_size, in_channels, seq_len, device, num_trials=10):
    """
    计算模型的推理时间（Inference Time）。
    
    参数:
        model: 需要测量的神经网络模型
        batch_size: 批量大小
        in_channels: 输入通道数
        seq_len: 输入序列长度
        device: 运行设备 (CPU/GPU)
        num_trials: 运行多少次取平均

    返回:
        平均推理时间 (ms) 或 None（如果显存溢出）
    """
    model.eval()
    x = torch.randn(batch_size, in_channels, seq_len, device=device)

    # 预热运行（避免首次运行慢导致误差）
    try:
        for _ in range(5):
            _ = model(x)

        # 正式测量
        times = []
        for _ in range(num_trials):
            torch.cuda.synchronize() if device.type == "cuda" else None  # 计时前同步 GPU
            start_time = time.time()
            _ = model(x)  # 执行推理
            torch.cuda.synchronize() if device.type == "cuda" else None  # 计时后同步 GPU
            times.append((time.time() - start_time) * 1000)  # 转换成毫秒

        return sum(times) / len(times)  # 返回平均推理时间 (ms)
    
    except torch.cuda.OutOfMemoryError:
        print(f"[OOM] seq_len={seq_len} => returning None")
        torch.cuda.empty_cache()  # 清理缓存，防止后续任务崩溃
        return None

def create_model(name, in_channels, config, seq_len):
    """工厂函数：返回对应模型实例，确保模型适配不同的 seq_len"""
    if name == "BrainMamba1":
        return BrainMamba1(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"])
    elif name == "BrainMamba2":
        return BrainMamba2(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"])
    elif name == "BrainMamba1_multibranch":
        return BrainMamba1_multibranch(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"])
    elif name == "BrainMamba2_multibranch":
        return BrainMamba2_multibranch(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"])
    elif name == "EEG2Rep":
        config_eeg2rep['Data_shape'] = [config["batch_size"], in_channels, seq_len]
        return EEG2Rep(config_eeg2rep, num_classes=in_channels)
    elif name == "EEGNet":
        return EEGNet(in_channels=in_channels, seq_len=seq_len, num_classes=2)
    elif name == "BrainAttn_multibranch":
        return BrainAttn_multibranch(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"], scale_factor=1)
    elif name == "MAEEGReconstruction":
        return MAEEGReconstruction(
            input_channel=in_channels, embed_size=256, 
            downsample_size=[2, 2],  # 让下采样动态变化
            kernel_size=[3, 5], dropout=0.2, transformer_embed_size=256, 
            heads=8, forward_neuron=512, num_transformers=4
        )
    elif name == "BENDR":
        return BENDR(in_channels=in_channels, encoder_dim=512, num_layers=8, num_heads=8, ff_dim=3076)
    else:
        raise ValueError(f"Unknown model name {name}")

def main():
    model_names = [
            "BrainMamba2_multibranch",
            "MAEEGReconstruction",
            #"EEGNet",
            #"BrainMamba1_multibranch",
            #"BIOT",
            #"CNNTransformer",
            "BrainAttn_multibranch",
            "EEG2Rep",  
            "BENDR"
        ]


    sequence_lengths = [256, 1280, 2560, 3200, 3840, 3968, 4480, 5120, 7280, 10000]  
    batch_size = 64
    in_channels = 16

    # 存储显存 & 推理时间数据
    memory_data = {name: {} for name in model_names}
    inference_data = {name: {} for name in model_names}

    for name in model_names:
        for seq_len in sequence_lengths:
            print(f"\n[INFO] 正在初始化 {name}，seq_len={seq_len}")

            # **为每个 seq_len 重新实例化模型**
            model = create_model(name, in_channels, config, seq_len).to(device)

            # 计算显存占用
            mem_mb = measure_memory(model, batch_size, in_channels, seq_len, device)
            memory_data[name][seq_len] = mem_mb

            # 计算推理时间
            inference_time = measure_inference_time(model, batch_size, in_channels, seq_len, device)
            inference_data[name][seq_len] = inference_time  # 可能返回 None

    # 输出显存数据
    print("\n=== 显存占用 (MB) ===")
    for name, data in memory_data.items():
        print(f"{name}: {data}")

    # 输出推理时间
    print("\n=== 各模型推理时间 (ms) ===")
    for name, data in inference_data.items():
        print(f"{name}: {data}")

    # 绘制推理时间 vs 序列长度
    plt.figure(figsize=(8, 6))
    for name in model_names:
        y_arr = [inference_data[name][seq_len] for seq_len in sequence_lengths if inference_data[name][seq_len] is not None]
        x_arr = [seq_len for seq_len in sequence_lengths if inference_data[name][seq_len] is not None]
        plt.plot(x_arr, y_arr, label=name, marker="o")

    plt.xlabel("Sequence Length")
    plt.ylabel("Inference Time (ms)")
    plt.title("Inference Time vs Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("inference_time_vs_seq_len.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()