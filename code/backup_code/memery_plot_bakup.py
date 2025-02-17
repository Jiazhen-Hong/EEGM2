import matplotlib.pyplot as plt
import numpy as np
import torch
from models import BrainMamba1, BrainMamba2, BrainMamba1_multibranch, BrainMamba2_multibranch, UNet
from models.BrainAttn_multibranch import *
from Github_models.EEG2Rep_models.models import EEG2Rep
from Github_models.EEGNet import EEGNet
from Github_models.MAEEG import *

from Github_models.BIOT_models.biot import UnsupervisedPretrain, BIOTClassifier
from Github_models.BIOT_models.cnn_transformer import CNNTransformer
#from Github_models.EEGPT_models.engine_pretraining import LitEEGPT
#from Github_models.EEGPT_models.configs import MODELS_CONFIGS, get_config
#from Github_models.EEGConformer_models.eeg_conformer import *  
#from Github_models.BENDR.reproduction_JH import BENDR
from Github_models.BENDR.reproduction_JH_selfTransformer import BENDR
# #from Github_models.BENDR.Trainer_BENDR import BENDR
# from Github_models.BENDR.Trainer_BENDR_JH_selfTransformer import BENDR

import os
import json

# torch.backends.cuda.enable_mem_efficient_sdp(False)  # 关闭内存优化
# torch.backends.cuda.enable_flash_sdp(False)    
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_math_sdp(False)

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

    # Conformer 需要特殊的输入形状
    if isinstance(model, EEGNet):
        x = torch.randn(batch_size, in_channels, 1, seq_len, device=device)  
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
    # elif name == "BIOT":
    #     return UnsupervisedPretrain(emb_size=256, heads=8, depth=4, n_channels=in_channels, n_fft=200, hop_length=100)
    elif name == "BIOT":
        # # 让 STFT 的 `n_fft` 固定，但 `hop_length` 变化，确保 Transformer 看到更长的序列
        # n_fft = 200  # 固定 200
        # hop_length = max(20, seq_len // 40)  # 让 hop_length 变小，使时间维度变长
        # return UnsupervisedPretrain(
        #     emb_size=256, heads=8, depth=4, n_channels=in_channels, n_fft=n_fft, hop_length=hop_length
        # )
        return BIOTClassifier(
                emb_size=256,
                heads=8,
                depth=4,
                n_classes=2,
                n_fft=200,
                hop_length=100
                )

    elif name == "CNNTransformer":
        return CNNTransformer(in_channels=in_channels, n_classes=2, fft=200, steps=5, dropout=0.2, nhead=64, emb_size=256, n_segments=5)
    elif name == "BrainAttn_multibranch":
        return BrainAttn_multibranch(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"], scale_factor=1)
    # elif name == "EEGPT":
    #     tag = "base3"
    #     variant = "D"
    #     return LitEEGPT(get_config(**(MODELS_CONFIGS[tag])), USE_LOSS_A=(variant!="A"), USE_LN=(variant!="B"), USE_SKIP=(variant!="C"))
    # elif name == "EEGConformer":
    #     return Conformer(emb_size=200, depth=6, n_classes=2, num_heads=5, seq_len=seq_len)
    elif name == "EEG2Rep":
        # need adjust data shape in config_eeg2rep
        config_eeg2rep['Data_shape'] = [config["batch_size"], in_channels, seq_len]
        return EEG2Rep(config_eeg2rep, num_classes=in_channels)
    elif name == "EEGNet":
        return EEGNet(in_channels=in_channels, seq_len=seq_len, num_classes=2)
    elif name == "MAEEGReconstruction":
        return MAEEGReconstruction(
            input_channel=in_channels, embed_size=256, 
            downsample_size=[1, 1],  # 让下采样动态变化
            kernel_size=[3, 5], dropout=0.2, transformer_embed_size=256, 
            heads=8, forward_neuron=512, num_transformers=4
        )
    elif name == "BENDR":
        return BENDR(in_channels=in_channels, encoder_dim=512, num_layers=8, num_heads=8, ff_dim=3076)
        # return BENDR(
        #     in_channels=in_channels,  # EEG 通道数
        #     encoder_dim=512,  # Encoder 维度
        #     num_layers=8,  # Transformer 层数
        #     num_heads=8,  # 注意力头数
        #     ff_dim=3076,  # 前馈层维度
        #     num_classes=2  # 二分类
        #    )
    else:
        raise ValueError(f"Unknown model name {name}")

def main():
    model_names = [
        "BENDR",
        "BrainMamba2_multibranch",
        "MAEEGReconstruction",
        #"EEGNet",
        #"BrainMamba1_multibranch",
        #"BIOT",
        #"CNNTransformer",
        "BrainAttn_multibranch",
        "EEG2Rep",  
    ]

    sequence_lengths = [256, 1280, 2560, 3200, 3840, 4480, 5120]  
    batch_size = 64
    in_channels = 16

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
    plt.title("Memory Usage vs Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mem_usage_vs_seq_len_all.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()