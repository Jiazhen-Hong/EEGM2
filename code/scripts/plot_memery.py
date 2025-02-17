import matplotlib.pyplot as plt
import numpy as np
import torch
import json
from models import BrainMamba1, BrainMamba2, BrainMamba1_multibranch, BrainMamba2_multibranch, BrainMamba2EncoderOnly, UNet
from models.BrainAttn_multibranch import *
from Github_models.EEG2Rep_models.models import EEG2Rep
from Github_models.EEGNet import EEGNet
from Github_models.MAEEG import *
from Github_models.BIOT_models.biot import UnsupervisedPretrain, BIOTClassifier
from Github_models.BIOT_models.cnn_transformer import CNNTransformer
from Github_models.BENDR.reproduction_JH_selfTransformer import BENDR
from datetime import datetime
import os
from utility import setup_logging

logpath =  f'../results_paper/Memery-Usage/Memery-Usage_{datetime.now().strftime("%H%M%S")}/'
if not os.path.exists(logpath):
    os.makedirs(logpath)
logger, log_file = setup_logging(logpath)



# 读取配置文件
with open("config_downstream.json", "r") as f:  
    config = json.load(f)

# 读取 EEG2Rep 的配置
with open("./Github_models/EEG2Rep_models/config_eeg2rep.json", "r") as f:  
    config_eeg2rep = json.load(f)

device = torch.device(config["GPU_device"])

def measure_memory(model, batch_size, in_channels, seq_len, device):
    """测量模型的显存占用"""
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    
    x = torch.randn(batch_size, in_channels, seq_len, device=device)
    try:
        with torch.no_grad():
            _ = model(x)
        if device.type == "cuda":
            return torch.cuda.max_memory_allocated(device) / 1024**2
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[OOM] seq_len={seq_len} => returning None")
            torch.cuda.empty_cache()
            return None
        raise e

def create_model(name, in_channels, config, seq_len):
    """创建模型实例，确保每个模型适配不同的 seq_len"""
    if name == "BrainMamba1":
        return BrainMamba1(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"])
    elif name == "BrainMamba2":
        return BrainMamba2(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"])
    elif name == "BrainMamba1_multibranch":
        return BrainMamba1_multibranch(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"])
    elif name == "EEGM2":
        return BrainMamba2_multibranch(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"])
    elif name == "EEGM2(Light)":
        return BrainMamba2EncoderOnly(in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"])
    elif name == "BIOT":
        n_fft = min(200, seq_len) 
        hop_length = max(100, n_fft // 2)
        return BIOTClassifier(emb_size=256, heads=8, depth=4, n_classes=2, n_fft= n_fft, hop_length=hop_length)
    elif name == "CNNTransformer":
        return CNNTransformer(in_channels=in_channels, n_classes=2, fft=200, steps=5, dropout=0.2, nhead=64, emb_size=256, n_segments=5)
    elif name == "EEGM2-S5":
        return BrainAttn_multibranch(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"], scale_factor=1)
    elif name == "EEG2Rep":
        config_eeg2rep['Data_shape'] = [config["batch_size"], in_channels, seq_len]
        return EEG2Rep(config_eeg2rep, num_classes=in_channels)
    elif name == "EEGNet":
        return EEGNet(in_channels=in_channels, seq_len=seq_len, num_classes=2)
    elif name == "MAEEG":
        return MAEEGReconstruction(input_channel=in_channels, embed_size=256, downsample_size=[1, 1], kernel_size=[3, 5], dropout=0.2, transformer_embed_size=256, heads=8, forward_neuron=512, num_transformers=4)
    elif name == "BENDR":
        return BENDR(in_channels=in_channels, encoder_dim=512, num_layers=8, num_heads=8, ff_dim=3076)
    else:
        raise ValueError(f"Unknown model name {name}")

def main():
    model_names = ["EEGM2", "EEGM2(Light)", "EEGM2-S5", "EEG2Rep", "BENDR", "MAEEG", "BIOT"]
    #sequence_lengths = [250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
    sequence_lengths = [50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
    #sequence_lengths = [250, 500, 1000]
    batch_size, in_channels = 64, 16
    config["batch_size"] = batch_size
    

    memory_data = {name: {} for name in model_names}
    param_counts = {name: {} for name in model_names}

    for name in model_names:
        for seq_len in sequence_lengths:
            model = create_model(name, in_channels, config, seq_len).to(device)
            param_counts[name][seq_len] = sum(p.numel() for p in model.parameters())
            memory_data[name][seq_len] = measure_memory(model, batch_size, in_channels, seq_len, device)

    logger.info("=== Param Counts ===")
    for name in model_names:
        for seq_len in sequence_lengths:
            logger.info(f"{name} (seq_len={seq_len}): {param_counts[name][seq_len]} params")

    logger.info("=== Memory Usage (MB) ===")
    for name, data in memory_data.items():
        logger.info(f"{name}: {data}")


    plt.figure(figsize=(10, 8))
    # colors = {
    #     "EEGM2": "darkred",           # 深红色
    #     "EEGM2(Encoder)": "red",  
    #     "EEGM2-S5": "lightcoral",            
    #     "EEG2Rep": "blue",            # 蓝色
    #     "BENDR": "purple",            # 紫色
    #     "MAEEG": "green",             # 绿色
    #     "BIOT": "orange"              # 橙色
    #     }
    colors = {
        "EEGM2": "lightcoral",           # 暗红色 (Dark Red)
        "EEGM2(Light)": "red",  # 砖红色 (Firebrick)
        "EEGM2-S5": "#A52A2A",        # 棕红色 (Brown)
        "EEG2Rep": "#4169E1",         # 皇家蓝 (Royal Blue)
        "BENDR": "#4B0082",           # 靛青色 (Indigo)
        "MAEEG": "green",           # 暗绿色 (Dark Green)
        "BIOT": "#FF8C00"             # 暗橙色 (Dark Orange)
    }

    styles = {
        "EEGM2": "o-",
        "EEGM2(Light)": "s-",
        "EEGM2-S5": "d-",
        "EEG2Rep": "^-",
        "BENDR": "+-",
        "MAEEG": "*-",
        "BIOT": "x-"
        }
    

    # for idx, name in enumerate(model_names):
    #     y_values = [memory_data[name][seq] for seq in sequence_lengths]
    #     plt.plot(sequence_lengths, y_values, styles[idx % len(styles)], linewidth=2, markersize=8, label=name)
    
    for name in model_names:
        y_values = [memory_data[name][seq] for seq in sequence_lengths if memory_data[name][seq] is not None]
        x_values = [seq for seq in sequence_lengths if memory_data[name][seq] is not None]
        
        plt.plot(x_values, y_values, styles[name], color=colors[name], alpha=0.8,
                linewidth=2, markersize=10, label=name)

    plt.xlabel("Length of Sequence", fontsize=20)
    plt.ylabel("Peak Memory Usage (MB)", fontsize=20)
    #plt.title("Memory Usage vs Sequence Length", fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle='-.', linewidth=0.5)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{logpath}/mem_usage_vs_seq_len.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
