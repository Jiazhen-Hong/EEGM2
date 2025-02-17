import matplotlib.pyplot as plt
import numpy as np
import torch
import time
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

logpath =  f'../results_paper/Inference-Time/Inference-Time_{datetime.now().strftime("%H%M%S")}/'
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

# def measure_inference_time(model, batch_size, in_channels, seq_len, device, num_trials=10):
#     model.eval()
#     x = torch.randn(batch_size, in_channels, seq_len, device=device)
#     try:
#         for _ in range(5):
#             _ = model(x)
#         times = []
#         for _ in range(num_trials):
#             torch.cuda.synchronize() if device.type == "cuda" else None
#             start_time = time.time()
#             _ = model(x)
#             torch.cuda.synchronize() if device.type == "cuda" else None
#             times.append((time.time() - start_time) * 1000)
#         return sum(times) / len(times)
#     except torch.cuda.OutOfMemoryError:
#         print(f"[OOM] seq_len={seq_len} => returning None")
#         torch.cuda.empty_cache()
#         return None

def measure_inference_time(model, batch_size, in_channels, seq_len, device, num_trials=10):
    """
    测量推理时间，同时返回每个样本的推理时间 (ms/sample)。
    """
    model.eval()
    x = torch.randn(batch_size, in_channels, seq_len, device=device)
    try:
        # 预热 GPU 以确保时间测量准确
        for _ in range(5):
            _ = model(x)
        
        # 测量推理时间
        times = []
        for _ in range(num_trials):
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.time()
            _ = model(x)
            torch.cuda.synchronize() if device.type == "cuda" else None
            times.append((time.time() - start_time) * 1000)  # 转换为 ms
        
        avg_inference_time_ms = sum(times) / len(times)  # 平均推理时间 (ms/batch)
        
        # 计算每样本的推理时间
        ms_per_sample = avg_inference_time_ms / batch_size  # 每个样本的推理时间 (ms/sample)

        return ms_per_sample
    except torch.cuda.OutOfMemoryError:
        print(f"[OOM] seq_len={seq_len} => returning None")
        torch.cuda.empty_cache()
        return None



def create_model(name, in_channels, config, seq_len):
    if name == "EEGM2":
        return BrainMamba2_multibranch(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"])
    elif name == "EEGM2-S5":
        return BrainAttn_multibranch(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"], scale_factor=1)
    elif name == "EEGM2(Encoder)":
            return BrainMamba2EncoderOnly(in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"])
    elif name == "EEG2Rep":
        config_eeg2rep['Data_shape'] = [config["batch_size"], in_channels, seq_len]
        return EEG2Rep(config_eeg2rep, num_classes=in_channels)
    elif name == "MAEEG":
        return MAEEGReconstruction(input_channel=in_channels, embed_size=256, downsample_size=[2, 2], kernel_size=[3, 5], dropout=0.2, transformer_embed_size=256, heads=8, forward_neuron=512, num_transformers=4)
    elif name == "BENDR":
        return BENDR(in_channels=in_channels, encoder_dim=512, num_layers=8, num_heads=8, ff_dim=3076)
    else:
        raise ValueError(f"Unknown model name {name}")

def main():
    model_names = ["EEGM2", "EEGM2-S5", "EEGM2(Encoder)","EEG2Rep", "BENDR", "MAEEG"]
    sequence_lengths = [250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
    batch_size, in_channels = 64, 16
    config["batch_size"] = batch_size

    inference_data = {name: {} for name in model_names}
    
    for name in model_names:
        for seq_len in sequence_lengths:
            logger.info(f"Initializing {name} with seq_len={seq_len}")
            model = create_model(name, in_channels, config, seq_len).to(device)
            ms_per_sample = measure_inference_time(model, batch_size, in_channels, seq_len, device)
            inference_data[name][seq_len] = ms_per_sample
    
    logger.info("=== Inference Time (ms/sample) ===")
    for name, data in inference_data.items():
        logger.info(f"{name}: {data}")
    
    plt.figure(figsize=(10, 8))
    styles = ['o-', 's--', 'd-.', '^:', 'x-', '*-']
    for idx, name in enumerate(model_names):
        y_values = [inference_data[name][seq] for seq in sequence_lengths if inference_data[name][seq] is not None]
        x_values = [seq for seq in sequence_lengths if inference_data[name][seq] is not None]
        plt.plot(x_values, y_values, styles[idx % len(styles)], linewidth=3, markersize=8, label=name)
    
    plt.xlabel("Sequence Length", fontsize=20)
    plt.ylabel("Inference Time (ms/sample)", fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle='-.', linewidth=0.5)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{logpath}/inference_time_vs_seq_len.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()