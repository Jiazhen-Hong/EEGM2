"""
Created on Mon Dec 21 04:12:39 2024

@author: jiazhen@emotiv.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2 # JH download 11/14/2024, https://github.com/state-spaces/mamba (Complie with Numpy 1.x only, also required)


class SelfSupervisedMambaModel(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super(SelfSupervisedMambaModel, self).__init__()
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)  

    def forward(self, x):
        residual = x
        x = self.mamba(x)
        x = self.norm(x)
        x = x + residual        
        return x

class BrainMamba2(nn.Module):
    """
    一个简化版的 3-layer Encoder + 3-layer Decoder 结构示例。

    注意：
    1. 将原有 encoder4、pool4 去掉；
    2. 瓶颈层从 512 改为 256；
    3. 将 decoder4、decodeMamba4 去掉；
    4. 其余部分保持与原先类似的写法和操作顺序；
    5. 需要在 forward() 里相应地删除 x4, x4p 等相关张量的操作。
    """
    def __init__(self, in_channels, out_channels, d_state, d_conv, expand, scale_factor=1, logger=None):
        super(BrainMamba2, self).__init__()
        self.scale_factor = scale_factor
        base_channels = 64 // self.scale_factor

        # 1) Input embedding
        self.input_embedding = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        # 2) Encoder Layers (共 3 层)
        # -- Encoder1
        self.encoder1 = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            SelfSupervisedMambaModel(d_model=base_channels, d_state=d_state, d_conv=d_conv, expand=expand),
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # -- Encoder2
        self.encoder2 = nn.Conv1d(base_channels, 128 // self.scale_factor, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # -- Encoder3
        self.encoder3 = nn.Conv1d(128 // self.scale_factor, 256 // self.scale_factor, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 3) Bottleneck (这里改为 256 通道)
        self.bottleneck = nn.Sequential(
            nn.Linear(256 // self.scale_factor, 256 // self.scale_factor),
            SelfSupervisedMambaModel(d_model=256 // self.scale_factor, d_state=d_state, d_conv=d_conv, expand=expand),
            nn.Linear(256 // self.scale_factor, 256 // self.scale_factor),
        )

        # 4) Decoder Layers (共 3 层, 去掉原先的第4层)
        # -- Decoder3 (原本对应前面的 encoder3)
        self.decoder3 = nn.Conv1d(
            (256 // self.scale_factor) + (256 // self.scale_factor), 
            256 // self.scale_factor, 
            kernel_size=3, 
            padding=1
        )
        self.decodeMamba3 = SelfSupervisedMambaModel(
            d_model=256 // self.scale_factor, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )

        # -- Decoder2
        self.decoder2 = nn.Conv1d(
            (128 // self.scale_factor) + (256 // self.scale_factor), 
            128 // self.scale_factor, 
            kernel_size=3, 
            padding=1
        )
        self.decodeMamba2 = SelfSupervisedMambaModel(
            d_model=128 // self.scale_factor, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )

        # -- Decoder1
        self.decoder1 = nn.Conv1d(
            (64 // self.scale_factor) + (128 // self.scale_factor), 
            64 // self.scale_factor, 
            kernel_size=3, 
            padding=1
        )

        # 5) 最终输出
        self.onput_embedding = nn.Conv1d(64 // self.scale_factor, out_channels, kernel_size=1)
        self.logger = logger  
        self.logged_input_shapes = False

    def forward(self, x):
        # ------ Input embedding ------
        if self.logger and not self.logged_input_shapes:
            self.logger.info(f"Input data shape before 1D CNN: {x.shape}")

        x = self.input_embedding(x)   # (batch, base_channels, time)
        if self.logger and not self.logged_input_shapes:
            self.logger.info(f"Input embedding shape after 1D CNN: {x.shape}")
            self.logged_input_shapes = True

        # ------ Encoder 1 ------
        x = x.permute(0, 2, 1)        # (B, C, T) -> (B, T, C)
        x1 = self.encoder1(x)         # (B, T, base_channels)
        x1 = x1.permute(0, 2, 1)      # (B, T, C) -> (B, C, T)
        x1p = self.pool1(x1)          # 下采样 => (B, base_channels, T/2)

        # ------ Encoder 2 ------
        x2 = self.encoder2(x1p)       # (B, 128//scale_factor, T/2)
        x2p = self.pool2(x2)          # 下采样 => (B, 128//scale_factor, T/4)

        # ------ Encoder 3 ------
        x3 = self.encoder3(x2p)       # (B, 256//scale_factor, T/4)
        x3p = self.pool3(x3)          # 下采样 => (B, 256//scale_factor, T/8)

        # ------ Bottleneck (256 通道) ------
        x3p = x3p.permute(0, 2, 1)    # (B, C, T) -> (B, T, C)
        bottleneck = self.bottleneck(x3p)  # (B, T/8, 256)
        bottleneck = bottleneck.permute(0, 2, 1)  # -> (B, 256, T/8)

        # 这里上采样到 x3 的时间维度 (T/4)
        bottleneck = F.interpolate(bottleneck, size=x3.size(2), mode='linear', align_corners=False)

        # ------ Decoder 3 ------
        # Skip: x3 + bottleneck => cat 通道
        d3 = torch.cat([x3, bottleneck], dim=1)   # (B, 256+256, T/4)
        d3 = self.decoder3(d3)                    # -> (B, 256, T/4)
        d3 = d3.permute(0, 2, 1)                  # (B, C, T) -> (B, T, C)
        d3 = self.decodeMamba3(d3)                # -> (B, T, 256)
        d3 = d3.permute(0, 2, 1)                  # -> (B, 256, T/4)

        # 上采样到 x2 的时间维度 (T/2)
        d3 = F.interpolate(d3, size=x2.size(2), mode='linear', align_corners=False)

        # ------ Decoder 2 ------
        d2 = torch.cat([x2, d3], dim=1)           # (B, 128+256, T/2)
        d2 = self.decoder2(d2)                    # -> (B, 128, T/2)
        d2 = d2.permute(0, 2, 1)                  # (B, C, T) -> (B, T, C)
        d2 = self.decodeMamba2(d2)                # -> (B, T, 128)
        d2 = d2.permute(0, 2, 1)                  # -> (B, 128, T/2)

        # 上采样到 x1 的时间维度 (T)
        d2 = F.interpolate(d2, size=x1.size(2), mode='linear', align_corners=False)

        # ------ Decoder 1 ------
        d1 = torch.cat([x1, d2], dim=1)           # (B, 64+128, T)
        d1 = self.decoder1(d1)                    # -> (B, 64, T)

        # ------ Output embedding ------
        out = self.onput_embedding(d1)            # -> (B, out_channels, T)
        return out
