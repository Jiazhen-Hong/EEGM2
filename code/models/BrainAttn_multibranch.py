"""
Example of BrainMamba1_parallel with multi-scale input embedding (kernel_sizes=1,3,7).
Created on Mon Dec 21 04:12:39 2024

@author: jiazhen@emotiv.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2  # ensure you have mamba_ssm installed

# ------------------------------------------------------------------------------
# 1) 多分支输入嵌入：并行卷积核大小分别为 1, 3, 7
# ------------------------------------------------------------------------------
class MultiBranchInputEmbedding(nn.Module):
    """
    同时用 kernel_size = 1, 3, 7 的三支卷积；
    最后拼接通道后，用一个 1x1 卷积做融合，得到与 out_channels 相同的通道数。
    """
    def __init__(self, in_channels, out_channels):
        super(MultiBranchInputEmbedding, self).__init__()
        # 三个并行分支
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)

        # 用 1x1 卷积把 3*out_channels -> out_channels
        self.fuse = nn.Conv1d(3 * out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        """
        x 形状: [batch, in_channels, time]
        输出形状: [batch, out_channels, time]
        """
        b1 = self.branch1(x)  # => [B, out_channels, T]
        b3 = self.branch3(x)  # => [B, out_channels, T]
        b7 = self.branch7(x)  # => [B, out_channels, T]

        out = torch.cat([b1, b3, b7], dim=1)  # => [B, 3*out_channels, T]
        out = self.fuse(out)                  # => [B, out_channels, T]
        return out
    


# ------------------------------------------------------------------------------
# 2) Mamba2 模型，与之前相同
# ------------------------------------------------------------------------------
class SelfSupervisedAttnModel(nn.Module):
    """
    与 SelfSupervisedMambaModel 类似的接口：
      - 输入: x, shape=(B,T,d_model)
      - 输出: shape=(B,T,d_model)
    不同之处：这里用 multihead self-attention 做映射。
    """
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True  # 让输入/输出是 (B,T,C)
        )
        self.norm = nn.LayerNorm(d_model)
        # 你也可以再加一个 FFN 等, 看你需求

    def forward(self, x):
        """
        x: (B,T,d_model)
        return: (B,T,d_model)
        """
        residual = x
        # 做自注意力
        attn_out, _ = self.mha(x, x, x)  # self-attn
        x = self.norm(attn_out + residual)
        return x

# ------------------------------------------------------------------------------
# 3) 主体网络：BrainMamba1_parallel，替换 input_embedding 为多分支结构
# ------------------------------------------------------------------------------
from torch.nn import functional as F
# 假设你把 SelfSupervisedAttnModel 放到同一个文件或别的地方
# from your_attn_file import SelfSupervisedAttnModel

class BrainAttn_multibranch(nn.Module):
    """
    与 BrainMamba2_multibranch 一样的 Encoder-Decoder骨架。
    但把 SelfSupervisedMambaModel 换成 SelfSupervisedAttnModel
    """
    def __init__(self, in_channels, out_channels, d_state, d_conv, expand, scale_factor=1):
        super().__init__()
        self.scale_factor = scale_factor
        base_channels = 64 // self.scale_factor

        # 多分支输入嵌入
        self.input_embedding = MultiBranchInputEmbedding(in_channels, base_channels)

        # --- 3层 Encoder ---
        # Encoder1
        self.encoder1 = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            SelfSupervisedAttnModel(d_model=base_channels, n_heads=4, dropout=0.1),
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Encoder2
        self.encoder2 = nn.Conv1d(base_channels, 128 // self.scale_factor, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Encoder3
        self.encoder3 = nn.Conv1d(128 // self.scale_factor, 256 // self.scale_factor, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Linear(256 // self.scale_factor, 256 // self.scale_factor),
            SelfSupervisedAttnModel(d_model=256 // self.scale_factor, n_heads=4, dropout=0.1),
            nn.Linear(256 // self.scale_factor, 256 // self.scale_factor),
        )

        # --- 3层 Decoder ---
        self.decoder3 = nn.Conv1d((256 // self.scale_factor)+(256 // self.scale_factor),
                                  256 // self.scale_factor, kernel_size=3, padding=1)
        self.decodeAttn3 = SelfSupervisedAttnModel(d_model=256 // self.scale_factor, n_heads=4, dropout=0.1)

        self.decoder2 = nn.Conv1d((128 // self.scale_factor)+(256 // self.scale_factor),
                                  128 // self.scale_factor, kernel_size=3, padding=1)
        self.decodeAttn2 = SelfSupervisedAttnModel(d_model=128 // self.scale_factor, n_heads=4, dropout=0.1)

        self.decoder1 = nn.Conv1d((64 // self.scale_factor)+(128 // self.scale_factor),
                                  64 // self.scale_factor, kernel_size=3, padding=1)

        # 输出embedding
        self.onput_embedding = nn.Conv1d(64 // self.scale_factor, out_channels, kernel_size=1)

    def forward(self, x):
        # 与 BrainMamba2_multibranch 一样
        # 1) input embedding
        x = self.input_embedding(x) # => (B, base_channels, T)

        # 2) encoder1
        x = x.permute(0, 2, 1)    # => (B, T, C)
        x1 = self.encoder1(x)     # => (B, T, base_channels)
        x1 = x1.permute(0, 2, 1)  # => (B, base_channels, T)
        x1p = self.pool1(x1)      # => (B, base_channels, T/2)

        # 3) encoder2
        x2 = self.encoder2(x1p)   # => (B,128, T/2)
        x2p = self.pool2(x2)      # => (B,128, T/4)

        # 4) encoder3
        x3 = self.encoder3(x2p)   # => (B,256, T/4)
        x3p = self.pool3(x3)      # => (B,256, T/8)

        # 5) bottleneck
        x3p = x3p.permute(0,2,1)  # => (B, T/8, 256)
        bottleneck = self.bottleneck(x3p)  # => (B, T/8, 256)
        bottleneck = bottleneck.permute(0,2,1)  # => (B,256, T/8)

        # upsample to T/4
        bottleneck = F.interpolate(bottleneck, size=x3.size(2), mode='linear', align_corners=False)

        # 6) decoder3
        d3 = torch.cat([x3, bottleneck], dim=1) # => (B,512, T/4)
        d3 = self.decoder3(d3)                  # => (B,256, T/4)
        d3 = d3.permute(0,2,1)                  # => (B,T/4,256)
        d3 = self.decodeAttn3(d3)               # => (B,T/4,256)
        d3 = d3.permute(0,2,1)                  # => (B,256,T/4)

        # upsample to T/2
        d3 = F.interpolate(d3, size=x2.size(2), mode='linear', align_corners=False)

        # 7) decoder2
        d2 = torch.cat([x2, d3], dim=1)         # => (B,128+256, T/2)
        d2 = self.decoder2(d2)                  # => (B,128, T/2)
        d2 = d2.permute(0,2,1)                  
        d2 = self.decodeAttn2(d2)               # => (B,T/2,128)
        d2 = d2.permute(0,2,1)                  # => (B,128,T/2)

        # upsample to T
        d2 = F.interpolate(d2, size=x1.size(2), mode='linear', align_corners=False)

        # 8) decoder1
        d1 = torch.cat([x1, d2], dim=1)         # => (B,64+128, T)
        d1 = self.decoder1(d1)                  # => (B,64, T)

        out = self.onput_embedding(d1)          # => (B,out_channels,T)
        return out