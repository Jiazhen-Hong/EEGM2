import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce

# --------------- Convolutional Transformer for EEG decoding -------------------

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv1d(1, 40, kernel_size=25, stride=1, padding=12),
            nn.Conv1d(40, 40, kernel_size=1, stride=1),
            nn.BatchNorm1d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv1d(40, emb_size, kernel_size=1, stride=1),  
            Rearrange('b e t -> b t e'),  # 重新排列数据
        )

    def forward(self, x):
        kernel_size = min(75, x.shape[-1])  # **动态计算 kernel_size**
        avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=15)
        x = avg_pool(self.shallownet(x))
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            energy = energy.masked_fill(~mask, -float('inf'))

        att = F.softmax(energy / math.sqrt(self.emb_size), dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=10, drop_p=0.5, forward_expansion=4):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, forward_expansion, drop_p),
                nn.Dropout(drop_p)
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[
            TransformerEncoderBlock(emb_size) for _ in range(depth)
        ])


class ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        if x.ndim == 2:  # (batch_size, emb_size)
            out = self.clshead(x)
        else:  # (batch_size, seq_len, emb_size)
            x = reduce(x, 'b n e -> b e', 'mean')  # 只有 3D 时才做 Reduce
            out = self.clshead(x)
        return x, out


class Conformer(nn.Module):
    def __init__(self, in_channels=58, emb_size=40, depth=6, n_classes=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(emb_size)
        self.transformer = TransformerEncoder(depth, emb_size)
        self.classifier = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = x.mean(dim=1)  # (B, 1, T)
        x = self.patch_embed(x)
        x = self.transformer(x)

        print(f"Shape before classifier: {x.shape}")  # **调试信息**
        return self.classifier(x)