import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2

class MultiBranchInputEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiBranchInputEmbedding, self).__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.fuse = nn.Conv1d(3 * out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b7 = self.branch7(x)
        out = torch.cat([b1, b3, b7], dim=1)
        out = self.fuse(out)
        return out

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

class EEGM2_Light(nn.Module):
    def __init__(self, in_channels, d_state, d_conv, expand, scale_factor=1):
        super(EEGM2_Light, self).__init__()
        self.scale_factor = scale_factor
        base_channels = 64 // self.scale_factor

        self.input_embedding = MultiBranchInputEmbedding(in_channels, base_channels)
        
        self.encoder1 = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            SelfSupervisedMambaModel(d_model=base_channels, d_state=d_state, d_conv=d_conv, expand=expand),
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.encoder2 = nn.Conv1d(base_channels, 128 // self.scale_factor, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.encoder3 = nn.Conv1d(128 // self.scale_factor, 256 // self.scale_factor, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.input_embedding(x)
        x = x.permute(0, 2, 1)
        x1 = self.encoder1(x)
        x1 = x1.permute(0, 2, 1)
        x1p = self.pool1(x1)
        
        x2 = self.encoder2(x1p)
        x2p = self.pool2(x2)
        
        x3 = self.encoder3(x2p)
        x3p = self.pool3(x3)
        
        return x3p
