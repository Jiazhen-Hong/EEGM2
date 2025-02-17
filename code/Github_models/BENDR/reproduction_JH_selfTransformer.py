import torch
import torch.nn as nn
import torch.nn.functional as F

########################
# ----- JH edit -----
########################
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        assert embed_size % heads == 0, "Embed size 必须能被 heads 整除"
        self.heads = heads
        self.head_dim = embed_size // heads

        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        q = self.query(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

        # **确保计算的是标准 Attention，O(N^2)**
        energy = torch.einsum("bhqd, bhkd -> bhqk", q, k) / (self.head_dim ** 0.5)
        attention = torch.softmax(energy, dim=-1)
        out = torch.einsum("bhqk, bhvd -> bhqd", attention, v).transpose(1, 2)
        return self.fc_out(out.reshape(batch_size, seq_len, embed_dim))
    
class TransformerBlock(nn.Module):
    """ 自定义 Transformer Block """
    def __init__(self, embed_size, heads, forward_neuron):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.linear1 = nn.Linear(embed_size, forward_neuron)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(forward_neuron, embed_size)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        out1 = self.attention(x)
        out1 = self.layer_norm1(out1 + x)
        out2 = self.relu(self.linear1(out1))
        out2 = self.linear2(out2)
        out = self.layer_norm2(out2 + out1)
        return out

class TransformerEncoder(nn.Module):
    """ 多层 Transformer Encoder """
    def __init__(self, embed_size, heads, forward_neuron, num_transformers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, forward_neuron) for _ in range(num_transformers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

########################
# ----- BENDR -----
########################
class BENDREncoder(nn.Module):
    """ BENDR Encoder: 1D CNN 降维 """
    def __init__(self, in_channels=1, encoder_dim=512):
        super(BENDREncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 64),
            nn.GELU(),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.GELU(),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU(),

            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.GELU(),

            nn.Conv1d(512, encoder_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, encoder_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv_layers(x)


#####################
# BENDR w/o conv
#####################
class BENDREncoder(nn.Module):
    """ BENDR Encoder without downsampling """
    def __init__(self, in_channels=1, encoder_dim=512):
        super(BENDREncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, encoder_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, encoder_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv_layers(x)  # (batch, encoder_dim, seq_len)

class BENDR(nn.Module):
    """ Full BENDR Model with 自定义 Transformer """
    def __init__(self, in_channels=1, encoder_dim=512, num_layers=8, num_heads=8, ff_dim=3076):
        super(BENDR, self).__init__()
        self.encoder = BENDREncoder(in_channels, encoder_dim)
        self.transformer = TransformerEncoder(embed_size=encoder_dim, heads=num_heads, forward_neuron=ff_dim, num_transformers=num_layers)

    def forward(self, x):
        x = self.encoder(x)  # (batch, encoder_dim, seq_len//32)
        x = x.permute(0, 2, 1)  # (batch, seq_len//32, encoder_dim) for Transformer
        x = self.transformer(x)
        return x

###############################################################