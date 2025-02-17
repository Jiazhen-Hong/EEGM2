import math
import torch
import torch.nn as nn

# ----- 1️⃣ MAEEG Convolution Module -----
class MAEEGConvolution(nn.Module):
    def __init__(self, input_channel, output_channel, downsample_size, kernel_size, dropout):
        super().__init__()
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size]
        if not isinstance(downsample_size, (list, tuple)):
            downsample_size = [downsample_size]
        assert len(kernel_size) == len(downsample_size)

        # 让 kernel size 为奇数，避免 padding 复杂性
        kernel_size = [k if k % 2 else k + 1 for k in kernel_size]

        self.encoder = nn.Sequential()
        for i, (kernel, downsample) in enumerate(zip(kernel_size, downsample_size)):
            self.encoder.add_module(
                f"ConvBlock_{i}", nn.Sequential(
                    nn.Conv1d(input_channel, output_channel, kernel_size=kernel, stride=downsample, padding=kernel // 2),
                    nn.Dropout(dropout),
                    nn.GroupNorm(output_channel // 2, output_channel),
                    nn.GELU()
                )
            )
            input_channel = output_channel

    def forward(self, x):
        return self.encoder(x).transpose(1, 2)  # 转置后使得时间维度在第二维

# ----- 2️⃣ Self-Attention Module -----
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        assert embed_size % heads == 0, "Embedding size必须能被head整除"
        self.heads = heads
        self.head_dim = embed_size // heads
        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        q = self.query(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

        energy = torch.einsum("bhqd, bhkd -> bhqk", q, k) / math.sqrt(self.head_dim)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(energy, dim=-1)

        out = torch.einsum("bhqk, bhvd -> bhqd", attention, v).transpose(1, 2)
        return self.fc_out(out.reshape(batch_size, seq_len, embed_dim))

# ----- 3️⃣ Transformer Encoder -----
# class TransformerEncoder(nn.Module):
#     def __init__(self, embed_size, heads, forward_neuron, num_transformers):
#         super().__init__()
#         self.layers = nn.TransformerEncoder(
#             encoder_layer=nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads, dim_feedforward=forward_neuron),
#             num_layers=num_transformers
#         )

#     def forward(self, x):
#         return self.layers(x)

########################
# ----- JH edit -----
########################
class TransformerBlock(nn.Module):
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
    def __init__(self, embed_size, heads, forward_neuron, num_transformers):
        super().__init__()
        # 正确地添加多个 TransformerBlock，而不是递归调用 TransformerEncoder
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, forward_neuron) for _ in range(num_transformers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
########################
########################





# ----- 4️⃣ MAEEG Reconstruction -----
class MAEEGReconstruction(nn.Module):
    def __init__(self, input_channel, embed_size, downsample_size, kernel_size, dropout, transformer_embed_size, heads, forward_neuron, num_transformers):
        super().__init__()
        self.encoder = MAEEGConvolution(input_channel, embed_size, downsample_size, kernel_size, dropout)
        self.transformer = TransformerEncoder(embed_size=embed_size, heads=heads, forward_neuron=forward_neuron, num_transformers=num_transformers)
        self.decoder = nn.ConvTranspose1d(embed_size, input_channel, kernel_size=3, stride=2, padding=1, output_padding=1)

    # def forward(self, x):
    #     x = self.encoder(x)
    #     print(f"[DEBUG] Transformer 输入形状: {x.shape}")
    #     x = self.transformer(x)
    #     return self.decoder(x.transpose(1, 2))

    def forward(self, x):
        encoded = self.encoder(x)
        representation = self.transformer(encoded)
        reconstructed = self.decoder(representation.transpose(1, 2))  # (batch, embed_dim, seq_len')

        # **确保 reconstructed 的 seq_len 和输入 x 的 seq_len 一致**
        if reconstructed.shape[2] != x.shape[2]:
            reconstructed = nn.functional.interpolate(reconstructed, size=x.shape[2], mode='linear', align_corners=False)

        return reconstructed

# ----- 5️⃣ MAEEG Classification -----
# class MAEEGClassification(nn.Module):
#     def __init__(self, input_channel, embed_size, downsample_size, kernel_size, dropout, transformer_embed_size, heads, forward_neuron, num_transformers, num_classes):
#         super().__init__()
#         self.encoder = MAEEGConvolution(input_channel, embed_size, downsample_size, kernel_size, dropout)
#         self.transformer = TransformerEncoder(embed_size=embed_size, heads=heads, forward_neuron=forward_neuron, num_transformers=num_transformers)
#         self.fc = nn.Linear(embed_size, num_classes)

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.transformer(x)
#         return self.fc(x.mean(dim=1))  # Global Average Pooling

class MAEEGClassification(nn.Module):
    def __init__(self, 
                 input_channel=14, embed_size=128, downsample_size=[2, 2, 2], kernel_size=[3, 3, 3], 
                 dropout=0.1, transformer_embed_size=128, heads=8, forward_neuron=256, 
                 num_transformers=4, num_classes=2):
        super().__init__()
        self.encoder = MAEEGConvolution(input_channel, embed_size, downsample_size, kernel_size, dropout)
        self.transformer = TransformerEncoder(embed_size, heads, forward_neuron, num_transformers)
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))  # Global Average Pooling