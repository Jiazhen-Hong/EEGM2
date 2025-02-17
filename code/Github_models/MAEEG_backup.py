# Provide by Soheila, Navid 02/06/2025
# 
#
#
import math
import torch
import torch.nn as nn
import numpy as np


class MAEEGConvolution(nn.Module):
    def __init__(self, input_channel, output_channel, downsample_size, kernel_size, dropout):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel

        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size]
        if not isinstance(downsample_size, (list, tuple)):
            downsample_size = [downsample_size]
        assert len(kernel_size) == len(downsample_size)

        # Centerable convolutions make life simpler
        kernel_size = [e if e % 2 else e + 1 for e in kernel_size]
        self.kernel_size = kernel_size
        self.downsample_size = downsample_size

        self.encoder = nn.Sequential()

        for i, (kernel, downsample) in enumerate(zip(kernel_size, downsample_size)):
            self.encoder.add_module(f"MAEEG encoder {i}", nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=kernel, stride=downsample, padding=kernel//2),
                nn.Dropout(dropout),
                nn.GroupNorm(output_channel // 2, output_channel),
                nn.GELU()
            ))
            input_channel = output_channel


    def forward(self, x):
        out = self.encoder(x)
        out = torch.transpose(out, 1, 2)
        return out

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim*heads == embed_size), "Embed size needs to be diveded by heads."

        # Define linear projection
        self.queries = []
        self.keys = []
        self.values = []
        for i in range(self.heads):
            self.queries.append(nn.Linear(self.embed_size, self.head_dim, bias=False))
            self.keys.append(nn.Linear(self.embed_size, self.head_dim, bias=False))
            self.values.append(nn.Linear(self.embed_size, self.head_dim, bias=False))

        self.linear = nn.Linear(self.heads*self.head_dim, self.embed_size, bias=False)

    def forward(self, x, mask=None):
        attention_outs = []
        for i in range(self.heads):
            query = self.queries[i](x)
            key = self.keys[i](x)
            value = self.values[i](x)
            dot_prod = torch.einsum("iqd,ikd->iqk", query, key)
            if mask is not None:
                dot_prod = dot_prod.masked_fill(mask == 0, float(-1e20))
            attention = torch.softmax(dot_prod / self.head_dim**(1/2), dim=2)
            attention_out = torch.einsum("iqk,ikd->iqd", attention, value)
            attention_outs.append(attention_out)
        attention_outs = torch.cat(attention_outs, 2)
        out = self.linear(attention_outs)
        return out

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

class PositionalEncoding(nn.Module):
    def __init__(self, timesteps, embed_size):
        super().__init__()
        self.pe = torch.zeros(timesteps, embed_size)
        for t in range(timesteps):
            for i in range(embed_size):
                if i % 2 == 0:
                    self.pe[t, i] = torch.sin(torch.tensor(t / (10000)**(i / embed_size)))
                else:
                    self.pe[t, i] = torch.cos(torch.tensor(t / (10000)**(i / embed_size)))
        self.pe.requires_grad = False

    def forward(self, x):
        return x + self.pe

def positional_encoding(x, timesteps, embed_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pe = torch.zeros(timesteps, embed_size)
    pe = pe.to(device)
    for t in range(timesteps):
        for i in range(embed_size):
            if i % 2 == 0:
                pe[t, i] = torch.sin(torch.tensor(t / (10000) ** (i / embed_size)))
            else:
                pe[t, i] = torch.cos(torch.tensor(t / (10000) ** (i / embed_size)))
    pe.requires_grad = False
    out = x + pe
    return out



class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, transformer_embed_size, heads, forward_neuron, num_transformers):
        super().__init__()
        self.embed_size = embed_size
        self.input_conditioning = nn.Conv1d(in_channels=embed_size, out_channels=transformer_embed_size, kernel_size=1)
        self.transformer_blocks = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=transformer_embed_size, nhead=heads,
                                                                                                 dim_feedforward=forward_neuron,),
                                                        num_layers=num_transformers)
        self.output_conditioning = nn.Conv1d(in_channels=transformer_embed_size, out_channels=embed_size, kernel_size=1)

    def forward(self, x):
        out = positional_encoding(x, x.shape[1], self.embed_size)
        out = torch.transpose(out, 1, 2)
        out = self.input_conditioning(out)
        out = torch.transpose(out, 1, 2)
        out = self.transformer_blocks(out)
        out = torch.transpose(out, 1, 2)
        out = self.output_conditioning(out)
        out = torch.transpose(out, 1, 2)
        return out


class MAEEGConvDecoder(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, upsample_size, dropout, output_size):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.output_size = output_size

        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size]
        if not isinstance(upsample_size, (list, tuple)):
            upsample_size = [upsample_size]
        assert len(kernel_size) == len(upsample_size)

        # Centerable convolutions make life simpler
        kernel_size = [e if e % 2 else e + 1 for e in kernel_size]
        self.kernel_size = kernel_size[::-1]
        self.upsample_size = upsample_size[::-1]

        self.decoder = nn.Sequential()

        for i, (kernel, upsample) in enumerate(zip(self.kernel_size, self.upsample_size)):
            output_padding = upsample - 1 if self.output_size[i] % upsample == 0 else self.output_size[i] % upsample - 1
            if i != len(kernel_size) - 1:
                self.decoder.add_module(f"MAEEG decoder {i}", nn.Sequential(
                    nn.ConvTranspose1d(input_channel, input_channel, kernel_size=kernel, stride=upsample,
                                       padding=kernel//2, output_padding=output_padding),
                    nn.Dropout(dropout),
                    nn.GroupNorm(input_channel // 2, input_channel),
                    nn.GELU()
                ))
            else:
                self.decoder.add_module(f"MAEEG decoder {i}", nn.Sequential(
                    nn.ConvTranspose1d(input_channel, output_channel, kernel_size=kernel, stride=upsample,
                                       padding=kernel//2, output_padding=output_padding),
                ))

    def forward(self, x):
        out = torch.transpose(x, 1, 2)
        out = self.decoder(out)
        return out


class MAEEGReconstruction(nn.Module):
    def __init__(self, input_channel, embed_size, downsample_size, kernel_size, dropout,
                 transformer_embed_size, heads, forward_neuron, num_transformers, mask_p, mask_chunk, output_size):
        super().__init__()
        self.maeeg_encoder = MAEEGConvolution(input_channel=input_channel, output_channel=embed_size,
                                              downsample_size=downsample_size, kernel_size=kernel_size,
                                              dropout=dropout)
        self.transformer_encoder = TransformerEncoder(embed_size=embed_size, transformer_embed_size=transformer_embed_size, heads=heads, forward_neuron=forward_neuron,
                                                      num_transformers=num_transformers)
        self.maeeg_decoder = MAEEGConvDecoder(input_channel=embed_size, output_channel=input_channel,
                                              upsample_size=downsample_size, kernel_size=kernel_size,
                                              dropout=dropout, output_size=output_size)
        self.mask_p = mask_p
        self.mask_chunk = mask_chunk

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = self.maeeg_encoder(x)
        batch, len_seq, len_feature = out.shape
        self.mask = make_mask(shape=(batch, len_seq), p=self.mask_p, total=len_seq,
                              chunk=self.mask_chunk, allow_no_inds=False)
        mask_replacement = torch.nn.Parameter(torch.zeros(len_feature), requires_grad=True)
        mask_replacement = mask_replacement.to(device)
        out[self.mask] = mask_replacement
        out = self.transformer_encoder(out)
        out = self.maeeg_decoder(out)
        return out

    def save_convolution(self, path):
        torch.save(self.maeeg_encoder.state_dict(), path)

    def save_transformer(self, path):
        torch.save(self.transformer_encoder.state_dict(), path)


class MAEEGReconstructionV2(nn.Module):
    def __init__(self, input_channel, embed_size, downsample_size, kernel_size, dropout,
                 transformer_embed_size, heads, forward_neuron, num_transformers, mask_p, mask_chunk, output_size):
        super().__init__()
        self.maeeg_encoder = MAEEGConvolution(input_channel=input_channel, output_channel=embed_size,
                                              downsample_size=downsample_size, kernel_size=kernel_size,
                                              dropout=dropout)
        self.transformer_encoder = TransformerEncoder(embed_size=embed_size, transformer_embed_size=transformer_embed_size, heads=heads, forward_neuron=forward_neuron,
                                                      num_transformers=num_transformers)
        self.maeeg_decoder = MAEEGConvDecoder(input_channel=embed_size, output_channel=input_channel,
                                              upsample_size=downsample_size, kernel_size=kernel_size,
                                              dropout=dropout, output_size=output_size)
        self.mask_p = mask_p
        self.mask_chunk = mask_chunk

    def forward(self, x):
        out = self.maeeg_encoder(x)
        out = self.transformer_encoder(out)
        out = self.maeeg_decoder(out)
        return out

    def forward_mask(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = self.maeeg_encoder(x)
        batch, len_seq, len_feature = out.shape
        self.mask = make_mask(shape=(batch, len_seq), p=self.mask_p, total=len_seq,
                              chunk=self.mask_chunk, allow_no_inds=False)
        mask_replacement = torch.nn.Parameter(torch.zeros(len_feature), requires_grad=True)
        mask_replacement = mask_replacement.to(device)
        out[self.mask] = mask_replacement
        out = self.transformer_encoder(out)
        out = self.maeeg_decoder(out)
        return out

    def save_convolution(self, path):
        torch.save(self.maeeg_encoder.state_dict(), path)

    def save_transformer(self, path):
        torch.save(self.transformer_encoder.state_dict(), path)



class BinaryClassifier(nn.Module):
    def __init__(self, fc_neuron, dropout):
        super().__init__()
        self.fc_neuron = fc_neuron
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=1, out_features=fc_neuron)
        self.linear2 = nn.Linear(in_features=fc_neuron, out_features=1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = self.flatten(x)
        batch, in_features = out.shape
        if self.linear1.in_features == 1:
            self.linear1 = nn.Linear(in_features=in_features, out_features=self.fc_neuron, device=device)
        out = self.relu(self.linear1(out))
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


class BinaryClassifierV2(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features=1, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = self.flatten(x)
        batch, in_features = out.shape
        out = self.dropout(out)
        if self.linear.in_features == 1:
            self.linear = nn.Linear(in_features=in_features, out_features=1, device=device)
        out = self.linear(out)
        out = self.sigmoid(out)
        return out


class MAEEGClassification(nn.Module):
    def __init__(self, input_channel, embed_size, downsample_size,
                 kernel_size, dropout, transformer_embed_size, heads, forward_neuron,
                 num_transformers, fc_neuron, dropout_classifier, use_transformer=False):
        super().__init__()
        self.maeeg_encoder = MAEEGConvolution(input_channel=input_channel, output_channel=embed_size,
                                              downsample_size=downsample_size, kernel_size=kernel_size,
                                              dropout=dropout)
        self.transformer_encoder = TransformerEncoder(embed_size=embed_size, transformer_embed_size=transformer_embed_size, heads=heads, forward_neuron=forward_neuron,
                                                      num_transformers=num_transformers)
        self.classifier = BinaryClassifier(fc_neuron=fc_neuron, dropout=dropout_classifier)
        self.use_transformer = use_transformer

    def load_convolution_encoder(self, path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.maeeg_encoder.load_state_dict(torch.load(path, map_location=device))

    def load_transformer_encoder(self, path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transformer_encoder.load_state_dict(torch.load(path, device))

    def freeze_convolution(self):
        for param in self.maeeg_encoder.parameters():
            param.requires_grad = False

    def freeze_transformer(self):
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.maeeg_encoder(x)
        if self.use_transformer:
            out = self.transformer_encoder(out)
        out = self.classifier(out)
        return out


class MAEEGClassificationV2(nn.Module):
    def __init__(self, input_channel, embed_size, downsample_size,
                 kernel_size, dropout, transformer_embed_size, heads, forward_neuron,
                 num_transformers, dropout_classifier, use_transformer=False):
        super().__init__()
        self.maeeg_encoder = MAEEGConvolution(input_channel=input_channel, output_channel=embed_size,
                                              downsample_size=downsample_size, kernel_size=kernel_size,
                                              dropout=dropout)
        self.transformer_encoder = TransformerEncoder(embed_size=embed_size, transformer_embed_size=transformer_embed_size, heads=heads,
                                                      forward_neuron=forward_neuron, num_transformers=num_transformers)
        self.classifier = BinaryClassifierV2(dropout_classifier)
        self.use_transformer = use_transformer

    def load_convolution_encoder(self, path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.maeeg_encoder.load_state_dict(torch.load(path, map_location=device))

    def load_transformer_encoder(self, path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transformer_encoder.load_state_dict(torch.load(path, map_location=device))

    def freeze_convolution(self):
        for param in self.maeeg_encoder.parameters():
            param.requires_grad = False

    def freeze_transformer(self):
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.maeeg_encoder(x)
        if self.use_transformer:
            out = self.transformer_encoder(out)
        out = self.classifier(out)
        return out


# I don't use this class
class ReconstructionLoss(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, input, target):
        batches, masked_t = self.mask.shape
        loss = 0
        loss_fn = nn.MSELoss()
        for batch in range(batches):
            for t in range(masked_t):
                loss += loss_fn(input[batch][t], target[batch][t])
        return loss


def make_span_from_seeds(seeds, span, total=None):
    inds = list()
    # Loop for masked indices
    for seed in seeds:
        for i in range(seed, seed + span):
            if total is not None and i >= total:
                break
            # At least, there is a span between indices so that only the head indices can get masked
            elif i not in inds:
                inds.append(int(i))
    return np.array(inds)


def make_mask(shape, p, total, chunk, allow_no_inds=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize mask tensor
    mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)
    span = math.ceil(total*p) // chunk
    assert span > 0 and span*chunk + (chunk-1) <= total, "chunk is too large."

    for i in range(shape[0]):
        choice = list()
        flag = False
        while not allow_no_inds and p > 0 and flag is False:
            # Get nonzero indices (get masked indices given a probability)
            choice = np.random.choice(total-(span-1), chunk, replace=False)
            choice = np.sort(choice)
            flag = True
            for j in range(len(choice)):
                if j != len(choice) - 1 and choice[j+1] - choice[j] <= span:
                    flag = False

        # Get final mask tensor
        mask[i, make_span_from_seeds(choice, span, total=total)] = True
        mask = mask.to(device)

    return mask


def get_time_input_sizes_after_convolution(downsample_size, time_input_size):
    input_size = [time_input_size]
    output_size = time_input_size
    for i in range(len(downsample_size)):
        output_size = math.ceil(output_size / downsample_size[i])
        if i != len(downsample_size) - 1:
            input_size.append(output_size)
    input_size = input_size[::-1]
    return input_size
