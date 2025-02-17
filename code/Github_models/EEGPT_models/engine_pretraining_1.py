import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from functools import partial
from typing import Any, Dict, Optional

# ---- 引入 “动态” EEGTransformer 等，需要你先把 model文件命名好
# 你可以在同一文件内贴上 “dynamic EEGTransformer” 的实现
# 也可以 from .modeling_pretraining import EEGTransformer
# 这里示例内嵌

########################################
# 1) 定义 trunc_normal_ 与 EEGTransformer (动态)
########################################
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2*l - 1, 2*u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.)).add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob=drop_prob
    def drop_path(self,x):
        if self.drop_prob<=0. or not self.training:
            return x
        keep_prob=1-self.drop_prob
        shape=(x.shape[0],)+(1,)*(x.ndim-1)
        random_tensor=keep_prob+torch.rand(shape, dtype=x.dtype, device=x.device)
        mask=random_tensor.floor()
        return x.div(keep_prob)*mask
    def forward(self,x):
        return self.drop_path(x)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        hidden_features=hidden_features or in_features
        out_features=out_features or in_features
        self.fc1=nn.Linear(in_features, hidden_features)
        self.act=nn.GELU()
        self.fc2=nn.Linear(hidden_features, out_features)
        self.drop=nn.Dropout(drop)
    def forward(self,x):
        x=self.fc1(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.fc2(x)
        x=self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, is_causal=False):
        super().__init__()
        self.num_heads=num_heads
        self.head_dim=dim//num_heads
        self.qkv=nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop=attn_drop
        self.proj=nn.Linear(dim, dim)
        self.proj_drop=nn.Dropout(proj_drop)
        self.is_causal=is_causal
    def forward(self,x):
        B,T,C=x.shape
        qkv=self.qkv(x).reshape(B,T,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        out=torch.nn.functional.scaled_dot_product_attention(
            q,k,v,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0,
            is_causal=self.is_causal
        )
        out=out.transpose(1,2).reshape(B,T,C)
        out=self.proj(out)
        out=self.proj_drop(out)
        return out

class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0., attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1=nn.LayerNorm(dim)
        self.attn=Attention(dim,num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path=DropPath(drop_path)
        self.norm2=nn.LayerNorm(dim)
        hidden=int(dim*mlp_ratio)
        self.mlp=MLP(dim, hidden, dim, drop=drop)
    def forward(self,x):
        x=x+self.drop_path(self.attn(self.norm1(x)))
        x=x+self.drop_path(self.mlp(self.norm2(x)))
        return x

class EEGTransformer(nn.Module):
    """
    动态 patchify => (B,C,T)->(B*N, C+1, dim)
    """
    def __init__(self, patch_size=64, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4.0,
                 drop_rate=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.patch_size=patch_size
        self.proj=nn.Conv2d(1, embed_dim, kernel_size=(1,patch_size), stride=(1,patch_size))
        self.chan_embed=nn.Embedding(64, embed_dim)  # 64 as max channel? or bigger
        dpr=[ x.item() for x in torch.linspace(0,drop_path,depth)]
        self.blocks=nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  drop=drop_rate, attn_drop=attn_drop, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm=nn.LayerNorm(embed_dim)
        self.summary=nn.Parameter(torch.zeros(1,1,embed_dim))
        trunc_normal_(self.summary,std=0.02)

    def forward(self,x):
        """
        x: shape (B,C,T)
        return: shape (B*N, embed_dim) after transformer
        """
        B,C,T=x.shape
        if T%self.patch_size!=0:
            raise ValueError(f"T={T} not multiple of patch_size={self.patch_size}")
        N=T//self.patch_size
        # patchify
        x=x.unsqueeze(1)  # =>(B,1,C,T)
        x=self.proj(x)    # =>(B,embed_dim,C,N)
        x=x.permute(0,3,2,1) # =>(B,N,C,embed_dim)
        # + channel embed
        # (for simplicity, assume c<64, or you can clamp index)
        c_idx=torch.arange(C,device=x.device).long()
        ce=self.chan_embed(c_idx).view(1,1,C,-1)
        x=x+ce
        B_new,N_new,C_new,D=x.shape
        x=x.contiguous().view(B_new*N_new,C_new,D)
        # add summary token
        s=self.summary.repeat(B_new*N_new,1,1)
        x=torch.cat([x,s],dim=1) # => (B*N, C+1, D)
        # pass blocks
        for blk in self.blocks:
            x=blk(x)
        x=self.norm(x)
        # (B*N, C+1, D), we can return entire x or just the last
        return x.reshape(B_new*N_new, -1)

########################################
# 2) 定义一个 LitEEGPT, 不要 predictor / reconstructor
########################################
class LitEEGPT(pl.LightningModule):
    def __init__(self, models_configs, USE_LOSS_A=True, USE_LN=True, USE_SKIP=True):
        super().__init__()
        # 只要一个 encoder
        confE=models_configs['encoder']
        self.encoder=EEGTransformer(
            patch_size=64, # or confE["patch_size"] if you define
            embed_dim=confE["embed_dim"],
            depth=confE["depth"],
            num_heads=confE["num_heads"],
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop=0.0,
            drop_path=0.0
        )
    def forward(self,x):
        """
        x shape (B,C,T)
        return shape (B*N, (C+1)*embed_dim) or similar
        """
        return self.encoder(x)

# 其余训练相关都不写, 这样 memory check 直接 forward(x) 就OK


# 你也可以在最底部 if __name__=="__main__": 放  dataset / trainer, 但不影响 memory usage
if __name__=="__main__":
    print("This is a minimal engine_pretraining.py to test forward pass. No dataset loading here.")