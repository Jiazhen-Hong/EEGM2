import math
import torch
import torch.nn as nn
import torch.nn.functional as F

########################################
#  Channel Index
########################################
CHANNEL_DICT = {k.upper():v for v,k in enumerate(
    [
     'FP1','FPZ','FP2','AF7','AF3','AF4','AF8',
     'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
     'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
     'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
     'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
     'P7','P5','P3','P1','PZ','P2','P4','P6','P8',
     'PO7','PO3','POZ','PO4','PO8',
     'O1','OZ','O2'
    ]
)}

########################################
#  Truncated Normal init
########################################
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x/math.sqrt(2.))) / 2.
    with torch.no_grad():
        l= norm_cdf((a-mean)/std)
        u= norm_cdf((b-mean)/std)
        tensor.uniform_(2*l-1,2*u-1)
        tensor.erfinv_()
        tensor.mul_(std*math.sqrt(2.)).add_(mean)
        tensor.clamp_(min=a,max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor,mean,std,a,b)


########################################
#  apply_mask, apply_mask_t
########################################
def apply_mask(mask, x):
    """
    x: (B,N,C,D)
    mask: (mN,mC)
    """
    B,N,C,D= x.shape
    if len(mask.shape)==2:
        mN,mC= mask.shape
        mask_keep= mask.reshape(1,mN*mC,1).repeat(B,1,D)
        out= torch.gather(x.reshape(B,N*C,D), dim=-2,index=mask_keep)
        out= out.view(B,mN,mC,D)
    else:
        mN= mask.shape[0]
        mask_keep= mask.reshape(1,mN,1).repeat(B,1,D)
        out= torch.gather(x.reshape(B,N*C,D), dim=-2,index=mask_keep)
    return out

def apply_mask_t(mask_t, x):
    """
    x: (B,N,D)
    mask_t: (mN,)
    """
    B,N,D= x.shape
    mN= mask_t.shape[0]
    mask_keep= mask_t.view(1,mN,1).repeat(B,1,D)
    out= torch.gather(x, dim=1, index=mask_keep)
    return out

########################################
#  DropPath, MLP, Attention, Block
########################################
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob= drop_prob
    def drop_path(self,x):
        if self.drop_prob<=0. or not self.training:
            return x
        keep_prob=1.-self.drop_prob
        shape=(x.shape[0],)+(1,)*(x.ndim-1)
        r= torch.rand(shape,dtype=x.dtype,device=x.device)+ keep_prob
        mask= r.floor()
        return x.div(keep_prob)*mask
    def forward(self,x):
        return self.drop_path(x)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 drop=0.):
        super().__init__()
        hidden_features= hidden_features or in_features
        out_features= out_features or in_features
        self.fc1= nn.Linear(in_features, hidden_features)
        self.act= nn.GELU()
        self.fc2= nn.Linear(hidden_features, out_features)
        self.drop= nn.Dropout(drop)
    def forward(self,x):
        x= self.fc1(x)
        x= self.act(x)
        x= self.drop(x)
        x= self.fc2(x)
        x= self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 is_causal=False, use_rope=False, return_attention=False):
        super().__init__()
        self.num_heads= num_heads
        self.head_dim= dim//num_heads
        self.use_rope= use_rope
        self.qkv= nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop= attn_drop
        self.proj= nn.Linear(dim,dim)
        self.proj_drop= nn.Dropout(proj_drop)
        self.is_causal= is_causal
        self.return_attention= return_attention

    def forward(self,x, freqs=None):
        B,T,C= x.shape
        qkv= self.qkv(x).reshape(B,T,3,self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        q,k,v= qkv[0],qkv[1],qkv[2]
        # e.g. scaled_dot_product_attention
        out= torch.nn.functional.scaled_dot_product_attention(
            q,k,v,
            dropout_p= (self.attn_drop if self.training else 0),
            is_causal= self.is_causal
        )
        out= out.transpose(1,2).reshape(B,T,C)
        out= self.proj(out)
        out= self.proj_drop(out)
        return out

class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1= nn.LayerNorm(dim)
        self.attn= Attention(dim,num_heads,qkv_bias,attn_drop,drop)
        self.drop_path= DropPath(drop_path)
        self.norm2= nn.LayerNorm(dim)
        hidden= int(dim*mlp_ratio)
        self.mlp= MLP(dim,hidden,dim,drop)
    def forward(self,x, freqs=None):
        x= x + self.drop_path(self.attn(self.norm1(x), freqs))
        x= x + self.drop_path(self.mlp(self.norm2(x)))
        return x


########################################
# EEGTransformer (Encoder)
########################################
class EEGTransformer(nn.Module):
    """
    动态 patchify => (B,C,T)-> conv2d => shape (B,embed_dim,C,N)
      => permute => (B,N,C,embed_dim)
      => flatten =>(B*N,C,embed_dim) => + summary => pass blocks => ...
    并且 `self.num_patches=(C, N)` 被 predictor / reconstructor 引用

    同时有 prepare_chan_ids() 方便 engine_pretraining里调用
    """
    def __init__(self,
                 img_size=(58, 256*4),
                 patch_size=64,
                 depth=12,
                 num_heads=8,
                 mlp_ratio=4.0,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 qkv_bias=True,
                 embed_dim=768,
                 init_std=0.02,
                 **kwargs
                ):
        super().__init__()
        self.img_size= img_size   # (58,1024) as default
        self.patch_size= patch_size

        # 先写个 conv2d patchify
        self.proj= nn.Conv2d(1, embed_dim, kernel_size=(1,patch_size),
                             stride=(1,patch_size))
        # channel embed
        self.chan_embed= nn.Embedding(len(CHANNEL_DICT), embed_dim)

        # 构造 transformer block
        dpr= [ x.item() for x in torch.linspace(0,drop_path_rate, depth)]
        self.blocks= nn.ModuleList([
            Block(dim=embed_dim,num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm= nn.LayerNorm(embed_dim)
        self.summary= nn.Parameter(torch.zeros(1,1,embed_dim))
        trunc_normal_(self.summary,std=init_std)

        # 初始化
        self.apply(self._init_weights)
        self.fix_init_weight()

        # num_patches 默认
        C0= self.img_size[0]
        T0= self.img_size[1]
        N0= T0// patch_size
        self.num_patches= (C0, N0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0*(layer_id)))
        for i,blk in enumerate(self.blocks):
            rescale(blk.attn.proj.weight.data,i+1)
            rescale(blk.mlp.fc2.weight.data,i+1)

    def _init_weights(self,m):
        if isinstance(m,nn.Linear):
            trunc_normal_(m.weight,std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias,0)
            nn.init.constant_(m.weight,1.0)
        elif isinstance(m,nn.Conv2d):
            trunc_normal_(m.weight,std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.Embedding):
            nn.init.normal_(m.weight,std=0.02)

    def prepare_chan_ids(self, channel_list):
        """ 跟原始版本一样 """
        chan_ids=[]
        for ch in channel_list:
            ch= ch.upper().strip('.')
            assert ch in CHANNEL_DICT, f"{ch} not in channel dict"
            chan_ids.append( CHANNEL_DICT[ch] )
        return torch.tensor(chan_ids).unsqueeze(0).long()

    def forward(self, x, chan_ids=None, mask_x=None):
        """
        x: (B,C,T)
        1) T must be multiple of self.patch_size
        2) update self.num_patches= (C,N) => predictor/reconstructor可用
        """
        B,C,T= x.shape
        if T%self.patch_size!=0:
            raise ValueError(f"T={T} not multiple of patch_size={self.patch_size}")
        N= T// self.patch_size
        self.num_patches= (C,N)

        # patchify
        x= x.unsqueeze(1) # =>(B,1,C,T)
        x= self.proj(x)   # =>(B,embed_dim,C,N)
        x= x.permute(0,3,2,1).contiguous() # =>(B,N,C,embed_dim)

        # + channel embed
        if chan_ids is None:
            chan_ids= torch.arange(C,device=x.device,dtype=torch.long)
        c_embed= self.chan_embed(chan_ids)  # =>(C, embed_dim)
        x= x+ c_embed.view(1,1,C,-1)

        # mask_x?
        if mask_x is not None:
            x= apply_mask(mask_x, x)
        B_new,N_new,C_new,D= x.shape
        x= x.view(B_new*N_new, C_new, D)

        # summary
        s= self.summary.repeat(B_new*N_new,1,1)  # =>(B*N,1,embed_dim)
        x= torch.cat([x,s],dim=1)               # =>(B*N,C+1,embed_dim)

        # pass blocks
        for blk in self.blocks:
            x= blk(x)

        # layernorm & flatten
        x= self.norm(x)
        x= x.reshape(B_new*N_new, -1)
        return x


########################################
# EEGTransformerPredictor
########################################
class EEGTransformerPredictor(nn.Module):
    def __init__(self,
                 num_patches,
                 embed_dim=768,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 **kwargs):
        super().__init__()
        self.num_patches= num_patches  # (C,N)
        self.linear_in= nn.Linear(embed_dim, embed_dim)
        dpr= [ x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks= nn.ModuleList([
            Block(dim=embed_dim,num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm= nn.LayerNorm(embed_dim)

    def forward(self, x, mask_x=None):
        """
        x: shape (B, ???)
        """
        # for demonstration
        # shape => pass blocks => ...
        x= self.linear_in(x)
        for blk in self.blocks:
            x= blk(x)
        x= self.norm(x)
        #  returning z, comb_z
        return x, x


########################################
# EEGTransformerReconstructor
########################################
class EEGTransformerReconstructor(nn.Module):
    def __init__(self,
                 num_patches,
                 patch_size=64,
                 embed_dim=768,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 **kwargs):
        super().__init__()
        self.num_patches= num_patches
        self.patch_size= patch_size
        dpr= [ x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks= nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm= nn.LayerNorm(embed_dim)
        self.proj= nn.Linear(embed_dim, patch_size)

    def forward(self, x, chan_ids=None, mask_y=None):
        """
        x: shape (B, ???)
        Return shape => ...
        """
        for blk in self.blocks:
            x= blk(x)
        x= self.norm(x)
        x= self.proj(x)
        return x