# engine_pretraining.py

import os
import math
import random
import copy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pytorch_lightning.utilities.types import STEP_OUTPUT
from functools import partial
import numpy as np
import torchvision

# 从 configs 中拿超参: max_lr, max_epochs, steps_per_epoch 等
from EEGPT_models.configs import max_lr, max_epochs, batch_size, devices
# 这里的 steps_per_epoch 需要你在 configs 或者训练脚本中自行计算. 
# 如果你只做 memory 测试可用个占位.

# 从 modeling_pretraining 导入所有组件
from EEGPT_models.modeling_pretraining import (
    EEGTransformer,
    EEGTransformerPredictor,
    EEGTransformerReconstructor,
    apply_mask
)

# 你可能有一个“全局通道列表”
use_channels_names = [
    'FP1','FPZ','FP2','AF3','AF4',
    'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
    'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
    'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
    'P7','P5','P3','P1','PZ','P2','P4','P6','P8',
    'PO7','PO3','POZ','PO4','PO8',
    'O1','OZ','O2',
]

# 你也可能有个自定义函数
def grad_logger(named_params):
    """
    简化版示例: 用于记录梯度信息
    """
    grads = []
    for n, p in named_params:
        if p.grad is not None:
            grads.append(p.grad.detach().abs().mean().item())
    if len(grads)==0:
        return type("GradStats",(), dict(
            first_layer=0, last_layer=0, min=0, max=0
        )) 
    return type("GradStats",(), dict(
        first_layer=grads[0],
        last_layer=grads[-1],
        min=min(grads),
        max=max(grads)
    ))

###########################################################
# Optional: CosineWDSchedule, WarmupCosineSchedule
# 这里根据你之前的需求贴上:
###########################################################
class CosineWDSchedule:
    """示例: 每个 step 调用一次 step(), 主要调 weight_decay等."""
    def __init__(self, optimizer, ref_wd=1e-6, final_wd=1e-6, T_max=1000):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self.global_step = 0

    def step(self):
        alpha = self.global_step / self.T_max
        wd = self.ref_wd + (self.final_wd - self.ref_wd) * alpha
        for group in self.optimizer.param_groups:
            if 'WD_exclude' not in group or not group['WD_exclude']:
                group['weight_decay'] = wd
        self.global_step +=1

###########################################################
#  LitEEGPT
###########################################################
class LitEEGPT(pl.LightningModule):
    """
    包含 encoder(EEGTransformer), predictor(EEGTransformerPredictor),
    reconstructor(EEGTransformerReconstructor). 以及
    forward_target / forward_context / make_masks / training_step / ...
    """
    def __init__(self, models_configs, USE_LOSS_A=True, USE_LN=True, USE_SKIP=True):
        super().__init__()
        self.USE_LOSS_A = USE_LOSS_A
        self.USE_LN     = USE_LN
        self.USE_SKIP   = USE_SKIP

        # 1) 构建 encoder
        self.encoder = EEGTransformer(
            img_size=(58, 256*4),  # 例如 58通道 x 1024 时间 
            patch_size=64,
            depth=models_configs["encoder"]["depth"],
            num_heads=models_configs["encoder"]["num_heads"],
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            qkv_bias=True,
            embed_dim=models_configs["encoder"]["embed_dim"],
            init_std=0.02,
        )

        # 2) 构建 predictor
        self.predictor = EEGTransformerPredictor(
            num_patches = self.encoder.num_patches,
            embed_dim   = models_configs["predictor"]["embed_dim"],
            depth       = models_configs["predictor"]["depth"],
            num_heads   = models_configs["predictor"]["num_heads"],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
        )

        # 3) 构建 reconstructor
        self.reconstructor = EEGTransformerReconstructor(
            num_patches = self.encoder.num_patches,
            patch_size  =64, 
            embed_dim   = models_configs["reconstructor"]["embed_dim"],
            depth       = models_configs["reconstructor"]["depth"],
            num_heads   = models_configs["reconstructor"]["num_heads"],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
        )

        # 4) target encoder
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad=False

        # 5) default channel-ids
        self.chans_id = self.encoder.prepare_chan_ids(use_channels_names)

        self.loss_fn  = nn.MSELoss()

        # (以下各种scheduler, 见 configure_optimizers)

    def forward(self, x):
        """
        x: shape (B, C, T)
        仅做 forward 测试 / memory usage
        返回 encoder 输出. 真实训练则要 predictor & reconstructor
        """
        # 只跑 encoder => 返回 shape (B*N, embed_dim*(C+1))之类(示例)
        # 取决于 EEGTransformer 的 forward
        z = self.encoder(x, self.chans_id.to(x.device))
        return z

    ##########################################
    #  构建 make_masks / forward_target / forward_context
    ##########################################
    def make_masks(self, num_patchs, mC_x=12, p_n_y=0.5, p_c_y=0.2):
        """
        参考之前: num_patchs=(C,N).
        这里随机生成 mask_x / mask_y
        """
        C,N = num_patchs
        while True:
            mask_x=[]
            mask_y=[]
            mask_y_bx=[]
            for i in range(N):
                c_idx= torch.randperm(C)+ i*C
                if random.random()>p_n_y:
                    mask_x.append(c_idx[:mC_x])
                    mask_y_bx.append(c_idx[mC_x:])
                else:
                    mask_y.append(c_idx)
            if len(mask_x)==0: continue
            if len(mask_y_bx)==0: continue
            mask_y_bx= torch.cat(mask_y_bx,dim=0)
            mask_y_bx= mask_y_bx[torch.rand(mask_y_bx.shape)<p_c_y]
            if len(mask_y_bx)==0: continue
            break
        mask_x= torch.stack(mask_x,dim=0)
        mask_y= torch.cat(mask_y+[mask_y_bx],dim=0)
        return mask_x, mask_y

    def forward_target(self, x, mask_y):
        """
        先 target_encoder(x) => h
        然后把 x reshape成(??), apply_mask => y
        """
        with torch.no_grad():
            h= self.target_encoder(x, self.chans_id.to(x.device))
            h= F.layer_norm(h, (h.size(-1),))
            C,N= self.encoder.num_patches  # run-time shape
            # 这里假设 x.shape => (B,C*, T*), 只要 (C*, T*)= (C* block_size_c, N* block_size_n)
            # 你之前写的 assert x.shape[-1]%N==0 ...
            if x.shape[-2]!=C:
                raise ValueError(f"x.shape[-2]={x.shape[-2]}, but encoder.num_patches[0]={C}")
            if x.shape[-1] % N!=0:
                raise ValueError(f"x.shape[-1]={x.shape[-1]}, but N={N} not dividing it.")
            block_size_c= x.shape[-2]//C
            block_size_n= x.shape[-1]//N

            # x => (B, C, block_size_c, N, block_size_n)
            x= x.view(x.shape[0], C, block_size_c, N, block_size_n)
            x= x.permute(0,3,1,2,4).contiguous() # =>(B,N,C, block_size_c, block_size_n)
            # => (B,C,N, block_size_c * block_size_n)
            x= x.view(x.shape[0], C, N, block_size_c*block_size_n)
            y= apply_mask(mask_y.to(x.device), x)
            if self.USE_LN:
                y= F.layer_norm(y,(y.size(-1),))
            return h,y

    def forward_context(self, x, mask_x, mask_y):
        """
        1) encoder => z
        2) predictor => z, comb_z
        3) reconstructor => r
        """
        z= self.encoder(x, self.chans_id.to(x.device), mask_x=mask_x)
        z, comb_z= self.predictor(z, mask_x=mask_x)
        if not self.USE_SKIP:
            comb_z= z
        r= self.reconstructor(comb_z, self.chans_id.to(x.device), mask_y=mask_y)
        return z, r

    ##########################################
    #  train / val step
    ##########################################
    def training_step(self, batch, batch_idx):
        """
        1) make masks
        2) forward_target => h,y
        3) forward_context => z,r
        4) loss => MSE
        """
        x, _= batch  # (B, C, T)
        mask_x, mask_y= self.make_masks(self.encoder.num_patches)  # (C,N)
        h,y= self.forward_target(x, mask_y)
        z,r= self.forward_context(x, mask_x, mask_y)
        loss1= self.loss_fn(h,z)
        loss2= self.loss_fn(y,r)
        if self.USE_LOSS_A:
            loss= loss1+loss2
        else:
            loss= loss2

        self.log('train_loss1', loss1, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_loss2', loss2, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_loss',  loss,  on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,_= batch
        mask_x, mask_y= self.make_masks(self.encoder.num_patches)
        h,y= self.forward_target(x,mask_y)
        z,r= self.forward_context(x,mask_x,mask_y)
        loss1= self.loss_fn(h,z)
        loss2= self.loss_fn(y,r)
        if self.USE_LOSS_A:
            loss= loss1+loss2
        else:
            loss= loss2
        self.log('valid_loss1', loss1, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_loss2', loss2, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_loss',  loss,  on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        self.wd_scheduler.step()
        return super().on_train_batch_start(batch,batch_idx)

    def on_train_batch_end(self, outputs:STEP_OUTPUT, batch, batch_idx):
        # log grad stats
        gstats= grad_logger(self.encoder.named_parameters())
        self.log('grad_stats.first_layer',gstats.first_layer, on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.last_layer', gstats.last_layer,  on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.min',        gstats.min,         on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.max',        gstats.max,         on_epoch=True, on_step=False, sync_dist=True)

        # momentum update of target encoder
        with torch.no_grad():
            m= next(self.momentum_scheduler)
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m)* param_q.detach().data)

        return super().on_train_batch_end(outputs,batch,batch_idx)

    def on_load_checkpoint(self, checkpoint: Dict[str,Any]):
        res= super().on_load_checkpoint(checkpoint)
        self.configure_optimizers()
        return res

    ########################################
    #  optimizer / scheduler
    ########################################
    def configure_optimizers(self):
        param_groups= [
            {
                'params':(p for n,p in self.encoder.named_parameters() if ('bias' not in n) and (len(p.shape)!=1))
            },{
                'params':(p for n,p in self.predictor.named_parameters() if ('bias' not in n) and (len(p.shape)!=1))
            },{
                'params':(p for n,p in self.reconstructor.named_parameters() if ('bias' not in n) and (len(p.shape)!=1))
            },{
                'params':(p for n,p in self.encoder.named_parameters() if ('bias' in n) or (len(p.shape)==1)),
                'WD_exclude': True,
                'weight_decay':0
            },{
                'params':(p for n,p in self.predictor.named_parameters() if ('bias' in n) or (len(p.shape)==1)),
                'WD_exclude': True,
                'weight_decay':0
            },{
                'params':(p for n,p in self.reconstructor.named_parameters() if ('bias' in n) or (len(p.shape)==1)),
                'WD_exclude': True,
                'weight_decay':0
            }
        ]
        optimizer= torch.optim.AdamW(param_groups, lr=6e-5)

        # 你原先的一些全局
        # e.g. steps_per_epoch= ...
        steps_per_epoch= 1000  # 占位; 你需要在外部根据 train_loader 计算
        lr_scheduler= torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr= 6e-5,
            steps_per_epoch= steps_per_epoch,
            epochs= max_epochs,
            div_factor=2,
            final_div_factor=8,
            pct_start= 0.2
        )
        lr_dict= {
            'scheduler': lr_scheduler,
            'interval':'step',
            'frequency':1,
            'monitor':'valid_loss',
            'strict':True,
            'name': None
        }

        # WD scheduler
        self.wd_scheduler= CosineWDSchedule(
            optimizer, ref_wd=1e-6, final_wd=1e-6, T_max= (max_epochs*steps_per_epoch)
        )
        # momentum
        ema= [0.996,1.0]
        self.momentum_scheduler= (
            ema[0] + i*(ema[1]-ema[0])/(steps_per_epoch*max_epochs)
            for i in range( steps_per_epoch*max_epochs+1)
        )
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )


###########################################################
# 你也可以在 if __name__=="__main__" 写训练逻辑
###########################################################
if __name__=="__main__":
    print("Running as main => e.g. dataset or trainer code here. \n"
          "If you only do memory usage test, you can ignore this block.")
    # 这里放任何可能的训练 or debug
    pass