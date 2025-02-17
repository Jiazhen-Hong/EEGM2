# configs.py

import torch
import torchvision
import math
import random
import os

def load_fn(x):
    """
    Data loading function for DatasetFolder loader=...
    This is fine to keep top-level because it won't run
    until used by the actual dataset creation.
    """
    x = torch.load(x)
    
    window_length = 4 * 256
    data_length = x.shape[1]

    max_start_index = data_length - window_length
    if max_start_index > 0:
        index = random.randint(0, max_start_index)
        x = x[:, index : index + window_length]
    x = x.to(torch.float)
    return x


# -----------------------
# Hyperparameters
# -----------------------
max_epochs = 200
max_lr     = 5e-4
batch_size = 64
devices    = [0]


# -----------------------
# Model configs
# -----------------------
MODELS_CONFIGS = {
    "tiny1": {
        "embed_dim": 64,
        "embed_num": 1,
        "depth": [2, 2, 4],
        "num_heads": 4
    },
    "tiny2": {
        "embed_dim": 64,
        "embed_num": 4,
        "depth": [2, 2, 4],
        "num_heads": 4
    },
    "tiny3": {
        "embed_dim": 64,
        "embed_num": 4,
        "depth": [8, 8, 8],
        "num_heads": 4
    },
    "little": {
        "embed_dim": 128,
        "embed_num": 4,
        "depth": [8, 8, 8],
        "num_heads": 4
    },
    "base1": {
        "embed_dim": 256,
        "embed_num": 1,
        "depth": [6, 6, 6],
        "num_heads": 4
    },
    "base2": {
        "embed_dim": 256,
        "embed_num": 4,
        "depth": [8, 8, 8],
        "num_heads": 4
    },
    "base3": {
        "embed_dim": 512,
        "embed_num": 1,
        "depth": [6, 6, 6],
        "num_heads": 8
    },
    "large": {
        "embed_dim": 512,
        "embed_num": 4,
        "depth": [8, 8, 8],
        "num_heads": 8
    },
}


def get_config(embed_dim=512, embed_num=4, depth=[8, 8, 8], num_heads=4):
    """
    Return a dict describing the sub-config for encoder/predictor/reconstructor.
    """
    models_configs = {
        'encoder': {
            'embed_dim': embed_dim,
            'embed_num': embed_num,
            'depth': depth[0],
            'num_heads': num_heads,
        },
        'predictor': {
            'embed_dim': embed_dim,
            'embed_num': embed_num,
            'predictor_embed_dim': embed_dim,
            'depth': depth[1],
            'num_heads': num_heads,
        },
        'reconstructor': {
            'embed_dim': embed_dim,
            'embed_num': embed_num,
            'reconstructor_embed_dim': embed_dim,
            'depth': depth[2],
            'num_heads': num_heads,
        },
    }
    return models_configs


# --------------------------------------------------------
# The dataset creation is put inside a if __name__=="__main__" block
# --------------------------------------------------------
if __name__ == "__main__":
    print("[configs.py] Running as main script => we load dataset + define loaders")
    
    train_folder = "../datasets/pretrain/merged/TrainFolder/"
    valid_folder = "../datasets/pretrain/merged/ValidFolder/"
    
    # Only do this if the directories exist or if you intend to run training
    if os.path.isdir(train_folder) and os.path.isdir(valid_folder):
        train_dataset = torchvision.datasets.DatasetFolder(
            root=train_folder,
            loader=load_fn,
            extensions=['.edf'])
        valid_dataset = torchvision.datasets.DatasetFolder(
            root=valid_folder,
            loader=load_fn,
            extensions=['.edf'])

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False
        )

        steps_per_epoch = math.ceil(len(train_loader) / len(devices))

        tag = "tiny1"
        variant = "D"

        print("Now you can do your training or debugging if you want.")
        print(f"Found {len(train_dataset)} train samples, {len(valid_dataset)} val samples.")
        print(f"steps_per_epoch={steps_per_epoch}, tag={tag}, variant={variant}")

    else:
        print(f"Directory {train_folder} or {valid_folder} not found. Skipping dataset creation.")
        # Or raise an error if you *must* have them