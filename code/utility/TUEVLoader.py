import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class TUEVLoader(Dataset):
    """
    一个用于读取 TUEV (.pkl) 数据的 Dataset:
      - root: 存放 .pkl 文件的目录
      - file_list: 该目录下的 .pkl 文件列表(如 ["xxxx-0.pkl", "xxxx-1.pkl", ...])
    """

    def __init__(self, root, file_list):
        super().__init__()
        self.root = root
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # 构造 .pkl 文件的完整路径
        pkl_path = os.path.join(self.root, self.files[index])
        
        # 读取 pkl
        with open(pkl_path, "rb") as f:
            sample = pickle.load(f)
        
        # TUEV 里通常存的是:
        # {
        #   "signal": shape=(16, N),  # EEG数据
        #   "offending_channel": shape=(1,)
        #   "label": shape=(1,)       # 事件标签(可能是0,1,2...等)
        # }
        X = sample["signal"]
        # 如果 label 存在于 shape=(1,), 取第0个即可得到整型
        Y = sample["label"][0]   # 例如 => 0 or 1

        # ⚠️ 让类别索引从 0 开始 (原来是1-6)
        Y = Y - 1  # 现在类别变为 {0, 1, 2, 3, 4, 5}

        # 简单量级归一化：按 95% 分位数缩放
        # 如果不需要可注释掉
        # 这里 axis=-1 表示对最后一个维度做分位数操作，也就是对每个通道单独归一化
        X = X / (
            np.quantile(np.abs(X), q=0.95, axis=-1, keepdims=True) + 1e-8
        )

        # 转成 PyTorch Tensor
        X = torch.FloatTensor(X)  # shape=(16, N)
        Y = torch.tensor(Y, dtype=torch.long)

        return X, Y