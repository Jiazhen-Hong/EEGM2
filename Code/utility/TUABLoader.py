"""
Created on Tue Nov 14 18:12:39 2024
@author: jiazhen@emotiv.com
"""
import pickle  
from torch.utils.data import Dataset, DataLoader  
import torch
import os 
import numpy as np

class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, tuab_data):
        root, files = tuab_data
        self.root = root
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        X = X[:16, :] # TODO 
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y