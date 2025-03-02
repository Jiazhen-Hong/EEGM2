"""
Created on Tue Nov 14 18:12:39 2024
@author: jiazhen@emotiv.com
"""

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

class MSEWithSpectralLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        """
        MSE + Spectral Loss
        Args:
            alpha (float): Weight for MSE Loss
            beta (float): Weight for Spectral Loss
        """
        super(MSEWithSpectralLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        # Time domain loss (MSE)
        mse = self.mse_loss(y_pred, y_true)
        
        # Frequency domain loss (Spectral Loss)
        y_true_fft = torch.fft.rfft(y_true, dim=-1)
        y_pred_fft = torch.fft.rfft(y_pred, dim=-1)
        
        spectral_loss = torch.mean(torch.abs(y_true_fft - y_pred_fft)**2)
        
        # Combine the losses
        total_loss = self.alpha * mse + self.beta * spectral_loss
        return total_loss


class L1WithSpectralLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(L1WithSpectralLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()

    def forward(self, y_pred, y_true):
        l1_loss = self.l1_loss(y_pred, y_true)
        y_true_fft = torch.fft.rfft(y_true, dim=-1)
        y_pred_fft = torch.fft.rfft(y_pred, dim=-1)
        spectral_loss = torch.mean(torch.abs(y_true_fft - y_pred_fft) ** 2)
        total_loss = self.alpha * l1_loss + self.beta * spectral_loss
        return total_loss
    
