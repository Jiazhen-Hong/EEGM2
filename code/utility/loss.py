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

# Example usage:
#criterion = MSEWithSpectralLoss(alpha=1.0, beta=0.5)  # Adjust alpha and beta as needed
#loss = criterion(predicted_signal, target_signal)


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
    

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 控制难分类样本的权重
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)  # 普通交叉熵损失
        pt = torch.exp(-ce_loss)  # 计算 pt（预测正确的概率）
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # Focal Loss 公式

        if self.alpha is not None:
            focal_loss *= self.alpha[labels]  # 适配类别权重

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        


class BalancedCELoss(nn.Module):
    def __init__(self, beta=0.99):
        super(BalancedCELoss, self).__init__()
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        probs = torch.softmax(logits, dim=1)  # 计算类别概率
        effective_num = 1.0 - self.beta ** probs.sum(dim=0)  # 计算类别的有效样本数
        weights = (1.0 - self.beta) / (effective_num + 1e-6)  # 计算类别权重
        weights = weights / weights.sum() * len(weights)  # 归一化
        loss = self.ce_loss(logits, labels) * weights[labels]  # 加权损失
        return loss.mean()
    

class CBLoss(nn.Module):
    def __init__(self, beta=0.999, num_classes=6):
        super(CBLoss, self).__init__()
        self.beta = beta
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        samples_per_class = torch.tensor([513, 6986, 4761, 946, 8721, 48288]).float().to(logits.device)
        effective_num = 1.0 - self.beta ** samples_per_class
        class_weights = (1.0 - self.beta) / (effective_num + 1e-6)
        class_weights = class_weights / class_weights.sum() * self.num_classes

        ce_loss = self.ce_loss(logits, labels)
        cb_loss = class_weights[labels] * ce_loss
        return cb_loss.mean()