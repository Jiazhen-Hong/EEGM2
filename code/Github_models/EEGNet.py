import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, in_channels=22, seq_len=1024, num_classes=2):
        super(EEGNet, self).__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len

        # **🔹 Layer 1: 修改 `in_channels` 以适配 EEG 多通道输入**
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=(1, 64), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # **🔹 Layer 2: Depthwise Conv2D + Pooling**
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))  
        self.conv2 = nn.Conv2d(16, 4, (1, 32))  # **注意这里 `in_channels=16`**
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d((1, 4))

        # **🔹 Layer 3: Separable Conv2D + Pooling**
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (1, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((1, 4))

        # **🔹 FC Layer (动态计算输入大小)**
        self._compute_fc_size()
        self.fc1 = nn.Linear(self.final_fc_size, num_classes)  

    def _compute_fc_size(self):
        """ 计算全连接层的输入大小 """
        with torch.no_grad():
            x = torch.rand(1, self.in_channels, 1, self.seq_len)  # [Batch, Channels, 1, seq_len]
            x = self._forward_features(x)
            self.final_fc_size = x.numel()  # 计算展平后的维度

    def _forward_features(self, x):
        """ 计算前几层的特征图大小 """
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)

        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.sigmoid(self.fc1(x))
        return x

# **🔹 测试不同 `seq_len` 和 `in_channels`**
if __name__ == "__main__":
    model = EEGNet(in_channels=22, seq_len=1024, num_classes=2)  # 22 通道, 1024 时间点
    print(model)

    # 测试输入数据
    x = torch.rand(4, 22, 1, 1024)  # [Batch=4, Channels=22, 1, Sequence Length=1024]
    y = model(x)
    print(f"Output shape: {y.shape}")  # 输出 (4, 2)