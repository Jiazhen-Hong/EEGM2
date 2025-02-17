
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class BENDREncoder(nn.Module):
#     """ BENDR Encoder: 1D CNN 层用于下采样 EEG 信号 """
#     def __init__(self, in_channels=14, encoder_dim=1536):
#         super(BENDREncoder, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv1d(in_channels, 512, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(512),
#             nn.GELU(),

#             nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(512),
#             nn.GELU(),

#             nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(512),
#             nn.GELU(),

#             nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(512),
#             nn.GELU(),

#             nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(512),
#             nn.GELU(),

#             nn.Conv1d(512, encoder_dim, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(encoder_dim),
#             nn.GELU(),
#         )

#     def forward(self, x):
#         return self.conv_layers(x)

# class BENDRTransformer(nn.Module):
#     """ BENDR Transformer 负责时间序列建模 + LayerDrop """
#     def __init__(self, embed_dim=1536, num_layers=8, num_heads=8, ff_dim=3076, layer_drop=0.1):
#         super(BENDRTransformer, self).__init__()
#         self.layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, activation='gelu', batch_first=True)
#             for _ in range(num_layers)
#         ])
#         self.layer_drop = layer_drop

#     def forward(self, x):
#         for layer in self.layers:
#             if not self.training or torch.rand(1).item() >= self.layer_drop:
#                 x = layer(x)
#         return x

# class BENDR(nn.Module):
#     """ BENDR: CNN Encoder + Transformer + 分类头 """
#     def __init__(self, in_channels=14, encoder_dim=1536, num_layers=8, num_heads=8, ff_dim=3076, num_classes=2, pretrain=False):
#         super(BENDR, self).__init__()
#         self.encoder = BENDREncoder(in_channels, encoder_dim)
#         self.transformer = BENDRTransformer(encoder_dim, num_layers, num_heads, ff_dim)

#         self.pretrain = pretrain  # 是否是预训练模式

#         if not pretrain:
#             self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_dim))
#             self.classifier = nn.Linear(encoder_dim, num_classes)
#         else:
#             self.decoder = nn.Sequential(
#                 nn.ConvTranspose1d(encoder_dim, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
#                 nn.BatchNorm1d(512),
#                 nn.GELU(),
#                 nn.ConvTranspose1d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
#                 nn.BatchNorm1d(256),
#                 nn.GELU(),
#                 nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
#                 nn.BatchNorm1d(128),
#                 nn.GELU(),
#                 nn.ConvTranspose1d(128, 14, kernel_size=3, stride=2, padding=1, output_padding=1),
#                 nn.Sigmoid(),  # 归一化输出
#             )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = x.permute(0, 2, 1)  # 变成 (batch, seq_len, encoder_dim)
#         x = self.transformer(x)

#         if self.pretrain:
#             x = x.permute(0, 2, 1)  # 变回 (batch, encoder_dim, seq_len//32)
#             return self.decoder(x)  # 反卷积恢复原始信号
        
#         cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)  # 在第一维度添加 CLS Token
#         x = self.transformer(x)
#         x = x[:, 0, :]  # 取 CLS Token 的输出
#         return self.classifier(x)

# import os
# import logging
# import torch.optim as optim
# import os
# import logging
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from sklearn.metrics import roc_auc_score 

# class BENDRTrainer:
#     def __init__(self, train_loader, val_loader, test_loader, device, result_path, pre_epoch=50):
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.test_loader = test_loader
#         self.device = device
#         self.result_path = result_path
#         self.logger = logging.getLogger(__name__)

#         # 配置
#         self.num_pretrain_epochs = pre_epoch  # 预训练 50 轮
#         self.num_finetune_epochs = 10  # 微调 10 轮
#         self.learning_rate = 0.001
#         self.mask_ratio = 0.3  # 预训练时的掩码比例

#         # 预训练和微调的存储路径
#         self.pretrain_model_path = os.path.join(self.result_path, "best_pretrain_model.pth")
#         self.finetune_model_path = os.path.join(self.result_path, "best_finetune_model.pth")

#     def apply_mask(self, data):
#         """ 在预训练时应用随机掩码 """
#         mask = torch.rand_like(data) > self.mask_ratio  # 0.3 的概率被掩码
#         masked_data = data * mask  # 只保留未被掩码的部分
#         return masked_data, mask

#     def pretrain(self):
#         """ 第 1 阶段：BENDR 预训练 """
#         model = BENDR(num_classes=2).to(self.device)  # **这里关键，不加分类头**
#         optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
#         criterion = nn.MSELoss()
#         best_loss = float("inf")

#         for epoch in range(self.num_pretrain_epochs):
#             model.train()
#             running_loss = 0.0

#             for batch in self.train_loader:
#                 if isinstance(batch, (tuple, list)):  
#                     data = batch[0]  # 只获取数据部分
#                 else:
#                     data = batch
                
#                 data = data.to(self.device)  # 确保数据转移到 GPU

#                 # 应用掩码
#                 masked_data, mask = self.apply_mask(data)

#                 optimizer.zero_grad()
#                 reconstructed = model(masked_data)  # **这里应该输出完整的 EEG 表示，而不是分类结果**

#                 # # **确保 reconstructed 形状匹配 data**
#                 if reconstructed.shape != data.shape:
#                     self.logger.error(f"Shape mismatch: reconstructed {reconstructed.shape}, data {data.shape}")
#                     continue  # 跳过当前 batch 避免崩溃

#                 # **只计算被掩码部分的损失**
#                 loss = criterion(reconstructed * mask, data * mask)
#                 loss.backward()
#                 optimizer.step()

#                 running_loss += loss.item()

#             avg_loss = running_loss / len(self.train_loader)
#             self.logger.info(f"Pretrain Epoch [{epoch + 1}/{self.num_pretrain_epochs}], Loss: {avg_loss:.4f}")

#             if avg_loss < best_loss:
#                 best_loss = avg_loss
#                 torch.save(model.state_dict(), self.pretrain_model_path)
#                 self.logger.info(f"Saved best pretrain model with Loss: {best_loss:.4f}")

#         self.logger.info("Pretraining completed.")

#     def finetune(self):
#         """ 第 2 阶段：BENDR 分类微调 """
#         model = BENDR(num_classes=2).to(self.device)  # **这里加上分类头**
#         optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
#         criterion = nn.CrossEntropyLoss()
#         best_val_loss = float("inf")

#         # 加载预训练的 Encoder
#         pretrained_dict = torch.load(self.pretrain_model_path)
#         classifier_dict = model.state_dict()
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in classifier_dict}
#         classifier_dict.update(pretrained_dict)
#         model.load_state_dict(classifier_dict)

#         self.logger.info("Loaded pretrained BENDR Encoder for classification task.")

#         for epoch in range(self.num_finetune_epochs):
#             model.train()
#             running_loss = 0.0

#             for data, labels in self.train_loader:
#                 data, labels = data.to(self.device), labels.to(self.device)
#                 optimizer.zero_grad()

#                 classification_output = model(data)
#                 loss = criterion(classification_output, labels)
#                 loss.backward()
#                 optimizer.step()

#                 running_loss += loss.item()

#             avg_loss = running_loss / len(self.train_loader)
#             self.logger.info(f"Fine-tune Epoch [{epoch + 1}/{self.num_finetune_epochs}], Loss: {avg_loss:.4f}")

#             # 计算验证集损失
#             model.eval()
#             val_loss = 0.0
#             with torch.no_grad():
#                 for data, labels in self.val_loader:
#                     data, labels = data.to(self.device), labels.to(self.device)
#                     classification_output = model(data)
#                     loss = criterion(classification_output, labels)
#                     val_loss += loss.item()

#             val_loss /= len(self.val_loader)
#             self.logger.info(f"Validation Loss after epoch {epoch + 1}: {val_loss:.4f}")

#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 torch.save(model.state_dict(), self.finetune_model_path)
#                 self.logger.info(f"Saved best fine-tune model with Validation Loss: {best_val_loss:.4f}")

#         self.logger.info("Fine-tuning completed.")

#     def test(self):
#         """ 测试模型性能 """
#         model = BENDR(num_classes=2).to(self.device)
#         model.load_state_dict(torch.load(self.finetune_model_path))
#         model.eval()
#         self.logger.info("Loaded best fine-tune model for testing.")

#         correct, total = 0, 0
#         all_labels, all_probs = [], []

#         with torch.no_grad():
#             for data, labels in self.test_loader:
#                 data, labels = data.to(self.device), labels.to(self.device)
#                 classification_output = model(data)
#                 probs = torch.softmax(classification_output, dim=1)[:, 1]  # 二分类
#                 _, predicted = torch.max(classification_output.data, 1)

#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#                 all_labels.extend(labels.cpu().numpy())
#                 all_probs.extend(probs.cpu().numpy())

#         test_accuracy = 100 * correct / total
#         test_auroc = roc_auc_score(all_labels, all_probs)

#         self.logger.info(f"Test Set Accuracy: {test_accuracy:.2f}%")
#         self.logger.info(f"Test Set AUROC: {test_auroc:.4f}")



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import logging
from sklearn.metrics import roc_auc_score

# --------------------------------------
# **BENDR Encoder (CNN 下采样层)**
# --------------------------------------
class BENDREncoder(nn.Module):
    def __init__(self, in_channels=14, encoder_dim=1536):
        super(BENDREncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.Conv1d(512, encoder_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(encoder_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv_layers(x)

# --------------------------------------
# **BENDR Transformer**
# --------------------------------------
class BENDRTransformer(nn.Module):
    def __init__(self, embed_dim=1536, num_layers=8, num_heads=8, ff_dim=3076, layer_drop=0.1):
        super(BENDRTransformer, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, activation='gelu', batch_first=True)
            for _ in range(num_layers)
        ])
        self.layer_drop = layer_drop

    def forward(self, x):
        for layer in self.layers:
            if not self.training or torch.rand(1).item() >= self.layer_drop:
                x = layer(x)
        return x

# --------------------------------------
# **完整的 BENDR 模型**
# --------------------------------------
class BENDR(nn.Module):
    def __init__(self, in_channels=14, encoder_dim=1536, num_layers=8, num_heads=8, ff_dim=3076, num_classes=None):
        super(BENDR, self).__init__()
        self.encoder = BENDREncoder(in_channels, encoder_dim)
        self.transformer = BENDRTransformer(encoder_dim, num_layers, num_heads, ff_dim)

        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Linear(encoder_dim, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.transformer(x)

        if self.num_classes is None:
            return x
        return self.classifier(x[:, 0, :])

# --------------------------------------
# **BENDR Trainer**
# --------------------------------------
class BENDRTrainer:
    def __init__(self, train_loader, val_loader, test_loader, device, result_path, pre_epoch=50):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.result_path = result_path
        self.logger = logging.getLogger(__name__)

        self.num_pretrain_epochs = pre_epoch
        self.num_finetune_epochs = 10
        self.learning_rate = 0.001
        self.mask_ratio = 0.3

        self.pretrain_model_path = os.path.join(self.result_path, "best_pretrain_model.pth")
        self.finetune_model_path = os.path.join(self.result_path, "best_finetune_model.pth")

    def apply_mask(self, data):
        """ 生成 mask, 使其适用于 Transformer 输入 """
        batch_size, channels, seq_len = data.shape  # [64, 14, 256]

        # **计算经过 CNN Encoder 下采样后的序列长度**
        reduced_seq_len = seq_len // 32  # 256 -> 8

        # **在 Transformer 维度生成 mask**
        mask = torch.rand((batch_size, reduced_seq_len), device=data.device) > self.mask_ratio  # Shape: [64, 8]

        return data, mask  # `mask` 现在是 `[batch_size, 8]`

    def contrastive_loss(self, z, mask):
        """ 对比学习损失 (仅在 `seq_len//32` 维度进行 Mask) """
        batch_size, seq_len, dim = z.shape  # [64, 8, 1536]
        z = F.normalize(z, dim=-1)  # 归一化

        # **确保 mask 维度正确**
        mask = mask.unsqueeze(-1).expand(-1, -1, dim)  # Shape: [64, 8, 1536]

        # **只在 mask 位置计算 loss**
        pos_pairs = z[mask].view(-1, dim)  # 保证是 2D: [N_pos, 1536]
        neg_pairs = z[~mask].view(-1, dim)  # 保证是 2D: [N_neg, 1536]

        if pos_pairs.shape[0] == 0 or neg_pairs.shape[0] == 0:
            return torch.tensor(0.0, device=z.device)

        pos_sim = torch.exp(torch.sum(pos_pairs * pos_pairs, dim=-1) / 0.1)
        
        # **修正 `neg_sim` 计算**
        neg_sim = torch.exp(torch.matmul(pos_pairs, neg_pairs.T) / 0.1)  # 确保是 2D 矩阵

        loss = -torch.log(pos_sim / (pos_sim + torch.sum(neg_sim, dim=-1)))
        return loss.mean()

    def pretrain(self):
        model = BENDR(num_classes=None).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
        best_loss = float("inf")

        for epoch in range(self.num_pretrain_epochs):
            model.train()
            running_loss = 0.0

            for batch in self.train_loader:
                data = batch[0].to(self.device) if isinstance(batch, (tuple, list)) else batch.to(self.device)
                optimizer.zero_grad()
                masked_data, mask = self.apply_mask(data)
                z = model(masked_data)
                loss = self.contrastive_loss(z, mask)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            self.logger.info(f"Pretrain Epoch [{epoch + 1}/{self.num_pretrain_epochs}], Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), self.pretrain_model_path)

        self.logger.info("Pretraining completed.")

    def finetune(self):
        model = BENDR(num_classes=2).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        best_val_loss = float("inf")

        pretrained_dict = torch.load(self.pretrain_model_path)
        classifier_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in classifier_dict}
        classifier_dict.update(pretrained_dict)
        model.load_state_dict(classifier_dict)

        for epoch in range(self.num_finetune_epochs):
            model.train()
            running_loss = 0.0

            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                classification_output = model(data)
                loss = criterion(classification_output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            self.logger.info(f"Fine-tune Epoch [{epoch + 1}/{self.num_finetune_epochs}], Loss: {avg_loss:.4f}")

            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                torch.save(model.state_dict(), self.finetune_model_path)

        self.logger.info("Fine-tuning completed.")

    def test(self):
        model = BENDR(num_classes=2).to(self.device)
        model.load_state_dict(torch.load(self.finetune_model_path))
        model.eval()

        correct, total = 0, 0
        all_labels, all_probs = [], []

        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                classification_output = model(data)
                probs = torch.softmax(classification_output, dim=1)[:, 1]
                _, predicted = torch.max(classification_output.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        test_accuracy = 100 * correct / total
        test_auroc = roc_auc_score(all_labels, all_probs)
        self.logger.info(f"Test Set Accuracy: {test_accuracy:.2f}%")
        self.logger.info(f"Test Set AUROC: {test_auroc:.4f}")