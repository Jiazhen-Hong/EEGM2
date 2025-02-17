import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
from sklearn.metrics import roc_auc_score
from Github_models.MAEEG import MAEEGClassification, MAEEGReconstruction


class MAEEGTrainer:
    def __init__(self, train_loader, val_loader, test_loader, device, result_path, pre_epoch):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.result_path = result_path
        self.logger = logging.getLogger(__name__)

        # 配置
        self.num_pretrain_epochs = pre_epoch  # 预训练 50 轮
        self.num_finetune_epochs = 10  # 微调 50 轮
        self.learning_rate = 0.001

        # 预训练和微调的存储路径
        self.pretrain_model_path = os.path.join(self.result_path, "best_pretrain_model.pth")
        self.finetune_model_path = os.path.join(self.result_path, "best_finetune_model.pth")

    def pretrain(self):
        """ 第 1 阶段：自监督预训练 (EEG Reconstruction) """
        model = MAEEGReconstruction(
            input_channel=14,
            embed_size=128,
            downsample_size=[2, 2, 2],
            kernel_size=[3, 3, 3],
            dropout=0.1,
            transformer_embed_size=128,
            heads=8,
            forward_neuron=256,
            num_transformers=4
        ).to(self.device)

        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        best_loss = float("inf")

        for epoch in range(self.num_pretrain_epochs):
            model.train()
            running_loss = 0.0

            for batch in self.train_loader:
                if isinstance(batch, (tuple, list)):  
                    data = batch[0]  # 只获取数据部分
                else:
                    data = batch
                
                data = data.to(self.device)  # 确保数据转移到 GPU

                optimizer.zero_grad()
                reconstructed = model(data)
                loss = criterion(reconstructed, data)  # 计算重构损失
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            self.logger.info(f"Pretrain Epoch [{epoch + 1}/{self.num_pretrain_epochs}], Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), self.pretrain_model_path)
                self.logger.info(f"Saved best pretrain model with Loss: {best_loss:.4f}")

        self.logger.info("Pretraining completed.")

    def finetune(self):
        """ 第 2 阶段：分类微调 """
        model = MAEEGClassification(num_classes=2).to(self.device)  # 假设分类 5 个类别
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        best_val_accuracy = 0.0

        # 加载预训练的 Encoder
        pretrained_dict = torch.load(self.pretrain_model_path)
        classifier_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in classifier_dict}
        classifier_dict.update(pretrained_dict)
        model.load_state_dict(classifier_dict)

        self.logger.info("Loaded pretrained MAEEG Encoder for classification task.")

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

            self.logger.info(f"Fine-tune Epoch [{epoch + 1}/{self.num_finetune_epochs}], Loss: {running_loss / len(self.train_loader):.4f}")

            # 计算验证集准确率
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for data, labels in self.val_loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    classification_output = model(data)
                    _, predicted = torch.max(classification_output.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total
            self.logger.info(f"Validation Accuracy after epoch {epoch + 1}: {val_accuracy:.2f}%")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), self.finetune_model_path)
                self.logger.info(f"Saved best fine-tune model with Validation Accuracy: {best_val_accuracy:.2f}%")

        self.logger.info("Fine-tuning completed.")

    def test(self):
        """ 测试模型性能 """
        model = MAEEGClassification().to(self.device)
        model.load_state_dict(torch.load(self.finetune_model_path))
        model.eval()
        self.logger.info("Loaded best fine-tune model for testing.")

        correct, total = 0, 0
        all_labels, all_probs = [], []

        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                classification_output = model(data)
                probs = torch.softmax(classification_output, dim=1)[:, 1]  # 二分类
                _, predicted = torch.max(classification_output.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        test_accuracy = 100 * correct / total
        test_auroc = roc_auc_score(all_labels, all_probs)

        self.logger.info(f"Test Set Accuracy: {test_accuracy:.2f}%")
        self.logger.info(f"Test Set AUROC: {test_auroc:.4f}")
        print(f"Test Set Accuracy: {test_accuracy:.2f}%")
        print(f"Test Set AUROC: {test_auroc:.4f}")

